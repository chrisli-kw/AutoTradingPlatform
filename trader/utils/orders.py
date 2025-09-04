import time
import logging
import pandas as pd
from shioaji import constant, contracts
from datetime import datetime
from collections import namedtuple

from . import get_contract
from .objects.data import TradeData
from .callback import CallbackHandler
from .positions import FuturesMargin, TradeDataHandler, WatchListTool
from .database import db
from .database.tables import TradingStatement, SecurityInfo
from ..config import API, TimeSimTradeStockEnd, StrategyList, Cost


class OrderTool(FuturesMargin):
    def __init__(self, account_name: str):
        super().__init__()
        self.WatchListTool = WatchListTool(account_name=account_name)
        self.account_name = account_name
        self.OrderInfo = namedtuple(
            typename="OrderInfo",
            field_names=[
                'action_type',
                'action', 'target', 'quantity',
                'order_cond', 'octype',
                'daytrade_short', 'reason'
            ],
            defaults=['', '', '', 0, '', '', False, '']
        )
        self.MsgOrder = namedtuple(
            typename='MsgOrder',
            field_names=['operation', 'order', 'status', 'contract']
        )
        self.OrderTable = pd.DataFrame(columns=[
            'Time', 'market', 'code', 'action',
            'price', 'quantity', 'amount',
            'order_cond', 'order_lot', 'leverage',
            'op_type', 'account_id', 'msg',

        ])

    @staticmethod
    def is_new_order(operation: dict):
        return operation['op_code'] == '00' or operation['op_msg'] == ''

    @staticmethod
    def is_cancel_order(operation: dict):
        return operation['op_type'] == 'Cancel'

    @staticmethod
    def is_insufficient_quota(operation: dict):
        return operation['op_code'] == '88' and '此證券配額張數不足' in operation['op_msg']

    @staticmethod
    def sign_(action: str):
        if action in ['Sell', 'Cover']:
            return -1
        return 1

    def get_cost_price(self, target: str, price: float, order_cond: str):
        '''取得股票的進場價'''

        if order_cond == 'ShortSelling':
            return price

        df = db.query(SecurityInfo, self.WatchListTool.match_target(target))
        cost_price = df.cost_price.values[0] if not df.empty else 0
        return cost_price

    @staticmethod
    def content_attr(content: namedtuple, attr: str, default=None):
        '''Get attribute from content namedtuple or dict'''
        if isinstance(content, dict):
            return content.get(attr, default)
        return getattr(content, attr, default)

    def _account_id(self, content, market='Stocks'):
        action = self.content_attr(content, 'action')
        if isinstance(content, dict):
            if market == 'Stocks' and action == 'Sell':
                return content['account_id']
            return content['account']['account_id']
        return f'simulate-{self.account_name}'

    def _order_lot(self, content):
        if isinstance(content, dict):
            return content.get('order_lot', '')
        elif content.quantity < 1000:
            return 'IntradayOdd'
        return 'Common'

    def generate_data(self, target: str, content: namedtuple):
        config = TradeDataHandler.getStrategyConfig(target)
        if config is None:
            return

        market = config.market
        is_real_trade = isinstance(content, dict)

        action = self.content_attr(content, 'action')
        order_cond = self.content_attr(content, 'order_cond', default='')
        op_type = self.content_attr(content, 'oc_type', default='')
        price = content.get('price', 0) if is_real_trade else 0
        if price == 0:
            price = TradeDataHandler.getQuotesNow(target)['price']

        is_stock = market == 'Stocks'
        sign = self.sign_(action if is_stock else op_type)

        if is_stock:
            quantity = self.get_stock_quantity(content)
            amount = self.get_stock_amount(
                target, price*sign, quantity, order_cond)
        else:
            if is_real_trade:
                quantity = content['quantity']
            else:
                quantity = self.get_sell_quantity(content)
            amount = self.get_open_margin(target, quantity)*sign

        order_data = {
            'Time': datetime.now(),
            'mode': TradeData.Account.Mode,
            'market': market,
            'code': target,
            'action': action,
            'price': price*sign,
            'quantity': quantity,
            'amount': amount,
            'order_cond': order_cond,
            'order_lot': self._order_lot(content),
            'op_type': op_type,
            'account_id': self._account_id(content, market),
            'msg': self.content_attr(content, 'reason', default='')
        }
        return order_data

    def check_leverage(self, target: str, mode='long'):
        '''取得個股的融資/融券成數'''

        # strategy = TradeData.Securities.Strategy.get(target)
        # conf = StrategyList.Config.get(strategy)
        # mode = conf.mode

        if mode in ['long', 'MarginTrading']:
            return TradeData.Stocks.Leverage.Long.get(target, 0)
        elif mode in ['short', 'ShortSelling']:
            return 1 - TradeData.Stocks.Leverage.Short.get(target, 0)
        return 0

    def get_stock_amount(self, target: str, price: float, quantity: int, mode='long'):
        '''Calculate the amount of stock orders.'''

        leverage = self.check_leverage(target, mode)

        if price < 0:
            price = -price
            cost_price = self.get_cost_price(target, price, mode)
            return -(price - cost_price*leverage)*quantity*(1 - Cost.STOCK_FEE_RATE)

        fee = max(price*quantity*Cost.STOCK_FEE_RATE, 20)
        return price*quantity*(1 - leverage) + fee

    def get_stock_quantity(self, content: namedtuple):
        if isinstance(content, dict):
            quantity = content['quantity']
            if content['action'] == 'Buy' and content['order_lot'] == 'Common':
                quantity *= 1000
        else:
            quantity = 1000*self.get_sell_quantity(content)
        return quantity

    def get_sell_quantity(self, content: namedtuple):
        contract = get_contract(content.target)
        if content.quantity >= 1000 and isinstance(contract, contracts.Stock):
            return content.quantity/1000
        return content.quantity

    def check_order_status(self, order_result, is_stock: bool = True):
        '''確認委託狀態'''
        time.sleep(0.1)

        if is_stock:
            API.update_status(API.stock_account)
        else:
            API.update_status(API.futopt_account)
        status = order_result.status.status
        if status not in ['PreSubmitted', 'Filled']:
            msg = order_result.status.msg
            logging.warning(f'Order not submitted/filled: {msg}')

    def appendOrder(self, target: str, content: namedtuple):
        '''Add new order data to OrderTable'''

        order_cond = self.content_attr(content, 'order_cond', default='Cash')
        order_data = self.generate_data(target, content)
        order_data['leverage'] = self.check_leverage(target, order_cond)

        db.add_data(TradingStatement, **order_data)
        return order_data

    def deleteOrder(self, code: str):
        '''Delete order data from OrderTable'''
        self.OrderTable = self.OrderTable[self.OrderTable.code != code]

    def checkEnoughToPlace(self, target: str):
        '''Check if current placed amount is under target limit.'''

        conf = TradeDataHandler.getStrategyConfig(target)
        df = self.OrderTable[self.OrderTable.market == conf.market]

        if conf.market == 'Stocks':
            return df.amount.sum() < TradeData.Account.DesposalMoney
        return df.amount.sum() < TradeData.Account.DesposalMargin

    def filterOrderTable(self, market: str):
        '''Filter OrderTable by market'''
        return self.OrderTable[self.OrderTable.market == market].copy()

    def place_order(self, content: namedtuple):
        logging.debug(f'[OrderState.Content|{content}|')

        target = content.target

        if target not in TradeData.BidAsk:
            return

        contract = TradeData.Contracts.get(target, get_contract(target))
        is_stock = isinstance(contract, contracts.Stock)
        quantity = self.get_sell_quantity(content)
        price_type = 'MKT'
        price = 0
        order_lot = 'IntradayOdd' if content.quantity < 1000 and is_stock else 'Common'

        if is_stock:
            bid_ask = TradeData.BidAsk[target]
            bid_ask = bid_ask.bid_price if content.action == 'Sell' else bid_ask.ask_price

            # 零股交易
            if 0 < content.quantity < 1000:
                price_type = 'LMT'
                price = bid_ask[1]

            # 整股交易
            else:
                if datetime.now() >= TimeSimTradeStockEnd:
                    price_type = 'LMT'
                    price = bid_ask[1]
                elif target in TradeData.Stocks.Punish:
                    price_type = 'LMT'
                    price = bid_ask[3]
                elif contract.exchange == 'OES':
                    price_type = 'LMT'
                    price = TradeDataHandler.getQuotesNow(target)['price']

        # 下單
        if TradeData.Account.Simulate:
            order_data = self.appendOrder(target, content)
            return order_data

        else:
            # #ff0000 批次下單的張數 (股票>1000股的單位為【張】) #ff0000
            q = 5 if order_lot == 'Common' else quantity
            enough_to_place = self.checkEnoughToPlace(target)
            while quantity > 0 and enough_to_place:
                order = API.Order(
                    # 價格 (市價單 = 0)
                    price=price,
                    # 數量 (最小1張; 零股最小50股 or 全部庫存)
                    quantity=min(quantity, q),
                    # 動作: 買進/賣出
                    action=content.action,
                    # 市價單/限價單
                    price_type=price_type,
                    # ROD:當天都可成交
                    order_type=constant.OrderType.ROD if is_stock else constant.OrderType.IOC,
                    # 委託類型: 現股/融資
                    order_cond=content.order_cond if is_stock else 'Cash',
                    # 整張或零股
                    order_lot=order_lot,
                    # {Auto, New, Cover, DayTrade}(自動、新倉、平倉、當沖)
                    octype='Auto' if is_stock else content.octype,
                    account=API.stock_account if is_stock else API.futopt_account,
                    # 先賣後買: True, False
                    daytrade_short=content.daytrade_short,
                )
                result = API.place_order(contract, order)
                # self.check_order_status(result, is_stock)
                quantity -= q

    def StockOrder(self, msg: dict):
        stock = msg['contract']['code']
        order = msg['order']
        operation = msg['operation']

        c3 = order['action'] == 'Buy'
        TradeDataHandler.update_deal_list(stock, order['action'])

        leverage = self.check_leverage(stock, order['order_cond'])

        if self.is_new_order(operation) and c3:
            self.appendOrder(stock, order)

        # 若融資配額張數不足，改現股買進 ex: '此證券配額張數不足，餘額 0 張（證金： 0 ）'
        elif self.is_insufficient_quota(operation):
            q_balance = operation['op_msg'].split(' ')
            if len(q_balance) > 1:
                q_balance = int(q_balance[1])
                infos = dict(action=order['action'], target=stock)

                # 若本日還沒有下過融資且剩餘券數為0，才可以改下現股
                if q_balance == 0 and stock not in TradeData.Stocks.Bought:
                    orderinfo = self.OrderInfo(
                        quantity=1000 *
                        int(order['quantity']*(1-leverage)),
                        order_cond='Cash',
                        **infos
                    )
                    self.place_order(orderinfo)

                elif q_balance > 0:
                    orderinfo = self.OrderInfo(
                        quantity=q_balance,
                        order_cond=order['order_cond'],
                        **infos
                    )
                    self.place_order(orderinfo)

        # 若刪單成功就自清單移除
        if self.is_cancel_order(operation):
            self.deleteOrder(stock)
            if c3:
                TradeDataHandler.update_deal_list(stock, 'Cancel')

        # 更新監控庫存
        df = db.query(SecurityInfo, self.WatchListTool.match_target(stock))
        order['oc_type'] = 'New' if df.empty else 'Cover'
        self.WatchListTool.update_monitor(order['oc_type'], msg)

    def StockDeal(self, msg: dict):
        msg = CallbackHandler.update_stock_msg(msg)

        action = msg['action']
        if action == 'Sell':
            stock = msg['code']
            TradeDataHandler.update_deal_list(stock, action)
            self.appendOrder(stock, msg)

    def FuturesOrder(self, msg: dict):
        msg = CallbackHandler().update_futures_msg(msg)

        order = msg['order']
        symbol = CallbackHandler.fut_symbol(msg)
        operation = msg['operation']

        if self.is_new_order(operation):
            TradeDataHandler.update_deal_list(symbol, order['oc_type'])
            self.appendOrder(symbol, order)

        # 若刪單成功就自清單移除
        if self.is_cancel_order(operation):
            self.deleteOrder(symbol)
            TradeDataHandler.update_deal_list(symbol, 'Cancel')

        # 更新監控庫存
        self.WatchListTool.update_monitor(order['oc_type'], msg)
