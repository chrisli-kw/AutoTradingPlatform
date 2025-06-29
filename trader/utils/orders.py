import time
import logging
import pandas as pd
from datetime import datetime
from collections import namedtuple

from ..config import API, Cost
from . import concat_df
from .objects.data import TradeData
from .positions import FuturesMargin, TradeDataHandler
from .database import db
from .database.tables import TradingStatement


class OrderTool(FuturesMargin):
    def __init__(self, account_name: str):
        super().__init__()
        self.account_name = account_name
        self.OrderInfo = namedtuple(
            typename="OrderInfo",
            field_names=[
                'action_type',
                'action', 'target', 'quantity',
                'order_cond', 'octype', 'pos_target',
                'pos_balance', 'daytrade_short', 'reason'
            ],
            defaults=['', '', '', 0, '', '', 0, 0, False, '']
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

    @staticmethod
    def get_cost_price(target: str, price: float, order_cond: str):
        '''取得股票的進場價'''

        if order_cond == 'ShortSelling':
            return price

        if target in TradeData.Stocks.Info.code.values:
            cost_price = TradeData.Stocks.Info.set_index(
                'code').cost_price.to_dict()
            return cost_price[target]
        return 0

    def _account_id(self, content, market='Stocks'):
        is_real_trade = isinstance(content, dict)
        if is_real_trade:
            action = content['action'] if is_real_trade else content.action
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

    def generate_data(self, target: str, content, market='Stocks'):
        is_stock = market == 'Stocks'
        is_real_trade = isinstance(content, dict)
        action = content['action'] if is_real_trade else content.action

        if is_real_trade:
            order_cond = content.get('order_cond', '')
            op_type = content.get('oc_type', '')
            price = content.get('price', 0)
        else:
            order_cond = content.order_cond
            op_type = content.octype
            price = 0

        if price == 0:
            price = TradeDataHandler.getQuotesNow(target)['price']

        sign = self.sign_(action if is_stock else op_type)

        if is_stock:
            quantity = self.get_stock_quantity(content, market)
            amount = self.get_stock_amount(
                target, price*sign, quantity, order_cond)
        else:
            if is_real_trade:
                quantity = content['quantity']
            else:
                quantity = self.get_sell_quantity(content, market)
            amount = self.get_open_margin(target, quantity)*sign

        order_data = {
            'Time': datetime.now(),
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
            'msg': '' if is_real_trade else content.reason
        }
        return order_data

    def check_leverage(self, target: str, mode='long'):
        '''取得個股的融資/融券成數'''
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

    def get_stock_quantity(self, content: namedtuple, market='Stocks'):
        if isinstance(content, dict):
            quantity = content['quantity']
            if content['action'] == 'Buy' and content['order_lot'] == 'Common':
                quantity *= 1000
        else:
            quantity = 1000*self.get_sell_quantity(content, market)
        return quantity

    def get_sell_quantity(self, content: namedtuple, market: str = 'Stocks'):
        '''根據庫存, 剩餘部位比例, 賣出比例，反推賣出量(張)'''
        if market == 'Stocks':
            q_before = int(content.quantity/1000)
            condition = content.pos_balance > 0 and content.quantity >= 1000
        else:
            q_before = content.quantity
            condition = content.pos_balance > 0 and content.quantity != 0

        if condition:
            quantity = int(q_before/content.pos_balance *
                           abs(content.pos_target))
            quantity = 1000*max(min(quantity, q_before), 1)
            return max(round(quantity/1000), 1)
        return q_before

    def check_order_status(self, order_result, market: str = 'Stocks'):
        '''確認委託狀態'''
        time.sleep(0.1)
        if market == 'Stocks':
            API.update_status(API.stock_account)
        else:
            API.update_status(API.futopt_account)
        status = order_result.status.status
        if status not in ['PreSubmitted', 'Filled']:
            msg = order_result.status.msg
            logging.warning(f'Order not submitted/filled: {msg}')

    def appendOrder(self, target: str, content, market='Stocks'):
        '''Add new order data to OrderTable'''

        if isinstance(content, dict):
            order_cond = content.get('order_cond', '')
        else:
            order_cond = content.order_cond

        order_data = self.generate_data(target, content, market)
        order_data['leverage'] = self.check_leverage(target, order_cond)

        self.OrderTable = concat_df(
            self.OrderTable,
            pd.DataFrame([order_data])
        )
        return order_data

    def deleteOrder(self, code: str):
        '''Delete order data from OrderTable'''
        self.OrderTable = self.OrderTable[self.OrderTable.code != code]

    def checkEnoughToPlace(self, market: str, target: str):
        '''Check if current placed amount is under target limit.'''
        df = self.OrderTable[self.OrderTable.market == market]
        return df.amount.sum() < target

    def filterOrderTable(self, market: str):
        '''Filter OrderTable by market'''
        return self.OrderTable[self.OrderTable.market == market].copy()

    def output_statement(self):
        '''Export trading statement'''

        if db.HAS_DB:
            if 'order_cond' in self.OrderTable.columns:
                self.OrderTable.order_cond.fillna('', inplace=True)
            if 'order_lot' in self.OrderTable.columns:
                self.OrderTable.order_lot.fillna('', inplace=True)
            if 'leverage' in self.OrderTable.columns:
                self.OrderTable.leverage.fillna(-1, inplace=True)
            if 'op_type' in self.OrderTable.columns:
                self.OrderTable.op_type.fillna('', inplace=True)
            db.dataframe_to_DB(self.OrderTable, TradingStatement)

    def read_statement(self, account: str = ''):
        '''Import trading statement'''

        if db.HAS_DB:
            df = db.query(
                TradingStatement,
                TradingStatement.account_id == account
            )
        else:
            df = self.OrderTable

        df = df.drop_duplicates()
        return df

    def update_pos_target(self, order: namedtuple, is_empty: False):
        if is_empty and order.action_type == 'Close':
            position = order.pos_target
            if position == 100 or position >= order.pos_balance:
                order = order._replace(pos_target=100)

        return order
