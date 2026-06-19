import time
import logging
import pandas as pd
import shioaji as sj
from datetime import datetime
from collections import namedtuple

from . import get_contract
from .objects.data import TradeData
from .callback import CallbackHandler
from .options import OptionOrderFactory
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
                'daytrade_short', 'reason',
                'price', 'price_type', 'order_type',
                'combo_legs', 'order_label'
            ],
            defaults=[
                '', '', '', 0, '', '', False, '',
                None, None, None, None, ''
            ]
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
        self._auto_order_callbacks = []
        self._auto_order_fills = []
        self._futures_order_meta = {}
        self._option_order_labels = {}
        self.Options = OptionOrderFactory(self.OrderInfo)

    def option_order_info(self, *args, **kwargs):
        return self.Options.order_info(*args, **kwargs)

    def option_combo_order_info(self, *args, **kwargs):
        return self.Options.combo_order_info(*args, **kwargs)

    @staticmethod
    def _value(value):
        return getattr(value, 'value', value)

    @staticmethod
    def _is_retry_order_status(status: str):
        return status in ['Cancelled', 'Inactive', 'Failed']

    def _option_order_label(self, content):
        return self.content_attr(content, 'order_label', '') or ''

    def _set_option_order_status(
            self,
            label: str,
            status: str,
            content=None,
            result=None,
            msg: dict = None
    ):
        if not label:
            return

        current = TradeData.Futures.OptionOrderStatus.get(label, {})
        data = {
            **current,
            'label': label,
            'status': status,
            'updated_at': datetime.now(),
        }

        if content is not None:
            data.update({
                'target': self.content_attr(content, 'target', ''),
                'action': self.content_attr(content, 'action', ''),
                'quantity': self.content_attr(content, 'quantity', 0),
                'price': self.content_attr(content, 'price', None),
                'price_type': self.content_attr(content, 'price_type', None),
                'order_type': self.content_attr(content, 'order_type', None),
                'octype': self.content_attr(content, 'octype', ''),
                'combo_legs': self.content_attr(content, 'combo_legs', None),
                'reason': self.content_attr(content, 'reason', ''),
            })

        if result is not None:
            status_info = getattr(result, 'status', None)
            order = getattr(result, 'order', None)
            result_status = self._value(getattr(status_info, 'status', ''))
            data.update({
                'trade_status': result_status,
                'status_msg': getattr(status_info, 'msg', None),
                'deal_quantity': getattr(status_info, 'deal_quantity', None),
                'cancel_quantity': getattr(
                    status_info, 'cancel_quantity', None),
                'order_id': getattr(order, 'id', None),
                'seqno': getattr(order, 'seqno', None),
                'ordno': getattr(order, 'ordno', None),
            })

        if msg is not None:
            operation = msg.get('operation', {})
            order = msg.get('order', {})
            status_info = msg.get('status', {})
            data.update({
                'operation': operation.get('op_type'),
                'op_code': operation.get('op_code'),
                'op_msg': operation.get('op_msg'),
                'order_id': order.get('id', data.get('order_id')),
                'seqno': order.get('seqno', data.get('seqno')),
                'ordno': order.get('ordno', data.get('ordno')),
                'deal_quantity': status_info.get(
                    'deal_quantity', data.get('deal_quantity')),
                'cancel_quantity': status_info.get(
                    'cancel_quantity', data.get('cancel_quantity')),
            })

        TradeData.Futures.OptionOrderStatus[label] = data

    def _set_option_order_result_status(self, label: str, content, result):
        if not label or result is None:
            return

        status_info = getattr(result, 'status', None)
        status = self._value(getattr(status_info, 'status', ''))
        deal_quantity = getattr(status_info, 'deal_quantity', 0) or 0
        order_quantity = getattr(
            status_info,
            'order_quantity',
            self.content_attr(content, 'quantity', 0)
        ) or 0

        if status == 'Filled' or (
            order_quantity and deal_quantity >= order_quantity
        ):
            label_status = 'Filled'
        elif self._is_retry_order_status(status):
            label_status = 'Retry'
        else:
            label_status = 'Submitted'

        self._set_option_order_status(
            label,
            label_status,
            content=content,
            result=result
        )

    def _register_option_order_result(self, label: str, result):
        if not label or result is None:
            return

        order = getattr(result, 'order', None)
        for key in ['id', 'seqno', 'ordno']:
            value = getattr(order, key, None)
            if value:
                self._option_order_labels[value] = label

    @staticmethod
    def _order_key(target: str, action: str, oc_type: str, quantity: int):
        return (
            target,
            action or '',
            oc_type or '',
            int(quantity or 0)
        )

    def _register_auto_order(self, target: str, action: str, oc_type: str, quantity: int):
        key = self._order_key(target, action, oc_type, quantity)
        self._auto_order_callbacks.append((datetime.now(), key))
        self._auto_order_fills.append({
            'time': datetime.now(),
            'target': target,
            'action': action or '',
            'oc_type': oc_type or '',
            'remaining': int(quantity or 0),
        })

    def _prune_auto_orders(self, max_age: int = 60):
        now = datetime.now()
        self._auto_order_callbacks = [
            item for item in self._auto_order_callbacks
            if (now - item[0]).total_seconds() <= max_age
        ]
        self._auto_order_fills = [
            item for item in self._auto_order_fills
            if (
                (now - item['time']).total_seconds() <= max_age and
                item.get('remaining', 0) > 0
            )
        ]

    def _pop_auto_order(self, key: tuple):
        for i, item in enumerate(self._auto_order_callbacks):
            if item[1] == key:
                self._auto_order_callbacks.pop(i)
                return True
        return False

    def _pop_auto_order_like(self, target: str, action: str, oc_type: str):
        for i, item in enumerate(self._auto_order_callbacks):
            key = item[1]
            if key[:3] == (target, action or '', oc_type or ''):
                self._auto_order_callbacks.pop(i)
                return True
        return False

    def _consume_auto_order(self, target: str, order: dict):
        self._prune_auto_orders()

        action = order.get('action', '')
        oc_type = order.get('oc_type', '')
        key = self._order_key(
            target,
            action,
            oc_type,
            order.get('quantity', 0)
        )
        if self._pop_auto_order(key):
            return True

        stock_key = self._order_key(
            target,
            action,
            '',
            order.get('quantity', 0)
        )
        if self._pop_auto_order(stock_key):
            return True

        return (
            self._pop_auto_order_like(target, action, oc_type) or
            self._pop_auto_order_like(target, action, '')
        )

    def _consume_auto_order_fill(
            self,
            target: str,
            action: str,
            oc_type: str,
            quantity: int
    ):
        self._prune_auto_orders()

        quantity = int(quantity or 0)
        if quantity <= 0:
            return False

        for item in self._auto_order_fills:
            same_order = (
                item['target'] == target and
                item['action'] == (action or '') and
                item['oc_type'] in [oc_type or '', '']
            )
            if not same_order:
                continue

            item['remaining'] -= quantity
            if item['remaining'] <= 0:
                self._auto_order_fills.remove(item)
            return True

        return False

    @staticmethod
    def is_new_order(operation: dict):
        return operation['op_code'] == '00' or operation['op_msg'] == ''

    @staticmethod
    def is_new_order_submit(operation: dict):
        return (
            operation.get('op_type') == 'New' and
            operation.get('op_code') == '00' and
            operation.get('op_msg') == ''
        )

    @staticmethod
    def is_accepted_order_operation(operation: dict):
        return (
            operation.get('op_code') == '00' and
            operation.get('op_type') != 'Cancel'
        )

    @staticmethod
    def is_cancel_order(operation: dict):
        return operation['op_type'] == 'Cancel'

    def _set_futures_order_meta(self, msg: dict, target: str, is_auto_order: bool):
        order = msg['order']
        existing = self._get_futures_order_meta(order)
        order_label = existing.get('order_label', '')
        if not order_label:
            for key in ['id', 'seqno', 'ordno']:
                value = order.get(key)
                if value in self._option_order_labels:
                    order_label = self._option_order_labels[value]
                    break

        data = {
            'order_msg': msg,
            'target': target,
            'oc_type': order.get('oc_type', existing.get('oc_type', '')),
            'action': order.get('action', existing.get('action', '')),
            'quantity': order.get('quantity', existing.get('quantity', 0)),
            'filled_quantity': existing.get('filled_quantity', 0),
            'is_auto_order': existing.get('is_auto_order', is_auto_order),
            'order_label': order_label,
        }
        logging.info(f'[FuturesOrder.Callback]{msg}')
        for key in ['ordno', 'seqno', 'id']:
            value = order.get(key)
            if value:
                self._futures_order_meta[value] = data

        if order_label:
            operation = msg.get('operation', {})
            if operation.get('op_type') == 'Cancel':
                self._set_option_order_status(
                    order_label, 'Retry', msg=msg)
            elif operation.get('op_type') == 'Reject':
                self._set_option_order_status(
                    order_label, 'Retry', msg=msg)
            elif self.is_new_order_submit(operation):
                self._set_option_order_status(
                    order_label, 'Submitted', msg=msg)

    def _get_futures_order_meta(self, msg: dict):
        for key in ['ordno', 'seqno', 'id', 'trade_id']:
            value = msg.get(key)
            if value and value in self._futures_order_meta:
                return self._futures_order_meta[value]
        return {}

    def _update_futures_order_filled(self, msg: dict, meta: dict):
        if not meta:
            return

        meta['filled_quantity'] = (
            meta.get('filled_quantity', 0) + int(msg.get('quantity', 0) or 0)
        )
        order_label = meta.get('order_label', '')
        if order_label:
            status = 'Filled'
            if meta['filled_quantity'] < int(meta.get('quantity', 0) or 0):
                status = 'PartFilled'
            self._set_option_order_status(order_label, status, msg=msg)

        if meta['filled_quantity'] < int(meta.get('quantity', 0) or 0):
            return

        order = meta.get('order_msg', {}).get('order', {})
        self._remove_futures_order_meta(order)

    def _remove_futures_order_meta(self, order: dict):
        for key in ['ordno', 'seqno', 'id']:
            value = order.get(key)
            if value:
                self._futures_order_meta.pop(value, None)

    def _infer_futures_oc_type(self, target: str, action: str):
        df = db.query(SecurityInfo, self.WatchListTool.match_target(target))
        if df.empty:
            return 'New'

        current_action = df.iloc[0].action
        if current_action == action:
            return 'New'
        return 'Cover'

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
        if content.quantity >= 1000 and isinstance(contract, sj.Stock):
            return content.quantity/1000
        return content.quantity

    def check_order_status(
            self,
            order_result,
            is_stock: bool = True,
            is_combo: bool = False,
            content=None,
            order_label: str = ''
    ):
        '''確認委託狀態'''
        time.sleep(0.1)

        if is_combo:
            API.update_combostatus(API.futopt_account)
        elif is_stock:
            API.update_status(API.stock_account)
        else:
            API.update_status(API.futopt_account)
        status = self._value(order_result.status.status)
        if order_label:
            self._set_option_order_result_status(
                order_label,
                content,
                order_result
            )
        if status not in ['PreSubmitted', 'Filled']:
            msg = order_result.status.msg
            logging.warning(f'Order not submitted/filled: {msg}')
        return status

    def appendOrder(self, target: str, content: namedtuple):
        '''Add new order data to OrderTable'''

        order_cond = self.content_attr(content, 'order_cond', default='Cash')
        order_data = self.generate_data(target, content)
        if order_data is None:
            return

        order_data['leverage'] = self.check_leverage(target, order_cond)

        db.add_data(TradingStatement, **order_data)
        return order_data

    def deleteOrder(self, code: str):
        '''Delete order data from OrderTable'''
        self.OrderTable = self.OrderTable[self.OrderTable.code != code]

    def checkEnoughToPlace(self, target: str):
        '''Check if current placed amount is under target limit.'''

        conf = TradeDataHandler.getStrategyConfig(target)
        if conf is None:
            return True

        df = self.OrderTable[self.OrderTable.market == conf.market]

        if conf.market == 'Stocks':
            return df.amount.sum() < TradeData.Account.DesposalMoney
        return df.amount.sum() < TradeData.Account.DesposalMargin

    def filterOrderTable(self, market: str):
        '''Filter OrderTable by market'''
        return self.OrderTable[self.OrderTable.market == market].copy()

    @staticmethod
    def _enum_value(enum_class, value):
        if value is None or isinstance(value, enum_class):
            return value
        return getattr(enum_class, str(value))

    def _stock_order(
            self,
            content: namedtuple,
            price: float,
            quantity: int,
            price_type: str,
            order_lot: str
    ):
        return sj.StockOrder(
            price=price,
            quantity=quantity,
            action=self._enum_value(sj.Action, content.action),
            price_type=self._enum_value(sj.StockPriceType, price_type),
            order_type=self._enum_value(
                sj.OrderType,
                self.content_attr(content, 'order_type', 'ROD') or 'ROD'
            ),
            order_cond=self._enum_value(
                sj.StockOrderCond, content.order_cond or 'Cash'),
            order_lot=self._enum_value(sj.StockOrderLot, order_lot),
            account=API.stock_account,
            daytrade_short=content.daytrade_short,
        )

    def _futures_order(
            self,
            content: namedtuple,
            price: float,
            quantity: int,
            price_type: str
    ):
        return sj.FuturesOrder(
            price=price,
            quantity=quantity,
            action=self._enum_value(sj.Action, content.action),
            price_type=self._enum_value(sj.FuturesPriceType, price_type),
            order_type=self._enum_value(
                sj.OrderType,
                self.content_attr(content, 'order_type', 'IOC') or 'IOC'
            ),
            octype=self._enum_value(
                sj.FuturesOCType, content.octype or 'Auto'),
            account=API.futopt_account,
        )

    def _combo_order(self, content: namedtuple):
        return sj.ComboOrder(
            price=self.content_attr(content, 'price'),
            quantity=content.quantity,
            price_type=self._enum_value(
                sj.FuturesPriceType,
                self.content_attr(content, 'price_type', 'LMT') or 'LMT'),
            order_type=self._enum_value(
                sj.OrderType,
                self.content_attr(content, 'order_type', 'IOC') or 'IOC'
            ),
            octype=self._enum_value(
                sj.FuturesOCType, content.octype or 'Auto'),
            account=API.futopt_account,
        )

    def _combo_leg_contract(self, leg: dict):
        contract = leg.get('contract')
        if contract is not None:
            return contract
        return get_contract(leg['target'])

    def _combo_contract(self, legs: list):
        return sj.ComboContract(legs=[
            sj.ComboBase.from_contract(
                self._combo_leg_contract(leg),
                action=self._enum_value(sj.Action, leg['action'])
            )
            for leg in legs
        ])

    def place_combo_order(self, content: namedtuple):
        '''Place an options combo order.'''

        if TradeData.Account.Simulate:
            return self.appendOrder(content.target, content)

        combo_legs = self.content_attr(content, 'combo_legs')
        if not combo_legs:
            raise ValueError('Combo order requires combo_legs.')
        if self.content_attr(content, 'price') is None:
            raise ValueError('Combo order requires net price.')

        combo = self._combo_contract(combo_legs)
        order = self._combo_order(content)
        order_label = self._option_order_label(content)
        if order_label:
            self._set_option_order_status(
                order_label, 'Submitted', content=content)

        self._register_auto_order(
            content.target,
            content.action,
            content.octype,
            content.quantity
        )
        try:
            result = API.place_comboorder(combo, order)
            self._register_option_order_result(order_label, result)
            self.check_order_status(
                result,
                is_stock=False,
                is_combo=True,
                content=content,
                order_label=order_label
            )
            return result
        except Exception:
            self._set_option_order_status(
                order_label, 'Retry', content=content)
            raise

    def place_order(self, content: namedtuple):
        logging.debug(f'[OrderState.Content|{content}|')

        target = content.target

        if self.content_attr(content, 'combo_legs'):
            return self.place_combo_order(content)

        contract = TradeData.Contracts.get(target, get_contract(target))
        is_stock = isinstance(contract, sj.Stock)
        quantity = self.get_sell_quantity(content)
        price_type = self.content_attr(content, 'price_type', 'MKT') or 'MKT'
        price = self.content_attr(content, 'price', 0) or 0
        order_lot = 'IntradayOdd' if content.quantity < 1000 and is_stock else 'Common'

        if is_stock and target not in TradeData.BidAsk:
            return

        if (not is_stock) and target not in TradeData.BidAsk and not price and price_type != 'MKT':
            return

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
            result = None
            order_label = self._option_order_label(content)
            if order_label:
                self._set_option_order_status(
                    order_label, 'Submitted', content=content)

            while quantity > 0 and enough_to_place:
                order_quantity = min(quantity, q)
                if is_stock:
                    order = self._stock_order(
                        content, price, order_quantity, price_type, order_lot)
                else:
                    order = self._futures_order(
                        content, price, order_quantity, price_type)
                self._register_auto_order(
                    target,
                    content.action,
                    '' if is_stock else content.octype,
                    order_quantity
                )
                try:
                    result = API.place_order(contract, order)
                    self._register_option_order_result(order_label, result)
                    self.check_order_status(
                        result,
                        is_stock=is_stock,
                        content=content,
                        order_label=order_label
                    )
                except Exception:
                    self._set_option_order_status(
                        order_label, 'Retry', content=content)
                    raise
                quantity -= order_quantity

            return result

    def StockOrder(self, msg: dict):
        code = msg['contract']['code']
        order = msg['order']
        operation = msg['operation']

        conf = TradeDataHandler.getStrategyConfig(code)
        if conf and code in conf.FILTER_OUT:
            return

        c3 = order['action'] == 'Buy'
        TradeDataHandler.update_deal_list(code, order['action'])

        leverage = self.check_leverage(code, order['order_cond'])

        if self.is_new_order(operation) and c3:
            self.appendOrder(code, order)

        # 若融資配額張數不足，改現股買進 ex: '此證券配額張數不足，餘額 0 張（證金： 0 ）'
        elif self.is_insufficient_quota(operation):
            q_balance = operation['op_msg'].split(' ')
            if len(q_balance) > 1:
                q_balance = int(q_balance[1])
                infos = dict(action=order['action'], target=code)

                # 若本日還沒有下過融資且剩餘券數為0，才可以改下現股
                if q_balance == 0 and code not in TradeData.Stocks.Bought:
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
            self.deleteOrder(code)
            if c3:
                TradeDataHandler.update_deal_list(code, 'Cancel')

        # 更新監控庫存
        df = db.query(SecurityInfo, self.WatchListTool.match_target(code))
        order['oc_type'] = 'New' if df.empty else 'Cover'
        is_auto_order = self._consume_auto_order(code, order)
        self.WatchListTool.update_monitor(
            order['oc_type'],
            msg,
            sync_position=(not is_auto_order and self.is_new_order(operation))
        )

    def StockDeal(self, msg: dict):
        msg = CallbackHandler.update_stock_msg(msg)

        action = msg['action']
        code = msg['code']

        conf = TradeDataHandler.getStrategyConfig(code)
        if conf and code in conf.FILTER_OUT:
            return

        if action == 'Sell':
            TradeDataHandler.update_deal_list(code, action)
            self.appendOrder(code, msg)

    def FuturesOrder(self, msg: dict):
        msg = CallbackHandler().update_futures_msg(msg)

        order = msg['order']
        symbol = CallbackHandler.fut_symbol(msg)
        operation = msg['operation']

        conf = TradeDataHandler.getStrategyConfig(symbol)
        if conf and symbol in conf.FILTER_OUT:
            return

        if self.is_accepted_order_operation(operation):
            is_auto_order = self._consume_auto_order(symbol, order)
            self._set_futures_order_meta(msg, symbol, is_auto_order)
            if self.is_new_order_submit(operation):
                TradeDataHandler.update_deal_list(symbol, order['oc_type'])
                self.appendOrder(symbol, order)

        # 若刪單成功就自清單移除
        if self.is_cancel_order(operation):
            self.deleteOrder(symbol)
            self._remove_futures_order_meta(order)
            TradeDataHandler.update_deal_list(symbol, 'Cancel')

    def FuturesDeal(self, msg: dict):
        symbol = CallbackHandler.fut_deal_symbol(msg)

        conf = TradeDataHandler.getStrategyConfig(symbol)
        if conf and symbol in conf.FILTER_OUT:
            return

        meta = self._get_futures_order_meta(msg)
        oc_type = meta.get('oc_type')
        is_auto_order = meta.get('is_auto_order', False)
        action = msg.get('action', '')
        quantity = msg.get('quantity', 0)
        auto_fill_consumed = False

        if not oc_type:
            for action_type in ['New', 'Cover', '']:
                if self._consume_auto_order_fill(
                        symbol,
                        action,
                        action_type,
                        quantity
                ):
                    is_auto_order = True
                    oc_type = action_type
                    auto_fill_consumed = True
                    break

        if not oc_type:
            oc_type = self._infer_futures_oc_type(
                symbol, msg.get('action', ''))
        elif is_auto_order and not auto_fill_consumed:
            self._consume_auto_order_fill(symbol, action, oc_type, quantity)

        data = {
            'code': symbol,
            'order': {
                'action': action,
                'quantity': quantity,
                'price': msg.get('price', 0),
                'order_cond': 'Cash',
            }
        }
        self.WatchListTool.update_monitor(
            oc_type,
            data,
            sync_position=not is_auto_order
        )
        self._update_futures_order_filled(msg, meta)
