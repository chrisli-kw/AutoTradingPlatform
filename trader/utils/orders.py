import time
import logging
import pandas as pd
from collections import namedtuple
from .. import API


class OrderTool:
    OrderInfo = namedtuple(
        "OrderInfo",
        [
            'action', 'target', 'quantity',
            'order_cond', 'octype', 'pos_target',
            'pos_balance', 'daytrade_short', 'reason'
        ],

        defaults=['', '', 0, '', '', 0, 0, False, '']
    )
    MsgOrder = namedtuple('MsgOrder', ['operation', 'order', 'status', 'contract'])
    tftOrder = pd.DataFrame(columns=[
        'Time', 'code', 'action', 'price', 'quantity', 'amount',
        'order_cond', 'order_lot', 'leverage', 'account_id', 'msg'
    ])
    fOrder = pd.DataFrame(
        columns=['Time', 'code', 'action', 'price', 'quantity', 'amount', 'op_type', 'account_id', 'msg'])

    def get_sell_quantity(self, content: namedtuple, market: str = 'Stocks'):
        '''根據庫存, 剩餘部位比例, 賣出比例，反推賣出量(張)'''
        if market == 'Stocks':
            q_before = int(content.quantity/1000)
            condition = content.pos_balance > 0 and content.quantity >= 1000
        else:
            q_before = content.quantity
            condition = content.pos_balance > 0 and content.quantity != 0

        if condition:
            quantity = int(q_before/content.pos_balance*abs(content.pos_target))
            quantity = 1000*max(min(quantity, q_before), 1)
            return max(round(quantity/1000), 1)
        return q_before

    def check_order_status(self, order_result):
        '''確認委託狀態'''
        time.sleep(0.1)
        API.update_status(API.stock_account)
        status = order_result.status.status
        if status not in ['PreSubmitted', 'Filled']:
            logging.warning('order not submitted/filled')

    def append_tftOrder(self, order_data: dict):
        '''新增一筆股票交易明細'''
        self.tftOrder = pd.concat([self.tftOrder, pd.DataFrame([order_data])])

    def append_fOrder(self, order_data: dict):
        '''新增一筆期權交易明細'''
        self.fOrder = pd.concat([self.fOrder, pd.DataFrame([order_data])])
