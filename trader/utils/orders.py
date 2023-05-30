import os
import time
import logging
import pandas as pd
from collections import namedtuple

from .. import API
from . import save_csv
from .database import db
from .database.tables import TradingStatement


class OrderTool:
    OrderInfo = namedtuple(
        typename="OrderInfo",
        field_names=[
            'action', 'target', 'quantity',
            'order_cond', 'octype', 'pos_target',
            'pos_balance', 'daytrade_short', 'reason'
        ],
        defaults=['', '', 0, '', '', 0, 0, False, '']
    )
    MsgOrder = namedtuple(
        typename='MsgOrder', 
        field_names=['operation', 'order', 'status', 'contract']
    )
    OrderTable = pd.DataFrame(columns=[
        'Time', 'market', 'code', 'action', 
        'price', 'quantity', 'amount',
        'order_cond', 'order_lot', 'leverage', 
        'op_type', 'account_id', 'msg',
        
    ])
    
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

    def appendOrder(self, order_data: dict):
        '''Add new order data to OrderTable'''
        self.OrderTable = pd.concat([
            self.OrderTable, 
            pd.DataFrame([order_data])
        ])

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

    def output_statement(self, filename: str=''):
        '''儲存對帳單'''

        if db.HAS_DB:
            db.dataframe_to_DB(self.OrderTable, TradingStatement)
        else:
            if os.path.exists(filename):
                statement = pd.read_csv(filename)  
            else:
                statement = pd.DataFrame()
            statement = pd.concat([statement, self.OrderTable])
            save_csv(statement, filename)