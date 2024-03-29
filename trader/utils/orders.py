import time
import logging
import pandas as pd
from collections import namedtuple

from ..config import API, PATH
from . import concat_df
from .file import FileHandler
from .database import db
from .database.tables import TradingStatement


class OrderTool(FileHandler):
    OrderInfo = namedtuple(
        typename="OrderInfo",
        field_names=[
            'action_type',
            'action', 'target', 'quantity',
            'order_cond', 'octype', 'pos_target',
            'pos_balance', 'daytrade_short', 'reason'
        ],
        defaults=['', '', '', 0, '', '', 0, 0, False, '']
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

    def appendOrder(self, order_data: dict):
        '''Add new order data to OrderTable'''
        self.OrderTable = concat_df(
            self.OrderTable,
            pd.DataFrame([order_data])
        )

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

    def output_statement(self, filename: str = ''):
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
        else:
            statement = self.read_and_concat(filename, self.OrderTable)
            self.save_table(statement, filename)

    def read_statement(self, account: str = ''):
        '''Import trading statement'''

        if db.HAS_DB:
            df = db.query(
                TradingStatement,
                TradingStatement.account_id == account
            )
        else:
            filename = f"{PATH}/stock_pool/statement_{account.split('-')[-1]}.csv"
            df = self.read_table(filename, df_default=self.OrderTable)
            df = df[df.account_id == account]
            df = df.astype({
                'price': float,
                'quantity': float,
                'amount': float,
                'leverage': float
            })
            df.Time = pd.to_datetime(df.Time)

        df = df.drop_duplicates()
        return df

    def update_pos_target(self, order: namedtuple, is_empty: False):
        if is_empty and order.action_type == 'Close':
            position = order.pos_target
            if position == 100 or position >= order.pos_balance:
                order = order._replace(pos_target=100)

        return order
