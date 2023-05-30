import pandas as pd
import logging
from collections import namedtuple
from .. import PATH, TODAY_STR
from ..utils.database import db
from ..utils.database.tables import PutCallRatioList, ExDividendTable


class StrategyTool:
    def __init__(self):
        self.Action = namedtuple(
            typename="Action", 
            field_names=['position', 'reason', 'msg', 'price'], 
            defaults=[0, '', '', 0]
        )
        self.pc_ratio = self.get_put_call_ratio()
        self.dividends = self.get_ex_dividends_list()
        self.Funcs = {
            'Open': {  # action
                '當沖': {},  # tradeType
                '非當沖': {}
            },
            'Close': {
                '當沖': {},
                '非當沖': {}
            }
        }
        self.QuantityFunc = {}

    def mapFunction(self, action: str, tradeType: str, strategy: str):
        has_action = action in self.Funcs
        has_tradeType = tradeType in self.Funcs[action]
        has_strategy = strategy in self.Funcs[action][tradeType]
        if has_action and has_tradeType and has_strategy:
            return self.Funcs[action][tradeType][strategy]
        return self.__DoNothing__

    def mapQuantities(self, strategy: str):

        def default_quantity(**kwargs):
            return 1, 499

        if strategy in self.QuantityFunc:
            return self.QuantityFunc[strategy]

        return default_quantity

    def __DoNothing__(self, **kwargs):
        return self.Action()

    def _get_value(self, df: pd.DataFrame, stockid: str, col: str):
        tb = df[df.name == stockid]

        if tb.shape[0]:
            return tb[col].values[-1]

        return 0

    def get_ex_dividends_list(self):
        '''取得當日除權息股票清單'''

        if db.HAS_DB:
            df = db.query(ExDividendTable)
            return df[df.Date == TODAY_STR].set_index('Code').CashDividend.to_dict()

        try:
            df = pd.read_csv(f'{PATH}/exdividends.csv')
            df.Code = df.Code.astype(str).str.zfill(4)
            return df[df.Date == TODAY_STR].set_index('Code').CashDividend.to_dict()
        except:
            logging.warning('==========exdividends.csv不存在，無除權息股票清單==========')
            return {}

    def get_put_call_ratio(self):
        '''取得前一個交易日收盤後的Put-Call ratio'''
        if db.HAS_DB:
            pc_ratio = db.query(PutCallRatioList.PutCallRatio)
            if pc_ratio.shape[0]:
                return pc_ratio.PutCallRatio.values[-1] 
            return 100
        
        try:
            pc_ratio = pd.read_csv(f'{PATH}/put_call_ratio.csv')
            pc_ratio = pc_ratio.sort_values('Date')
            return pc_ratio.PutCallRatio.values[-1]
        except:
            logging.warning('==========put_call_ratio.csv不存在，無前一交易日的Put/Call Ratio==========')
            return 100
