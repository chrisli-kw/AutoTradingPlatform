import logging
import pandas as pd
from datetime import datetime
from collections import namedtuple

from .. import file_handler
from ..config import PATH, TODAY_STR, StrategyList
from ..utils.database import db
from ..utils.database.tables import PutCallRatioList, ExDividendTable


class StrategyTool:
    def __init__(self, **kwargs):
        self.set_config(**kwargs)
        self.Action = namedtuple(
            typename="Action",
            field_names=['position', 'reason', 'msg', 'price', 'action'],
            defaults=[0, '', '', 0, 'Buy']
        )
        self.pc_ratio = self.get_put_call_ratio()
        self.dividends = self.get_ex_dividends_list()
        self.STRATEGIES_STOCK = pd.DataFrame(
            columns=['name', 'long_weight', 'short_weight']
        )
        self.STRATEGIES_FUTURES = pd.DataFrame(
            columns=['name', 'long_weight', 'short_weight']
        )
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
        self.revert_action = {}
        self.n_categories = None

    def set_config(self, **kwargs):
        self.account_name = kwargs.get('account_name', 'unknown')
        self.hold_day = kwargs.get('hold_day', 20)
        self.is_simulation = kwargs.get('is_simulation', True)

        self.stock_limit_type = kwargs.get('stock_limit_type', 'Constant')
        self.stock_limit_long = kwargs.get('stock_limit_long', 0)
        self.stock_limit_short = kwargs.get('stock_limit_short', 0)
        self.stock_model_version = kwargs.get('stock_model_version', '1.0.0')

        self.futures_limit_type = kwargs.get('futures_limit_type', 'Constant')
        self.futures_limit = kwargs.get('futures_limit', 0)
        self.futures_model_version = kwargs.get(
            'futures_model_version', '1.0.0')

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

    def update_indicators(self, now: datetime, kbars: dict):
        pass

    def update_StrategySet_data(self, target: str):
        pass

    def setNStockLimitLong(self, KBars: dict = None):
        '''
        Set the number limit of securities of a portfolio can hold
        for a long strategy
        '''

        if self.is_simulation:
            return 3000
        elif self.stock_limit_type != 'constant':
            return self.stock_limit_long
        return self.stock_limit_long

    def setNStockLimitShort(self, KBars: dict = None):
        '''
        Set the number limit of securities of a portfolio can hold
        for a short strategy
        '''

        if self.is_simulation:
            return 3000
        elif self.stock_limit_type != 'constant':
            return self.stock_limit_short
        return self.stock_limit_short

    def setNFuturesLimit(self, KBars: dict = None):
        '''Set the number limit of securities of a portfolio can hold'''
        return 0

    def _get_value(self, data: pd.DataFrame, stockid: str, col: str):
        if isinstance(data, pd.DataFrame):
            # Check if the dataframe is empty
            if data.empty:
                logging.warning(f"Dataframe is empty for stockid: {stockid}")
                return 0

            # Check if there's any missing value
            if data.tail(1).isnull().values.any():
                logging.warningv(
                    f"Dataframe contains NaN values for stockid: {stockid} when getting {col} value --> ")
                logging.warning(f'{data.tail().to_string()}')
                data = data.fillna(0)

            tb = data[data.name == stockid].copy()

            if tb.shape[0]:
                return tb[col].values[-1]
            return 0

        return data[stockid].get(col)

    def get_ex_dividends_list(self):
        '''取得當日除權息股票清單'''

        if db.HAS_DB:
            df = db.query(ExDividendTable)
            return df[df.Date == TODAY_STR].set_index('Code').CashDividend.to_dict()

        try:
            df = file_handler.read_table(f'{PATH}/exdividends.csv')
            df.Code = df.Code.astype(str).str.zfill(4)
            return df[df.Date == TODAY_STR].set_index('Code').CashDividend.to_dict()
        except:
            logging.warning('==========exdividends.csv不存在，無除權息股票清單==========')
            return {}

    def get_strategy_list(self, market='Stocks'):
        if market == 'Stocks':
            strategies = self.STRATEGIES_STOCK.copy()
        else:
            strategies = self.STRATEGIES_FUTURES.copy()
        return strategies.sort_values(ascending=False).name.to_list()

    def get_put_call_ratio(self):
        '''Get the latest Put-Call ratio'''
        if db.HAS_DB:
            pc_ratio = db.query(PutCallRatioList.PutCallRatio)
            if pc_ratio.shape[0]:
                return pc_ratio.PutCallRatio.values[-1]
            return 100

        try:
            pc_ratio = file_handler.read_table(f'{PATH}/put_call_ratio.csv')
            pc_ratio = pc_ratio.sort_values('Date')
            return pc_ratio.PutCallRatio.values[-1]
        except:
            logging.warning(
                '==========put_call_ratio.csv does not exist, the trader will run without Put/Call Ratio==========')
            return 100

    def transfer_position(self, inputs: dict, kbars: dict, **kwargs):
        target = inputs['symbol']
        return self.Action(100, '轉倉', f'{target} 轉倉-Cover')

    def isLong(self, strategy: str):
        '''Check if a strategy is a long strategy.'''
        return strategy in StrategyList.Long

    def isShort(self, strategy: str):
        '''Check if a strategy is a short strategy.'''
        return strategy in StrategyList.Short

    def isDayTrade(self, strategy: str):
        '''Check if a strategy is a day-trade strategy.'''
        return strategy in StrategyList.DayTrade

    def export_strategy_data(self):
        pass
