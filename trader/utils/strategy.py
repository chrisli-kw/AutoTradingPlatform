import logging
import numpy as np
import pandas as pd

from importlib import import_module

from .objects import Action
from ..config import TODAY_STR, StrategyList
from ..utils.database import db
from ..utils.database.tables import ExDividendTable
from ..utils.objects.data import TradeData


class StrategyTool:
    def __init__(self, env=None):
        self.account_name = env.ACCOUNT_NAME

        self.stock_limit_type = env.N_STOCK_LIMIT_TYPE
        self.stock_limit_long = env.N_LIMIT_LS
        self.stock_limit_short = env.N_LIMIT_SS
        self.stock_model_version = env.STOCK_MODEL_VERSION

        self.futures_limit_type = env.N_FUTURES_LIMIT_TYPE
        self.futures_limit = env.N_FUTURES_LIMIT
        self.futures_model_version = env.FUTURES_MODEL_VERSION

        self.is_simulation = env.MODE == 'Simulation'

        self.strategy_configs = {
            s: import_module(f'trader.scripts.StrategySet.{s}').StrategyConfig(self.account_name) for s in StrategyList.All
        }

    def mapFunction(self, action: str, strategy: str):
        if strategy in self.strategy_configs:
            conf = self.strategy_configs.get(strategy)
            return getattr(conf, action)

        return self.__DoNothing__

    def mapQuantities(self, strategy: str):

        def default_quantity(**kwargs):
            # (target_quantity, quantity_limit)
            return 1, 499

        if strategy in self.strategy_configs:
            return self.strategy_configs[strategy].Quantity

        return default_quantity

    def __DoNothing__(self, **kwargs):
        return Action()

    def update_StrategySet_data_(self, target: str):
        for conf in self.strategy_configs.values():
            if hasattr(conf, 'update_StrategySet_data'):
                conf.update_StrategySet_data(target)

    def setNStockLimitLong(self, **kwargs):
        '''
        Set the number limit of securities of a portfolio can hold
        for a long strategy
        '''

        if self.is_simulation:
            limit = 3000
        elif self.stock_limit_type != 'constant':
            limit = self.stock_limit_long
        else:
            limit = self.stock_limit_long
        self.stock_limit_long = limit
        return limit

    def setNStockLimitShort(self, **kwargs):
        '''
        Set the number limit of securities of a portfolio can hold
        for a short strategy
        '''

        if self.is_simulation:
            limit = 3000
        elif self.stock_limit_type != 'constant':
            limit = self.stock_limit_short
        else:
            limit = self.stock_limit_short
        self.stock_limit_short = limit
        return limit

    def setNFuturesLimit(self, **kwargs):
        '''Set the number limit of securities of a portfolio can hold'''
        if self.futures_limit_type != 'constant':
            limit = self.futures_limit
        else:
            limit = self.futures_limit
        self.futures_limit = limit
        return limit

    def _get_value(self, data: pd.DataFrame, stockid: str, col: str):
        if isinstance(data, pd.DataFrame):
            # Check if the dataframe is empty
            if data.empty:
                logging.warning(f"Dataframe is empty for stockid: {stockid}")
                return 0

            # Check if there's any missing value
            if data.tail(1).isnull().values.any():
                logging.warning(
                    f"Dataframe contains NaN values for stockid: {stockid} when getting {col} value")
                # logging.warning(f'{data.tail().to_string()}')
                data = data.fillna(0)

            tb = data[data.name == stockid].copy()

            if tb.shape[0]:
                return tb[col].values[-1]
            return 0

        return data.get(stockid, {}).get(col, 0)

    def get_ex_dividends_list(self):
        '''取得當日除權息股票清單'''

        if db.HAS_DB:
            df = db.query(ExDividendTable)
        else:
            df = pd.DataFrame(columns=['Date', 'Code', 'CashDividend'])

        df = df[df.Date == TODAY_STR].set_index('Code').CashDividend.to_dict()
        TradeData.Stocks.Dividends = df

    def get_pos_balance(self, strategy: str, raise_pos=False):
        if strategy not in self.strategy_configs:
            return 100

        conf = self.strategy_configs.get(strategy)
        if raise_pos:
            return 100*(conf.raise_qty/conf.max_qty)
        return 100*(conf.open_qty/conf.max_qty)

    def transfer_position(self, inputs: dict, **kwargs):
        target = inputs['symbol']
        return Action(100, '轉倉', f'{target} 轉倉-Cover')

    def isLong(self, strategy: str):
        '''Check if a strategy is a long strategy.'''
        return strategy in StrategyList.Long

    def isShort(self, strategy: str):
        '''Check if a strategy is a short strategy.'''
        return strategy in StrategyList.Short

    def isDayTrade(self, strategy: str):
        '''Check if a strategy is a day-trade strategy.'''
        return strategy in StrategyList.DayTrade

    def isRaiseQty(self, strategy: str):
        if strategy not in self.strategy_configs:
            return False

        conf = self.strategy_configs.get(strategy)
        position = conf.positions
        if not position.entries:
            return False

        return (
            conf.raiseQuota and
            conf.raise_qty <= conf.max_qty - position.total_qty
        )

    def export_strategy_data_(self):
        for conf in self.strategy_configs.values():
            if hasattr(conf, 'export_strategy_data'):
                conf.export_strategy_data()

    def append_monitor_list_(self, monitor_list: list):
        for conf in self.strategy_configs.values():
            if hasattr(conf, 'append_monitor_list'):
                monitor_list = conf.append_monitor_list(monitor_list)
        return np.unique(monitor_list).tolist()
