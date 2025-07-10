import logging
import numpy as np
import pandas as pd
from importlib import import_module

from .objects import Action
from ..config import TODAY_STR, StrategyList
from ..utils.database import db
from ..utils.database.tables import ExDividendTable
from ..utils.objects.data import TradeData


def import_strategy(account_name: str, strategy: str):
    module_path = f'trader.scripts.StrategySet.{strategy}'
    conf = import_module(module_path).StrategyConfig(account_name)
    return conf


class StrategyTool:
    def __init__(self, env=None):
        self.account_name = env.ACCOUNT_NAME

        self.stock_model_version = env.STOCK_MODEL_VERSION
        self.futures_model_version = env.FUTURES_MODEL_VERSION

        self.is_simulation = env.MODE == 'Simulation'
        self.check_can_stock()
        self.check_can_futures()

    def check_can_stock(self):
        markets = [conf.market for conf in StrategyList.Config.values()]
        TradeData.Stocks.CanTrade = any(x == 'Stocks' for x in markets)

    def check_can_futures(self):
        markets = [conf.market for conf in StrategyList.Config.values()]
        TradeData.Futures.CanTrade = any(x == 'Futures' for x in markets)

    def mapFunction(self, action: str, strategy: str):
        if strategy in StrategyList.Config:
            conf = StrategyList.Config.get(strategy)
            return getattr(conf, action)

        return self.__DoNothing__

    def mapQuantities(self, strategy: str):

        def default_quantity(**kwargs):
            # (target_quantity, quantity_limit)
            return 1, 499

        if strategy in StrategyList.Config:
            return StrategyList.Config[strategy].Quantity

        return default_quantity

    def __DoNothing__(self, **kwargs):
        return Action()

    def update_StrategySet_data_(self, target: str):
        for conf in StrategyList.Config.values():
            if hasattr(conf, 'update_StrategySet_data'):
                conf.update_StrategySet_data(target)

    def set_position_limit(self):
        if self.is_simulation:
            TradeData.Stocks.LimitLong = 3000
            TradeData.Stocks.LimitShort = 3000
            TradeData.Futures.Limit = 3000

        for conf in StrategyList.Config.values():
            targets = getattr(conf, 'Targets', [])
            limit = getattr(conf, 'PositionLimit', len(targets))

            if conf.market == 'Stocks' and conf.mode == 'long':
                TradeData.Stocks.LimitLong += limit

            if conf.market == 'Stocks' and conf.mode == 'short':
                TradeData.Stocks.LimitShort += limit

            if conf.market == 'Futures':
                TradeData.Futures.Limit += limit

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
        if strategy not in StrategyList.Config:
            return 100

        conf = StrategyList.Config.get(strategy)
        if not hasattr(conf, 'raise_qty'):
            return 100

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

        conf = StrategyList.Config.get(strategy)
        return getattr(conf, 'DayTrade', False)

    def isRaiseQty(self, strategy: str):
        if strategy not in StrategyList.Config:
            return False

        conf = StrategyList.Config.get(strategy)
        position = conf.positions
        if not position.entries:
            return False

        return (
            conf.raiseQuota and
            conf.raise_qty <= conf.max_qty - position.total_qty
        )

    def export_strategy_data_(self):
        for conf in StrategyList.Config.values():
            if hasattr(conf, 'export_strategy_data'):
                conf.export_strategy_data()

    def append_monitor_list_(self, monitor_list: list):
        for conf in StrategyList.Config.values():
            if hasattr(conf, 'append_monitor_list'):
                monitor_list = conf.append_monitor_list(monitor_list)
        return np.unique(monitor_list).tolist()
