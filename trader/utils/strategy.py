import logging
import numpy as np
import pandas as pd

from .objects import Action
from .positions import TradeDataHandler
from ..config import TODAY_STR, StrategyList
from ..utils.database import db
from ..utils.database.tables import ExDividendTable, SecurityInfo
from ..utils.objects.data import TradeData


class StrategyTool:
    def __init__(self, env=None):
        self.account_name = env.ACCOUNT_NAME
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

    def set_position_limit(self):
        if TradeData.Account.Simulate:
            TradeData.Stocks.LimitLong = 3000
            TradeData.Stocks.LimitShort = 3000
            TradeData.Futures.Limit = 3000

        for conf in StrategyList.Config.values():
            targets = getattr(conf, 'Targets', [])

            default_limit = len(targets)
            if getattr(conf, 'raiseQuota', False):
                default_limit += 1

            limit = getattr(conf, 'PositionLimit', default_limit)

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

    def get_pos_balance(self, target: str, raise_pos=False):
        conf = TradeDataHandler.getStrategyConfig(target)
        if conf is None:
            return 100

        if not hasattr(conf, 'raise_qty'):
            return 100

        max_qty = conf.max_qty.get(target, 1)
        if raise_pos:
            return 100*(conf.raise_qty/max_qty)
        return 100*(conf.open_qty/max_qty)

    def transfer_position(self, inputs: dict, **kwargs):
        target = inputs['symbol']
        df = db.query(
            SecurityInfo,
            SecurityInfo.mode == TradeData.Account.Mode,
            SecurityInfo.market == 'Futures',
            SecurityInfo.code == target
        )
        action = df.action.values[0]
        action = 'Sell' if action == 'Buy' else 'Buy'
        quantity = df.quantity.sum() if not df.empty else 0
        return Action(action, f'{target} 轉倉-Cover', quantity)

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

    def isRaiseQty(self, target: str):
        conf = TradeDataHandler.getStrategyConfig(target)
        if conf is None:
            return False

        position = conf.positions
        entries = [e for e in position.entries if e['name'] == target]
        if not entries:
            return False

        return (
            conf.raiseQuota and
            conf.raise_qty <= conf.max_qty.get(
                target) - position.total_qty.get(target, 0)
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
