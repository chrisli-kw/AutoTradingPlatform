import pandas as pd
from datetime import datetime
from collections import namedtuple

from ..config import PATH
from . import get_contract
from .time import TimeTool
from .file import FileHandler
from .database import db
from .database.tables import Watchlist


class WatchListTool(TimeTool, FileHandler):

    def __init__(self, account_name):
        self.account_name = account_name
        self.MatchAccount = Watchlist.account == self.account_name
        self.watchlist_file = f'watchlist_{account_name}'
        self.watchlist = self.get_watchlist()

    def get_watchlist(self):
        """Load watchlist data"""
        if db.HAS_DB:
            df = db.query(Watchlist, Watchlist.account == self.account_name)
            return df

        df = self.read_table(
            filename=f'{PATH}/stock_pool/{self.watchlist_file}.csv',
            df_default=pd.DataFrame(columns=[
                'account', 'market', 'code', 'buyday',
                'bsh', 'position', 'strategy'
            ])
        )
        df.code = df.code.astype(str)
        df.buyday = pd.to_datetime(df.buyday.astype(str))
        df.position = df.position.fillna(100)
        df.strategy = df.strategy.fillna('Unknown')
        return df

    def _append_watchlist(self, market: str, orderinfo: namedtuple, quotes: dict, strategy_pool: dict = None):
        '''Add new stock data to watchlist'''
        if isinstance(orderinfo, str):
            # Manual trading
            target = orderinfo
            position = 100
            cost_price = quotes
            check_quantity = True
        else:
            # Auto trading
            target = orderinfo.target
            position = abs(orderinfo.pos_target)
            cost_price = quotes.NowTargets[target]['price']
            check_quantity = orderinfo.quantity != 0

        if target not in self.watchlist.code.values and check_quantity:
            data = {
                'account': self.account_name,
                'market': market,
                'code': target,
                'buyday': datetime.now(),
                'bsh': cost_price,
                'position': position,
                'strategy': strategy_pool[target] if strategy_pool and target in strategy_pool else 'unknown'
            }
            self.watchlist = pd.concat([self.watchlist, pd.DataFrame([data])])

            if db.HAS_DB:
                db.add_data(Watchlist, **data)

    def init_watchlist(self, stocks: pd.DataFrame, strategy_pool: dict):
        stocks = stocks[~stocks.code.isin(self.watchlist.code)].code.values

        # Add stocks to watchlist if it is empty.
        if not self.watchlist.shape[0] and stocks.shape[0]:
            for stock in stocks:
                cost_price = get_contract(stock).reference
                self._append_watchlist(
                    'Stocks', stock, cost_price, strategy_pool)

    def update_watchlist(self, codeList: list):
        '''Update watchlist data when trading time is closed'''

        # Update watchlist position if there's any stock both exists
        # in stock account and watchlist but position <= 0.
        condi1 = self.watchlist.code.isin(codeList)
        condi2 = (self.watchlist.position <= 0)
        self.watchlist.loc[~condi1 | condi2, 'position'] = 0
        self.watchlist.loc[condi1 & condi2, 'position'] = 100

        if db.HAS_DB:
            code1 = self.watchlist[~condi1 | condi2].code.values
            condition = Watchlist.code.in_(code1), self.MatchAccount
            db.update(Watchlist, {'position': 0}, *condition)

            code2 = self.watchlist[condi1 & condi2].code.values
            condition = Watchlist.code.in_(code2), self.MatchAccount
            db.update(Watchlist, {'position': 100}, condition)

        self.remove_from_watchlist()

    def remove_from_watchlist(self):
        '''Delete Watchlist data where position <= 0.'''

        self.watchlist = self.watchlist[self.watchlist.position > 0]
        if db.HAS_DB:
            db.delete(Watchlist, Watchlist.position <= 0, self.MatchAccount)

    def update_watchlist_position(self, order: namedtuple, quotes: dict, strategy_pool: dict = None):
        target = order.target
        position = order.pos_target

        if target in self.watchlist.code.values:
            condition = self.watchlist.code == target
            if order.action_type == 'Open':
                self.watchlist.loc[condition, 'position'] += position
            else:
                self.watchlist.loc[condition, 'position'] -= position

            if db.HAS_DB and condition.sum():
                position = self.watchlist.loc[condition, 'position'].values[0]
                condition = Watchlist.code.in_([target]), self.MatchAccount
                db.update(Watchlist, {'position': position}, *condition)

            self.remove_from_watchlist()
        else:
            market = 'Stocks' if not order.octype else 'Futures'
            self._append_watchlist(market, order, quotes, strategy_pool)

    def save_watchlist(self, df: pd.DataFrame):
        if db.HAS_DB:
            codes = db.query(Watchlist.code, self.MatchAccount).code.values
            tb = df[~df.code.isin(codes)]
            db.dataframe_to_DB(tb, Watchlist)
        else:
            self.save_table(
                df=df,
                filename=f'{PATH}/stock_pool/{self.watchlist_file}.csv',
                saveEmpty=True
            )
