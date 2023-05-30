import pandas as pd
from datetime import datetime
from collections import namedtuple

from .. import PATH, TODAY
from . import save_csv, get_contract, db
from .time import TimeTool
from .database.tables import Watchlist

if db.has_db:
    db.create_table(Watchlist)


class WatchListTool(TimeTool):

    def __init__(self, account_name):
        self.account_name = account_name
        self.MatchAccount = Watchlist.account == self.account_name
        self.watchlist_file = f'watchlist_{account_name}'
        self.watchlist = self._open_watchlist()

    def _open_watchlist(self):
        """Load watchlist data"""
        if db.has_db:
            df = db.query(Watchlist, Watchlist.account == self.account_name)
            df = df.drop(['pk_id', 'create_time'], axis=1)
            return df

        try:
            df = pd.read_csv(f'{PATH}/stock_pool/{self.watchlist_file}.csv')
        except:
            columns = ['account', 'market', 'code', 'buyday', 'bsh', 'position', 'strategy']
            df = pd.DataFrame(columns=columns)
            self.save_watchlist(df)

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
            cost_price = quotes[target]['price']
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

            if db.has_db:
                db.add_data(Watchlist, **data)

    def init_watchlist(self, stocks: pd.DataFrame, strategy_pool: dict):
        stocks = stocks[~stocks.code.isin(self.watchlist.code)].code.values

        # Add stocks to watchlist if it is empty.
        if not self.watchlist.shape[0] and stocks.shape[0]:
            for stock in stocks:
                cost_price = get_contract(stock).reference
                self._append_watchlist('Stocks', stock, cost_price, strategy_pool)

    def update_watchlist(self, df_stocks: pd.DataFrame):
        '''Update watchlist data when trading time is closed'''

        df1 = self.watchlist[self.watchlist.market == 'Stocks']
        df2 = self.watchlist[self.watchlist.market == 'Futures']

        # Update watchlist if there's any new stock added in stock account.
        condi1 = (df_stocks.yd_quantity == 0) & (df_stocks.quantity > 0)
        condi2 = ~df_stocks.code.isin(df1.code)
        _stocks = df_stocks[condi1 | condi2]

        # Only update those which are manual traded (TODO: delete)
        for stock in _stocks.code.values:
            if stock in df1.code.values:
                condition = (df1.code == stock) & (df1.strategy.isnull())
                df1.loc[condition, ['buyday', 'position']] = [TODAY, 100]

        # Update watchlist position if there's any stock both exists
        # in stock account and watchlist but position <= 0.
        condi1 = df1.code.isin(df_stocks.code)
        condi2 = (df1.position <= 0)

        df1.loc[~condi1 | condi2, 'position'] = 0
        df1.loc[condi1 & condi2, 'position'] = 100
        self.watchlist = pd.concat([df1, df2])
        self.watchlist = self.watchlist[self.watchlist.position > 0]

        if db.has_db:

            code1 = df1[~condi1 | condi2].code.values
            condition = Watchlist.code.in_(code1), self.MatchAccount
            db.update(Watchlist, {'position': 0}, *condition)

            code2 = df1[condi1 & condi2].code.values
            condition = Watchlist.code.in_(code2), self.MatchAccount
            db.update(Watchlist, {'position': 100}, condition)

            # delete Watchlist data where position <= 0
            db.delete(Watchlist, Watchlist.position <= 0, self.MatchAccount)

    def save_watchlist(self, df: pd.DataFrame):
        # TODO either save_csv or dataframe_to_DB
        save_csv(df, f'{PATH}/stock_pool/{self.watchlist_file}.csv', saveEmpty=True)
        if db.has_db:
            codes = db.query(Watchlist.code, self.MatchAccount).code.values
            tb = df[~df.code.isin(codes)]
            db.dataframe_to_DB(tb, Watchlist)
