import logging
import pandas as pd
from datetime import datetime
from collections import namedtuple

from ..config import PATH
from . import get_contract, concat_df
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

        # stock
        self.stocks_to_monitor = {}
        self.stock_bought = []
        self.stock_sold = []

        # futures
        self.futures_to_monitor = {}
        self.futures_opened = []
        self.futures_closed = []

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
            self.watchlist = concat_df(self.watchlist, pd.DataFrame([data]))

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
        # if order.action_type == 'Close':
        #     if order.pos_target == 100 or order.pos_target >= order.pos_balance:
        #         order = order._replace(pos_target=100)

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
        elif order.action_type == 'Open':
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

    def check_is_empty(self, target, market='Stocks'):
        if market == 'Stocks':
            data = self.stocks_to_monitor[target]
            return (data['quantity'] <= 0 or data['position'] <= 0)
        else:
            data = self.futures_to_monitor[target]
            return (data['order']['quantity'] <= 0 or data['position'] <= 0)

    def reset_monitor_list(self, target: str, market='Stocks', day_trade=False):
        if market == 'Stocks':
            self.stocks_to_monitor.pop(target, None)
            if day_trade:
                self.stocks_to_monitor[target] = None
        else:
            self.futures_to_monitor.pop(target, None)
            if day_trade:
                self.futures_to_monitor[target] = None

    def update_deal_list(self, target: str, action_type: str, market='Stocks'):
        '''更新下單暫存清單'''

        if market == 'Stocks':
            if action_type == 'Sell' and len(target) == 4 and target not in self.stock_sold:
                self.stock_sold.append(target)

            if action_type == 'Buy' and len(target) == 4 and target not in self.stock_bought:
                self.stock_bought.append(target)

            if action_type == 'Cancel' and target in self.stock_bought:
                self.stock_bought.remove(target)
        else:
            if action_type == 'New' and target not in self.futures_opened:
                self.futures_opened.append(target)

            if action_type == 'Cover' and target not in self.futures_closed:
                self.futures_closed.append(target)

            if action_type == 'Cancel' and target in self.futures_opened:
                self.futures_opened.remove(target)

    def update_position_quantity(self, action: str, data: dict, position: float = 100):
        '''更新監控庫存(成交回報)'''
        target = data['code']
        if action in ['Buy', 'Sell']:
            if self.stocks_to_monitor[target] is not None:
                # TODO: 部分進場
                stage = 'update stocks_to_monitor'
                quantity = data['quantity']
                self.stocks_to_monitor[target]['position'] -= position
                self.stocks_to_monitor[target]['quantity'] -= quantity
            else:
                stage = 'add stocks_to_monitor'
                self.stocks_to_monitor[target] = data

        # New, Cover
        else:
            if self.futures_to_monitor[target] is not None:
                stage = 'update futures_to_monitor'
                quantity = data['order']['quantity']

                self.futures_to_monitor[target]['position'] -= position
                self.futures_to_monitor[target]['order']['quantity'] -= quantity
            elif action == 'New':
                stage = 'add futures_to_monitor'
                self.futures_to_monitor[target] = data
            else:
                stage = 'None'

        logging.debug(f'【更新】{stage}|{action} {target}|')
