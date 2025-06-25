import os
import logging
import pandas as pd
from datetime import datetime
from collections import namedtuple

from ..config import PATH, TODAY_STR, API, Cost
from . import get_contract, concat_df
from .time import time_tool
from .file import file_handler
from .objects.data import TradeData
from .database import db
from .database.tables import Watchlist


class WatchListTool:

    def __init__(self, account_name: str):
        self.account_name = account_name
        self.MatchAccount = Watchlist.account == self.account_name
        self.watchlist_file = f'watchlist_{account_name}'
        self.watchlist = self.get_watchlist()

    def get_watchlist(self):
        """Load watchlist data"""
        if db.HAS_DB:
            df = db.query(Watchlist, Watchlist.account == self.account_name)
            return df

        df = file_handler.Process.read_table(
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

    def _append_watchlist(self, market: str, orderinfo: namedtuple, quotes: dict = 0):
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
            cost_price = TradeData.Quotes.NowTargets[target]['price']
            check_quantity = orderinfo.quantity != 0

        if target not in self.watchlist.code.values and check_quantity:
            if market == 'Stocks':
                strategy_pool = TradeData.Stocks.Strategy
            else:
                strategy_pool = TradeData.Futures.Strategy
            data = {
                'account': self.account_name,
                'market': market,
                'code': target,
                'buyday': datetime.now(),
                'bsh': cost_price,
                'position': position,
                'strategy': strategy_pool.get(target, 'unknown')
            }
            self.watchlist = concat_df(self.watchlist, pd.DataFrame([data]))

            if db.HAS_DB:
                db.add_data(Watchlist, **data)

    def init_watchlist(self, stocks: pd.DataFrame):
        stocks = stocks[~stocks.code.isin(self.watchlist.code)].code.values

        # Add stocks to watchlist if it is empty.
        if not self.watchlist.shape[0] and stocks.shape[0]:
            for stock in stocks:
                cost_price = get_contract(stock).reference
                self._append_watchlist('Stocks', stock, cost_price)

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

    def merge_info(self, tbInfo: pd.DataFrame):
        tbInfo = tbInfo.merge(
            self.watchlist,
            how='left',
            on=['account', 'market', 'code']
        )
        tbInfo.position.fillna(100, inplace=True)
        return tbInfo

    def remove_from_watchlist(self):
        '''Delete Watchlist data where position <= 0.'''

        self.watchlist = self.watchlist[self.watchlist.position > 0]
        if db.HAS_DB:
            db.delete(Watchlist, Watchlist.position <= 0, self.MatchAccount)

    def update_watchlist_position(self, order: namedtuple):
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
            self._append_watchlist(market, order)

    def save_watchlist(self, df: pd.DataFrame):
        if db.HAS_DB:
            codes = db.query(Watchlist.code, self.MatchAccount).code.values
            tb = df[~df.code.isin(codes)]
            db.dataframe_to_DB(tb, Watchlist)
        else:
            file_handler.Process.save_table(
                df=df,
                filename=f'{PATH}/stock_pool/{self.watchlist_file}.csv',
                saveEmpty=True
            )


class TradeDataHandler:
    @staticmethod
    def check_is_empty(target, market='Stocks'):
        if market == 'Stocks':
            data = TradeData.Stocks.Monitor.get(target, {})
        else:
            data = TradeData.Futures.Monitor.get(target, {})
        quantity = data.get('order', {}).get('quantity', 0)
        position = data.get('position', 0)

        is_empty = (quantity <= 0 or position <= 0)
        logging.debug(
            f'[Monitor List]Check|{market}|{target}|{is_empty}|quantity: {quantity}; position: {position}|')
        return is_empty

    @staticmethod
    def reset_monitor(target: str, market='Stocks', day_trade=False):
        if market == 'Stocks':
            TradeData.Stocks.Monitor.pop(target, None)
            if day_trade:
                logging.debug(f'[Monitor List]Reset|Stocks|{target}|')
                TradeData.Stocks.Monitor[target] = None
        else:
            TradeData.Futures.Monitor.pop(target, None)
            if day_trade:
                logging.debug(f'[Monitor List]Reset|Futures|{target}|')
                TradeData.Futures.Monitor[target] = None

    @staticmethod
    def update_deal_list(target: str, action_type: str, market='Stocks'):
        '''更新下單暫存清單'''

        logging.debug(f'[Monitor List]{action_type}|{market}|{target}|')
        if market == 'Stocks':
            if action_type == 'Sell' and len(target) == 4 and target not in TradeData.Stocks.Sold:
                TradeData.Stocks.Sold.append(target)

            if action_type == 'Buy' and len(target) == 4 and target not in TradeData.Stocks.Bought:
                TradeData.Stocks.Bought.append(target)

            if action_type == 'Cancel' and target in TradeData.Stocks.Bought:
                TradeData.Stocks.Bought.remove(target)
        elif market == 'Futures':
            if action_type == 'New' and target not in TradeData.Futures.Opened:
                TradeData.Futures.Opened.append(target)

            if action_type == 'Cover' and target not in TradeData.Futures.Closed:
                TradeData.Futures.Closed.append(target)

            if action_type == 'Cancel' and target in TradeData.Futures.Opened:
                TradeData.Futures.Opened.remove(target)

    @staticmethod
    def update_monitor(action: str, data: dict, position: float = 100):
        '''更新監控庫存(成交回報)'''
        target = data['code']
        quantity_ = position_ = 0
        if action in ['Open', 'Close']:
            if TradeData.Stocks.Monitor[target] is not None:
                quantity = data['quantity']

                if action == 'New':
                    stage = 'Add|Stocks'
                    position *= -1
                    quantity *= -1
                else:
                    stage = 'Update|Stocks'

                TradeData.Stocks.Monitor[target]['position'] -= position
                TradeData.Stocks.Monitor[target]['quantity'] -= quantity
            else:
                stage = 'Add|Stocks'
                TradeData.Stocks.Monitor[target] = data

            if 'None' not in stage:
                position_ = TradeData.Stocks.Monitor[target]['position']
                quantity_ = TradeData.Stocks.Monitor[target]['quantity']

        # New, Cover
        else:
            if TradeData.Futures.Monitor[target] is not None:
                quantity = data['order']['quantity']

                if action == 'New':
                    stage = 'Add|Futures'
                    position *= -1
                    quantity *= -1
                else:
                    stage = 'Update|Futures'

                TradeData.Futures.Monitor[target]['position'] -= position
                TradeData.Futures.Monitor[target]['order']['quantity'] -= quantity
            elif action == 'New':
                stage = 'Add|Futures'

                date = TODAY_STR.replace('-', '/')
                data['contract'] = get_contract(target)
                data['isDue'] = date == data['contract'].delivery_date
                TradeData.Futures.Monitor[target] = data
            else:
                stage = 'None|Futures'

            if 'None' not in stage:
                position_ = TradeData.Futures.Monitor[target]['position']
                quantity_ = TradeData.Futures.Monitor[target]['order']['quantity']

        logging.debug(
            f'[Monitor List]{stage}|{target}|{action}|quantity: {quantity_}; position: {position_}|')

    @staticmethod
    def getQuotesNow(target: str):
        if target in TradeData.Quotes.NowIndex:
            return TradeData.Quotes.NowIndex[target]
        elif target in TradeData.Quotes.NowTargets:
            return TradeData.Quotes.NowTargets[target]
        return -1


class FuturesMargin:
    def __init__(self) -> None:
        self.full_path_name = './lib/indexMarging.csv'
        self.margin_table = None

    def get_margin_table(self, type='dict'):
        '''Get futures margin table'''

        if not os.path.exists(self.full_path_name):
            logging.warning(
                f'File not found: {self.full_path_name}, any action requiring the margin info may be inaccurate. Try to execute the runCrawlIndexMargin task to acquire the margin table.')
            if type == 'dict':
                return dict()
            return pd.DataFrame()

        df = file_handler.Process.read_table(self.full_path_name)

        codes = [[f.code, f.symbol, f.name]
                 for m in API.Contracts.Futures for f in m]
        codes = pd.DataFrame(codes, columns=['code', 'symbol', 'name'])
        codes = codes.set_index('name').symbol.to_dict()

        month = time_tool.GetDueMonth()[-2:]
        df['code'] = (df.商品別 + month).map(codes)
        df = df.dropna().set_index('code')

        if type == 'dict':
            return df.原始保證金.to_dict()
        return df

    def get_open_margin(self, target: str, quantity: int):
        '''Calculate the amount of margin for opening a position'''

        if self.margin_table and target in self.margin_table:
            fee = getattr(Cost, f'FUTURES_FEE_{target[:3]}', 100)
            return self.margin_table[target]*quantity + fee
        return 0

    def transfer_margin(self, target_old: str, target_new: str):
        '''Add new target margin on the futures due days'''

        if self.margin_table is None:
            return

        self.margin_table[target_new] = self.margin_table[target_old]


class Position:
    def __init__(self):
        self.entries = []  # [{'price': float, 'time': datetime}]
        # [{'price': float, 'time': datetime, 'reason': str}]
        self.exits = []
        self.total_qty = 0
        self.total_profit = 0.0

    def open(self, price, time, qty=1):
        self.entries.append({'price': price, 'time': time, 'qty': qty})
        self.total_qty += qty

    def close(self, price, time, reason='', qty=1):
        qty = min(qty, self.total_qty)

        closed_qty = 0
        profit = 0.0
        while closed_qty < qty and self.entries:
            if self.entries[0]['qty'] > qty:
                entry = self.entries[0]
                e_qty = qty
                self.entries[0]['qty'] -= qty
            else:
                entry = self.entries.pop(0)
                e_qty = entry['qty']

            closed_qty += e_qty
            entry_price = entry['price']
            profit = (price - entry_price) * e_qty
            self.exits.append({
                'entry_price': entry_price,
                'exit_price': price,
                'profit': profit,
                'qty': e_qty,
                'open_time': entry['time'],
                'close_time': time,
                'reason': reason
            })
        self.total_qty -= closed_qty
        self.total_profit += profit
        return profit

    def is_open(self):
        return self.total_qty > 0
