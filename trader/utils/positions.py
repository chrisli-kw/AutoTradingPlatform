import os
import logging
import pandas as pd
from datetime import datetime
from collections import namedtuple

from ..config import TODAY_STR, API, Cost
from . import get_contract
from .database import db
from .database.tables import SecurityInfo
from .time import time_tool
from .file import file_handler
from .objects.data import TradeData


class WatchListTool:

    def __init__(self, account_name: str):
        self.account_name = account_name
        self.MatchAccount = SecurityInfo.account == self.account_name
        self.watchlist_file = f'watchlist_{account_name}'

    def append(self, market: str, orderinfo: namedtuple, quotes: dict = 0):
        '''Add new stock data to watchlist'''

        if not db.HAS_DB:
            return

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

        if db.query(SecurityInfo, SecurityInfo.code == target).empty and check_quantity:
            strategy_pool = TradeData.Securities.Strategy
            data = {  # TODO: 新增欄位資料
                'account': self.account_name,
                'market': market,
                'code': target,
                'timestamp': datetime.now(),
                'position': position,
                'strategy': strategy_pool.get(target, 'unknown')
            }

            db.add_data(SecurityInfo, **data)

    def merge_info(self, tbInfo: pd.DataFrame):
        # TODO: delete

        watchlist = db.query(SecurityInfo)
        tbInfo = tbInfo.merge(
            watchlist,
            how='left',
            on=['account', 'market', 'code']
        )
        tbInfo.position.fillna(100, inplace=True)
        return tbInfo

    def update_position(self, order: namedtuple):
        target = order.target
        position = order.pos_target

        condition = SecurityInfo.code == target, self.MatchAccount
        watchlist = db.query(SecurityInfo, *condition)

        if watchlist.empty and order.action_type == 'Open':
            market = 'Stocks' if not order.octype else 'Futures'
            self.append(market, order)

        elif not watchlist.empty:
            db_position = watchlist.position.values[0]

            if order.action_type != 'Open':
                position *= -1

            db_position += position

            if db_position > 0:
                db.update(SecurityInfo, {'position': position}, *condition)
            else:
                db.delete(
                    SecurityInfo, SecurityInfo.position <= 0, self.MatchAccount)


class TradeDataHandler:
    @staticmethod
    def check_is_empty(target, market='Stocks'):
        data = TradeData.Securities.Monitor.get(target, {})
        quantity = data.get('order', {}).get('quantity', 0)
        position = data.get('position', 0)

        is_empty = (quantity <= 0 or position <= 0)
        logging.debug(
            f'[Monitor List]Check|{target}|{is_empty}|quantity: {quantity}; position: {position}|')
        return is_empty

    @staticmethod
    def reset_monitor(target: str, market='Stocks', day_trade=False):
        TradeData.Securities.Monitor.pop(target, None)
        if day_trade:
            logging.debug(f'[Monitor List]Reset|Securities|{target}|')
            TradeData.Securities.Monitor[target] = None

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
            if TradeData.Securities.Monitor[target] is not None:
                quantity = data['quantity']

                if action == 'New':
                    stage = 'Add|Target'
                    position *= -1
                    quantity *= -1
                else:
                    stage = 'Update|Target'

                TradeData.Securities.Monitor[target]['position'] -= position
                TradeData.Securities.Monitor[target]['quantity'] -= quantity
            else:
                stage = 'Add|Target'
                TradeData.Securities.Monitor[target] = data

            if 'None' not in stage:
                position_ = TradeData.Securities.Monitor[target]['position']
                quantity_ = TradeData.Securities.Monitor[target]['quantity']

        # New, Cover
        else:
            if TradeData.Securities.Monitor[target] is not None:
                quantity = data['order']['quantity']

                if action == 'New':
                    stage = 'Add|Target'
                    position *= -1
                    quantity *= -1
                else:
                    stage = 'Update|Target'

                TradeData.Securities.Monitor[target]['position'] -= position
                TradeData.Securities.Monitor[target]['order']['quantity'] -= quantity
            elif action == 'New':
                stage = 'Add|Target'

                date = TODAY_STR.replace('-', '/')
                data['contract'] = get_contract(target)
                data['isDue'] = date == data['contract'].delivery_date
                TradeData.Securities.Monitor[target] = data
            else:
                stage = 'None|Target'

            if 'None' not in stage:
                position_ = TradeData.Securities.Monitor[target]['position']
                quantity_ = TradeData.Securities.Monitor[target]['order']['quantity']

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
