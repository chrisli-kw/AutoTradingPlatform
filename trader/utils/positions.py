import os
import logging
import pandas as pd
from datetime import datetime
from collections import namedtuple

from ..config import API, StrategyList, Cost
from .database import db
from .database.tables import SecurityInfo, PositionTable
from .time import time_tool
from .file import file_handler
from .objects.data import TradeData


class WatchListTool:

    def __init__(self, account_name: str):
        self.account_name = account_name
        self.MatchAccount = SecurityInfo.account == self.account_name
        self.watchlist_file = f'watchlist_{account_name}'

    def append(self, market: str, orderinfo: namedtuple):
        # TODO: delete
        '''Add new stock data to watchlist'''

        if not db.HAS_DB:
            return

        if isinstance(orderinfo, str):
            # Manual trading
            target = orderinfo
            position = 100
            check_quantity = True
        else:
            # Auto trading
            target = orderinfo.target
            position = 100
            check_quantity = orderinfo.quantity != 0

        if db.query(SecurityInfo, SecurityInfo.code == target).empty and check_quantity:
            strategy_pool = TradeData.Securities.Strategy
            data = {
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
        position = 100

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

    def match_target(self, target: str):
        return SecurityInfo.code == target, SecurityInfo.account == self.account_name

    def check_is_empty(self, target: str):
        '''Check if the target is empty in the monitor list'''
        condition = self.match_target(target)
        data = db.query(SecurityInfo, *condition)
        data = data.to_dict('record')[0] if not data.empty else {}
        quantity = data.get('quantity', 0)
        position = data.get('position', 0)

        is_empty = (quantity <= 0 or position <= 0)
        logging.debug(
            f'[Monitor List]Check|{target}|{is_empty}|quantity: {quantity}; position: {position}|')
        return is_empty

    def update_monitor(self, action: str, data: dict):
        '''更新監控庫存(成交回報)'''

        target = data['code']
        conf = TradeDataHandler.getStrategyConfig(target)
        condition = self.match_target(target)
        df = db.query(SecurityInfo, *condition)
        if not df.empty:
            quantity = data['order']['quantity']

            if action == 'New':
                position = int(100*conf.raise_qty/conf.max_qty)
                position *= -1
                quantity *= -1
            else:
                position = int(100*conf.stop_loss_qty/conf.max_qty)

            df['position'] -= position
            df['quantity'] -= quantity
            db.update(SecurityInfo, df.iloc[0].to_dict(), *condition)
            self.check_remove_monitor(target)

        else:
            data = dict(
                account=self.account_name,
                market='Stocks' if conf.market == 'Stocks' else 'Futures',
                code=target,
                action=action,
                quantity=data['order']['quantity'],
                cost_price=data['order']['price'],
                last_price=TradeDataHandler.getQuotesNow(
                    target).get('price', 0),
                pnl=0,
                yd_quantity=0,
                order_cond=data['order'].get('order_cond', 'Cash'),
                timestamp=datetime.now(),
                position=int(100*conf.open_qty/conf.max_qty),
                strategy=TradeData.Securities.Strategy.get(
                    target, 'unknown')
            )
            db.add_data(SecurityInfo, **data)

    def check_remove_monitor(self, target: str):
        is_empty = self.check_is_empty(self.account_name, target)
        if is_empty:
            logging.debug(f'[Monitor List]Remove|{target}|')

            TradeDataHandler.reset_monitor(target)
            condition = self.match_target(target)
            db.delete(SecurityInfo, *condition)
        return is_empty


class TradeDataHandler:
    @staticmethod
    def reset_monitor(target: str):
        TradeData.Securities.Monitor.pop(target, None)

    @staticmethod
    def update_deal_list(target: str, action_type: str):
        '''更新下單暫存清單'''

        logging.debug(f'[Monitor List]{action_type}|{target}|')
        if action_type in ['Buy', 'Sell']:
            if action_type == 'Sell' and len(target) == 4 and target not in TradeData.Stocks.Sold:
                TradeData.Stocks.Sold.append(target)

            if action_type == 'Buy' and len(target) == 4 and target not in TradeData.Stocks.Bought:
                TradeData.Stocks.Bought.append(target)

        elif action_type in ['New', 'Cover']:
            if action_type == 'New' and target not in TradeData.Futures.Opened:
                TradeData.Futures.Opened.append(target)

            if action_type == 'Cover' and target not in TradeData.Futures.Closed:
                TradeData.Futures.Closed.append(target)

        elif action_type == 'Cancel':
            if target in TradeData.Stocks.Bought:
                TradeData.Stocks.Bought.remove(target)

            if target in TradeData.Futures.Opened:
                TradeData.Futures.Opened.remove(target)

    @staticmethod
    def getQuotesNow(target: str):
        if target in TradeData.Quotes.NowIndex:
            return TradeData.Quotes.NowIndex[target]
        elif target in TradeData.Quotes.NowTargets:
            return TradeData.Quotes.NowTargets[target]
        return -1

    @staticmethod
    def getStrategyConfig(target: str):
        strategy = TradeData.Securities.Strategy.get(target)
        conf = StrategyList.Config.get(strategy)
        return conf

    @staticmethod
    def getFuturesQuota():
        '''更新可開倉的期貨標的數'''

        open_deals = len(TradeData.Futures.Opened)
        close_deals = len(TradeData.Futures.Closed)
        df = db.query(SecurityInfo, SecurityInfo.market == 'Futures')
        quota = TradeData.Futures.Limit-df.shape[0] - open_deals + close_deals
        return quota

    @staticmethod
    def getStocksQuota(mode: str):
        '''更新可開倉的股票標的數'''

        strategies = TradeData.Securities.Strategy

        buy_deals = 0
        for s in TradeData.Stocks.Bought:
            strategy = strategies.get(s)
            conf = StrategyList.Config.get(strategy)
            if conf.mode == 'long':
                buy_deals += 1

        sell_deals = 0
        for s in TradeData.Stocks.Sold:
            strategy = strategies.get(s)
            conf = StrategyList.Config.get(strategy)
            if conf.mode == 'long':
                sell_deals += 1

        if mode == 'long':
            quota = abs(TradeData.Stocks.LimitLong) - \
                TradeData.Stocks.N_Long - buy_deals + sell_deals
        else:
            quota = abs(TradeData.Stocks.LimitShort) - \
                TradeData.Stocks.N_Short + buy_deals - sell_deals
        return quota


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
    def __init__(self, account_name: str, strategy: str, sim_trade: bool = False):
        self.account_name = account_name
        self.strategy = strategy
        self.sim_trade = sim_trade
        self.entries = []
        # [{
        #     'account': str,
        #     'name': str,
        #     'price': float,
        #     'timestamp': time
        #     'quantity': int,
        #     'reason': str  # 建倉、加碼、停損、停利
        # }]
        self.exits = []
        self.total_qty = 0
        self.total_profit = 0.0

        if not sim_trade:
            df = db.query(
                PositionTable,
                PositionTable.account == account_name,
                PositionTable.strategy == strategy
            )
            self.entries = df.to_dict('records')
            self.total_qty = df.quantity.sum() if not df.empty else 0

    def open(self, inputs: dict):
        self.entries.append(inputs)
        self.total_qty += inputs['quantity']

        if not self.sim_trade:
            db.add_data(PositionTable, **inputs)

    def close(self, inputs: dict):
        price = inputs['price']
        qty = min(inputs['quantity'], self.total_qty)
        reason = inputs.get('reason', '平倉')

        closed_qty = 0
        profit = 0.0
        entries = [e for e in self.entries if e['name'] == inputs['name']]
        while closed_qty < qty and entries:
            if entries[0]['quantity'] > qty:
                entry = entries[0]
                e_qty = qty
                entry['quantity'] -= qty
                self.entries[self.entries.index(entry)] = entry
                self.update_entries(entry)
            else:
                entry = entries.pop(0)
                entry = self.entries.pop(self.entries.index(entry))
                e_qty = entry['quantity']
                self.delete_entries(entry)

            closed_qty += e_qty
            entry_price = entry['price']
            profit = (price - entry_price) * e_qty
            self.exits.append({
                'entry_price': entry_price,
                'exit_price': price,
                'profit': profit,
                'qty': e_qty,
                'open_time': entry['timestamp'],
                'close_time': inputs['timestamp'],
                'reason': reason
            })
        self.total_qty -= closed_qty
        self.total_profit += profit
        return profit

    def is_open(self):
        return self.total_qty > 0

    def delete_entries(self, inputs: dict):
        '''Delete entries from the position table'''

        if not db.HAS_DB:
            return

        if self.sim_trade:
            return

        condition = (
            PositionTable.account == inputs['account'],
            PositionTable.strategy == inputs['strategy'],
            PositionTable.name == inputs['name'],
            PositionTable.timestamp == inputs['timestamp']
        )
        db.delete(PositionTable, *condition)

    def update_entries(self, inputs: dict):
        '''Update the position table in the database'''

        if not db.HAS_DB:
            return

        if self.sim_trade:
            return

        condition = (
            PositionTable.account == inputs['account'],
            PositionTable.strategy == inputs['strategy'],
            PositionTable.name == inputs['name'],
            PositionTable.timestamp == inputs['timestamp']
        )
        db.update(PositionTable, inputs, *condition)
