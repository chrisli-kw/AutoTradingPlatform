import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from collections import namedtuple

from .database import db
from .time import time_tool
from .file import file_handler
from .notify import Notification
from .objects.data import TradeData
from .database.tables import SecurityInfo, PositionTable
from ..config import API, StrategyList, Cost, NotifyConfig


class WatchListTool:

    def __init__(self, account_name: str):
        self.account_name = account_name

    def update_position(self, order: namedtuple):
        # TODO: delete this function
        target = order.target
        position = 100
        watchlist = self.get_match_info(target)

        if watchlist.empty and order.action_type == 'Open':
            market = 'Stocks' if not order.octype else 'Futures'
            self.append(market, order)

        elif not watchlist.empty:
            db_position = watchlist.position.values[0]

            if order.action_type != 'Open':
                position *= -1

            db_position += position

            condition = self.match_target(target)
            if db_position > 0:
                db.update(SecurityInfo, {'position': position}, condition)
            else:
                condition &= SecurityInfo.position <= 0
                db.delete(SecurityInfo, condition)

    def match_target(self, target: str):
        condition = SecurityInfo.mode == TradeData.Account.Mode
        condition &= SecurityInfo.account == self.account_name
        condition &= SecurityInfo.code == target
        return condition

    def get_match_info(self, target):
        '''Get the watchlist info for the target'''
        condition = self.match_target(target)
        df = db.query(SecurityInfo, condition)
        return df

    def check_is_empty(self, target: str):
        '''Check if the target is empty in the monitor list'''

        data = self.get_match_info(target)
        data = data.to_dict('records')[0] if not data.empty else {}
        quantity = data.get('quantity', 0)
        position = data.get('position', 0)

        is_empty = (quantity == 0)
        if is_empty:
            logging.warning(
                f'[Monitor List]Check|{target}|{is_empty}|quantity: {quantity}; position: {position}|')
        return is_empty

    @staticmethod
    def _futures_cost_price(
            current_quantity: int,
            current_cost: float,
            fill_delta: int,
            fill_price: float
    ):
        new_quantity = current_quantity + fill_delta
        if not new_quantity:
            return 0
        if not current_quantity or current_quantity * new_quantity < 0:
            return fill_price
        if current_quantity * fill_delta > 0:
            old_amount = abs(current_quantity) * current_cost
            fill_amount = abs(fill_delta) * fill_price
            return (old_amount + fill_amount) / abs(new_quantity)
        return current_cost

    def _update_futures_monitor(
            self,
            target: str,
            data: dict,
            conf,
            df,
            oc_type: str = ''
    ):
        order = data['order']
        trade_id = data.get('trade_id') or order.get('trade_id')
        combo_tag = data.get('combo_tag') or order.get('combo_tag')
        is_open = oc_type in ['New', 'NewPosition', 'DayTrade']
        action = TradeDataHandler.normalize_action(order.get('action'))
        fill_quantity = abs(int(order.get('quantity', 0) or 0))
        fill_delta = TradeDataHandler.signed_quantity(
            fill_quantity, action, market='Futures')
        current_quantity = int(df.iloc[0].quantity) if not df.empty else 0
        new_quantity = current_quantity + fill_delta
        current_cost = float(df.iloc[0].cost_price) if not df.empty else 0
        fill_price = float(order.get('price', 0) or 0)
        cost_price = self._futures_cost_price(
            current_quantity, current_cost, fill_delta, fill_price)
        strategy_quantity = TradeDataHandler.strategy_quantity(
            new_quantity,
            mode=getattr(conf, 'mode', 'long'),
            action=action,
            market='Futures'
        )
        max_qty = conf.max_qty.get(target, 1) if conf else 1
        position = int(100*strategy_quantity/max_qty) if max_qty > 0 else 0

        condition = self.match_target(target)
        if new_quantity == 0:
            if not df.empty:
                db.delete(SecurityInfo, condition)
            return new_quantity

        position_action = 'Buy' if new_quantity > 0 else 'Sell'
        if df.empty:
            quote = TradeDataHandler.getQuotesNow(target)
            security_data = dict(
                mode=TradeData.Account.Mode,
                account=self.account_name,
                market='Futures',
                code=target,
                action=position_action,
                quantity=new_quantity,
                cost_price=cost_price,
                last_price=quote.get('price', 0),
                pnl=0,
                yd_quantity=0,
                order_cond=order.get('order_cond', 'Cash'),
                timestamp=datetime.now(),
                position=position,
                strategy=TradeData.Securities.Strategy.get(target, 'unknown')
            )
            if is_open and trade_id:
                security_data['trade_id'] = trade_id
            if is_open and combo_tag:
                security_data['combo_tag'] = combo_tag
            db.add_data(SecurityInfo, **security_data)
        else:
            update_data = {
                'action': position_action,
                'quantity': new_quantity,
                'cost_price': cost_price,
                'position': position,
            }
            if is_open and trade_id:
                update_data['trade_id'] = trade_id
            if is_open and combo_tag:
                update_data['combo_tag'] = combo_tag
            if current_quantity * new_quantity < 0:
                update_data['timestamp'] = datetime.now()
            db.update(SecurityInfo, update_data, condition)
        return new_quantity

    def update_monitor(
            self,
            oc_type: str,
            data: dict,
            sync_position: bool = False,
            notify_position: bool = False
    ):
        '''更新監控庫存(成交回報)'''

        if oc_type in ['NewPosition', 'DayTrade']:
            oc_type = 'New'

        try:
            target = data['code']
        except KeyError:
            target = data['contract']['code']
        conf = TradeDataHandler.getStrategyConfig(target)

        if conf is None:
            return

        df = self.get_match_info(target)
        if conf.market == 'Futures':
            broker_quantity = self._update_futures_monitor(
                target, data, conf, df, oc_type=oc_type)
        elif not df.empty:
            max_qty = conf.max_qty.get(target, 1) if conf else 1
            quantity = data['order']['quantity']

            if oc_type == 'New':
                if max_qty > 0 and max_qty > 0:
                    position = int(100*conf.raise_qty/max_qty)
                else:
                    position = 0
                position *= -1
                quantity *= -1
            else:
                if max_qty > 0 and max_qty > 0:
                    position = int(100*conf.stop_loss_qty/max_qty)
                else:
                    position = 0

            df['position'] -= position
            df['quantity'] -= quantity
            condition = self.match_target(target)
            db.update(SecurityInfo, df.iloc[0].to_dict(), condition)
            self.check_remove_monitor(target)
            broker_quantity = int(df.iloc[0].quantity)

        else:
            max_qty = conf.max_qty.get(target, 1) if conf else 1
            quote = TradeDataHandler.getQuotesNow(target)
            data = dict(
                mode=TradeData.Account.Mode,
                account=self.account_name,
                market='Stocks' if conf.market == 'Stocks' else 'Futures',
                code=target,
                action=data['order']['action'],
                quantity=data['order']['quantity'],
                cost_price=data['order']['price'],
                last_price=quote.get('price', 0),
                pnl=0,
                yd_quantity=0,
                order_cond=data['order'].get('order_cond', 'Cash'),
                timestamp=datetime.now(),
                position=int(100*conf.open_qty/max_qty) if max_qty > 0 else 0,
                strategy=TradeData.Securities.Strategy.get(target, 'unknown')
            )
            db.add_data(SecurityInfo, **data)
            broker_quantity = int(data['quantity'])

        if sync_position:
            self.sync_strategy_position(
                target,
                oc_type,
                data,
                broker_quantity=broker_quantity,
                notify_position=notify_position
            )

    def check_remove_monitor(self, target: str):
        is_empty = self.check_is_empty(target)
        if is_empty:
            condition = self.match_target(target)
            db.delete(SecurityInfo, condition)
        return is_empty

    def sync_strategy_position(
            self,
            target: str,
            oc_type: str,
            data: dict,
            broker_quantity: int = None,
            notify_position: bool = False,
            sync_reason: str = 'deal callback sync'
    ):
        conf = TradeDataHandler.getStrategyConfig(target)
        if conf is None or getattr(conf, 'positions', None) is None:
            return

        order = data.get('order', data)
        fill_quantity = abs(int(
            order.get('quantity', data.get('quantity', 0)) or 0))
        price = order.get('price', data.get(
            'price', data.get('cost_price', 0))) or 0
        if price == 0:
            price = TradeDataHandler.getQuotesNow(target).get('price', 0)

        strategy = TradeData.Securities.Strategy.get(target)
        if broker_quantity is None:
            df = self.get_match_info(target)
            broker_quantity = int(df.iloc[0].quantity) if not df.empty else 0

        market = getattr(conf, 'market', 'Futures')
        position_action = (
            'Buy' if broker_quantity > 0 else
            'Sell' if broker_quantity < 0 else
            TradeDataHandler.normalize_action(order.get('action'))
        )
        desired_quantity = TradeDataHandler.strategy_quantity(
            broker_quantity,
            mode=getattr(conf, 'mode', 'long'),
            action=position_action,
            market=market
        )

        inputs = {
            'mode': TradeData.Account.Mode,
            'account': self.account_name,
            'strategy': strategy,
            'name': target,
            'timestamp': datetime.now(),
            'price': price,
            'quantity': 0,
            'reason': sync_reason,
        }

        conf.positions.reload()
        current_quantity = int(conf.positions.total_qty.get(target, 0) or 0)
        quantity_delta = desired_quantity - current_quantity
        if quantity_delta > 0:
            inputs['quantity'] = quantity_delta
            avg_cost = conf.positions.open(inputs)
        elif quantity_delta < 0:
            inputs['quantity'] = abs(quantity_delta)
            avg_cost = conf.positions.close(inputs)
        else:
            avg_cost = conf.positions.average_cost()

        if hasattr(conf, 'avg_cost'):
            conf.avg_cost = avg_cost

        if notify_position:
            entries = [e for e in conf.positions.entries if e['name'] == target]
            name = entries[0]['name'] if entries else target
            total_qty = sum(e['quantity'] for e in entries)
            Notification(NotifyConfig, account=self.account_name).post_human_deal(
                name, oc_type, fill_quantity, total_qty)


class TradeDataHandler:
    @staticmethod
    def normalize_action(action):
        return getattr(action, 'value', action) or ''

    @staticmethod
    def signed_quantity(quantity: int, action: str, market='Futures'):
        quantity = abs(int(quantity or 0))
        action = TradeDataHandler.normalize_action(action)
        if market == 'Futures' and action == 'Sell':
            return -quantity
        return quantity

    @staticmethod
    def strategy_quantity(
            broker_quantity: int,
            mode='long',
            action='',
            market='Futures'
    ):
        quantity = int(broker_quantity or 0)
        action = TradeDataHandler.normalize_action(action)
        if market == 'Futures':
            direction = -1 if mode == 'short' else 1
            return max(quantity * direction, 0)
        expected_action = 'Sell' if mode == 'short' else 'Buy'
        return max(quantity, 0) if action == expected_action else 0

    @staticmethod
    def reset_monitor(target: str):
        TradeData.Securities.Monitor[target] = None

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
        return {}

    @staticmethod
    def getOptionOrderStatus(label: str):
        return TradeData.Futures.OptionOrderStatus.get(label, {})

    @staticmethod
    def getStrategyConfig(target: str):
        strategy = TradeData.Securities.Strategy.get(target)
        conf = StrategyList.Config.get(strategy)
        return conf

    @staticmethod
    def getFuturesQuota(account: str):
        '''更新可開倉的期貨標的數'''

        open_deals = len(TradeData.Futures.Opened)
        close_deals = len(TradeData.Futures.Closed)
        df = db.query(
            SecurityInfo,
            SecurityInfo.mode == TradeData.Account.Mode,
            SecurityInfo.account == account,
            SecurityInfo.market == 'Futures'
        )
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

    @staticmethod
    def unify_monitor_data(account_name: str):
        watchlist = WatchListTool(account_name)
        for code, strategy in TradeData.Securities.Strategy.items():
            if code not in TradeData.Securities.Monitor:
                TradeData.Securities.Monitor.update({code: None})

            conf = TradeDataHandler.getStrategyConfig(code)
            if conf is None:
                continue

            condition_info = (
                SecurityInfo.mode == TradeData.Account.Mode,
                SecurityInfo.account == account_name,
                SecurityInfo.code == code
            )
            condition_table = (
                PositionTable.mode == TradeData.Account.Mode,
                PositionTable.account == account_name,
                PositionTable.name == code,
                PositionTable.strategy == strategy
            )

            df = db.query(SecurityInfo, *condition_info)

            monitor_data = TradeData.Securities.Monitor.get(code)
            if monitor_data is None:
                if (
                    getattr(conf, 'positions', None) is not None and
                    conf.positions.total_qty.get(code, 0)
                ):
                    db.delete(PositionTable, *condition_table)
                    conf.positions.reload()
                continue

            data = monitor_data.copy()
            broker_quantity = int(data.get('quantity', 0) or 0)
            market = getattr(conf, 'market', data.get('market', 'Futures'))
            strategy_quantity = TradeDataHandler.strategy_quantity(
                broker_quantity,
                mode=getattr(conf, 'mode', 'long'),
                action=data.get('action', ''),
                market=market
            )
            max_qty = conf.max_qty.get(code, 1)
            data.update({
                'mode': TradeData.Account.Mode,
                'timestamp': data.get('timestamp') or datetime.now(),
                'position': int(100*strategy_quantity/max_qty)
                if max_qty > 0 else 0,
                'strategy': strategy,
            })

            columns = SecurityInfo.__table__.columns.keys()
            security_data = {
                key: value for key, value in data.items()
                if key in columns and key not in ['pk_id', 'create_time']
            }
            if df.empty:
                db.add_data(SecurityInfo, **security_data)
            else:
                db.update(SecurityInfo, security_data, *condition_info)

            if code in getattr(conf, 'FILTER_OUT', []):
                continue

            sync_data = {
                'quantity': abs(broker_quantity),
                'action': data.get('action', ''),
                'price': data.get('cost_price', 0),
                'cost_price': data.get('cost_price', 0),
            }
            watchlist.sync_strategy_position(
                code,
                'Sync',
                sync_data,
                broker_quantity=broker_quantity,
                notify_position=False,
                sync_reason='unify monitor data sync'
            )


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
    def __init__(self, account_name: str, strategy: str, backtest: bool = False):
        self.account_name = account_name
        self.strategy = strategy
        self.backtest = backtest
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
        self.total_qty = {}
        self.total_profit = 0.0

        if not backtest:
            self.reload()

    def reload(self):
        if self.backtest:
            return self

        df = db.query(
            PositionTable,
            PositionTable.mode == TradeData.Account.Mode,
            PositionTable.account == self.account_name,
            PositionTable.strategy == self.strategy
        )
        self.entries = df.to_dict('records')
        self.total_qty = df.groupby(
            'name').quantity.sum().to_dict() if not df.empty else {}
        return self

    def average_cost(self):
        if self.entries:
            costs = [e.get('price', 0) for e in self.entries]
            qtys = [e.get('quantity', 0) for e in self.entries]
            try:
                return np.average(costs, weights=qtys)
            except ZeroDivisionError:
                return 0
        return np.inf

    def open(self, inputs: dict):
        self.entries.append(inputs)

        name = inputs['name']
        total_qty = self.total_qty.get(name, 0)
        self.total_qty[name] = total_qty + inputs['quantity']

        if not self.backtest and inputs.get('quantity', 0) > 0:
            db.add_data(PositionTable, **inputs)

        return self.average_cost()

    def close(self, inputs: dict):
        name = inputs['name']
        price = inputs['price']
        qty = min(inputs.get('quantity', 0), self.total_qty.get(name, 0))
        reason = inputs.get('reason', '平倉')

        closed_qty = 0
        total_profit = 0.0

        entries = [e for e in self.entries if e['name'] == name]

        while closed_qty < qty and entries:
            entry = entries[0]
            remaining_qty = qty - closed_qty

            if entry['quantity'] > remaining_qty:
                e_qty = remaining_qty
                entry['quantity'] -= e_qty
                self.update_entries(entry)
            else:
                entries.pop(0)
                self.entries.remove(entry)
                e_qty = entry['quantity']
                self.delete_entries(entry)

            closed_qty += e_qty

            entry_price = entry['price']
            profit = (price - entry_price) * e_qty
            total_profit += profit

            self.exits.append({
                'entry_price': entry_price,
                'exit_price': price,
                'profit': profit,
                'qty': e_qty,
                'open_time': entry['timestamp'],
                'close_time': inputs['timestamp'],
                'open_reason': entry.get('reason', '建倉'),
                'reason': reason
            })

        self.total_qty[name] = self.total_qty.get(name, 0) - closed_qty
        self.total_profit += total_profit

        return self.average_cost()

    def is_open(self, name: str):
        return self.total_qty.get(name, 0) > 0

    def query_condition(self, inputs: dict):
        '''Query condition for the position table in the database'''

        return (
            PositionTable.mode == inputs['mode'],
            PositionTable.account == inputs['account'],
            PositionTable.strategy == inputs['strategy'],
            PositionTable.name == inputs['name'],
            PositionTable.timestamp == inputs['timestamp']
        )

    def delete_entries(self, inputs: dict):
        '''Delete entries from the position table'''

        if not db.HAS_DB:
            return

        if self.backtest:
            return

        condition = self.query_condition(inputs)
        logging.warning('Delete entries from the position table')
        db.delete(PositionTable, *condition)

    def update_entries(self, inputs: dict):
        '''Update the position table in the database'''

        if not db.HAS_DB:
            return

        if self.backtest:
            return

        condition = self.query_condition(inputs)
        db.update(PositionTable, inputs, *condition)
