import os
import ssl
import time
import logging
import numpy as np
import pandas as pd
from sys import platform
from typing import Dict
from collections import namedtuple
from datetime import datetime, timedelta

from shioaji import constant
from . import API, PATH, TODAY, TODAY_STR, __version__
from .config import StrategyLong, StrategyShort, StrategyLongDT, StrategyShortDT
from .config import FEE_RATE, TStart, TEnd, TTry, TimeStartStock, TimeStartFuturesDay
from .config import TimeEndFuturesDay, TimeStartFuturesNight, TimeEndFuturesNight
from .utils import get_contract, save_csv
from .utils.kbar import KBarTool
from .utils.accounts import AccountInfo
from .utils.watchlist import WatchListTool
from .utils.cipher import CipherTool
from .utils.notify import Notification
from .utils.orders import OrderTool
from .utils.database import db
from .utils.database.redis import RedisTools
from .utils.database.tables import SecurityInfoStocks, SecurityInfoFutures
from .utils.subscribe import Subscriber
from .strategies.long import LongStrategy
from .strategies.short import ShortStrategy


ssl._create_default_https_context = ssl._create_unverified_context


class StrategyExecutor(AccountInfo, WatchListTool, KBarTool, OrderTool, RedisTools, Subscriber):
    def __init__(self, config=None, kbar_script=None):
        self.ct = CipherTool(decrypt=True, encrypt=False)
        self.CONFIG = config

        # 交易帳戶設定
        self.ACCOUNT_NAME = self.getENV('ACCOUNT_NAME')
        self.__API_KEY__ = self.getENV('API_KEY')
        self.__SECRET_KEY__ = self.getENV('SECRET_KEY')
        self.__ACCOUNT_ID__ = self.getENV('ACCOUNT_ID', 'code')

        # 股票使用者設定
        self.KBAR_START_DAYay = self.getENV('KBAR_START_DAYay', 'date')
        self.MODE = self.getENV('MODE')
        self.MARKET = self.getENV('MARKET')
        self.FILTER_OUT = self.getENV('FILTER_OUT', 'list')
        self.STRATEGY_STOCK = self.getENV('STRATEGY_STOCK', 'list')
        self.PRICE_THRESHOLD = self.getENV('PRICE_THRESHOLD', 'int')
        self.INIT_POSITION = self.getENV('INIT_POSITION', 'int')
        self.POSITION_LIMIT_LONG = self.getENV('POSITION_LIMIT_LONG', 'int')
        self.POSITION_LIMIT_SHORT = self.getENV('POSITION_LIMIT_SHORT', 'int')
        self.N_STOCK_LIMIT_TYPE = self.getENV('N_STOCK_LIMIT_TYPE')
        self.N_LIMIT_LS = self.getENV('N_LIMIT_LS', 'int')
        self.N_LIMIT_SS = self.getENV('N_LIMIT_SS', 'int')
        self.BUY_UNIT = self.getENV('BUY_UNIT', 'int')
        self.BUY_UNIT_TYPE = self.getENV('BUY_UNIT_TYPE')
        self.ORDER_COND1 = self.getENV('ORDER_COND1')
        self.ORDER_COND2 = self.getENV('ORDER_COND2')
        self.LEVERAGE_LONG = {}
        self.LEVERAGE_SHORT = {}
        self.day_trade_cond = {
            'MarginTrading': 'ShortSelling',
            'ShortSelling': 'MarginTrading',
            'Cash': 'Cash'
        }
        self.simulation = self.MODE == 'Simulation'

        # 期貨使用者設定
        self.MODE_FUTURES = self.getENV('MODE_FUTURES')  # TODO: delete
        self.STRATEGY_FUTURES = self.getENV('STRATEGY_FUTURES', 'list')
        self.MARGIN_LIMIT = self.getENV('MARGIN_LIMIT', 'int')
        self.N_FUTURES_LIMIT_TYPE = self.getENV('N_FUTURES_LIMIT_TYPE')
        self.N_FUTURES_LIMIT = self.getENV('N_FUTURES_LIMIT', 'int')
        self.N_SLOT = self.getENV('N_SLOT', 'int')
        self.N_SLOT_TYPE = self.getENV('N_SLOT_TYPE')

        super().__init__()
        KBarTool.__init__(self, self.KBAR_START_DAYay)
        OrderTool.__init__(self)
        RedisTools.__init__(self)
        Subscriber.__init__(self)
        WatchListTool.__init__(self, self.ACCOUNT_NAME)

        # 股票可進場籌碼 (進場時判斷用)
        self.simulate_amount = np.iinfo(np.int64).max
        self.stocks = pd.DataFrame()
        self.desposal_money = 0
        self.stock_bought = []
        self.stock_sold = []
        self.n_stocks_long = 0
        self.n_stocks_short = 0
        self.total_market_value = 0
        self.punish_list = []
        self.pct_chg_DowJones = self.get_pct_chg_DowJones()
        self.n_categories = None

        # 期貨可進場籌碼 (進場時判斷用)
        self.futures_opened = []
        self.futures_closed = []
        self.n_futures = 0
        self.futures = pd.DataFrame()
        self.Futures_Code_List = {}
        self.margin_table = None

        # 交易相關
        self.can_stock = 'stock' in self.MARKET
        self.can_sell = self.MODE not in ['LongBuy', 'ShortBuy']
        self.can_buy = self.MODE not in ['LongSell', 'ShortSell']
        self.can_futures = 'futures' in self.MARKET
        self.stocks_to_monitor = {}
        self.futures_to_monitor = {}

        # 載入指標模組
        params = dict(
            account_name=self.ACCOUNT_NAME,
            hold_day=self.getENV('HOLD_DAY', 'int'),
            is_simulation=self.simulation,
            stock_limit_type=self.N_STOCK_LIMIT_TYPE,
            futures_limit_type=self.N_FUTURES_LIMIT_TYPE,
            stock_limit_long=self.N_LIMIT_LS,
            stock_limit_short=self.N_LIMIT_SS,
            futures_limit=self.N_FUTURES_LIMIT,
        )
        self.strategy_l = LongStrategy(**params)
        self.strategy_s = ShortStrategy(**params)
        self.notifier = Notification()
        self.set_kbar_scripts(kbar_script)

    def getENV(self, key: str, _type: str = 'text'):
        if self.CONFIG and key in self.CONFIG:
            env = self.CONFIG[key]

            if _type == 'int':
                return int(env)
            elif _type == 'list':
                if 'none' in env.lower():
                    return []
                return env.replace(' ', '').split(',')
            elif _type == 'date' and env:
                return pd.to_datetime(env)
            elif _type == 'code':
                return self.ct.decrypt(env)
            return env
        elif _type == 'int':
            return 0
        elif _type == 'list':
            return []
        return None

    def _set_trade_risks(self):
        '''設定交易風險值: 可交割金額、總市值'''

        cost_value = (self.stocks.quantity * self.stocks.cost_price).sum()
        pnl = self.stocks.pnl.sum()
        if self.simulation:
            account_balance = self.INIT_POSITION
            settle_info = pnl
        else:
            account_balance = self.balance()
            settle_info = self.settle_info(mode='info').iloc[1:, 1].sum()

        self.desposal_money = min(
            account_balance+settle_info, self.POSITION_LIMIT_LONG)
        self.total_market_value = self.desposal_money + cost_value + pnl

        logging.info(
            f'Desposal amount = {self.desposal_money} (limit: {self.POSITION_LIMIT_LONG})')

    def _set_margin_limit(self):
        '''計算可交割的保證金額，不可超過帳戶可下單的保證金額上限'''
        if self.simulation:
            self.desposal_margin = self.simulate_amount
            self.ProfitAccCount = self.simulate_amount
        else:
            self.get_account_margin()
        self.desposal_margin = min(self.desposal_margin, self.MARGIN_LIMIT)
        logging.info(f'權益總值: {self.ProfitAccCount}')
        logging.info(
            f'Margin available = {self.desposal_margin} (limit: {self.MARGIN_LIMIT})')

    def _set_leverage(self, stockids: list):
        '''
        取得個股融資成數資料，
        若帳戶設定為不可融資，則全部融資成數為0
        '''

        df = pd.DataFrame([self.get_leverage(s) for s in stockids])
        if df.shape[0]:
            df.loc[df.個股融券信用資格 == 'N', '融券成數'] = 100
            df.代號 = df.代號.astype(str)
            df.融資成數 /= 100
            df.融券成數 /= 100

            if self.ORDER_COND1 != 'Cash':
                self.LEVERAGE_LONG = df.set_index('代號').融資成數.to_dict()
            else:
                self.LEVERAGE_LONG = {code: 0 for code in stockids}

            if self.ORDER_COND2 != 'Cash':
                self.LEVERAGE_SHORT = df.set_index('代號').融券成數.to_dict()
            else:
                self.LEVERAGE_SHORT = {code: 1 for code in stockids}

        logging.info(f'Long leverages: {self.LEVERAGE_LONG}')
        logging.info(f'Short leverages: {self.LEVERAGE_SHORT}')

    def _set_futures_code_list(self):
        '''期貨商品代號與代碼對照表'''
        if self.can_futures and self.HAS_FUTOPT_ACCOUNT and self.Futures_Code_List == {}:
            logging.debug('Set Futures_Code_List')
            self.Futures_Code_List = {
                f.code: f.symbol for m in API.Contracts.Futures for f in m}

    def __order_callback(self, stat, msg):
        '''處理委託/成交回報'''

        if stat == constant.OrderState.StockOrder:
            stock = msg['contract']['code']
            order = msg['order']
            operation = msg['operation']

            c2 = operation['op_code'] == '00' or operation['op_msg'] == ''
            c3 = order['action'] == 'Buy'
            c4 = operation['op_code'] == '88' and '此證券配額張數不足' in operation['op_msg']
            c5 = len(stock) == 4

            if order['account']['account_id'] == self.account_id_stock:
                self.notifier.post_tftOrder(stat, msg)

                if c3 and c5 and stock not in self.stock_bought:
                    self.stock_bought.append(stock)

                leverage = self.check_leverage(stock, order['order_cond'])
                if c2 and c3:
                    # 記錄委託成功的買單
                    price = self.Quotes.NowTargets[stock]['price'] if stock in self.Quotes.NowTargets else order['price']
                    quantity = order['quantity']
                    quantity = quantity * \
                        1000 if order['order_lot'] == 'Common' else quantity
                    order_data = {
                        'Time': datetime.now(),
                        'market':'Stocks',
                        'code': stock,
                        'action': order['action'],
                        'price': price,
                        'quantity': quantity,
                        'amount': self.get_stock_amount(stock, price, quantity, order['order_cond']),
                        'order_cond': order['order_cond'],
                        'order_lot': order['order_lot'],
                        'leverage': leverage,
                        'account_id': order['account']['account_id']
                    }
                    self.appendOrder(order_data)

                # 若融資配額張數不足，改現股買進 ex: '此證券配額張數不足，餘額 0 張（證金： 0 ）'
                elif c4:
                    q_balance = operation['op_msg'].split(' ')
                    if len(q_balance) > 1:
                        q_balance = int(q_balance[1])
                        infos = dict(
                            action=order['action'], target=stock, pos_target=100, pos_balance=100)
                        # 若本日還沒有下過融資且剩餘券數為0，才可以改下現股
                        if q_balance == 0 and stock not in self.stock_bought:
                            orderinfo = self.OrderInfo(
                                quantity=1000 *
                                int(order['quantity']*(1-leverage)),
                                order_cond='Cash',
                                **infos
                            )
                            self._place_order(orderinfo, market='Stocks')

                        elif q_balance > 0:
                            orderinfo = self.OrderInfo(
                                quantity=q_balance,
                                order_cond=order['order_cond'],
                                **infos
                            )
                            self._place_order(orderinfo, market='Stocks')

                # 若刪單成功就自清單移除
                if operation['op_type'] == 'Cancel':
                    self.deleteOrder(stock)
                    if c3 and stock in self.stock_bought:
                        self.stock_bought.remove(stock)

        elif stat == constant.OrderState.StockDeal and msg['account_id'] == self.account_id_stock:
            stock = msg['code']
            msg.update({
                'position': 100,
                'yd_quantity': 0,
                'bst': datetime.now(),
                'cost_price': msg['price']
            })
            self.notifier.post_tftDeal(stat, msg)

            if msg['order_lot'] == 'Common':
                msg['quantity'] *= 1000

            quantity = msg['quantity']
            if msg['action'] == 'Sell':

                if stock not in self.stock_sold and len(stock) == 4:
                    self.stock_sold.append(stock)

                price = msg['price']
                leverage = self.check_leverage(stock, msg['order_cond'])
                cost_price = self.get_cost_price(
                    stock, price, msg['order_cond'])

                # 紀錄成交的賣單
                order_data = {
                    'Time': datetime.now(),
                    'market':'Stocks',
                    'code': stock,
                    'price': -price,
                    'quantity': quantity,
                    # 賣出金額 - 融資金額 - 手續費
                    'amount': -(price - cost_price*leverage)*quantity*(1 - FEE_RATE),
                    'order_cond': msg['order_cond'],
                    'order_lot': msg['order_lot'],
                    'leverage': leverage,
                    'account_id': msg['account_id']
                }
                self.appendOrder(order_data)

            # 更新監控庫存
            if not self.simulation:
                self.update_monitor_lists(stock, msg['action'], msg, quantity)

        elif stat == constant.OrderState.FuturesOrder:
            code = msg['contract']['code']
            symbol = code + msg['contract']['delivery_month']
            msg.update({
                'symbol': symbol,
                'cost_price': self.Quotes.NowTargets[symbol]['price'] if symbol in self.Quotes.NowTargets else 0,
                'bsh': max(self.Quotes.AllTargets[symbol]['price']) if self.Quotes.AllTargets[symbol]['price'] else 0,
                'bst': datetime.now(),
                'position': 100
            })
            order = msg['order']
            operation = msg['operation']

            if order['account']['account_id'] == self.account_id_futopt:
                self.notifier.post_fOrder(stat, msg)
                if operation['op_code'] == '00' or operation['op_msg'] == '':
                    self._update_futures_deal_list(symbol, order['oc_type'])

                    # 紀錄成交的賣單
                    price = -order['price'] if c4 else order['price']
                    sign = -1 if order['oc_type'] == 'Cover' else 1
                    quantity = order['quantity']
                    order_data = {
                        'Time': datetime.now(),
                        'market':'Futures',
                        'code': symbol,
                        'price': price*sign,
                        'quantity': quantity,
                        'amount': self.get_open_margin(symbol, quantity)*sign,
                        'op_type': order['oc_type'],
                        'account_id': order['account']['account_id']
                    }
                    self.appendOrder(order_data)

                # 若刪單成功就自清單移除
                if operation['op_type'] == 'Cancel':
                    self.deleteOrder(symbol)
                    if symbol in self.futures_opened:
                        self.futures_opened.remove(symbol)

                # 更新監控庫存
                if not self.simulation:
                    self.update_monitor_lists(
                        stock, operation['op_type'], msg, order['quantity'])

        elif stat == constant.OrderState.FuturesDeal:
            self.notifier.post_fDeal(stat, msg)

    def login_and_activate(self):
        # 登入
        self._login(self.__API_KEY__, self.__SECRET_KEY__, self.ACCOUNT_NAME)
        self.account_id_stock = API.stock_account.account_id
        logging.info(f'Stock account ID: {self.account_id_stock}')

        if self.HAS_FUTOPT_ACCOUNT:
            self.can_futures = 'futures' in self.MARKET
            self.account_id_futopt = API.futopt_account.account_id
            logging.info(f'Futures account ID: {self.account_id_futopt}')

        # 啟動憑證 (Mac 不需啟動)
        if platform != 'darwin':
            logging.info(f'Activate {self.ACCOUNT_NAME} CA')
            API.activate_ca(
                ca_path=f"./lib/ekey/551/{self.__ACCOUNT_ID__}/S/Sinopac.pfx",
                ca_passwd=self.__ACCOUNT_ID__,
                person_id=self.__ACCOUNT_ID__,
            )

        # 系統 callback 設定
        self._set_callbacks()

    def _set_callbacks(self):
        '''取得API回傳報價'''
        @API.on_tick_stk_v1()
        def stk_quote_callback_v1(exchange, tick):
            if tick.intraday_odd == 0 and tick.simtrade == 0:

                if tick.code not in self.Quotes.NowTargets:
                    logging.debug(f'First quote of {tick.code}')

                tick_data = self.stk_quote_v1(tick)
                # self.to_redis({tick.code: tick_data})

        @API.on_tick_fop_v1()
        def fop_quote_callback_v1(exchange, tick):
            try:
                if tick.simtrade == 0:
                    symbol = self.Futures_Code_List[tick.code]

                    if symbol not in self.Quotes.NowTargets:
                        logging.debug(f'First quote of {symbol}')

                    tick_data = self.fop_quote_v1(symbol, tick)
                    # self.to_redis({symbol: tick_data})
            except KeyError:
                logging.exception('KeyError: ')

        @API.quote.on_quote
        def quote_callback(topic: str, quote: dict):
            self.index_v0(quote)

        @API.quote.on_event
        def event_callback(resp_code: int, event_code: int, info: str, event: str):
            if 'Subscription Not Found' in info:
                logging.warning(info)

            else:
                logging.info(
                    f'Response code: {resp_code} | Event code: {event_code} | info: {info} | Event: {event}')

                if info == 'Session connect timeout' or event_code == 1:
                    time.sleep(5)
                    logging.info(f'登出系統: {API.logout()}')
                    logging.warning('Re-login')

                    time.sleep(5)
                    self._login(self.__API_KEY__,
                                self.__SECRET_KEY__, self.ACCOUNT_NAME)

        # 訂閱下單回報
        API.set_order_callback(self.__order_callback)

        # 訂閱五檔回報
        @API.on_bidask_stk_v1()
        def stk_quote_callback(exchange, bidask):
            self.BidAsk[bidask.code] = bidask

        @API.on_bidask_fop_v1()
        def fop_quote_callback(exchange, bidask):
            symbol = self.Futures_Code_List[bidask.code]
            self.BidAsk[symbol] = bidask

    def _log_and_notify(self, msg: str):
        '''將訊息加入log並推播'''
        logging.info(msg)
        self.notifier.post(f'\n{msg}', msgType='Monitor')

    def _filter_out_targets(self, market='Stocks'):
        '''過濾關注股票'''

        if market == 'Stocks':
            self.stocks = self.stocks[~self.stocks.code.isin(self.FILTER_OUT)]
        else:
            self.futures = self.futures[~self.futures.Code.isin(
                self.FILTER_OUT)]

    def init_stocks(self):
        '''初始化股票資訊'''

        if not self.can_stock:
            return None, []

        # 讀取選股清單
        stocks_pool = self.get_stock_pool()

        # 取得遠端庫存
        self.stocks = self.get_securityInfo('Stocks')

        # 取得策略清單
        strategy_w = self.find_strategy(stocks_pool, 'All')
        self.init_watchlist(self.stocks, strategy_w)

        # 庫存的處理
        self.stocks = self.stocks.merge(
            self.watchlist, 
            how='left', 
            on=['account', 'market', 'code']
        )
        self.stocks.position.fillna(100, inplace=True)
        strategies = self.find_strategy(stocks_pool, 'Stocks')

        # 剔除不堅控的股票
        self._filter_out_targets(market='Stocks')

        # 新增歷史K棒資料
        self.update_stocks_to_monitor(stocks_pool)
        all_targets = list(self.stocks_to_monitor)
        self.history_kbars(['TSE001', 'OTC101'] + all_targets)

        # 交易風險控制
        self.n_stocks_long = self.stocks[self.stocks.action == 'Buy'].shape[0]
        self.n_stocks_short = self.stocks[self.stocks.action ==
                                          'Sell'].shape[0]
        self.N_LIMIT_LS = self.strategy_l.setNStockLimitLong(KBars=self.KBars)
        self.N_LIMIT_SS = self.strategy_s.setNStockLimitShort(KBars=self.KBars)
        self.punish_list = self.get_punish_list().證券代號.to_list()
        self._set_leverage(all_targets)
        self._set_trade_risks()
        logging.debug(f'stocks_to_monitor: {self.stocks_to_monitor}')
        return strategies, all_targets

    def init_futures(self):
        '''初始化期貨資訊'''

        def preprocess_(df):
            if df.shape[0]:
                for c in ['Volume', 'ContractAverPrice', 'SettlePrice', 'RealPrice']:
                    df[c] = df[c].astype(float)
                    if c in ['ContractAverPrice', 'SettlePrice', 'RealPrice']:
                        df[c] = df.groupby('Code')[c].transform('mean')
                    else:
                        df[c] = df.groupby('Code')[c].transform(sum)

                df = df.drop_duplicates('Code')
                df = df.rename(columns={'ContractAverPrice': 'cost_price'})
                df.Code = df.Code.astype(str).map(self.Futures_Code_List)
                df.OrderBS = df.OrderBS.apply(
                    lambda x:'Buy' if x == 'B' else ('Sell' if x == 'S' else x))
                
                orders = df[['Volume', 'OrderBS']]
                orders = orders.rename(
                    columns={'Volume':'quantity', 'OrderBS':'action'})
                df['order'] = orders.to_dict('records')
            return df

        if not self.can_futures:
            return None, []

        # 讀取選股清單
        futures_pool = self.get_futures_pool()

        # 取得遠端庫存
        self.futures = self.get_securityInfo('Futures')
        self._set_futures_code_list()

        # 庫存的處理
        self.futures = preprocess_(self.futures)
        self.futures = self.futures.merge(
            self.watchlist, 
            how='left', 
            left_on=['Account', 'Market', 'Code'], 
            right_on=['account', 'market', 'code']
        )
        self.futures.position.fillna(100, inplace=True)
        
        self.n_futures = self.futures.shape[0]

        # 取得策略清單
        strategies = self.find_strategy(futures_pool, 'Futures')

        # 剔除不堅控的股票
        self._filter_out_targets(market='Futures')

        # update_futures_to_monitor
        self.futures.index = self.futures.Code
        self.futures_to_monitor.update(self.futures.to_dict('index'))
        self.futures_to_monitor.update({
            s: None for ids in futures_pool.values() for s in ids if s not in self.futures_to_monitor})

        # 新增歷史K棒資料
        all_futures = list(self.futures_to_monitor)
        self.history_kbars(all_futures)

        # 交易風險控制 TODO: add setNFuturesLimitLong
        self.N_FUTURES_LIMIT = self.strategy_s.setNFuturesLimitShort(KBars=self.KBars)
        self._set_margin_limit()
        self.margin_table = self.get_margin_table().原始保證金.to_dict()
        return strategies, all_futures

    def _update_position(self, order: namedtuple, strategies: Dict[str, str]):
        '''更新庫存部位比例'''

        action = order.action if not order.octype else order.octype
        target = order.target
        position = order.pos_target
        is_day_trade = strategies[target] in StrategyShortDT + StrategyLongDT

        # udpate watchlist
        self.update_watchlist_position(target, action, position)

        # append watchlist or update monitor list
        if action in ['Buy', 'Sell']:
            if target not in self.stocks.code.values:
                self._append_watchlist(
                    'Stocks', order, self.Quotes, strategies)
            else:
                self.stocks_to_monitor[target]['position'] -= abs(position)
        elif action == 'New':
            if target not in self.futures.Code.values:
                self._append_watchlist(
                    'Futures', order, self.Quotes, strategies)
        else:  # and target in self.futures.Code.values:
            self.futures_to_monitor[target]['position'] -= abs(position)

        # remove from monitor list
        if abs(position) == 100 or position >= order.pos_balance:
            if target in self.stocks_to_monitor and self.stocks_to_monitor[target]['position'] <= 0:
                # if is_day_trade: self.stocks_to_monitor[target] = None # TODO: 一天可以當沖進出很多次的判定
                self.remove_stock_monitor_list(target)

            if target in self.futures_to_monitor and self.futures_to_monitor[target]['position'] <= 0:
                if is_day_trade:
                    self.futures_to_monitor[target] = None
                else:
                    self.remove_futures_monitor_list(target)

    def update_stocks_to_monitor(self, stocks_pool: Dict[str, list]):
        '''更新買進/賣出股票監控清單'''

        def not_in_stocks(x):
            return (x not in self.stocks_to_monitor)

        df = self.stocks.copy()
        df.index = df.code
        if not self.simulation:
            df.order_cond = df.order_cond.apply(lambda x: x._value_)
        self.stocks_to_monitor = df.to_dict('index')

        for stra, stocks_ in stocks_pool.items():
            if (
                (self.can_buy and (stra in StrategyLong + StrategyLongDT)) or
                (self.can_sell and (stra in StrategyShort + StrategyShortDT))
            ):
                self.stocks_to_monitor.update(
                    {s: None for s in stocks_ if not_in_stocks(s)})

    def update_monitor_lists(self, target, action, data, quantity):
        '''更新監控庫存(成交回報)'''
        if action in ['Buy', 'Sell']:
            if target in self.stocks.code.values:
                logging.debug(
                    f'更新 stocks_to_monitor 【QUANTITY】: {action} {target}')
                self.stocks_to_monitor[target]['quantity'] -= quantity
            else:
                logging.debug(
                    f'更新 stocks_to_monitor 【DATA】: {action} {target}')
                self.stocks_to_monitor[target] = data

        # New, Cover
        else:
            if target in self.futures_to_monitor and self.futures_to_monitor[target]:
                logging.debug(
                    f'更新 futures_to_monitor 【QUANTITY】: {action} {target}')
                self.futures_to_monitor[target]['order']['quantity'] -= quantity
            else:
                logging.debug(
                    f'更新 futures_to_monitor 【DATA】: {action} {target}')
                self.futures_to_monitor[target] = data

    def _update_futures_deal_list(self, target, octype):
        '''更新期貨下單暫存清單'''
        if octype == 'New' and target not in self.futures_opened:
            self.futures_opened.append(target)

        if octype == 'Cover' and target not in self.futures_closed:
            self.futures_closed.append(target)

    def merge_buy_sell_lists(self, stocks_pool: Dict[str, str], market='Stocks'):
        '''合併進出場清單: 將庫存與選股清單，合併'''

        if market == 'Stocks' and self.stocks.shape[0]:
            sells = self.stocks.code.values
        elif market == 'Futures' and self.futures.shape[0]:
            sells = self.futures.Code.values
        else:
            sells = []

        all = sells.copy()
        for ids in stocks_pool.values():
            all = np.append(all, ids)

        return np.unique(all)

    def monitor_stocks(self, target: str, strategies: Dict[str, str]):
        if target in self.Quotes.NowTargets and self.Quotes.NowIndex:
            inputs = self.Quotes.NowTargets[target].copy()
            data = self.stocks_to_monitor[target]
            strategy = strategies[target]

            isInStocks = target in self.stocks.code.values
            is_long_strategy = strategy in StrategyLong + StrategyLongDT
            isSell = (
                # 做多賣出
                (data and 'action' in data and data['action'] == 'Buy') or
                # 做空賣出
                (not data and self.can_sell and not is_long_strategy)
            )

            # 建倉
            if data is None:
                mode = 'short' if isSell else 'long'

                actionType = 'Open'
                pos_balance = 0
                order_cond = self.check_order_cond(target, mode)
                quantity = self.get_quantity(target, strategy, order_cond, mode)
                enoughOpen = self.check_enough(target, quantity, mode)

            # 庫存
            else:
                actionType = 'Close'
                pos_balance = data['position']
                order_cond = data['order_cond']
                quantity = data['quantity']
                enoughOpen = False

            # 當沖做多賣出
            c1 = isSell and strategy in StrategyLongDT and target in self.stock_bought
            # 非當沖做空賣出(放空建倉)
            c2 = isSell and enoughOpen and strategy in StrategyShort
            # 非當沖做多賣出(庫存賣出)
            c3 = isSell and self.can_sell and isInStocks and strategy in StrategyLong
            # 當沖做空回補
            c4 = (not isSell) and strategy in StrategyShortDT and target in self.stock_sold
            # 非當沖做多買進
            c5 = (not isSell) and enoughOpen and strategy in StrategyLong
            # 非當沖做空回補
            c6 = (not isSell) and enoughOpen and isInStocks and strategy in StrategyShort
            # TODO 當沖做多買進
            # TODO 當沖做空賣出
            if quantity and (c1 or c2 or c3 or c4 or c5 or c6):
                is_day_trade = strategy in StrategyShortDT + StrategyLongDT and (c1 or c4)
                tradeType = '當沖' if is_day_trade else '非當沖'

                func = self.strategy_l if is_long_strategy else self.strategy_s
                func = func.mapFunction(actionType, tradeType, strategy)

                if data:
                    inputs.update(data)

                actionInfo = func(
                    inputs=inputs,
                    kbars=self.KBars,
                    indexQuotes=self.Quotes.NowIndex,
                    pct_chg_DowJones=self.pct_chg_DowJones
                )
                if actionInfo.position:
                    infos = dict(
                        action='Sell' if isSell else 'Buy',
                        target=target,
                        quantity=quantity,
                        order_cond=self.day_trade_cond[order_cond] if c1 or c4 else order_cond,
                        pos_target=-actionInfo.position if c2 or c4 or c6 else actionInfo.position,
                        pos_balance=pos_balance,
                        reason=actionInfo.msg,
                    )
                    self._log_and_notify(actionInfo.msg)
                    return self.OrderInfo(**infos)

        return self.OrderInfo(target=target)

    def monitor_futures(self, target: str, strategies: Dict[str, str]):
        '''檢查期貨是否符合賣出條件，回傳賣出部位(%)'''

        if target in self.Quotes.NowTargets and self.N_FUTURES_LIMIT != 0:
            inputs = self.Quotes.NowTargets[target].copy()
            data = self.futures_to_monitor[target]
            strategy = strategies[target]
            is_long_strategy = strategy in StrategyLong + StrategyLongDT

            # 建倉
            if data is None:
                mode = 'long' if is_long_strategy else 'short'

                octype = 'New'
                actionType = 'Open'
                quantity = self.get_open_slot(target, strategy, mode)
                enoughOpen = self._check_enough_open(target, quantity)
                position = 100
                action = 'Buy' if is_long_strategy else 'Sell'

            # 庫存
            else:
                octype = 'Cover'
                actionType = 'Close'
                enoughOpen = False
                position = data['position']
                if target in self.futures_opened:
                    quantity = data['order']['quantity']
                    action = data['order']['action']
                else:
                    quantity = data['Volume']
                    action = data['OrderBS']

            # 當沖做多建倉
            # 當沖做多平倉
            # 當沖做空建倉
            c3 = octype == 'New' and enoughOpen and not self.is_not_trade_day(
                inputs['datetime'])
            # 當沖做空平倉
            c4 = octype == 'Cover'  # TODO 平倉的判定條件
            # 非當沖做多建倉
            # 非當沖做多平倉
            # 非當沖做空建倉
            # 非當沖做空平倉

            if c3 or c4:
                is_day_trade = strategy in StrategyShortDT + StrategyLongDT
                tradeType = '當沖' if is_day_trade else '非當沖'
                func = self.strategy_l if is_long_strategy else self.strategy_s
                func = func.mapFunction(actionType, tradeType, strategy)

                if data:
                    inputs.update(data)

                actionInfo = func(
                    inputs=inputs,
                    kbars=self.KBars,
                    indexQuotes=self.Quotes.NowIndex,
                    allQuotes=self.Quotes.AllTargets,
                )
                if actionInfo.position:
                    self._log_and_notify(actionInfo.msg)
                    return self.OrderInfo(
                        action=action,
                        target=target,
                        quantity=quantity,
                        octype=octype,
                        pos_target=actionInfo.position,
                        pos_balance=position,
                        reason=actionInfo.msg
                    )

        return self.OrderInfo(target=target)

    def _place_order(self, content: namedtuple, market='Stocks'):
        logging.debug(f'【content: {content}】')

        target = content.target
        contract = get_contract(target)
        if target in self.BidAsk:
            quantity = self.get_sell_quantity(content, market)
            price_type = 'MKT'
            price = 0
            order_lot = 'IntradayOdd' if content.quantity < 1000 and market == 'Stocks' else 'Common'

            if market == 'Stocks':
                bid_ask = self.BidAsk[target]
                bid_ask = bid_ask.bid_price if content.action == 'Sell' else bid_ask.ask_price

                # 零股交易
                if 0 < content.quantity < 1000:
                    price_type = 'LMT'
                    price = bid_ask[1]

                # 整股交易
                else:
                    if datetime.now() >= TTry:
                        price_type = 'LMT'
                        price = bid_ask[1]
                    elif target in self.punish_list:
                        price_type = 'LMT'
                        price = bid_ask[3]
                    elif contract.exchange == 'OES':
                        price_type = 'LMT'
                        price = self.Quotes.NowTargets[target]['price']

                log_msg = f"【{target}下單內容: price={price}, quantity={quantity}, action={content.action}, price_type={price_type}, order_cond={content.order_cond}, order_lot={order_lot}】"
            else:
                log_msg = f"【{target}下單內容: price={price}, quantity={quantity}, action={content.action}, price_type={price_type}, order_cond={content.octype}, order_lot={order_lot}】"

            # 下單
            logging.debug(log_msg)
            if self.simulation and market == 'Stocks':
                price = self.Quotes.NowTargets[target]['price']
                quantity *= 1000
                leverage = self.check_leverage(target, content.order_cond)
                if content.action == 'Sell':
                    self.stock_sold.append(target)
                    cost_price = self.get_cost_price(
                        target, price, content.order_cond)
                    amount = -(price - cost_price*leverage) * \
                        quantity*(1 - FEE_RATE)
                else:
                    self.stock_bought.append(target)
                    amount = self.get_stock_amount(
                        target, price, quantity, content.order_cond)

                order_data = {
                    'Time': datetime.now(),
                    'market':'Stocks',
                    'code': target,
                    'action': content.action,
                    'price': -price if content.action == 'Sell' else price,
                    'quantity': quantity,
                    'amount': amount,
                    'order_cond': content.order_cond if market == 'Stocks' else 'Cash',
                    'order_lot': order_lot,
                    'leverage': leverage,
                    'account_id': 'simulate',
                    'msg': content.reason
                }
                self.appendOrder(order_data)

                # 更新監控庫存
                order_data.update({
                    'code': target,
                    'position': abs(content.pos_target),
                    'yd_quantity': 0,
                    'bst': datetime.now(),
                    'cost_price': abs(order_data['price'])
                })
                self.update_monitor_lists(
                    target, content.action, order_data, quantity)

                logging.debug('Place simulate order complete.')
                self.notifier.post(log_msg, msgType='Order')

            elif self.simulation and market == 'Futures':
                price = self.Quotes.NowTargets[target]['price']
                sign = -1 if content.octype == 'Cover' else 1
                order_data = {
                    'Time': datetime.now(),
                    'market':'Futures',
                    'code': target,
                    'action': content.action,
                    'price': price*sign,
                    'quantity': quantity,
                    'amount': self.get_open_margin(target, quantity)*sign,
                    'op_type': content.octype,
                    'account_id': 'simulate',
                    'msg': content.reason
                }
                self.appendOrder(order_data)
                self._update_futures_deal_list(target, content.octype)

                # 更新監控庫存
                bsh = max(self.Quotes.AllTargets[target]['price']) if self.Quotes.AllTargets[target]['price'] else price
                order_data.update({
                    'symbol': target,
                    'cost_price': abs(price),
                    'bsh': bsh,
                    'bst': datetime.now(),
                    'position': 100,
                    'order': {
                        'quantity': quantity,
                        'action': content.action
                    }
                })
                self.update_monitor_lists(
                    target, content.octype, order_data, quantity)

                logging.debug('Place simulate order complete.')
                self.notifier.post(log_msg, msgType='Order')
            else:
                # #ff0000 批次下單的張數 (股票>1000股的單位為【張】) #ff0000
                q = 5 if order_lot == 'Common' else quantity
                if market == 'Stocks':
                    target_ = self.desposal_money
                else:
                    target_ = self.desposal_margin

                enough_to_place = self.checkEnoughToPlace(market, target_)
                while quantity > 0 and enough_to_place:
                    order = API.Order(
                        # 價格 (市價單 = 0)
                        price=price,
                        # 數量 (最小1張; 零股最小50股 or 全部庫存)
                        quantity=min(quantity, q),
                        # 動作: 買進/賣出
                        action=content.action,
                        # 市價單/限價單
                        price_type=price_type,
                        # ROD:當天都可成交
                        order_type=constant.OrderType.ROD,
                        # 委託類型: 現股/融資
                        order_cond=content.order_cond if market == 'Stocks' else 'Cash',
                        # 整張或零股
                        order_lot=order_lot,
                        # {Auto, New, Cover, DayTrade}(自動、新倉、平倉、當沖)
                        octype='Auto' if market == 'Stocks' else content.octype,
                        account=API.stock_account if market == 'Stocks' else API.futopt_account,
                        # 先賣後買: True, False
                        daytrade_short=content.daytrade_short,
                    )
                    result = API.place_order(contract, order)
                    self.check_order_status(result)
                    quantity -= q

    def get_securityInfo(self, market='Stocks'):
        '''取得證券庫存清單'''

        if self.simulation:
            try:
                if db.HAS_DB and market == 'Stocks':
                    df = db.query(
                        SecurityInfoStocks, 
                        SecurityInfoStocks.account == self.ACCOUNT_NAME
                    ).drop(['pk_id', 'create_time'], axis=1)
                elif db.HAS_DB and market == 'Futures':
                    df = db.query(
                        SecurityInfoFutures, 
                        SecurityInfoFutures.Account == self.ACCOUNT_NAME
                    ).drop(['pk_id', 'create_time'], axis=1)
                else:
                    df = pd.read_pickle(
                        f'{PATH}/stock_pool/simulation_{market.lower()}_{self.ACCOUNT_NAME}.pkl')
            except:
                if market == 'Stocks':
                    df = self.df_securityInfo
                else:
                    df = self.df_futuresInfo

            df['account_id'] = 'simulate'

        else:
            if market == 'Stocks':
                df = self.securityInfo()
                return df[df.code.apply(len) == 4]

            df = self.get_openpositions()
        return df

    def _get_day_filter_out(self):
        '''取得交易當日全額交割股清單'''

        try:
            url = 'https://www.sinotrade.com.tw/Stock/Stock_3_8_3'
            df = pd.read_html(url)
            return df[0]
        except:
            logging.warning('查無全額交割股清單')
            return pd.DataFrame(columns=['股票代碼'])

    def get_selection_files(self):
        '''取得選股清單'''

        dirpath = f'{PATH}/selections'
        files = os.listdir(dirpath)
        files = [f for f in files if '.csv' in f]
        if files:
            df = pd.read_csv(f'{PATH}/selections/{files[0]}')
            df.date = pd.to_datetime(df.date)
            df.name = df.name.astype(str)

            # 排除不交易的股票
            # ### 全額交割股不買
            day_filter_out = self._get_day_filter_out()
            df = df[~df.name.isin(day_filter_out.股票代碼.values)]
            df = df[~df.name.isin(self.FILTER_OUT)]

            # 排除高價股
            df = df[df.Close <= self.PRICE_THRESHOLD]

            return df

        return pd.DataFrame()

    def get_margin_table(self):
        '''取得保證金清單'''
        df = pd.read_csv('./lib/indexMarging.csv',
                         encoding='big5').reset_index()
        df.columns = list(df.iloc[0, :])
        df = df.iloc[1:, :-2]
        df.原始保證金 = df.原始保證金.astype(int)

        codes = [[f.code, f.symbol, f.name]
                 for m in API.Contracts.Futures for f in m]
        codes = pd.DataFrame(codes, columns=['code', 'symbol', 'name'])
        codes = codes.set_index('name').symbol.to_dict()

        month = str((datetime.now() + timedelta(days=30)).month).zfill(2)
        df['code'] = (df.商品別 + month).map(codes)
        return df.dropna().set_index('code')

    def get_stock_pool(self):
        '''取得股票選股池'''

        pools = {st: [] for st in self.STRATEGY_STOCK}

        df = self.get_selection_files()
        day = self.last_business_day()

        if df.shape[0] > 1 and day == df.date.max():
            # 建立族群清單
            df['n_category'] = df.category.map(
                df.groupby('category').name.count().to_dict())
            self.n_categories = df[[
                'name', 'company_name', 'category', 'n_category']]
            self.n_categories = self.n_categories.sort_values(
                'n_category', ascending=False)
            self.n_categories = self.n_categories.set_index(
                'name').n_category.to_dict()

            df = df.dropna().sort_values('Close')

            # 族群清單按照策略權重 & pc_ratio 決定
            # 權重大的先加入，避免重複
            strategy_l = self.strategy_l.STRATEGIES.sort_values(
                'weight', ascending=False)
            strategy_s = self.strategy_s.STRATEGIES.sort_values(
                'weight', ascending=False)
            if self.strategy_l.pc_ratio >= 115:
                strategies = strategy_l.name.to_list() + strategy_s.name.to_list()
            else:
                strategies = strategy_s.name.to_list() + strategy_l.name.to_list()

            for s in strategies:
                if s in df.columns and s in pools:
                    stockids = df[df[s] == 1].name.to_list()
                    pools[s] = stockids
                    df = df[df[s] != 1]

        return pools

    def get_futures_pool(self):
        '''取得期權目標商品清單'''

        pools = {st: [] for st in self.STRATEGY_FUTURES}

        due_year_month = self.GetDueMonth(TODAY)
        indexes = {
            '放空小台': [f'MXF{due_year_month}'],
            '做多小台': [f'MXF{due_year_month}'],
            '放空大台': [f'TXF{due_year_month}'],
            '做多大台': [f'TXF{due_year_month}'],
        }
        pools.update(
            {st: indexes[st] if st in indexes else [] for st in pools})
        return pools

    def get_quantity(self, target: str, strategy: str, order_cond, mode='long'):
        '''計算進場股數'''

        if self.BUY_UNIT_TYPE == 'constant':
            return 1000*self.BUY_UNIT

        if mode == 'long':
            quantityFunc = self.strategy_l.mapQuantities(strategy)
        else:
            quantityFunc = self.strategy_s.mapQuantities(strategy)

        inputs = self.Quotes.NowTargets[target]
        quantity, quantity_limit = quantityFunc(
            inputs=inputs, kbars=self.KBars)
        leverage = self.check_leverage(target, order_cond)
        # TODO: 不可超過券商可融資餘額上限
        quantity = int(min(quantity, quantity_limit)/(1 - leverage))
        quantity = min(quantity, 499)
        return 1000*quantity

    def get_stock_amount(self, target: str, price: float, quantity: int, mode='long'):
        '''計算股票委託金額'''

        leverage = self.check_leverage(target, mode)
        fee = max(price*quantity*FEE_RATE, 20)
        return price*quantity*(1 - leverage) + fee

    def get_open_slot(self, target: str, strategy: str, mode='long'):
        '''計算買進口數'''

        if self.N_SLOT_TYPE == 'constant':
            return self.N_SLOT

        if mode == 'long':
            quantityFunc = self.strategy_l.mapQuantities(strategy)
        else:
            quantityFunc = self.strategy_s.mapQuantities(strategy)

        inputs = self.Quotes.NowTargets[target]
        slot, quantity_limit = quantityFunc(inputs=inputs, kbars=self.KBars)
        slot = int(min(slot, quantity_limit))
        slot = min(slot, 499)
        return slot

    def get_open_margin(self, target: str, quantity: int):
        '''計算期貨保證金額'''

        if target in self.Quotes.NowTargets and self.margin_table and target in self.margin_table:
            fee = 100
            return self.margin_table[target]*quantity + fee

        return 0

    def get_pct_chg_DowJones(self):
        '''取得道瓊指數前一天的漲跌幅'''

        start = self._strf_timedelta(TODAY, 30)
        dj = self.DowJones(start, TODAY_STR)
        if len(dj):
            return 100*round(dj[0]/dj[1] - 1, 4)
        return 0

    def get_cost_price(self, target: str, price: float, order_cond: str):
        '''取得股票的進場價'''

        if order_cond == 'ShortSelling':
            return price

        cost_price = self.stocks.set_index('code').cost_price.to_dict()
        if target in cost_price:
            return cost_price[target]
        return 0

    def check_leverage(self, target: str, mode='long'):
        '''取得個股的融資/融券成數'''
        if mode in ['long', 'MarginTrading'] and target in self.LEVERAGE_LONG:
            return self.LEVERAGE_LONG[target]
        elif mode in ['short', 'ShortSelling'] and target in self.LEVERAGE_SHORT:
            return 1 - self.LEVERAGE_SHORT[target]
        return 0

    def check_order_cond(self, target: str, mode='long'):
        '''檢查個股可否融資'''

        contract = get_contract(target)
        if mode == 'long':
            if self.ORDER_COND1 != 'Cash' and (self.LEVERAGE_LONG[target] == 0 or contract.margin_trading_balance == 0):
                return 'Cash'
            return self.ORDER_COND1
        else:
            if self.ORDER_COND2 != 'Cash' and (self.LEVERAGE_SHORT[target] == 1 or contract.short_selling_balance == 0):
                return 'Cash'
            return self.ORDER_COND2

    def check_enough(self, target: str, quantity: int, mode='long'):
        '''計算可買進的股票數量 & 金額'''

        if target not in self.Quotes.NowTargets:
            return False

        # 更新可買進的股票額度 TODO: buy_deals, sell_deals會合計多空股票數，使quota1, quota2無法精準
        buy_deals = len([s for s in self.stock_bought if len(s) == 4])
        sell_deals = len([s for s in self.stock_sold if len(s) == 4])
        quota1 = abs(self.N_LIMIT_LS) - self.n_stocks_long - \
            buy_deals + sell_deals
        quota2 = abs(self.N_LIMIT_SS) - self.n_stocks_short + \
            buy_deals - sell_deals

        # 更新已委託金額
        df = self.filterOrderTable('Stocks')
        df = df[df.code.apply(len) == 4]
        amount1 = df.amount.sum()
        amount2 = df[df.price > 0].amount.abs().sum()
        amount3 = df[df.price < 0].amount.abs().sum()

        cost_price = self.Quotes.NowTargets[target]['price']
        target_amount = self.get_stock_amount(
            target, cost_price, quantity, mode)

        # under day limit condition
        # 1. 不可超過可交割金額
        # 2. 不可大於帳戶可委託金額上限
        # 3. 不可超過股票數上限
        if mode == 'long':
            return (
                (amount1 + target_amount <= self.desposal_money) &
                (amount2 + target_amount <= self.POSITION_LIMIT_LONG) &
                (quota1 > 0)
            )

        return (
            (amount1 + target_amount <= self.desposal_money) &
            (amount2 + target_amount <= self.POSITION_LIMIT_LONG) &
            # 4. 不可超過可信用交易額度上限
            (amount3 + target_amount <= self.POSITION_LIMIT_SHORT) &
            (quota2 > 0)
        )

    def _check_enough_open(self, target: str, quantity: int):
        '''計算可開倉的期貨口數 & 金額'''

        # 更新可開倉的期貨標的數
        open_deals = len(self.futures_opened)
        close_deals = len(self.futures_closed)
        quota = abs(self.N_FUTURES_LIMIT) - \
            self.n_futures - open_deals + close_deals

        # 更新已委託金額
        df = self.filterOrderTable('Futures')
        amount1 = df.amount.sum()
        amount2 = df[df.price > 0].amount.sum()

        # under day limit condition
        # 1. 不可超過可交割保證金
        # 2. 不可大於帳戶可委託保證金上限
        # 3. 不可超過股票數上限
        target_amount = self.get_open_margin(target, quantity)
        return (
            (amount1 + target_amount <= self.desposal_margin) &
            (amount2 + target_amount <= self.MARGIN_LIMIT) &
            (quota > 0)
        )

    def find_strategy(self, stocks_pool: dict, market='Stocks') -> Dict[str, str]:
        '''
        找出買進股票對應的策略
        result = {
            'stockid':'strategy',
        }
        '''

        if market == 'Stocks':
            result = self.stocks.set_index('code').strategy.to_dict()
        elif market == 'Futures' and self.futures.shape[0]:
            result = self.futures.set_index('Code').strategy.to_dict()
        else:
            result = {}

        for s, stocks in stocks_pool.items():
            for stock in stocks:
                if stock not in result:
                    result[stock] = s
        return result

    def is_not_trade_day(self, now: datetime):
        '''檢查是否為非交易時段'''
        if self.can_futures:
            return (
                (now.weekday() in [5, 6]) or
                ((now > TimeEndFuturesNight) and (now < TimeStartFuturesDay)) or
                # TODO: 只在 05:00 ~ 08:45之間關閉
                ((now > TimeEndFuturesDay) and (now < TimeStartFuturesNight))
            )

        return (now.weekday() in [5, 6]) or not (now <= TEnd)

    def remove_stock_monitor_list(self, target: str):
        logging.debug(f'Remove【{target}】from stocks_to_monitor.')
        self.stocks_to_monitor.pop(target, None)

        if target in self.stocks.code.values:
            logging.debug(f'Remove【{target}】from self.stocks.')
            self.stocks = self.stocks[self.stocks.code != target]
            if db.HAS_DB:
                db.delete(
                    SecurityInfoStocks, 
                    SecurityInfoStocks.code == target, 
                    SecurityInfoStocks.account == self.ACCOUNT_NAME
                )

    def remove_futures_monitor_list(self, target: str):
        '''Remove futures from futures_to_monitor'''
        logging.debug(f'Remove【{target}】from futures_to_monitor.')
        self.futures_to_monitor.pop(target, None)

        if target in self.futures.Code.values:
            logging.debug(f'Remove【{target}】from self.futures.')
            self.futures = self.futures[self.futures.Code != target]
            if db.HAS_DB:
                db.delete(
                    SecurityInfoFutures, 
                    SecurityInfoFutures.Code == target, 
                    SecurityInfoFutures.Account == self.ACCOUNT_NAME
                )

    def run(self):
        '''執行自動交易'''

        strategy_s, all_stocks = self.init_stocks()
        strategy_f, all_futures = self.init_futures()
        self.subscribe_all(all_stocks+all_futures)

        now = datetime.now()
        if (not self.is_not_trade_day(now)) and now > TStart:
            self.update_today_previous_kbar(
                all_stocks, self.strategy_l.dividends)

        logging.info(f"Today's punish lis: {self.punish_list}")
        logging.info(
            f"Today's Ex-dividend stocks: {self.strategy_l.dividends}")
        logging.info(f"Previous Put/Call ratio: {self.strategy_l.pc_ratio}")
        logging.info(
            f"Previous percentage change in Dow Jones Index: {self.pct_chg_DowJones}")
        logging.info(f'Start to monitor, basic settings:')
        logging.info(f'Mode:{self.MODE}, Strategy:{self.STRATEGY_STOCK}')
        logging.info(
            f'[Stock Position] Long:{self.n_stocks_long}; Short:{self.n_stocks_short}')
        logging.info(
            f'[Stock Portfolio Limit] Long:{self.N_LIMIT_LS}; Short:{self.N_LIMIT_SS}')
        logging.info(
            f'[Stock Position Limit] Long:{self.POSITION_LIMIT_LONG}; Short: {self.POSITION_LIMIT_SHORT}')
        logging.info(
            f'Futures positions:{self.n_futures}, portfolio Limit:{self.N_FUTURES_LIMIT}')

        text = f"\n【開始監控】{self.ACCOUNT_NAME} 啟動完成({__version__})"
        text += f"\n【操盤模式】{self.MODE}\n【操盤策略】{self.STRATEGY_STOCK}"
        text += f"\n【前日行情】Put/Call: {self.strategy_l.pc_ratio}"
        text += f"\n【美股行情】道瓊({self.pct_chg_DowJones}%)"
        self.notifier.post(text, msgType='Monitor')

        # 開始監控
        while True:
            now = datetime.now()
            hour = now.hour

            if self.is_not_trade_day(now):
                logging.info('Non-trading time, stop monitoring')
                self._update_K5()
                self._update_K15()
                self._update_K30()
                self._update_K60(hour)
                break
            elif all(x == 0 for x in [
                self.n_stocks_long, self.n_stocks_short,
                self.N_LIMIT_LS, self.N_LIMIT_SS,
                self.N_FUTURES_LIMIT, self.n_futures
            ]):
                self._log_and_notify(f"【停止監控】{self.ACCOUNT_NAME} 無可監控清單")
                break

            # update K-bar data
            is_trading_time = (
                (self.can_futures and now > TimeStartFuturesDay + timedelta(seconds=30)) or
                (self.can_stock and now > TimeStartStock + timedelta(seconds=30))
            )
            if is_trading_time and now.second < 5:
                self._update_K1(self.strategy_l.dividends, quotes=self.Quotes)
                # TODO: all_stocks+all_futures
                self._set_target_quote_default(all_stocks+all_futures)
                self._set_index_quote_default()
                self._set_futures_code_list()
                self.strategy_l.update_indicators(now, self.KBars)
                self.strategy_s.update_indicators(now, self.KBars)

                if now.minute % 5 == 0:
                    self._update_K5()
                    self.balance(mode='debug')  # 防止斷線用 TODO:待永豐更新後刪除

                if now.minute % 15 == 0:
                    self._update_K15()

                if now.minute % 30 == 0:
                    self._update_K30()

                if now.minute == 0:
                    self._update_K60(hour-1)

            # TODO: merge stocks_to_monitor & futures_to_monitor
            if self.can_stock:
                for target in list(self.stocks_to_monitor):
                    order = self.monitor_stocks(target, strategy_s)
                    if order.pos_target:
                        self._place_order(order, market='Stocks')
                        self._update_position(order, strategy_s)

            if self.can_futures:
                for target in list(self.futures_to_monitor):
                    order = self.monitor_futures(target, strategy_f)
                    if order.pos_target:
                        self._place_order(order, market='Futures')
                        self._update_position(order, strategy_f)

            time.sleep(max(5 - (datetime.now() - now).total_seconds(), 0))

        time.sleep(10)
        self.unsubscribe_all({'Stocks': all_stocks, 'Futures': all_futures})
        self.output_files()

    def __save_simulate_securityInfo(self):
        '''儲存模擬交易模式下的股票庫存表'''
        if self.simulation:
            # 儲存庫存
            logging.debug(f'stocks_to_monitor: {self.stocks_to_monitor}')
            logging.debug(
                f'stocks shape: {self.stocks.shape}; watchlist shape: {self.watchlist.shape}')
            df = pd.DataFrame(
                {k: v for k, v in self.stocks_to_monitor.items() if v}).T
            if df.shape[0]:
                df = df[df.account_id == 'simulate']
                df = df.sort_values('code').reset_index()
                df['last_price'] = df.code.map(
                    {s: self.Quotes.NowTargets[s]['price'] for s in df.code})
                df['pnl'] = df.action.apply(lambda x: 1 if x == 'Buy' else -1)
                df['pnl'] = df.pnl*(df.last_price - df.cost_price)*df.quantity
                df.yd_quantity = df.quantity
                df = df[self.df_securityInfo.columns]
            else:
                df = self.df_securityInfo
            logging.debug(
                f'stocks shape: {df.shape}; watchlist shape: {self.watchlist.shape}')
            
            if db.HAS_DB:
                db.delete(
                    SecurityInfoStocks, 
                    SecurityInfoStocks.account == self.ACCOUNT_NAME
                )
                db.dataframe_to_DB(df, SecurityInfoStocks)
            else:
                df.to_pickle(
                    f'{PATH}/stock_pool/simulation_stocks_{self.ACCOUNT_NAME}.pkl')

    def __save_simulate_futuresInfo(self):
        '''儲存模擬交易模式下的期貨庫存表'''
        if self.simulation:
            # 儲存庫存
            logging.debug(f'futures_to_monitor: {self.futures_to_monitor}')
            df = pd.DataFrame(
                {k: v for k, v in self.futures_to_monitor.items() if v}).T
            if df.shape[0]:
                df = df[df.account_id == 'simulate']
                df = df.reset_index(drop=True)
                df = df.rename(columns={
                    'account_id': 'Account',
                    'bst': 'Date',
                    'symbol': 'CodeName',
                    'action': 'OrderBS',
                    'quantity': 'Volume',
                    'cost_price': 'ContractAverPrice',
                    'price': 'RealPrice',
                })
                df['Code'] = df.CodeName.apply(lambda x: get_contract(x).code)

                for c in self.df_futuresInfo.columns:
                    if c not in df.columns:
                        df[c] = 0 if c in ['ContractAverPrice',
                                           'SettlePrice', 'RealPrice'] else ''
                df = df[self.df_futuresInfo.columns]
            else:
                df = self.df_futuresInfo

            if db.HAS_DB:
                db.delete(
                    SecurityInfoFutures, 
                    SecurityInfoFutures.Account == self.ACCOUNT_NAME
                )
                db.dataframe_to_DB(df, SecurityInfoStocks)
            else:
                df.to_pickle(
                    f'{PATH}/stock_pool/simulation_futures_{self.ACCOUNT_NAME}.pkl')

    def output_files(self):
        '''停止交易時，輸出庫存資料 & 交易明細'''
        if 'position' in self.stocks.columns and not self.simulation:
            self.stocks = self.get_securityInfo('Stocks')
            # self.history_kbars(self.stocks[self.stocks.yd_quantity == 0].code)
            self.update_watchlist(self.stocks)
        self.save_watchlist(self.watchlist)
        self.output_statement(f'{PATH}/stock_pool/statement_{self.ACCOUNT_NAME}.csv')

        if (datetime.now().weekday() not in [5, 6]):
            for freq, df in self.KBars.items():
                if freq != '1D':
                    filename = f'{PATH}/Kbars/k{freq[:-1]}min_{self.ACCOUNT_NAME}.csv'
                    save_csv(df, filename)

        if self.can_stock:
            self.__save_simulate_securityInfo()
            
        if self.can_futures:
            self.__save_simulate_futuresInfo()

        time.sleep(5)
