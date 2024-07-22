import ssl
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict
from sys import platform
from shioaji import constant
from collections import namedtuple
from datetime import datetime, timedelta

from . import __version__
from . import exec, notifier, picker, crawler2, file_handler
from .config import API, PATH, TODAY, TODAY_STR, holidays, MonitorFreq
from .config import FEE_RATE, TEnd, TTry, TimeStartStock, TimeStartFuturesDay, TimeEndStock
from .config import TimeEndFuturesDay, TimeStartFuturesNight, TimeEndFuturesNight, TimeTransferFutures
from .utils import get_contract
from .utils.orders import OrderTool
from .utils.cipher import CipherTool
from .utils.accounts import AccountInfo
from .utils.subscribe import Subscriber
from .utils.positions import WatchListTool
from .utils.callback import CallbackHandler
from .utils.database import db
from .utils.database.tables import SecurityInfoStocks, SecurityInfoFutures
try:
    from .scripts.StrategySet import StrategySet as StrategySets
except:
    from .utils.strategy import StrategyTool as StrategySets


ssl._create_default_https_context = ssl._create_unverified_context


class StrategyExecutor(AccountInfo, WatchListTool, OrderTool, Subscriber):
    def __init__(self, config=None):
        self.ct = CipherTool(decrypt=True, encrypt=False)
        self.CONFIG = config

        # 交易帳戶設定
        self.ACCOUNT_NAME = self.getENV('ACCOUNT_NAME')
        self.__API_KEY__ = self.getENV('API_KEY')
        self.__SECRET_KEY__ = self.getENV('SECRET_KEY')
        self.__ACCOUNT_ID__ = self.getENV('ACCOUNT_ID', 'decrypt')
        self.__CA_PASSWD__ = self.getENV('CA_PASSWD', 'decrypt')

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
        self.STOCK_MODEL_VERSION = self.getENV('STOCK_MODEL_VERSION')
        self.simulation = self.MODE == 'Simulation'

        # 期貨使用者設定
        self.TRADING_PERIOD = self.getENV('TRADING_PERIOD')
        self.STRATEGY_FUTURES = self.getENV('STRATEGY_FUTURES', 'list')
        self.MARGIN_LIMIT = self.getENV('MARGIN_LIMIT', 'int')
        self.N_FUTURES_LIMIT_TYPE = self.getENV('N_FUTURES_LIMIT_TYPE')
        self.N_FUTURES_LIMIT = self.getENV('N_FUTURES_LIMIT', 'int')
        self.N_SLOT = self.getENV('N_SLOT', 'int')
        self.N_SLOT_TYPE = self.getENV('N_SLOT_TYPE')
        self.FUTURES_MODEL_VERSION = self.getENV('FUTURES_MODEL_VERSION')

        super().__init__()
        Subscriber.__init__(self, self.KBAR_START_DAYay)
        OrderTool.__init__(self)
        WatchListTool.__init__(self, self.ACCOUNT_NAME)

        # 股票可進場籌碼 (進場時判斷用)
        self.simulate_amount = np.iinfo(np.int64).max
        self.stocks = pd.DataFrame()
        self.desposal_money = 0
        self.stock_strategies = {}
        self.n_stocks_long = 0
        self.n_stocks_short = 0
        self.total_market_value = 0
        self.punish_list = []
        self.pct_chg_DowJones = self.get_pct_chg_DowJones()
        self.n_categories = None

        # 期貨可進場籌碼 (進場時判斷用)
        self.futures_strategies = {}
        self.futures_transferred = {}
        self.transfer_list = []
        self.n_futures = 0
        self.futures = pd.DataFrame()
        self.Futures_Code_List = {}
        self.margin_table = None

        # 交易相關
        self.can_stock = 'stock' in self.MARKET
        self.can_sell = self.MODE not in ['LongBuy', 'ShortBuy']
        self.can_buy = self.MODE not in ['LongSell', 'ShortSell']
        self.can_futures = 'futures' in self.MARKET

        # 載入指標模組
        self.set_scripts(StrategySets)

    def set_scripts(self, strategySet: object):
        self.StrategySet = strategySet(
            account_name=self.ACCOUNT_NAME,
            hold_day=self.getENV('HOLD_DAY', 'int'),
            is_simulation=self.simulation,
            stock_limit_type=self.N_STOCK_LIMIT_TYPE,
            stock_limit_long=self.N_LIMIT_LS,
            stock_limit_short=self.N_LIMIT_SS,
            stock_model_version=self.STOCK_MODEL_VERSION,
            futures_limit_type=self.N_FUTURES_LIMIT_TYPE,
            futures_limit=self.N_FUTURES_LIMIT,
            futures_model_version=self.FUTURES_MODEL_VERSION
        )

    def getENV(self, key: str, type_: str = 'text'):
        if self.CONFIG and key in self.CONFIG:
            env = self.CONFIG[key]

            if type_ == 'int':
                return int(env)
            elif type_ == 'list':
                if 'none' in env.lower():
                    return []
                return env.replace(' ', '').split(',')
            elif type_ == 'date' and env:
                return pd.to_datetime(env)
            elif type_ == 'decrypt':
                if not env or (not env[0].isdigit() and env[1:].isdigit()):
                    return env
                return self.ct.decrypt(env)
            return env
        elif type_ == 'int':
            return 0
        elif type_ == 'list':
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
            account_balance = 0
            self.desposal_margin = self.simulate_amount
            self.ProfitAccCount = self.simulate_amount
        else:
            account_balance = self.balance()
            self.get_account_margin()
        self.desposal_margin = min(
            account_balance+self.desposal_margin, self.MARGIN_LIMIT)
        logging.info(f'權益總值: {self.ProfitAccCount}')
        logging.info(
            f'Margin available = {self.desposal_margin} (limit: {self.MARGIN_LIMIT})')

    def _set_leverage(self, stockids: list):
        '''
        取得個股融資成數資料，
        若帳戶設定為不可融資，則全部融資成數為0
        '''

        df = pd.DataFrame([crawler2.get_leverage(s) for s in stockids])
        if df.shape[0]:
            df.columns = df.columns.str.replace(' ', '')
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
        if self.can_futures and self.Futures_Code_List == {}:
            logging.debug('Set Futures_Code_List')
            self.Futures_Code_List = {
                f.code: f.symbol for m in API.Contracts.Futures for f in m}

    def _order_callback(self, stat, msg):
        '''處理委託/成交回報'''

        if stat == constant.OrderState.StockOrder:
            stock = msg['contract']['code']
            order = msg['order']
            operation = msg['operation']

            c2 = operation['op_code'] == '00' or operation['op_msg'] == ''
            c3 = order['action'] == 'Buy'
            c4 = operation['op_code'] == '88' and '此證券配額張數不足' in operation['op_msg']

            if order['account']['account_id'] == self.account_id_stock:
                notifier.post_tftOrder(stat, msg)
                self.update_deal_list(stock, order['action'], 'Stocks')

                leverage = self.check_leverage(stock, order['order_cond'])
                if c2 and c3:
                    # 記錄委託成功的買單
                    price = order['price']
                    quantity = order['quantity']
                    if order['order_lot'] == 'Common':
                        quantity *= 1000
                    order_data = {
                        'Time': datetime.now(),
                        'market': 'Stocks',
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
                    if c3:
                        self.update_deal_list(stock, 'Cancel', 'Stocks')

        elif stat == constant.OrderState.StockDeal and msg['account_id'] == self.account_id_stock:
            notifier.post_tftDeal(stat, msg)

            stock = msg['code']
            msg = CallbackHandler.update_stock_msg(msg)

            quantity = msg['quantity']
            if msg['action'] == 'Sell':

                self.update_deal_list(stock, msg['action'], 'Stocks')

                price = msg['price']
                leverage = self.check_leverage(stock, msg['order_cond'])
                cost_price = self.get_cost_price(
                    stock, price, msg['order_cond'])

                # 紀錄成交的賣單
                order_data = {
                    'Time': datetime.now(),
                    'market': 'Stocks',
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
                self.update_position_quantity(msg['action'], msg)

        elif stat == constant.OrderState.FuturesOrder:
            notifier.post_fOrder(stat, msg)

            code = msg['contract']['code']
            symbol = code + msg['contract']['delivery_month']
            market = 'Futures' if msg['contract']['option_right'] == 'Future' else 'Options'
            if symbol not in self.Quotes.NowTargets:
                for k in self.Quotes.NowTargets:
                    if symbol in k:
                        symbol = k
            order = msg['order']
            operation = msg['operation']
            price = order['price']
            if price == 0:
                price = self.getQuotesNow(symbol)['price']
            msg.update({
                'symbol': symbol,
                'cost_price': price,
                'bst': datetime.now(),
                'position': 100
            })

            if order['account']['account_id'] == self.account_id_futopt:
                if operation['op_code'] == '00' or operation['op_msg'] == '':
                    self.update_deal_list(symbol, order['oc_type'], market)

                    # 紀錄成交的賣單
                    sign = -1 if order['oc_type'] == 'Cover' else 1
                    quantity = order['quantity']
                    order_data = {
                        'Time': datetime.now(),
                        'market': 'Futures',
                        'code': symbol,
                        'action': order['action'],
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
                    self.update_deal_list(symbol, 'Cancel', market)
                    if order['oc_type'] == 'New':
                        self.futures_to_monitor[symbol] = None

                # 更新監控庫存
                msg['code'] = symbol
                self.update_position_quantity(order['oc_type'], msg)

        elif stat == constant.OrderState.FuturesDeal:
            notifier.post_fDeal(stat, msg)

            code = msg['code']
            delivery_month = msg['delivery_month']
            symbol = code + delivery_month
            if symbol in self.futures_to_monitor and self.futures_to_monitor[symbol] is not None:
                price = msg['price']
                self.futures_to_monitor[symbol]['cost_price'] = price

    def login_and_activate(self):
        # 登入
        self._login(self.__API_KEY__, self.__SECRET_KEY__, self.ACCOUNT_NAME)
        self.account_id_stock = API.stock_account.account_id
        logging.info(f'Stock account ID: {self.account_id_stock}')

        if self.HAS_FUTOPT_ACCOUNT:
            self.can_futures = 'futures' in self.MARKET
            self.account_id_futopt = API.futopt_account.account_id
            self._set_futures_code_list()
            logging.info(f'Futures account ID: {self.account_id_futopt}')

        # 啟動憑證 (Mac 不需啟動)
        if platform != 'darwin':
            logging.info(f'Activate {self.ACCOUNT_NAME} CA')
            if self.__CA_PASSWD__:
                ca_passwd = self.__CA_PASSWD__
            else:
                ca_passwd = self.__ACCOUNT_ID__
            API.activate_ca(
                ca_path=f"./lib/ekey/551/{self.__ACCOUNT_ID__}/S/Sinopac.pfx",
                ca_passwd=ca_passwd,
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
                    logging.debug(f'[Quotes]First|{tick.code}|')

                tick_data = self.update_quote_v1(tick)
                # self.to_redis({tick.code: tick_data})

        @API.on_tick_fop_v1()
        def fop_quote_callback_v1(exchange, tick):
            try:
                if tick.simtrade == 0:
                    symbol = self.Futures_Code_List[tick.code]

                    if symbol not in self.Quotes.NowTargets:
                        logging.debug(f'[Quotes]First|{symbol}|')

                    tick_data = self.update_quote_v1(tick, code=symbol)
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
        API.set_order_callback(self._order_callback)

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
        notifier.post(f'\n{msg}', msgType='Monitor')

    def init_stocks(self):
        '''初始化股票資訊'''

        if not self.can_stock or datetime.now() > TimeEndStock:
            return []

        # 讀取選股清單
        self.stock_strategies = self.get_stock_pool()

        # 取得遠端庫存
        self.stocks = self.get_securityInfo('Stocks')

        # 取得策略清單
        self.init_watchlist(self.stocks, self.stock_strategies)

        # 庫存的處理
        self.stocks = self.stocks.merge(
            self.watchlist,
            how='left',
            on=['account', 'market', 'code']
        )
        self.stocks.position.fillna(100, inplace=True)
        self.stock_strategies.update(
            self.stocks.set_index('code').strategy.to_dict())

        # 剔除不堅控的股票
        self.stocks = self.stocks[~self.stocks.code.isin(self.FILTER_OUT)]

        # 新增歷史K棒資料
        self.update_stocks_to_monitor()
        all_targets = list(self.stocks_to_monitor)
        self.history_kbars(['TSE001', 'OTC101'] + all_targets)

        # 交易風險控制
        buy_condition = self.stocks.action == 'Buy'
        self.n_stocks_long = self.stocks[buy_condition].shape[0]
        self.n_stocks_short = self.stocks[~buy_condition].shape[0]
        self.N_LIMIT_LS = self.StrategySet.setNStockLimitLong(KBars=self.KBars)
        self.N_LIMIT_SS = self.StrategySet.setNStockLimitShort(
            KBars=self.KBars)
        self.punish_list = crawler2.get_punish_list().證券代號.to_list()
        self._set_leverage(all_targets)
        self._set_trade_risks()
        logging.debug(f'stocks_to_monitor: {self.stocks_to_monitor}')
        return all_targets

    def init_futures(self):
        '''初始化期貨資訊'''

        def preprocess_(df):
            if df.shape[0]:
                df['contract'] = df.code.apply(lambda x: get_contract(x))
                df['isDue'] = df.contract.apply(
                    lambda x: TODAY_STR.replace('-', '/') == x.delivery_date)
                df.code = df.contract.apply(lambda x: x.symbol)
                df['order'] = df[['quantity', 'action']].to_dict('records')
            return df

        if not self.can_futures:
            return []

        # 讀取選股清單
        self.futures_strategies = self.get_futures_pool()

        # 取得遠端庫存
        self.futures = self.get_securityInfo('Futures')

        # 庫存的處理
        self.futures = preprocess_(self.futures)
        self.futures = self.futures.merge(
            self.watchlist,
            how='left',
            on=['account', 'market', 'code']
        )
        self.futures.position.fillna(100, inplace=True)
        self.futures.index = self.futures.code

        # 剔除不堅控的股票
        self.futures = self.futures[~self.futures.code.isin(self.FILTER_OUT)]
        self.futures_strategies.update(self.futures.strategy.to_dict())
        self.n_futures = self.futures.shape[0]

        # update_futures_to_monitor
        self.futures_to_monitor.update(self.futures.to_dict('index'))
        self.futures_to_monitor.update({
            f: None for f in self.futures_strategies if f not in self.futures_to_monitor})

        # 新增歷史K棒資料
        all_futures = list(self.futures_to_monitor)
        self.history_kbars(all_futures)

        # 交易風險控制
        self.N_FUTURES_LIMIT = self.StrategySet.setNFuturesLimit(
            KBars=self.KBars)
        self._set_margin_limit()
        self.margin_table = self.get_margin_table().原始保證金.to_dict()
        logging.debug(f'futures_to_monitor: {self.futures_to_monitor}')
        return all_futures

    def _update_position(self, order: namedtuple, market: str, order_data: dict):
        '''
        Updating steps:
        1. update_position_quantity
        2. update_deal_list
        3. check_remove_monitor
        4. update_pos_target
        5. update_watchlist_position
        '''
        target = order.target

        # update monitor list position
        if self.simulation:
            if market == 'Stocks':
                action = order.action
                order_data.update({
                    'position': order.pos_target,
                    'bst': datetime.now(),
                    'cost_price': abs(order_data['price']),
                    'yd_quantity': 0,
                })
            else:
                action = order.octype
                order_data.update({
                    'position': order.pos_target,
                    'bst': datetime.now(),
                    'symbol': target,
                    'cost_price': abs(order_data['price']),
                    'order': {
                        'quantity': self.get_sell_quantity(order, market),
                        'action': order.action
                    }
                })

            self.update_position_quantity(action, order_data, order.pos_target)
            self.update_deal_list(target, action, market)

        # check monitor list
        is_empty = self.check_remove_monitor(target, order.action_type, market)
        order = self.update_pos_target(order, is_empty)

        # append watchlist or udpate watchlist position
        if market == 'Stocks':
            strategies = self.stock_strategies
        else:
            strategies = self.futures_strategies
        self.update_watchlist_position(order, self.Quotes, strategies)

    def update_stocks_to_monitor(self):
        '''更新買進/賣出股票監控清單'''

        df = self.stocks.copy()
        df.index = df.code
        if not self.simulation:
            df.order_cond = df.order_cond.apply(lambda x: x._value_)
        self.stocks_to_monitor.update(df.to_dict('index'))

        for stock, stra in self.stock_strategies.items():
            if (
                (self.can_buy and self.StrategySet.isLong(stra)) or
                (self.can_sell and self.StrategySet.isShort(stra))
            ) and (stock not in self.stocks_to_monitor):
                self.stocks_to_monitor.update({stock: None})

    def update_after_interfere(self, target: str, action_type: str, market):
        logging.info(
            f'[Monitor List]Interfere|{market}|{target}|{action_type}|')

        infos = dict(
            action_type=action_type,
            target=target,
            pos_target=100
        )
        order = self.OrderInfo(**infos)
        self.check_remove_monitor(target, action_type, market)

        if market == 'Stocks':
            strategies = self.stock_strategies
        else:
            strategies = self.futures_strategies
        self.update_watchlist_position(order, self.Quotes, strategies)

    def merge_buy_sell_lists(self, stocks_pool: Dict[str, str], market='Stocks'):
        '''合併進出場清單: 將庫存與選股清單，合併'''

        if market == 'Stocks' and self.stocks.shape[0]:
            sells = self.stocks.code.values
        elif market == 'Futures' and self.futures.shape[0]:
            sells = self.futures.code.values
        else:
            sells = []

        all = sells.copy()
        for ids in stocks_pool.values():
            all = np.append(all, ids)

        return np.unique(all)

    def monitor_stocks(self, target: str):
        if target in self.Quotes.NowTargets:
            inputs = self.getQuotesNow(target).copy()
            data = self.stocks_to_monitor[target]
            strategy = self.stock_strategies[target]
            isLongStrategy = self.StrategySet.isLong(strategy)
            isDTStrategy = self.StrategySet.isDayTrade(strategy)
            isSell = (
                # long selling
                (data and 'action' in data and data['action'] == 'Buy') or
                # short selling
                (not data and self.can_sell and not isLongStrategy)
            )

            # new position
            if data is None:
                mode = 'short' if isSell else 'long'

                actionType = 'Open'
                pos_balance = 100
                order_cond = self.check_order_cond(target, mode)
                quantity = self.get_quantity(target, strategy, order_cond)
                enoughOpen = self.check_enough(target, quantity, mode)

            # in-stock position
            else:
                actionType = 'Close'
                pos_balance = data['position']
                order_cond = data['order_cond']
                quantity = data['quantity']
                enoughOpen = False

            inStocks = target in self.stocks.code.values
            inDeal = target in self.stock_bought + self.stock_sold

            is_day_trade = isDTStrategy and inDeal and (not inStocks)
            isOpen = actionType == 'Open' and enoughOpen
            isClose = (
                (not isDTStrategy) and (not inDeal) and inStocks and
                (actionType == 'Close')
            )
            isDTClose = (is_day_trade and (actionType == 'Close'))

            if quantity > 0 and (isOpen or isClose or isDTClose):
                tradeType = '當沖' if is_day_trade else '非當沖'
                func = self.StrategySet.mapFunction(
                    actionType, tradeType, strategy)

                if data:
                    inputs.update(data)

                actionInfo = func(
                    inputs=inputs,
                    kbars=self.KBars,
                    Quotes=self.Quotes,
                    pct_chg_DowJones=self.pct_chg_DowJones
                )
                if actionInfo.position:
                    infos = dict(
                        action_type=actionType,
                        action='Sell' if isSell else 'Buy',
                        target=target,
                        quantity=quantity,
                        order_cond=self.day_trade_cond[order_cond] if is_day_trade else order_cond,
                        pos_target=actionInfo.position,
                        pos_balance=pos_balance,
                        reason=actionInfo.msg,
                    )
                    self._log_and_notify(actionInfo.msg)
                    return self.OrderInfo(**infos)
            elif (quantity <= 0 or pos_balance <= 0) and actionType == 'Close':
                self.update_after_interfere(target, actionType, 'Stocks')
                self.StrategySet.update_StrategySet_data(target)

        return self.OrderInfo(target=target)

    def monitor_futures(self, target: str):
        '''檢查期貨是否符合賣出條件，回傳賣出部位(%)'''

        if target in self.Quotes.NowTargets and self.N_FUTURES_LIMIT != 0:
            inputs = self.getQuotesNow(target).copy()
            data = self.futures_to_monitor[target]
            strategy = self.futures_strategies[target]

            # new position
            if data is None:
                actionType = 'Open'
                pos_balance = 100
                octype = 'New'
                quantity = self.get_open_slot(target, strategy)
                enoughOpen = self._check_enough_open(target, quantity)

            # in-stock position
            else:
                actionType = 'Close'
                pos_balance = data['position']
                octype = 'Cover'
                quantity = data['order']['quantity']
                enoughOpen = False

            if target in self.futures_transferred:
                msg = f'{target} 轉倉-New'
                infos = dict(
                    action_type=actionType,
                    action=self.futures_transferred[target]['action'],
                    target=target,
                    quantity=self.futures_transferred[target]['quantity'],
                    octype=octype,
                    pos_target=100,
                    pos_balance=0,
                    reason=msg
                )
                self._log_and_notify(msg)
                self.futures_transferred.pop(target)

                return self.OrderInfo(**infos)

            c1 = octype == 'New' and enoughOpen and not self.is_not_trade_day(
                inputs['datetime'])
            c2 = octype == 'Cover'
            if quantity > 0 and pos_balance > 0 and (c1 or c2):
                is_day_trade = self.StrategySet.isDayTrade(strategy)
                tradeType = '當沖' if is_day_trade else '非當沖'
                isTransfer = (
                    (actionType == 'Close') and
                    ('isDue' in data) and
                    data['isDue'] and
                    (datetime.now() > TimeTransferFutures)
                )
                if isTransfer:
                    func = self.StrategySet.transfer_position
                else:
                    func = self.StrategySet.mapFunction(
                        actionType, tradeType, strategy)

                if data:
                    inputs.update(data)

                actionInfo = func(
                    inputs=inputs,
                    kbars=self.KBars,
                    Quotes=self.Quotes,
                    pct_chg_DowJones=self.pct_chg_DowJones
                )
                if actionInfo.position:
                    if isTransfer:
                        new_contract = f'{target[:3]}{self.GetDueMonth()}'
                        self.margin_table[new_contract] = self.margin_table[target]
                        self.futures_to_monitor.update({new_contract: None})
                        self.history_kbars([new_contract])
                        self.subscribe_all([new_contract])
                        self.futures_transferred.pop(target, None)
                        self.futures_transferred.update({
                            new_contract: {
                                'quantity': quantity,
                                'action': data['order']['action']
                            }
                        })
                        self.futures_strategies[new_contract] = strategy
                        self.transfer_list.append(new_contract)

                    infos = dict(
                        action_type=actionType,
                        action=actionInfo.action,
                        target=target,
                        quantity=quantity,
                        octype=octype,
                        pos_target=actionInfo.position,
                        pos_balance=pos_balance,
                        reason=actionInfo.msg
                    )
                    self._log_and_notify(actionInfo.msg)
                    return self.OrderInfo(**infos)
            elif (quantity <= 0 or pos_balance <= 0) and actionType == 'Close':
                self.update_after_interfere(target, actionType, 'Futures')
                self.StrategySet.update_StrategySet_data(target)

        return self.OrderInfo(target=target)

    def _place_order(self, content: namedtuple, market='Stocks'):
        logging.debug(f'[OrderState.Content|{content}|')

        is_stock = market == 'Stocks'
        target = content.target
        contract = get_contract(target)
        if target in self.BidAsk:
            quantity = self.get_sell_quantity(content, market)
            price_type = 'MKT'
            price = 0
            order_lot = 'IntradayOdd' if content.quantity < 1000 and is_stock else 'Common'

            if is_stock:
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
                        price = self.getQuotesNow(target)['price']

                log_msg = f'[OrderState.Info]|{target}|price:{price}, quantity:{quantity}, action:{content.action}, price_type:{price_type}, order_cond:{content.order_cond}, order_lot:{order_lot}|'
            else:
                log_msg = f'[OrderState.Info]|{target}|price:{price}, quantity:{quantity}, action:{content.action}, price_type:{price_type}, order_cond:{content.octype}, order_lot:{order_lot}|'

            # 下單
            logging.debug(log_msg)
            if self.simulation and is_stock:
                price = self.getQuotesNow(target)['price']
                quantity *= 1000
                leverage = self.check_leverage(target, content.order_cond)
                if content.action == 'Sell':
                    cost_price = self.get_cost_price(
                        target, price, content.order_cond)
                    amount = -(price - cost_price*leverage) * \
                        quantity*(1 - FEE_RATE)
                else:
                    amount = self.get_stock_amount(
                        target, price, quantity, content.order_cond)

                sign = -1 if content.action == 'Sell' else 1
                order_data = {
                    'Time': datetime.now(),
                    'market': market,
                    'code': target,
                    'action': content.action,
                    'price': price*sign,
                    'quantity': quantity,
                    'amount': amount,
                    'order_cond': content.order_cond if is_stock else 'Cash',
                    'order_lot': order_lot,
                    'leverage': leverage,
                    'account_id': f'simulate-{self.ACCOUNT_NAME}',
                    'msg': content.reason
                }
                self.appendOrder(order_data)

                logging.debug('Place simulate order complete.')
                notifier.post(log_msg, msgType='Order')
                return order_data

            elif self.simulation and market == 'Futures':
                price = self.getQuotesNow(target)['price']
                sign = -1 if content.octype == 'Cover' else 1
                order_data = {
                    'Time': datetime.now(),
                    'market': market,
                    'code': target,
                    'action': content.action,
                    'price': price*sign,
                    'quantity': quantity,
                    'amount': self.get_open_margin(target, quantity)*sign,
                    'op_type': content.octype,
                    'account_id': f'simulate-{self.ACCOUNT_NAME}',
                    'msg': content.reason
                }
                self.appendOrder(order_data)

                logging.debug('Place simulate order complete.')
                notifier.post(log_msg, msgType='Order')
                return order_data

            else:
                # #ff0000 批次下單的張數 (股票>1000股的單位為【張】) #ff0000
                q = 5 if order_lot == 'Common' else quantity
                if is_stock:
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
                        order_type=constant.OrderType.ROD if is_stock else constant.OrderType.IOC,
                        # 委託類型: 現股/融資
                        order_cond=content.order_cond if is_stock else 'Cash',
                        # 整張或零股
                        order_lot=order_lot,
                        # {Auto, New, Cover, DayTrade}(自動、新倉、平倉、當沖)
                        octype='Auto' if is_stock else content.octype,
                        account=API.stock_account if is_stock else API.futopt_account,
                        # 先賣後買: True, False
                        daytrade_short=content.daytrade_short,
                    )
                    result = API.place_order(contract, order)
                    self.check_order_status(result, market)
                    quantity -= q

    def get_securityInfo(self, market='Stocks'):
        '''取得證券庫存清單'''

        if self.simulation:
            df_default = {
                'Stocks': self.df_securityInfo,
                'Futures': self.df_futuresInfo
            }
            try:
                if db.HAS_DB:
                    table = SecurityInfoStocks if market == 'Stocks' else SecurityInfoFutures
                    df = db.query(table, table.account == self.ACCOUNT_NAME)
                else:
                    df = file_handler.read_table(
                        f'{PATH}/stock_pool/simulation_{market.lower()}_{self.ACCOUNT_NAME}.pkl',
                        df_default=df_default[market]
                    )
            except:
                df = df_default[market]

            df['account_id'] = 'simulate'

        else:
            if market == 'Stocks':
                df = self.securityInfo()
                return df[df.code.apply(len) == 4]

            df = self.get_openpositions()
        return df

    def get_margin_table(self):
        '''取得保證金清單'''
        df = self.read_table('./lib/indexMarging.csv')

        codes = [[f.code, f.symbol, f.name]
                 for m in API.Contracts.Futures for f in m]
        codes = pd.DataFrame(codes, columns=['code', 'symbol', 'name'])
        codes = codes.set_index('name').symbol.to_dict()

        month = self.GetDueMonth()[-2:]
        df['code'] = (df.商品別 + month).map(codes)
        return df.dropna().set_index('code')

    def get_stock_pool(self):
        '''取得股票選股池。pools = {stockid: strategy}'''

        pools = {}

        df = picker.get_selection_files()
        if df.shape[0]:
            # 排除不交易的股票
            # ### 全額交割股不買
            day_filter_out = crawler2.get_CashSettle()
            df = df[~df.code.isin(day_filter_out.股票代碼.values)]
            df = df[~df.code.isin(self.FILTER_OUT)]

            # 排除高價股
            df = df[df.Close <= self.PRICE_THRESHOLD]

            df = df.sort_values('Close')

            # 建立族群清單
            n_category = df.groupby('category').code.count().to_dict()
            df['n_category'] = df.category.map(n_category)
            self.n_categories = (
                df.sort_values('n_category', ascending=False)
                .set_index('code').n_category.to_dict())

            # 族群清單按照策略權重 & pc_ratio 決定
            # 權重大的先加入，避免重複
            if self.StrategySet.pc_ratio >= 115:
                sort_order = ['long_weight', 'short_weight']
            else:
                sort_order = ['short_weight', 'long_weight']
            strategies = self.StrategySet.STRATEGIES_STOCK.sort_values(
                sort_order, ascending=False).name.to_list()

            for s in strategies:
                if s in df.Strategy.values and s in self.STRATEGY_STOCK:
                    code = df[df.Strategy == s].code
                    pools.update({stock: s for stock in code})
                    df = df[~df.code.isin(code)]

        return pools

    def get_futures_pool(self):
        '''取得期權目標商品清單。pools = {symbol: stragety}'''

        pools = {}

        due_year_month = self.GetDueMonth()
        for st in self.STRATEGY_FUTURES:
            if '小台' in st:
                pools.update({f'MXF{due_year_month}': st})
            elif '大台' in st:
                pools.update({f'TXF{due_year_month}': st})

        df = picker.get_selection_files()
        if df.shape[0]:
            # 排除不交易的股票
            df = df[~df.code.isin(self.FILTER_OUT)]

            df = df.sort_values('Close')

            # 族群清單按照策略權重 & pc_ratio 決定
            # 權重大的先加入，避免重複
            if self.StrategySet.pc_ratio >= 115:
                sort_order = ['long_weight', 'short_weight']
            else:
                sort_order = ['short_weight', 'long_weight']
            strategies = self.StrategySet.STRATEGIES_FUTURES.sort_values(
                sort_order, ascending=False).name.to_list()

            for s in strategies:
                if s in df.Strategy.values and s in self.STRATEGY_FUTURES:
                    code = df[df.Strategy == s].code
                    pools.update({stock: s for stock in code})
                    df = df[~df.code.isin(code)]

        return pools

    def get_quantity(self, target: str, strategy: str, order_cond: str):
        '''計算進場股數'''

        if self.BUY_UNIT_TYPE == 'constant':
            return 1000*self.BUY_UNIT

        quantityFunc = self.StrategySet.mapQuantities(strategy)

        inputs = self.getQuotesNow(target)
        quantity, quantity_limit = quantityFunc(
            inputs=inputs, kbars=self.KBars)
        leverage = self.check_leverage(target, order_cond)

        quantity = int(min(quantity, quantity_limit)/(1 - leverage))
        quantity = min(quantity, 499)

        contract = get_contract(target)
        if order_cond == 'MarginTrading':
            quantity = min(contract.margin_trading_balance, quantity)
        elif order_cond == 'ShortSelling':
            quantity = min(contract.short_selling_balance, quantity)

        return 1000*quantity

    def get_stock_amount(self, target: str, price: float, quantity: int, mode='long'):
        '''計算股票委託金額'''

        leverage = self.check_leverage(target, mode)
        fee = max(price*quantity*FEE_RATE, 20)
        return price*quantity*(1 - leverage) + fee

    def get_open_slot(self, target: str, strategy: str):
        '''計算買進口數'''

        if self.N_SLOT_TYPE == 'constant':
            return self.N_SLOT

        quantityFunc = self.StrategySet.mapQuantities(strategy)

        inputs = self.getQuotesNow(target)
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
        dj = crawler2.DowJones(start, TODAY_STR)
        if 'c' in dj and len(dj['c']):
            dj = dj['c']
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

        cost_price = self.getQuotesNow(target)['price']
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

    def is_not_trade_day(self, now: datetime):
        '''檢查是否為非交易時段'''
        is_holiday = pd.to_datetime(TODAY_STR) in holidays

        if self.can_futures:
            period = self.TRADING_PERIOD

            td = timedelta(minutes=6)
            if not is_holiday:
                if period == 'Day' and TimeStartFuturesDay - td <= now <= TimeEndFuturesDay:
                    return False

                if period == 'Night' and TimeStartFuturesNight - td <= now <= TimeEndFuturesNight:
                    return False

                if period == 'Both':
                    both_trade_periods = [
                        (TimeStartFuturesDay, TimeEndFuturesDay),
                        (TimeStartFuturesNight, TimeEndFuturesNight)
                    ]

                    for start, end in both_trade_periods:
                        if start - td <= now <= end:
                            return False

            return True

        return is_holiday or not (now <= TEnd)

    def is_break_loop(self, now: datetime):
        return (
            self.is_not_trade_day(now) or
            all(x == 0 for x in [
                self.n_stocks_long, self.n_stocks_short,
                self.N_LIMIT_LS, self.N_LIMIT_SS,
                self.N_FUTURES_LIMIT, self.n_futures
            ])
        )

    def check_remove_monitor(self, target: str, action_type: str, market='Stocks'):
        if action_type == 'Open':
            return False

        is_empty = self.check_is_empty(target, market)
        if is_empty:
            logging.info(f'[Monitor List]Remove|{market}|{target}|')
            if market == 'Stocks':
                day_trade = self.StrategySet.isDayTrade(
                    self.stock_strategies[target])
                self.reset_monitor_list(target, 'Stocks', day_trade=False)
                if target in self.stocks.code.values:
                    self.stocks = self.stocks[self.stocks.code != target]

            else:
                day_trade = self.StrategySet.isDayTrade(
                    self.futures_strategies[target])
                self.reset_monitor_list(target, 'Futures', day_trade=day_trade)

                if target in self.futures.code.values:
                    self.futures = self.futures[self.futures.code != target]
                    self.n_futures = self.futures.shape[0]

        if self.simulation and db.HAS_DB:
            table = SecurityInfoStocks if market == 'Stocks' else SecurityInfoFutures
            db.delete(
                table,
                table.code == target,
                table.account == self.ACCOUNT_NAME
            )

        return is_empty

    def loop_pause(self, freq=MonitorFreq):
        now = datetime.now()
        second = now.second
        microsecond = now.microsecond / 1e6

        # Calculate time to sleep until the next interval
        next_time = (second + microsecond) % freq
        sleep_time = freq - next_time

        if sleep_time < 0:
            sleep_time += freq

        time.sleep(sleep_time)

    def run(self):
        '''執行自動交易'''

        all_stocks = self.init_stocks()
        all_futures = self.init_futures()
        usage = round(API.usage().bytes/2**20, 2)
        self.subscribe_all(all_stocks+all_futures)

        logging.info(f"Current data usage: {usage}")
        logging.info(f"Today's punish lis: {self.punish_list}")
        logging.info(f"Stocks Ex-dividend: {self.StrategySet.dividends}")
        logging.info(f"Previous Put/Call ratio: {self.StrategySet.pc_ratio}")
        logging.info(f'Start to monitor, basic settings:')
        logging.info(f'Mode:{self.MODE}, Strategy: {self.STRATEGY_STOCK}')
        logging.info(f'[Stock Strategy] {self.stock_strategies}')
        logging.info(f'[Stock Position] Long: {self.n_stocks_long}')
        logging.info(f'[Stock Position] Short: {self.n_stocks_short}')
        logging.info(f'[Stock Portfolio Limit] Long: {self.N_LIMIT_LS}')
        logging.info(f'[Stock Portfolio Limit] Short: {self.N_LIMIT_SS}')
        logging.info(
            f'[Stock Position Limit] Long: {self.POSITION_LIMIT_LONG}')
        logging.info(
            f'[Stock Position Limit] Short: {self.POSITION_LIMIT_SHORT}')
        logging.info(f'[Stock Model Version] {self.STOCK_MODEL_VERSION}')
        logging.info(f'[Futures Strategy] {self.futures_strategies}')
        logging.info(f'[Futures position] {self.n_futures}')
        logging.info(f'[Futures portfolio Limit] {self.N_FUTURES_LIMIT}')
        logging.info(f'[Futures Model Version] {self.FUTURES_MODEL_VERSION}')

        text = f"\n【開始監控】{self.ACCOUNT_NAME} 啟動完成({__version__})"
        text += f"\n【操盤模式】{self.MODE}"
        text += f"\n【股票策略】{self.STRATEGY_STOCK}"
        text += f"\n【期貨策略】{self.STRATEGY_FUTURES}"
        text += f"\n【AI版本】Stock-{self.STOCK_MODEL_VERSION}; Futures:{self.FUTURES_MODEL_VERSION}"
        text += f"\n【前日行情】Put/Call: {self.StrategySet.pc_ratio}"
        text += f"\n【美股行情】道瓊({self.pct_chg_DowJones}%)"
        text += f"\n【數據用量】{usage}MB"
        notifier.post(text, msgType='Monitor')

        def periodic_updates():
            while True:
                self.loop_pause(freq=.5)
                now = datetime.now()

                if self.is_break_loop(now):
                    break

                # update K-bar data
                is_trading_time = (
                    (self.can_futures and now > TimeStartFuturesDay + timedelta(seconds=30)) or
                    (self.can_stock and now > TimeStartStock + timedelta(seconds=30))
                )
                if is_trading_time and now.second == 0 and now.microsecond/1e6 < .5:

                    if now.minute % 2 == 0:
                        self.updateKBars('2T')

                    if now.minute % 5 == 0:
                        self.updateKBars('5T')
                        # 防止斷線用 TODO:待永豐更新後刪除
                        balance = self.balance(mode='debug')
                        if balance == -1:
                            self._log_and_notify(
                                f"【連線異常】{self.ACCOUNT_NAME} 無法查詢餘額")

                    if now.minute % 15 == 0:
                        self.updateKBars('15T')

                    if now > TimeStartFuturesNight and now.minute % 30 == 0:
                        self.updateKBars('30T')
                    elif now > TimeStartFuturesDay and now.minute % 30 in [15, 45]:
                        self.updateKBars('30T')

                    if now > TimeStartFuturesNight and now.minute == 0:
                        self.updateKBars('60T')
                    elif now > TimeStartFuturesDay and now.minute == 45:
                        self.updateKBars('30T')

        # 開始監控
        exec.submit(periodic_updates)

        while True:
            self.loop_pause()
            now = datetime.now()

            if self.is_break_loop(now):
                break

            # TODO: merge stocks_to_monitor & futures_to_monitor
            for target in list(self.stocks_to_monitor):
                order = self.monitor_stocks(target)
                if order.pos_target:
                    order_data = self._place_order(order, market='Stocks')
                    self._update_position(order, 'Stocks', order_data)

            for target in list(self.futures_to_monitor):
                order = self.monitor_futures(target)
                if order.pos_target:
                    order_data = self._place_order(order, market='Futures')
                    self._update_position(order, 'Futures', order_data)

        logging.info('Non-trading time, stop monitoring')

        for scale in ['2T', '5T', '15T', '30T', '60T']:
            self.updateKBars(scale)

        if all(x == 0 for x in [
            self.n_stocks_long, self.n_stocks_short,
            self.N_LIMIT_LS, self.N_LIMIT_SS,
            self.N_FUTURES_LIMIT, self.n_futures
        ]):
            self._log_and_notify(f"【停止監控】{self.ACCOUNT_NAME} 無可監控清單")

        time.sleep(3)
        self.unsubscribe_all(all_stocks+all_futures)

    def simulator_update_securityInfo(self, df: pd.DataFrame, table):
        market = 'stocks' if 'stocks' in table.__tablename__ else 'futures'
        if db.HAS_DB:
            match_account = table.account == self.ACCOUNT_NAME
            codes = db.query(table.code, match_account).code.values
            tb = df[~df.code.isin(codes)]
            update_values = df[df.code.isin(codes)].set_index('code')

            # add new stocks
            db.dataframe_to_DB(tb, table)

            # update in-stocks
            update_values = update_values.to_dict('index')
            for target, values in update_values.items():
                condition = table.code == target, match_account
                db.update(table, values, *condition)
        else:
            self.save_table(
                df=df,
                filename=f'{PATH}/stock_pool/simulation_{market}_{self.ACCOUNT_NAME}.pkl'
            )

    def __save_simulate_securityInfo(self):
        '''儲存模擬交易模式下的股票庫存表'''
        if self.simulation:
            # 儲存庫存
            logging.debug(f'stocks_to_monitor: {self.stocks_to_monitor}')
            logging.debug(
                f'stocks shape: {self.stocks.shape}; watchlist shape: {self.watchlist.shape}')
            df = {k: v for k, v in self.stocks_to_monitor.items() if v}
            df = pd.DataFrame(df).T
            if df.shape[0]:
                # df = df[df.account_id == f'simulate-{self.ACCOUNT_NAME}']
                df = df[df.account_id.str.contains('simulate')]
                df = df.sort_values('code').reset_index()
                if self.is_not_trade_day(datetime.now()):
                    df['last_price'] = 0
                else:
                    df['last_price'] = df.code.map(
                        {s: self.getQuotesNow(s)['price'] for s in df.code})
                df['pnl'] = df.action.apply(lambda x: 1 if x == 'Buy' else -1)
                df['pnl'] = df.pnl*(df.last_price - df.cost_price)*df.quantity
                df.yd_quantity = df.quantity
                df['account'] = self.ACCOUNT_NAME
                df = df[self.df_securityInfo.columns]
            else:
                df = self.df_securityInfo
            logging.debug(
                f'stocks shape: {df.shape}; watchlist shape: {self.watchlist.shape}')

            if df.shape[0]:
                self.simulator_update_securityInfo(df, SecurityInfoStocks)

    def __save_simulate_futuresInfo(self):
        '''儲存模擬交易模式下的期貨庫存表'''
        if self.simulation:
            # 儲存庫存
            logging.debug(f'futures_to_monitor: {self.futures_to_monitor}')
            df = {k: v for k, v in self.futures_to_monitor.items() if v}
            df = pd.DataFrame(df).T
            if df.shape[0]:
                df = df[df.account_id.str.contains('simulate')]
                df = df.reset_index(drop=True)
                # df['id'] = np.arange(df.shape[0])
                if 'order' in df.columns:
                    df['direction'] = df.order.apply(lambda x: x['action'])
                else:
                    df['direction'] = df.action
                if self.is_not_trade_day(datetime.now()):
                    df['last_price'] = 0
                else:
                    df['last_price'] = df.code.map(
                        {s: self.getQuotesNow(s)['price'] for s in df.code})
                df['pnl'] = df.direction.apply(
                    lambda x: 1 if x == 'Buy' else -1)
                df['pnl'] = df.pnl*(df.last_price - df.cost_price)*df.quantity
                df['account'] = self.ACCOUNT_NAME
                df = df[self.df_futuresInfo.columns]
            else:
                df = self.df_futuresInfo

            if df.shape[0]:
                self.simulator_update_securityInfo(df, SecurityInfoFutures)

    def output_files(self):
        '''停止交易時，輸出庫存資料 & 交易明細'''
        if 'position' in self.stocks.columns and not self.simulation:
            codeList = self.get_securityInfo('Stocks').code.to_list()
            self.update_watchlist(codeList)

        self.save_watchlist(self.watchlist)
        self.output_statement(
            f'{PATH}/stock_pool/statement_{self.ACCOUNT_NAME}.csv')
        self.StrategySet.export_strategy_data()

        for freq, df in self.KBars.items():
            if freq != '1D':
                filename = f'{PATH}/Kbars/k{freq[:-1]}min_{self.ACCOUNT_NAME}.csv'
                self.save_table(df, filename)

        if self.can_stock:
            self.__save_simulate_securityInfo()

        if self.can_futures:
            self.__save_simulate_futuresInfo()

        time.sleep(1)
