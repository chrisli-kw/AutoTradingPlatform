import ssl
import time
import logging
import pandas as pd
from shioaji import constant
from collections import namedtuple
from datetime import datetime, timedelta

from . import __version__, notifier, picker
from .config import (
    API,
    PATH,
    TODAY_STR,
    MonitorFreq,
    TimeSimTradeStockEnd,
    TimeEndStock,
    TimeTransferFutures
)
from .utils import get_contract
from .utils.objects.data import TradeData
from .utils.orders import OrderTool
from .utils.time import time_tool
from .utils.crawler import crawler
from .utils.file import file_handler
from .utils.objects.env import UserEnv
from .utils.accounts import AccountInfo
from .utils.subscribe import Subscriber
from .utils.simulation import Simulator
from .utils.positions import WatchListTool, TradeDataHandler
from .utils.callback import CallbackHandler
from .utils.database import db
try:
    from .scripts.StrategySet import StrategySet as StrategySets
except:
    from .utils.strategy import StrategyTool as StrategySets


ssl._create_default_https_context = ssl._create_unverified_context


class StrategyExecutor(AccountInfo, WatchListTool, OrderTool, Subscriber):
    def __init__(self, account_name: str):
        self.env = UserEnv(account_name)

        super().__init__()
        Subscriber.__init__(self, self.env.KBAR_START_DAYay)
        OrderTool.__init__(self, self.env.ACCOUNT_NAME)
        WatchListTool.__init__(self, self.env.ACCOUNT_NAME)

        self.simulation = self.env.MODE == 'Simulation'
        self.simulator = Simulator(self.env.ACCOUNT_NAME)
        self.StrategySet = StrategySets(self.env)
        self.day_trade_cond = {
            'MarginTrading': 'ShortSelling',
            'ShortSelling': 'MarginTrading',
            'Cash': 'Cash'
        }
        self.desposal_money = 0
        self.total_market_value = 0
        self.punish_list = []

    def _set_trade_risks(self):
        '''設定交易風險值: 可交割金額、總市值'''

        df = TradeData.Stocks.Info.copy()
        cost_value = (df.quantity*df.cost_price).sum()
        pnl = df.pnl.sum()
        if self.simulation:
            account_balance = self.env.INIT_POSITION
            settle_info = pnl
        else:
            account_balance = self.balance()
            settle_info = self.settle_info(mode='info').iloc[1:, 1].sum()

        self.desposal_money = min(
            account_balance+settle_info, self.env.POSITION_LIMIT_LONG)
        self.total_market_value = self.desposal_money + cost_value + pnl

        logging.info(
            f'Desposal amount = {self.desposal_money} (limit: {self.env.POSITION_LIMIT_LONG})')

    def _set_margin_limit(self):
        '''計算可交割的保證金額，不可超過帳戶可下單的保證金額上限'''
        if self.simulation:
            account_balance = 0
            self.desposal_margin = self.simulator.simulate_amount
            self.ProfitAccCount = self.simulator.simulate_amount
        else:
            account_balance = self.balance()
            self.get_account_margin()
        self.desposal_margin = min(
            account_balance+self.desposal_margin, self.env.MARGIN_LIMIT)
        logging.info(
            f'[AccountInfo] Margin: total={self.ProfitAccCount}; available={self.desposal_margin}; limit={self.env.MARGIN_LIMIT}')

    def _set_leverage(self, stockids: list):
        '''
        取得個股融資成數資料，
        若帳戶設定為不可融資，則全部融資成數為0
        '''

        df = pd.DataFrame([crawler.FromHTML.Leverage(s) for s in stockids])
        if df.shape[0]:
            df.columns = df.columns.str.replace(' ', '')
            df.loc[df.個股融券信用資格 == 'N', '融券成數'] = 100
            df.代號 = df.代號.astype(str)
            df.融資成數 /= 100
            df.融券成數 /= 100

            if self.env.ORDER_COND1 != 'Cash':
                TradeData.Stocks.Leverage.Long = df.set_index(
                    '代號').融資成數.to_dict()
            else:
                TradeData.Stocks.Leverage.Long = {code: 0 for code in stockids}

            if self.env.ORDER_COND2 != 'Cash':
                TradeData.Stocks.Leverage.Short = df.set_index(
                    '代號').融券成數.to_dict()
            else:
                TradeData.Stocks.Leverage.Short = {
                    code: 1 for code in stockids}

        logging.info(f'Long leverages: {TradeData.Stocks.Leverage.Long}')
        logging.info(f'Short leverages: {TradeData.Stocks.Leverage.Short}')

    def _set_futures_code_list(self):
        '''期貨商品代號與代碼對照表'''
        if self.env.can_futures:
            logging.debug('Set Futures_Code_List')
            TradeData.Futures.CodeList.update({
                f.code: f.symbol for m in API.Contracts.Futures for f in m
            })

    def _order_callback(self, stat, msg):
        '''處理委託/成交回報'''

        if stat == constant.OrderState.StockOrder:
            stock = msg['contract']['code']
            order = msg['order']
            operation = msg['operation']

            c3 = order['action'] == 'Buy'
            if order['account']['account_id'] == self.account_id_stock:
                notifier.post_tftOrder(stat, msg)
                TradeDataHandler.update_deal_list(
                    stock, order['action'], 'Stocks')

                leverage = self.check_leverage(stock, order['order_cond'])

                if self.is_new_order(operation) and c3:
                    self.appendOrder(stock, order, 'Stocks')

                # 若融資配額張數不足，改現股買進 ex: '此證券配額張數不足，餘額 0 張（證金： 0 ）'
                elif self.is_insufficient_quota(operation):
                    q_balance = operation['op_msg'].split(' ')
                    if len(q_balance) > 1:
                        q_balance = int(q_balance[1])
                        infos = dict(
                            action=order['action'], target=stock, pos_target=100, pos_balance=100)
                        # 若本日還沒有下過融資且剩餘券數為0，才可以改下現股
                        if q_balance == 0 and stock not in TradeData.Stocks.Bought:
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
                if self.is_cancel_order(operation):
                    self.deleteOrder(stock)
                    if c3:
                        TradeDataHandler.update_deal_list(
                            stock, 'Cancel', 'Stocks')

        elif stat == constant.OrderState.StockDeal and msg['account_id'] == self.account_id_stock:
            notifier.post_tftDeal(stat, msg)
            msg = CallbackHandler.update_stock_msg(msg)

            action = msg['action']
            if action == 'Sell':
                stock = msg['code']
                TradeDataHandler.update_deal_list(stock, action, 'Stocks')
                self.appendOrder(stock, msg, 'Stocks')

            # 更新監控庫存
            TradeDataHandler.update_monitor(action, msg)

        elif stat == constant.OrderState.FuturesOrder:
            notifier.post_fOrder(stat, msg)
            msg = CallbackHandler().update_futures_msg(msg)

            order = msg['order']
            if order['account']['account_id'] == self.account_id_futopt:
                symbol = CallbackHandler.fut_symbol(msg)
                operation = msg['operation']
                market = 'Futures' if msg['contract']['option_right'] == 'Future' else 'Options'

                if self.is_new_order(operation):
                    TradeDataHandler.update_deal_list(
                        symbol, order['oc_type'], market)
                    self.appendOrder(symbol, order, market)

                # 若刪單成功就自清單移除
                if self.is_cancel_order(operation):
                    self.deleteOrder(symbol)
                    TradeDataHandler.update_deal_list(symbol, 'Cancel', market)
                    if order['oc_type'] == 'New':
                        TradeData.Futures.Monitor[symbol] = None

                # 更新監控庫存
                TradeDataHandler.update_monitor(order['oc_type'], msg)

        elif stat == constant.OrderState.FuturesDeal:
            notifier.post_fDeal(stat, msg)
            CallbackHandler.fDeal(msg)

    def login_and_activate(self):
        # 登入
        self.login_(self.env)
        self.account_id_stock = API.stock_account.account_id
        logging.info(
            f'[AccountInfo] Stock account ID: {self.account_id_stock}')

        if self.HAS_FUTOPT_ACCOUNT:
            self.env.can_futures = 'futures' in self.env.MARKET
            self.account_id_futopt = API.futopt_account.account_id
            self._set_futures_code_list()
            logging.info(
                f'[AccountInfo] Futures account ID: {self.account_id_futopt}')

        # 啟動憑證 (Mac 不需啟動)
        logging.info(f'[AccountInfo] Activate {self.env.ACCOUNT_NAME} CA')
        id = self.env.account_id()
        API.activate_ca(
            ca_path=f"./lib/ekey/551/{id}/S/Sinopac.pfx",
            ca_passwd=self.env.ca_passwd() if self.env.ca_passwd() else id,
            person_id=id,
        )

        # 系統 callback 設定
        self._set_callbacks()

    def _set_callbacks(self):
        '''取得API回傳報價'''
        @API.on_tick_stk_v1()
        def stk_quote_callback_v1(exchange, tick):
            if tick.intraday_odd == 0 and tick.simtrade == 0:

                if tick.code not in TradeData.Quotes.NowTargets:
                    logging.debug(f'[Quotes]First|{tick.code}|')

                tick_data = self.update_quote_v1(tick)
                # self.to_redis({tick.code: tick_data})

        @API.on_tick_fop_v1()
        def fop_quote_callback_v1(exchange, tick):
            try:
                if tick.simtrade == 0:
                    symbol = TradeData.Futures.CodeList[tick.code]

                    if symbol not in TradeData.Quotes.NowTargets:
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
            CallbackHandler.events(
                resp_code, event_code, info, event, self.env)

        # 訂閱下單回報
        API.set_order_callback(self._order_callback)

        # 訂閱五檔回報
        @API.on_bidask_stk_v1()
        def stk_quote_callback(exchange, bidask):
            TradeData.BidAsk[bidask.code] = bidask

        @API.on_bidask_fop_v1()
        def fop_quote_callback(exchange, bidask):
            symbol = TradeData.Futures.CodeList[bidask.code]
            TradeData.BidAsk[symbol] = bidask

    def _log_and_notify(self, msg: str):
        '''將訊息加入log並推播'''
        logging.info(msg)
        notifier.post(f'\n{msg}', msgType='Monitor')

    def init_stocks(self):
        '''初始化股票資訊'''

        if not self.env.can_stock or datetime.now() > TimeEndStock:
            return []

        # 讀取選股清單
        TradeData.Stocks.Strategy = self.get_securityPool('Stocks')

        # 取得遠端庫存
        info = self.get_securityInfo('Stocks')

        # 取得策略清單
        self.init_watchlist(info)

        # 庫存的處理
        info = self.merge_info(info)
        TradeData.Stocks.Strategy.update(
            info.set_index('code').strategy.to_dict())

        # 剔除不堅控的股票
        info = info[~info.code.isin(self.env.FILTER_OUT)]

        # 新增歷史K棒資料
        self.update_stocks_to_monitor()
        all_targets = list(TradeData.Stocks.Monitor)
        self.history_kbars(['TSE001', 'OTC101'] + all_targets)

        # 交易風險控制
        buy_condition = info.action == 'Buy'
        TradeData.Stocks.N_Long = info[buy_condition].shape[0]
        TradeData.Stocks.N_Short = info[~buy_condition].shape[0]
        TradeData.Stocks.Info = info
        self.env.N_LIMIT_LS = self.StrategySet.setNStockLimitLong()
        self.env.N_LIMIT_SS = self.StrategySet.setNStockLimitShort()
        self.punish_list = crawler.FromHTML.PunishList()
        self._set_leverage(all_targets)
        self._set_trade_risks()
        logging.debug(f'Stocks to monitor: {TradeData.Stocks.Monitor}')
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

        if not self.env.can_futures:
            return []

        # 讀取選股清單
        TradeData.Futures.Strategy = self.get_securityPool('Futures')

        # 取得遠端庫存
        info = self.get_securityInfo('Futures')

        # 庫存的處理
        info = preprocess_(info)
        info = self.merge_info(info)
        info.index = info.code

        # 剔除不堅控的股票
        info = info[~info.code.isin(self.env.FILTER_OUT)]
        TradeData.Futures.Strategy.update(info.strategy.to_dict())

        # update_futures_to_monitor
        TradeData.Futures.Monitor.update(info.to_dict('index'))
        TradeData.Futures.Monitor.update({
            f: None for f in TradeData.Futures.Strategy if f not in TradeData.Futures.Monitor})
        TradeData.Futures.Info = info

        # 新增歷史K棒資料
        all_futures = list(TradeData.Futures.Monitor)
        all_futures = self.StrategySet.append_monitor_list(all_futures)
        self.history_kbars(all_futures)

        # 交易風險控制
        self.env.N_FUTURES_LIMIT = self.StrategySet.setNFuturesLimit()
        self._set_margin_limit()
        self.margin_table = self.get_margin_table()
        logging.debug(f'Futures to monitor: {TradeData.Futures.Monitor}')
        return all_futures

    def _update_position(self, order: namedtuple, market: str, order_data: dict):
        '''
        Updating steps:
        1. update_monitor
        2. update_deal_list
        3. check_remove_monitor
        4. update_pos_target
        5. update_watchlist_position
        '''
        target = order.target

        # update monitor list position
        if self.simulation:
            self.simulator.update_position(order, market, order_data)

        # check monitor list
        is_empty = self.check_remove_monitor(target, order.action_type, market)
        order = self.update_pos_target(order, is_empty)

        # append watchlist or udpate watchlist position
        self.update_watchlist_position(order)

    def update_stocks_to_monitor(self):
        '''更新買進/賣出股票監控清單'''

        df = TradeData.Stocks.Info.copy()
        df.index = df.code
        if not self.simulation:
            df.order_cond = df.order_cond.apply(lambda x: x._value_)
        TradeData.Stocks.Monitor.update(df.to_dict('index'))

        for stock, stra in TradeData.Stocks.Strategy.items():
            if (
                (self.env.can_buy and self.StrategySet.isLong(stra)) or
                (self.env.can_sell and self.StrategySet.isShort(stra))
            ) and (stock not in TradeData.Stocks.Monitor):
                TradeData.Stocks.Monitor.update({stock: None})

    def update_after_interfere(self, target: str, action_type: str, market):
        logging.warning(
            f'[Monitor List]Interfere|{market}|{target}|{action_type}|')

        infos = dict(
            action_type=action_type,
            target=target,
            pos_target=100
        )
        order = self.OrderInfo(**infos)
        self.check_remove_monitor(target, action_type, market)
        self.update_watchlist_position(order)
        self.StrategySet.update_StrategySet_data(target)

    # def merge_buy_sell_lists(self, stocks_pool: Dict[str, str], market='Stocks'):
    #     # TODO remove
    #     '''合併進出場清單: 將庫存與選股清單，合併'''

    #     if market == 'Stocks' and TradeData.Stocks.Info.shape[0]:
    #         sells = TradeData.Stocks.Info.code.values
    #     elif market == 'Futures' and TradeData.Futures.Info.shape[0]:
    #         sells = TradeData.Futures.Info.code.values
    #     else:
    #         sells = []

    #     all = sells.copy()
    #     for ids in stocks_pool.values():
    #         all = np.append(all, ids)

    #     return np.unique(all)

    def monitor_stocks(self, target: str):
        if target in TradeData.Quotes.NowTargets:
            inputs = TradeDataHandler.getQuotesNow(target).copy()
            data = TradeData.Stocks.Monitor[target]
            strategy = TradeData.Stocks.Strategy[target]
            isLongStrategy = self.StrategySet.isLong(strategy)
            isDTStrategy = self.StrategySet.isDayTrade(strategy)
            isSell = (
                # long selling
                (data and 'action' in data and data['action'] == 'Buy') or
                # short selling
                (not data and self.env.can_sell and not isLongStrategy)
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

            inStocks = target in TradeData.Stocks.Info.code.values
            inDeal = target in TradeData.Stocks.Bought + TradeData.Stocks.Sold

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

                actionInfo = func(inputs=inputs)
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

        return self.OrderInfo(target=target)

    def monitor_futures(self, target: str):
        '''檢查期貨是否符合賣出條件，回傳賣出部位(%)'''

        if target in TradeData.Quotes.NowTargets and self.env.N_FUTURES_LIMIT != 0:
            inputs = TradeDataHandler.getQuotesNow(target).copy()
            data = TradeData.Futures.Monitor[target]
            strategy = TradeData.Futures.Strategy[target]

            # new position
            if data is None:
                actionType = 'Open'
                pos_balance = 100
                octype = 'New'
                quantity = self.get_quantity(target, strategy, 'Futures')
                enoughOpen = self._check_enough_open(target, quantity)

            # in-stock position
            else:
                actionType = 'Close'
                pos_balance = data['position']
                octype = 'Cover'
                quantity = data['order']['quantity']
                enoughOpen = False

            if target in TradeData.Futures.Transferred:
                msg = f'{target} 轉倉-New'
                infos = dict(
                    action_type=actionType,
                    action=TradeData.Futures.Transferred[target]['action'],
                    target=target,
                    quantity=TradeData.Futures.Transferred[target]['quantity'],
                    octype=octype,
                    pos_target=100,
                    pos_balance=0,
                    reason=msg
                )
                self._log_and_notify(msg)
                TradeData.Futures.Transferred.pop(target)

                return self.OrderInfo(**infos)

            c1 = octype == 'New' and enoughOpen and self.is_trading_time_(
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

                actionInfo = func(inputs=inputs)
                if actionInfo.position:
                    if isTransfer:
                        new_contract = f'{target[:3]}{time_tool.GetDueMonth()}'
                        self.transfer_margin(target, new_contract)
                        TradeData.Futures.Monitor.update({new_contract: None})
                        TradeData.Futures.Monitor.pop(target, None)
                        self.history_kbars([new_contract])
                        self.subscribe_all([new_contract])
                        TradeData.Futures.Transferred.update({
                            new_contract: {
                                'quantity': quantity,
                                'action': data['order']['action']
                            }
                        })
                        TradeData.Futures.Strategy[new_contract] = strategy

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

        return self.OrderInfo(target=target)

    def _place_order(self, content: namedtuple, market='Stocks'):
        logging.debug(f'[OrderState.Content|{content}|')

        target = content.target

        if target not in TradeData.BidAsk:
            return

        is_stock = market == 'Stocks'
        contract = get_contract(target)
        quantity = self.get_sell_quantity(content, market)
        price_type = 'MKT'
        price = 0
        order_lot = 'IntradayOdd' if content.quantity < 1000 and is_stock else 'Common'

        if is_stock:
            bid_ask = TradeData.BidAsk[target]
            bid_ask = bid_ask.bid_price if content.action == 'Sell' else bid_ask.ask_price

            # 零股交易
            if 0 < content.quantity < 1000:
                price_type = 'LMT'
                price = bid_ask[1]

            # 整股交易
            else:
                if datetime.now() >= TimeSimTradeStockEnd:
                    price_type = 'LMT'
                    price = bid_ask[1]
                elif target in self.punish_list:
                    price_type = 'LMT'
                    price = bid_ask[3]
                elif contract.exchange == 'OES':
                    price_type = 'LMT'
                    price = TradeDataHandler.getQuotesNow(target)['price']

        # 下單
        log_msg = f'[OrderState.Info]|{target}|price:{price}, quantity:{quantity}, action:{content.action}, price_type:{price_type}, order_cond:{content.order_cond if is_stock else content.octype}, order_lot:{order_lot}|'
        logging.debug(log_msg)
        if self.simulation:
            order_data = self.appendOrder(target, content, market)
            notifier.post(log_msg, msgType='Order')
            logging.debug('Place simulate order complete.')
            return order_data
        else:
            # #ff0000 批次下單的張數 (股票>1000股的單位為【張】) #ff0000
            q = 5 if order_lot == 'Common' else quantity
            target_ = self.desposal_money if is_stock else self.desposal_margin
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
            return self.simulator.securityInfo(self.env.ACCOUNT_NAME, market)
        return self.securityInfo(market)

    def get_securityPool(self, market='Stocks'):
        '''
        Get the target securities pool with the format:
          {code: strategy}
        '''

        pools = {}
        if market == 'Stocks':
            pools.update(self.env.FILTER_IN)
        else:
            due_year_month = time_tool.GetDueMonth()
            pools.update({
                f'{code}{due_year_month}': st for code, st in self.env.FILTER_IN.items()
            })

        df = picker.get_selection_files()
        if df.shape[0]:
            # 排除不交易的股票
            if market == 'Stocks':
                # 全額交割股不買
                day_filter_out = crawler.FromHTML.get_CashSettle()
                df = df[~df.code.isin(day_filter_out.股票代碼.values)]

                # 排除高價股
                df = df[df.Close <= self.env.PRICE_THRESHOLD]

                strategies = self.env.STRATEGY_STOCK
            else:
                strategies = self.env.STRATEGY_FUTURES

            df = df[~df.code.isin(self.env.FILTER_OUT)]
            df = df.sort_values('Close')

            strategies_ordered = self.StrategySet.get_strategy_list(market)
            for s in strategies_ordered:
                if s in df.Strategy.values and s in strategies:
                    code = df[df.Strategy == s].code
                    pools.update({stock: s for stock in code})
                    df = df[~df.code.isin(code)]

        return pools

    def get_quantity(self, target: str, strategy: str, order_cond: str):
        '''Calculate the quantity for opening a position'''

        if order_cond == 'Futures':
            if self.env.N_SLOT_TYPE == 'constant':
                return self.env.N_SLOT
        elif self.env.BUY_UNIT_TYPE == 'constant':
            return 1000*self.env.BUY_UNIT

        quantityFunc = self.StrategySet.mapQuantities(strategy)

        inputs = TradeDataHandler.getQuotesNow(target)
        quantity, quantity_limit = quantityFunc(inputs=inputs)
        leverage = self.check_leverage(target, order_cond)

        quantity = int(min(quantity, quantity_limit)/(1 - leverage))
        quantity = min(quantity, 499)

        if order_cond == 'Futures':
            return quantity

        contract = get_contract(target)
        if order_cond == 'MarginTrading':
            quantity = min(contract.margin_trading_balance, quantity)
        elif order_cond == 'ShortSelling':
            quantity = min(contract.short_selling_balance, quantity)
        return 1000*quantity

    def check_order_cond(self, target: str, mode='long'):
        '''檢查個股可否融資'''
        contract = get_contract(target)
        if mode == 'long':
            if self.env.ORDER_COND1 != 'Cash' and (TradeData.Stocks.Leverage.Long[target] == 0 or contract.margin_trading_balance == 0):
                return 'Cash'
            return self.env.ORDER_COND1
        else:
            if self.env.ORDER_COND2 != 'Cash' and (TradeData.Stocks.Leverage.Short[target] == 1 or contract.short_selling_balance == 0):
                return 'Cash'
            return self.env.ORDER_COND2

    def check_enough(self, target: str, quantity: int, mode='long'):
        '''計算可買進的股票數量 & 金額'''

        if target not in TradeData.Quotes.NowTargets:
            return False

        if mode == 'long':
            func = self.StrategySet.isLong
        else:
            func = self.StrategySet.isShort

        strategies = TradeData.Stocks.Strategy

        buy_deals = TradeData.Stocks.Bought
        buy_deals = len([s for s in buy_deals if func(strategies[s])])
        sell_deals = TradeData.Stocks.Sold
        sell_deals = len([s for s in sell_deals if func(strategies[s])])

        if mode == 'long':
            quota = abs(self.env.N_LIMIT_LS) - \
                TradeData.Stocks.N_Long - buy_deals + sell_deals
        else:
            quota = abs(self.env.N_LIMIT_SS) - \
                TradeData.Stocks.N_Short + buy_deals - sell_deals

        # 更新已委託金額
        df = self.filterOrderTable('Stocks')
        df = df[df.code.apply(len) == 4]
        amount1 = df.amount.sum()
        amount2 = df[df.price > 0].amount.abs().sum()
        amount3 = df[df.price < 0].amount.abs().sum()

        cost_price = TradeDataHandler.getQuotesNow(target)['price']
        target_amount = self.get_stock_amount(
            target, cost_price, quantity, mode)

        # under day limit condition
        # 1. 不可超過可交割金額
        # 2. 不可大於帳戶可委託金額上限
        # 3. 不可超過股票數上限
        if mode == 'long':
            return (
                (amount1 + target_amount <= self.desposal_money) &
                (amount2 + target_amount <= self.env.POSITION_LIMIT_LONG) &
                (quota > 0)
            )

        return (
            (amount1 + target_amount <= self.desposal_money) &
            (amount2 + target_amount <= self.env.POSITION_LIMIT_LONG) &
            # 4. 不可超過可信用交易額度上限
            (amount3 + target_amount <= self.env.POSITION_LIMIT_SHORT) &
            (quota > 0)
        )

    def _check_enough_open(self, target: str, quantity: int):
        '''計算可開倉的期貨口數 & 金額'''

        if target not in TradeData.Quotes.NowTargets:
            return False

        # 更新可開倉的期貨標的數
        open_deals = len(TradeData.Futures.Opened)
        close_deals = len(TradeData.Futures.Closed)
        quota = abs(self.env.N_FUTURES_LIMIT) - \
            TradeData.Futures.Info.shape[0] - open_deals + close_deals

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
            (amount2 + target_amount <= self.env.MARGIN_LIMIT) &
            (quota > 0)
        )

    def is_trading_time_(self, now: datetime):
        '''檢查是否為交易時段'''

        if self.env.can_futures:
            return time_tool.is_trading_time(
                now,
                td=timedelta(minutes=-6),
                market='Futures',
                period=self.env.TRADING_PERIOD
            )

        return time_tool.is_trading_time(
            now,
            td=timedelta(minutes=-20),
            market='Stocks'
        )

    def is_all_zero(self):
        return all(x == 0 for x in [
            TradeData.Stocks.N_Long,
            TradeData.Stocks.N_Short,
            self.env.N_LIMIT_LS,
            self.env.N_LIMIT_SS,
            self.env.N_FUTURES_LIMIT,
            TradeData.Futures.Info.shape[0]
        ])

    def is_break_loop(self, now: datetime):
        return not self.is_trading_time_(now) or self.is_all_zero()

    def check_remove_monitor(self, target: str, action_type: str, market='Stocks'):

        def remove_(df):
            return df[df.code != target]

        if action_type == 'Open':
            return False

        is_empty = TradeDataHandler.check_is_empty(target, market)
        if is_empty:
            logging.debug(f'[Monitor List]Remove|{market}|{target}|')
            if market == 'Stocks':
                day_trade = self.StrategySet.isDayTrade(
                    TradeData.Stocks.Strategy[target])
                TradeDataHandler.reset_monitor(target, market, day_trade=False)
                if target in TradeData.Stocks.Info.code.values:
                    TradeData.Stocks.Info = remove_(TradeData.Stocks.Info)

            else:
                day_trade = self.StrategySet.isDayTrade(
                    TradeData.Futures.Strategy[target])
                TradeDataHandler.reset_monitor(
                    target, market, day_trade=day_trade)

                if target in TradeData.Futures.Info.code.values:
                    TradeData.Futures.Info = remove_(TradeData.Futures.Info)

        if self.simulation:
            self.simulator.remove_from_info(target, self.account_name, market)

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

        logging.info(f'Start to monitor, basic settings:')
        logging.info(f'[AccountInfo] Current data usage: {usage}')
        logging.info(f'[AccountInfo] Mode: {self.env.MODE}')
        logging.info(f'[Stock Strategy] {TradeData.Stocks.Strategy}')
        logging.info(f'[Stock Position] Long: {TradeData.Stocks.N_Long}')
        logging.info(
            f'[Stock Position] Limit Long: {self.env.POSITION_LIMIT_LONG}')
        logging.info(f'[Stock Position] Short: {TradeData.Stocks.N_Short}')
        logging.info(
            f'[Stock Position] Limit Short: {self.env.POSITION_LIMIT_SHORT}')
        logging.info(f'[Stock Portfolio Limit] Long: {self.env.N_LIMIT_LS}')
        logging.info(f'[Stock Portfolio Limit] Short: {self.env.N_LIMIT_SS}')
        logging.info(f'[Stock Model Version] {self.env.STOCK_MODEL_VERSION}')

        logging.info(f'[Futures Strategy] {TradeData.Futures.Strategy}')
        logging.info(f'[Futures position] {TradeData.Futures.Info.shape[0]}')
        logging.info(f'[Futures portfolio Limit] {self.env.N_FUTURES_LIMIT}')
        logging.info(
            f'[Futures Model Version] {self.env.FUTURES_MODEL_VERSION}')

        text = f"\n【開始監控】{self.env.ACCOUNT_NAME} 啟動完成({__version__})"
        text += f"\n【操盤模式】{self.env.MODE}"
        text += f"\n【股票策略】{self.env.STRATEGY_STOCK}"
        text += f"\n【期貨策略】{self.env.STRATEGY_FUTURES}"
        text += f"\n【AI版本】Stock-{self.env.STOCK_MODEL_VERSION}; Futures:{self.env.FUTURES_MODEL_VERSION}"
        text += f"\n【前日行情】Put/Call: {self.StrategySet.pc_ratio}"
        text += f"\n【數據用量】{usage}MB"
        notifier.post(text, msgType='Monitor')

        # 開始監控
        while True:
            self.loop_pause()
            now = datetime.now()

            if self.is_break_loop(now):
                break

            # 防止斷線用 TODO:待永豐更新後刪除
            if now.minute % 10 == 0 and now.second == 0:
                balance = self.balance(mode='debug')
                if balance == -1:
                    self._log_and_notify(
                        f"【連線異常】{self.env.ACCOUNT_NAME} 無法查詢餘額")

            # update K-bar data
            if MonitorFreq <= now.second:
                for freq in [2, 5, 15, 30, 60]:
                    if now.minute % freq == 0:
                        self.updateKBars(f'{freq}T')

            # TODO: merge stocks_to_monitor & futures_to_monitor
            for target in list(TradeData.Stocks.Monitor):
                order = self.monitor_stocks(target)
                if order.pos_target:
                    order_data = self._place_order(order, market='Stocks')
                    self._update_position(order, 'Stocks', order_data)

            for target in list(TradeData.Futures.Monitor):
                order = self.monitor_futures(target)
                if order.pos_target:
                    order_data = self._place_order(order, market='Futures')
                    self._update_position(order, 'Futures', order_data)

        logging.info('Non-trading time, stop monitoring')

        for scale in ['2T', '5T', '15T', '30T', '60T']:
            self.updateKBars(scale)

        if self.is_all_zero():
            self._log_and_notify(f"【停止監控】{self.env.ACCOUNT_NAME} 無可監控清單")

        time.sleep(3)
        self.unsubscribe_all(all_stocks+all_futures)

    def output_files(self):
        '''停止交易時，輸出庫存資料 & 交易明細'''
        if 'position' in TradeData.Stocks.Info.columns and not self.simulation:
            codeList = self.get_securityInfo('Stocks').code.to_list()
            self.update_watchlist(codeList)

        account = self.env.ACCOUNT_NAME
        self.save_watchlist(self.watchlist)
        self.output_statement(f'{PATH}/stock_pool/statement_{account}.csv')
        self.StrategySet.export_strategy_data()

        for freq, df in TradeData.KBars.Freq.items():
            if freq != '1D':
                filename = f'{PATH}/Kbars/k{freq[:-1]}min_{account}.csv'
                file_handler.Process.save_table(df, filename)

        if self.simulation:
            self.simulator.save_securityInfo(self.env, 'Stocks')
            self.simulator.save_securityInfo(self.env, 'Futures')
        time.sleep(1)
