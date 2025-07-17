import ssl
import time
import logging
from shioaji import constant, contracts
from collections import namedtuple
from datetime import datetime, timedelta

from . import __version__, picker, exec
from .config import (
    API,
    TODAY_STR,
    MonitorFreq,
    TimeTransferFutures,
    StrategyList
)
from .utils import get_contract
from .utils.database import db
from .utils.database.tables import SecurityInfo, PositionTable
from .utils.time import time_tool
from .utils.crawler import crawler
from .utils.notify import notifier
from .utils.orders import OrderTool
from .utils.subscribe import Subscriber
from .utils.simulation import Simulator
from .utils.accounts import AccountHandler
from .utils.callback import CallbackHandler
from .utils.objects.data import TradeData
from .utils.positions import WatchListTool, TradeDataHandler
from .utils.strategy import StrategyTool


ssl._create_default_https_context = ssl._create_unverified_context


class StrategyExecutor(AccountHandler, Subscriber):
    def __init__(self, account_name: str):
        StrategyList.init_config(account_name)
        super().__init__(account_name)
        AccountHandler.__init__(self, account_name)
        Subscriber.__init__(self)

        self.Order = OrderTool(account_name)
        self.simulator = Simulator(account_name)
        self.WatchList = WatchListTool(account_name)
        self.StrategySet = StrategyTool(self.env)
        self.day_trade_cond = {
            'MarginTrading': 'ShortSelling',
            'ShortSelling': 'MarginTrading',
            'Cash': 'Cash'
        }

        self.punish_list = []

    def _order_callback(self, stat, msg):
        '''處理委託/成交回報'''

        if (
            TradeData.Account.Simulate or
            (
                stat == constant.OrderState.StockOrder and
                msg['order']['account']['account_id'] != API.stock_account.account_id
            ) or
            (
                stat == constant.OrderState.StockDeal and
                msg['account_id'] != API.stock_account.account_id
            ) or
            (
                stat == constant.OrderState.FuturesOrder and
                msg['order']['account']['account_id'] == API.futopt_account.account_id
            )
        ):
            return

        if stat == constant.OrderState.StockOrder:
            notifier.post_tftOrder(stat, msg)
            self.Order.StockOrder(msg)

        elif stat == constant.OrderState.StockDeal:
            notifier.post_tftDeal(stat, msg)
            self.Order.StockDeal(msg)

        elif stat == constant.OrderState.FuturesOrder:
            notifier.post_fOrder(stat, msg)
            self.Order.FuturesOrder(msg)

        elif stat == constant.OrderState.FuturesDeal:
            notifier.post_fDeal(stat, msg)
            CallbackHandler.FuturesDeal(msg)

    def init_account(self):
        # 登入
        self.login_(self.env)
        logging.info(
            f'[AccountInfo] Stock account ID: {API.stock_account.account_id}')

        if self.HAS_FUTOPT_ACCOUNT:
            self._set_futures_code_list()
            logging.info(
                f'[AccountInfo] Futures account ID: {API.futopt_account.account_id}')

        self.activate_ca_()

        # set callbacks
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
        notifier.send.post(f'\n{msg}')

    def _get_filter_out(self):
        filter_out = []
        for conf in StrategyList.Config.values():
            filter_out.extend(getattr(conf, 'FILTER_OUT', []))
        return filter_out

    def init_target_sets(self):
        '''初始化監控資訊'''

        # 讀取選股清單
        TradeData.Securities.Strategy = self.get_securityPool()

        # 取得遠端庫存
        info = self.get_securityInfo()
        info.index = info.code

        # 剔除不堅控的股票
        info = info[~info.code.isin(self._get_filter_out())]

        # 庫存的處理 (遠端有庫存，地端無庫存)
        tb = info[~info.code.isin(TradeData.Securities.Strategy.keys())].copy()
        tb['strategy'] = None
        TradeData.Securities.Strategy.update(tb.strategy.to_dict())

        # 新增歷史K棒資料
        TradeData.Securities.Monitor.update(info.to_dict('index'))

        for code, strategy in TradeData.Securities.Strategy.items():
            if code not in TradeData.Securities.Monitor:
                TradeData.Securities.Monitor.update({code: None})

            conf = TradeDataHandler.getStrategyConfig(code)

            # 若遠端無庫存，地端有庫存，刪除地端資料
            if (
                TradeData.Securities.Monitor.get(code) is None and
                conf.positions.entries
            ):
                db.delete(
                    PositionTable,
                    PositionTable.mode == TradeData.Account.Mode,
                    PositionTable.account == self.account_name,
                    PositionTable.name == code,
                    PositionTable.strategy == strategy
                )
                StrategyList.Config.get(strategy).positions.entries = []

        # 新增歷史K棒資料
        all_targets = list(TradeData.Securities.Monitor)
        all_targets = self.StrategySet.append_monitor_list_(all_targets)
        self.history_kbars(['TSE001', 'OTC101'] + all_targets)
        TradeData.Contracts.update({
            code: get_contract(code) for code in all_targets})

        # 交易風險控制
        buy_condition = (info.action == 'Buy') & (info.market == 'Stocks')
        TradeData.Stocks.N_Long = info[buy_condition].shape[0]
        TradeData.Stocks.N_Short = info[~buy_condition].shape[0]

        self.StrategySet.set_position_limit()
        self.StrategySet.get_ex_dividends_list()
        # self.punish_list = crawler.FromHTML.PunishList() # TODO ValueError: No tables found
        self._set_leverage(all_targets)
        self._set_trade_risks()
        self._set_margin_limit()
        self.margin_table = self.Order.get_margin_table()
        logging.debug(f'Targets to monitor: {TradeData.Securities.Monitor}')
        return all_targets

    def update_position_(self, order: namedtuple, order_data: dict):
        if TradeData.Account.Simulate:
            self.simulator.update_monitor(order, order_data)

    def monitor_targets(self, target: str):
        if target in TradeData.Quotes.NowTargets:
            inputs = TradeDataHandler.getQuotesNow(target).copy()
            data = db.query(SecurityInfo, self.WatchList.match_target(target))
            strategy = TradeData.Securities.Strategy[target]
            raise_pos = self.StrategySet.isRaiseQty(target)

            contract = TradeData.Contracts.get(target)
            is_stock = isinstance(contract, contracts.Stock)

            # new position
            if data.empty or (not data.empty and raise_pos):
                actionType = 'Open'
                octype = 'New'

                data = {}
                order_cond, quantity = self.get_quantity(
                    target, raise_pos=raise_pos)

                enoughOpen = self.check_enough(target, quantity)

            # in-stock position
            else:
                actionType = 'Close'
                octype = 'Cover'

                data = data.to_dict('records')[0]
                order_cond = data.get('order_cond', 'Cash')
                quantity = data.get('quantity', 0)

                duration = (datetime.now() - data['timestamp']).total_seconds()
                if is_stock and duration < 3600*4.5:
                    order_cond = self.day_trade_cond[order_cond]

                enoughOpen = False

            if target in TradeData.Futures.Transferred:
                msg = f'{target} 轉倉-New'
                infos = dict(
                    action_type=actionType,
                    action=TradeData.Futures.Transferred[target]['action'],
                    target=target,
                    quantity=TradeData.Futures.Transferred[target]['quantity'],
                    octype=octype,
                    reason=msg
                )
                self._log_and_notify(msg)
                TradeData.Futures.Transferred.pop(target)

                return self.Order.OrderInfo(**infos)

            isTransfer = (
                (not is_stock) and
                (actionType == 'Close') and
                (TODAY_STR.replace('-', '/') == contract.delivery_date) and
                (datetime.now() > TimeTransferFutures)
            )
            c1 = octype == 'New' and enoughOpen and self.is_trading_time_(
                inputs['datetime'])
            c2 = octype == 'Cover'
            if c1 or c2:
                if isTransfer:
                    func = self.StrategySet.transfer_position
                else:
                    func = self.StrategySet.mapFunction(actionType, strategy)

                if data:
                    inputs.update(data)

                actionInfo = func(inputs=inputs)
                if actionInfo.action:
                    if isTransfer:
                        new_contract = f'{target[:3]}{time_tool.GetDueMonth()}'
                        self.Order.transfer_margin(target, new_contract)
                        self.history_kbars([new_contract])
                        self.subscribe_all([new_contract])
                        TradeData.Futures.Transferred.update({
                            new_contract: {
                                'quantity': quantity,
                                'action': data['action']
                            }
                        })
                        TradeData.Securities.Strategy[new_contract] = strategy
                        TradeData.Securities.Monitor[new_contract] = None
                        TradeData.Securities.Monitor.pop(target, None)
                        # TradeData.Securities.Strategy.pop(target, None)
                        data = {
                            'mode': TradeData.Account.Mode,
                            'account': self.account_name,
                            'strategy': strategy,
                            'name': target,
                            'timestamp': datetime.now(),
                            'price': TradeDataHandler.getQuotesNow(target).get('price', 0),
                            'quantity': actionInfo.quantity,
                            'reason': actionInfo.reason
                        }
                        conf = TradeDataHandler.getStrategyConfig(new_contract)
                        conf.positions.close(data)

                    infos = dict(
                        action_type=actionType,
                        target=target,
                        action=actionInfo.action,
                        quantity=actionInfo.quantity,
                        order_cond=order_cond,
                        octype=octype,
                        reason=actionInfo.reason,
                    )
                    self._log_and_notify(actionInfo.reason)
                    return self.Order.OrderInfo(**infos)

            elif quantity <= 0 and actionType == 'Close':
                logging.warning(f'[Monitor List]Interfere|{target}|Close|')
                self.WatchList.check_remove_monitor(target)

        return self.Order.OrderInfo(target=target)

    def get_securityInfo(self):
        '''取得證券庫存清單'''

        if TradeData.Account.Simulate:
            return self.simulator.securityInfo(self.env.ACCOUNT_NAME)
        return self.securityInfo()

    def get_securityPool(self):
        '''Get the target securities pool with the format: {code: strategy}'''

        due_year_month = time_tool.GetDueMonth()
        day_filter_out = crawler.FromHTML.get_CashSettle()
        df = picker.get_selection_files()

        pools = {}
        for strategy, conf in StrategyList.Config.items():
            targets = getattr(conf, 'Targets', [])

            for code in targets:
                if conf.market == 'Stocks':
                    pools.update({code: strategy})
                else:
                    pools.update({f'{code}{due_year_month}': strategy})

            # 排除不交易的股票
            # --- 全額交割股不買
            df = df[~df.code.isin(day_filter_out.股票代碼.values)]

            # --- 排除高價股
            df = df[df.Close <= getattr(conf, 'PRICE_THRESHOLD', 0)]

            df = df[~df.code.isin(getattr(conf, 'FILTER_OUT', []))]
            df = df.sort_values('Close')

            if strategy in df.Strategy.values:
                code = df[df.Strategy == strategy].code
                pools.update({stock: strategy for stock in code})
                df = df[~df.code.isin(code)]

        return pools

    def get_quantity(self, target: str, raise_pos=False):
        '''Calculate the quantity for opening a position'''

        strategy = TradeData.Securities.Strategy.get(target)
        contract = TradeData.Contracts.get(target)
        quantityFunc = self.StrategySet.mapQuantities(strategy)

        inputs = TradeDataHandler.getQuotesNow(target)
        quantity, quantity_limit = quantityFunc(
            inputs=inputs, raise_pos=raise_pos, target=target)

        order_cond = self.check_order_cond(target)
        leverage = self.Order.check_leverage(target, order_cond)

        quantity = int(min(quantity, quantity_limit)/(1 - leverage))
        quantity = min(quantity, 499)

        if not isinstance(contract, contracts.Stock):
            # 單位: 口
            return order_cond, quantity

        # 單位: 股
        if order_cond == 'MarginTrading':
            quantity = min(contract.margin_trading_balance, quantity)
        elif order_cond == 'ShortSelling':
            quantity = min(contract.short_selling_balance, quantity)
        return order_cond, 1000*quantity

    def check_order_cond(self, target: str):
        '''檢查個股可否融資融券'''

        contract = TradeData.Contracts.get(target)
        if not isinstance(contract, contracts.Stock):
            return 'Cash'

        conf = TradeDataHandler.getStrategyConfig(target)
        mode = conf.mode
        margin_trading = getattr(conf, 'Margin_Trading', False)
        short_selling = getattr(conf, 'SHORT_SELLING', False)

        if (
            mode == 'long' and
            margin_trading and
            TradeData.Stocks.Leverage.Long.get(target, 0) != 0 and
            contract.margin_trading_balance > 0
        ):
            return 'MarginTrading'
        elif (
            mode == 'short' and
            short_selling and
            TradeData.Stocks.Leverage.Short.get(target, 1) != 1 and
            contract.short_selling_balance > 0
        ):
            return 'ShortSelling'
        else:
            return 'Cash'

    def check_enough(self, target: str, quantity: int):
        '''計算可買進的股票數量 & 金額'''

        if target not in TradeData.Quotes.NowTargets:
            return False

        contract = TradeData.Contracts.get(target)
        mode = TradeDataHandler.getStrategyConfig(target).mode

        if isinstance(contract, contracts.Stock):
            quota = TradeDataHandler.getStocksQuota(mode)
            df = self.Order.filterOrderTable('Stocks')
            df = df[df.code.apply(len) == 4]

            price = TradeDataHandler.getQuotesNow(target)['price']
            target_amount = self.Order.get_stock_amount(
                target, price, quantity, mode)
        else:
            quota = TradeDataHandler.getFuturesQuota()
            df = self.Order.filterOrderTable('Futures')
            target_amount = self.Order.get_open_margin(target, quantity)

        if quota <= 0:
            return False

        # 更新已委託金額
        amount1 = df.amount.sum() + target_amount
        amount2 = df[df.price > 0].amount.sum() + target_amount
        amount3 = df[df.price < 0].amount.abs().sum() + target_amount

        # under day limit condition
        # 1. 不可超過可交割金額
        # 2. 不可大於帳戶可委託金額上限
        # 3. 不可超過股票數上限
        if isinstance(contract, contracts.Stock):
            check_long = (
                (amount1 <= TradeData.Account.DesposalMoney) &
                (amount2 <= self.env.MARGING_TRADING_AMOUNT)
            )
            if mode == 'long':
                return check_long
            return (check_long & (amount3 <= self.env.SHORT_SELLING_AMOUNT))

        else:
            return (
                (amount1 <= TradeData.Account.DesposalMargin) &
                (amount2 <= self.env.MARGIN_AMOUNT)
            )

    def is_trading_time_(self, now: datetime):
        '''檢查是否為交易時段'''

        if TradeData.Futures.CanTrade:
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
            TradeData.Stocks.LimitLong,
            TradeData.Stocks.LimitShort,
            TradeData.Futures.Limit,
            db.query(SecurityInfo).shape[0]
        ])

    def is_break_loop(self, now: datetime):
        return not self.is_trading_time_(now) or self.is_all_zero()

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

        all_targets = self.init_target_sets()
        usage = round(API.usage().bytes/2**20, 2)
        self.subscribe_all(all_targets)

        logging.info(f'Start to monitor, basic settings:')
        logging.info(f'[AccountInfo] Current data usage: {usage}')
        logging.info(f'[AccountInfo] Mode: {TradeData.Account.Mode}')
        logging.info(f'[Security Strategy] {TradeData.Securities.Strategy}')
        logging.info(
            f'[Security position] {db.query(SecurityInfo).shape[0]}')
        logging.info(f'[Stock Position] Long: {TradeData.Stocks.N_Long}')
        logging.info(
            f'[Stock Position] Limit Long: {self.env.MARGING_TRADING_AMOUNT}')
        logging.info(f'[Stock Position] Short: {TradeData.Stocks.N_Short}')
        logging.info(
            f'[Stock Position] Limit Short: {self.env.SHORT_SELLING_AMOUNT}')
        logging.info(
            f'[Stock Portfolio Limit] Long: {TradeData.Stocks.LimitLong}')
        logging.info(
            f'[Stock Portfolio Limit] Short: {TradeData.Stocks.LimitShort}')

        logging.info(f'[Futures portfolio Limit] {TradeData.Futures.Limit}')

        text = f"\n【開始監控】{self.env.ACCOUNT_NAME} 啟動完成({__version__})"
        text += f"\n【操盤模式】{TradeData.Account.Mode}"
        text += f"\n【策略清單】{list(StrategyList.Config.keys())}"
        text += f"\n【數據用量】{usage}MB"
        for target, info in TradeData.Securities.Monitor.items():
            if isinstance(info, dict):
                text += f"\n【庫存部位】{target}: {info.get('action', '')} - {info.get('quantity', 0)}"
        notifier.send.post(text)

        def periodic_check():
            # Check if the connection is still alive
            # TODO: delete in the future
            if now.minute % 10 == 0 and now.second == 30:
                balance = self.balance(mode='debug')
                if balance == -1:
                    self._log_and_notify(
                        f"【連線異常】{self.env.ACCOUNT_NAME} 無法查詢餘額")

        # 開始監控
        while True:
            self.loop_pause()
            now = datetime.now()

            if self.is_break_loop(now):
                break

            exec.submit(periodic_check)

            # update K-bar data
            if MonitorFreq <= now.second:
                for freq in [2, 5, 15, 30, 60]:
                    if now.minute % freq == 0:
                        self.updateKBars(f'{freq}T')

            for target in list(TradeData.Securities.Monitor):
                order = self.monitor_targets(target)
                if order.action:
                    order_data = self.Order.place_order(order)
                    self.update_position_(order, order_data)

        logging.info('Non-trading time, stop monitoring')

        for scale in ['2T', '5T', '15T', '30T', '60T']:
            self.updateKBars(scale)

        if self.is_all_zero():
            self._log_and_notify(f"【停止監控】{self.env.ACCOUNT_NAME} 無可監控清單")

        time.sleep(3)
        self.unsubscribe_all(all_targets)

    def output_files(self):
        '''停止交易時，輸出庫存資料 & 交易明細'''
        self.StrategySet.export_strategy_data_()
