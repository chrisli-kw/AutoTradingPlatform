import os
import ssl
import sys
import time
import logging
import shioaji as sj
from collections import namedtuple
from datetime import datetime, timedelta

from . import __version__, picker, exec
from .config import (
    API,
    TODAY_STR,
    MonitorFreq,
    TimeTransferFutures,
    StrategyList,
    NotifyConfig
)
from .utils import get_contract
from .utils.database import db
from .utils.database.tables import SecurityInfo
from .utils.time import time_tool
from .utils.crawler import crawler
from .utils.notify import Notification
from .utils.orders import OrderTool
from .utils.subscribe import Subscriber
from .utils.simulation import Simulator
from .utils.accounts import AccountHandler
from .utils.callback import CallbackHandler
from .utils.objects.data import TradeData
from .utils.positions import WatchListTool, TradeDataHandler
from .utils.strategy import StrategyTool
from .utils.bot import TelegramBot
from .utils import runtime


ssl._create_default_https_context = ssl._create_unverified_context


class StrategyExecutor(AccountHandler, Subscriber):
    def __init__(self, account_name: str):
        super().__init__(account_name)
        AccountHandler.__init__(self, account_name)
        Subscriber.__init__(self)
        StrategyList.init_config(account_name)

        self.notifier = Notification(config=NotifyConfig, account=account_name)
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
        runtime.write_session(
            account_name,
            pid=os.getpid(),
            command=" ".join(sys.argv),
            cwd=os.getcwd(),
            started_at=runtime.now_text(),
            launcher="runtime",
        )

    def _callback_msg_to_dict(self, msg):
        if isinstance(msg, dict):
            return {k: self._callback_msg_to_dict(v) for k, v in msg.items()}

        if hasattr(msg, 'items') and callable(msg.items):
            return {k: self._callback_msg_to_dict(v) for k, v in msg.items()}

        if hasattr(msg, 'dict') and callable(msg.dict):
            return self._callback_msg_to_dict(msg.dict())

        return msg

    def _subscribe_trade_callbacks(self):
        for account in [API.stock_account, API.futopt_account]:
            if account is None:
                continue

            try:
                is_subscribed = API.subscribe_trade(account)
                logging.info(
                    f'[OrderCallback] subscribe_trade|{account.account_id}|{is_subscribed}')
            except Exception:
                logging.exception(
                    f'[OrderCallback] subscribe_trade failed|{account.account_id}|')

    @staticmethod
    def _safe_notify(callback, stat, msg):
        try:
            callback(stat, msg)
        except Exception:
            logging.exception(
                f'[OrderCallback] notification failed|{stat}|')

    def _order_callback(self, stat, msg):
        '''處理委託/成交回報'''

        msg = self._callback_msg_to_dict(msg)

        if (
            TradeData.Account.Simulate or
            (
                stat == sj.OrderState.StockOrder and
                msg['order']['account']['account_id'] != API.stock_account.account_id
            ) or
            (
                stat == sj.OrderState.StockDeal and
                msg['account_id'] != API.stock_account.account_id
            ) or
            (
                stat == sj.OrderState.FuturesOrder and
                msg['order']['account']['account_id'] != API.futopt_account.account_id
            ) or
            (
                stat == sj.OrderState.FuturesDeal and
                msg['account_id'] != API.futopt_account.account_id
            )
        ):
            return

        if stat == sj.OrderState.StockOrder:
            self._safe_notify(self.notifier.post_tftOrder, stat, msg)
            self.Order.StockOrder(msg)

        elif stat == sj.OrderState.StockDeal:
            self._safe_notify(self.notifier.post_tftDeal, stat, msg)
            self.Order.StockDeal(msg)

        elif stat == sj.OrderState.FuturesOrder:
            self._safe_notify(self.notifier.post_fOrder, stat, msg)
            self.Order.FuturesOrder(msg)

        elif stat == sj.OrderState.FuturesDeal:
            self._safe_notify(self.notifier.post_fDeal, stat, msg)
            self.Order.FuturesDeal(msg)

    def init_account(self):
        # 登入
        self.login_(self.env)
        logging.info(
            f'[AccountInfo] Stock account ID: {API.stock_account.account_id}')

        if API.futopt_account:
            self._set_futures_code_list()
            logging.info(
                f'[AccountInfo] Futures account ID: {API.futopt_account.account_id}')

        self.activate_ca_()

        # set callbacks
        @API.on_tick_stk_v1()
        def stk_quote_callback_v1(tick):
            if tick.intraday_odd == 0 and tick.simtrade == 0:

                if tick.code not in TradeData.Quotes.NowTargets:
                    logging.debug(f'[Quotes]First|{tick.code}|')

                tick_data = self.update_quote_v1(tick)
                # self.to_redis({tick.code: tick_data})

        @API.on_tick_fop_v1()
        def fop_quote_callback_v1(tick):
            try:
                if tick.simtrade == 0:
                    symbol = TradeData.Futures.CodeList.get(
                        tick.code, tick.code)

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
        self._subscribe_trade_callbacks()

        # 訂閱五檔回報
        @API.on_bidask_stk_v1()
        def stk_quote_callback(bidask):
            TradeData.BidAsk[bidask.code] = bidask

        @API.on_bidask_fop_v1()
        def fop_quote_callback(bidask):
            symbol = TradeData.Futures.CodeList.get(bidask.code, bidask.code)
            TradeData.BidAsk[symbol] = bidask

    def _log_and_notify(self, msg: str):
        '''將訊息加入log並推播'''
        logging.info(msg)
        self.notifier.send.post(f'\n{msg}')

    def _get_filter_out(self):
        filter_out = []
        for conf in StrategyList.Config.values():
            filter_out.extend(getattr(conf, 'FILTER_OUT', []))
        return filter_out

    @staticmethod
    def _is_filter_out_target(code: str, filter_outs: list):
        if code in filter_outs:
            return True
        if 'TXO' in filter_outs and code.startswith('TXO'):
            return True
        if 'Call' in filter_outs and code.endswith('C'):
            return True
        if 'Put' in filter_outs and code.endswith('P'):
            return True
        return False

    def init_target_sets(self):
        '''初始化監控資訊'''

        # 取得遠端庫存
        info = self.get_securityInfo()

        # 讀取選股清單
        TradeData.Securities.Strategy = self.get_securityPool(
            target_set=info.code.tolist())

        # 剔除不堅控的股票
        filter_outs = self._get_filter_out()
        info = info[
            ~info.code.apply(
                lambda code: self._is_filter_out_target(code, filter_outs)
            )
        ]

        # 設定監控清單
        TradeData.Securities.Monitor.update(
            TradeDataHandler.build_monitor_dict(info))
        TradeDataHandler.unify_monitor_data(self.account_name)
        db_info = db.query(
            SecurityInfo,
            SecurityInfo.mode == TradeData.Account.Mode,
            SecurityInfo.account == self.account_name
        )
        if not db_info.empty:
            db_info = db_info[
                ~db_info.code.apply(
                    lambda code: self._is_filter_out_target(
                        code, filter_outs)
                )
            ]
            TradeData.Securities.Monitor = TradeDataHandler.build_monitor_dict(
                db_info)

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
        if TradeData.Account.Simulate and order_data:
            self.simulator.update_monitor(order, order_data)

    @staticmethod
    def _action_orders(actionInfo):
        orders = getattr(actionInfo, 'orders', None)
        if orders is None:
            return []
        if isinstance(orders, dict):
            return [orders]
        return orders

    @staticmethod
    def _strip_order_type(order_spec: dict):
        spec = order_spec.copy()
        order_type = spec.pop('type', spec.pop('order_type_', 'option'))
        if 'label' in spec and 'order_label' not in spec:
            spec['order_label'] = spec.pop('label')
        return order_type, spec

    def _register_option_order_strategy(self, order: namedtuple, strategy: str):
        TradeData.Securities.Strategy[order.target] = strategy

        try:
            if order.combo_legs:
                for leg in order.combo_legs:
                    target = leg.get('target')
                    contract = leg.get('contract')
                    if target:
                        TradeData.Securities.Strategy[target] = strategy
                    if target and contract:
                        TradeData.Contracts[target] = contract
            else:
                if order.target not in TradeData.Contracts:
                    TradeData.Contracts[order.target] = get_contract(
                        order.target)
        except Exception:
            logging.exception(
                f'[OptionOrder] register strategy failed|{order.target}|')

    def _build_orders_from_specs(
            self,
            order_specs: list,
            strategy: str,
            actionType: str,
            octype: str,
            default_reason: str
    ):
        orders = []
        for order_spec in order_specs:
            order_type, spec = self._strip_order_type(order_spec)
            spec.setdefault('action_type', actionType)
            spec.setdefault('octype', octype)
            spec.setdefault('reason', default_reason)

            if order_type in ['option', 'single_option']:
                order = self.Order.option_order_info(**spec)
                TradeData.Contracts[order.target] = self.Order.Options.contract(
                    expiration=spec['expiration'],
                    strike=spec['strike'],
                    option_type=spec['option_type'],
                    underlying=spec.get('underlying', 'TX')
                )
            elif order_type in ['option_combo', 'combo']:
                order = self.Order.option_combo_order_info(**spec)
            else:
                raise ValueError(f'Unsupported order spec type: {order_type}')

            self._register_option_order_strategy(order, strategy)
            orders.append(order)

        return orders

    @staticmethod
    def _as_order_list(orders):
        if isinstance(orders, list):
            return orders
        return [orders]

    @staticmethod
    def _should_place_order(order: namedtuple):
        return bool(
            getattr(order, 'action', '') or
            getattr(order, 'combo_legs', None)
        )

    def monitor_targets(self, target: str):
        strategy = TradeData.Securities.Strategy[target]
        if target in TradeData.Quotes.NowTargets and strategy:
            inputs = TradeDataHandler.getQuotesNow(target).copy()
            monitor_data = TradeData.Securities.Monitor.get(target)
            if isinstance(monitor_data, list):
                position_records = monitor_data
            else:
                data = db.query(
                    SecurityInfo, self.WatchList.match_target(target))
                position_records = data.to_dict(
                    'records') if not data.empty else []

            contract = TradeData.Contracts.get(target)
            is_stock = isinstance(contract, sj.Stock)
            position_data = position_records[0] if position_records else {}
            if isinstance(monitor_data, list):
                broker_quantity = max([
                    abs(int(record.get('quantity', 0) or 0))
                    for record in position_records
                ] or [0])
                strategy_quantity = broker_quantity
            else:
                broker_quantity = int(position_data.get('quantity', 0) or 0)
                strategy_quantity = TradeDataHandler.strategy_quantity(
                    broker_quantity,
                    mode=getattr(
                        TradeDataHandler.getStrategyConfig(target),
                        'mode',
                        'long'
                    ),
                    action=position_data.get('action', ''),
                    market='Stocks' if is_stock else 'Futures'
                )

            # new position
            if strategy_quantity <= 0:
                actionType = 'Open'
                octype = 'New'

                order_cond, quantity = self.get_quantity(target)
                enoughOpen = self.check_enough(target, quantity)

            # in-stock position
            else:
                actionType = 'Close'
                octype = 'Cover'

                order_cond = position_data.get('order_cond', 'Cash')
                quantity = strategy_quantity

                duration = (
                    datetime.now() - position_data['timestamp']
                ).total_seconds()
                if is_stock and duration < 3600*4.5:
                    order_cond = self.day_trade_cond[order_cond]

                enoughOpen = False

            data = position_data.copy()
            if data:
                data['broker_quantity'] = broker_quantity
                data['quantity'] = strategy_quantity
                if isinstance(monitor_data, list):
                    data['combo_legs'] = position_records

            if target in TradeData.Futures.Transferred:
                quantity = TradeData.Futures.Transferred[target]['quantity']
                infos = dict(
                    action_type=actionType,
                    action=TradeData.Futures.Transferred[target]['action'],
                    target=target,
                    quantity=quantity,
                    octype=octype,
                    reason=f'{target} 轉倉-New'
                )
                msg = f'{target} 轉倉-New: {infos}'
                self._log_and_notify(msg)
                TradeData.Futures.Transferred.pop(target)
                try:
                    conf = TradeDataHandler.getStrategyConfig(target)
                    if hasattr(conf, 'transfer'):
                        conf.transfer(target, quantity)
                except:
                    logging.exception(f'Update position table failed:')

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
                order_specs = self._action_orders(actionInfo)
                if actionInfo.action or order_specs:
                    if (
                        actionType == 'Close' and
                        actionInfo.action and
                        not actionInfo.isRaiseQty
                    ):
                        requested_quantity = int(actionInfo.quantity or 0)
                        available_quantity = max(int(quantity or 0), 0)
                        close_quantity = min(
                            requested_quantity, available_quantity)
                        if close_quantity != requested_quantity:
                            logging.warning(
                                f'[Order Quantity]Cap close quantity|{target}|'
                                f'{requested_quantity}->{close_quantity}|')
                            actionInfo = actionInfo._replace(
                                quantity=close_quantity)
                        if close_quantity <= 0:
                            return self.Order.OrderInfo(target=target)

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
                        TradeData.Contracts[new_contract] = get_contract(
                            new_contract)
                        TradeData.Securities.Strategy[new_contract] = strategy
                        TradeData.Securities.Monitor[new_contract] = None
                        TradeData.Securities.Monitor.pop(target, None)
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

                    if order_specs:
                        reason = actionInfo.reason or f'{target}|{strategy}|Option order'
                        self._log_and_notify(reason)
                        return self._build_orders_from_specs(
                            order_specs,
                            strategy,
                            actionType,
                            octype,
                            reason
                        )

                    if actionInfo.isRaiseQty:
                        actionType = 'Open'
                        octype = 'New'
                        order_cond = self.check_order_cond(target)

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

    def get_securityPool(self, target_set: list = None):
        '''Get the target securities pool with the format: {code: strategy}'''

        due_year_month = time_tool.GetDueMonth()
        day_filter_out = crawler.FromHTML.get_CashSettle()
        df = picker.get_selection_files()
        filter_outs = self._get_filter_out()

        pools = {}
        for strategy, conf in StrategyList.Config.items():
            targets = getattr(conf, 'Targets', [])

            for code in targets:
                if conf.market == 'Stocks':
                    pools.update({code: strategy})
                elif 'TXO' in code[:3]:
                    for target in target_set or []:
                        if (
                            target.startswith('TXO') and
                            'TXO' not in getattr(conf, 'FILTER_OUT', []) and
                            target not in getattr(conf, 'FILTER_OUT', [])
                        ):
                            pools.update({target: strategy})
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

        # 庫存的處理 (遠端有庫存，地端無庫存)
        pools.update({
            target: None
            for target in target_set or []
            if (
                target not in pools and
                not self._is_filter_out_target(target, filter_outs)
            )
        })

        return pools

    def get_quantity(self, target: str):
        '''Calculate the quantity for opening a position'''

        strategy = TradeData.Securities.Strategy.get(target)
        contract = TradeData.Contracts.get(target)
        quantityFunc = self.StrategySet.mapQuantities(strategy)

        inputs = TradeDataHandler.getQuotesNow(target)
        quantity, quantity_limit = quantityFunc(inputs=inputs, target=target)

        order_cond = self.check_order_cond(target)
        leverage = self.Order.check_leverage(target, order_cond)

        quantity = int(min(quantity, quantity_limit)/(1 - leverage))
        quantity = min(quantity, 499)

        if not isinstance(contract, sj.Stock):
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
        if not isinstance(contract, sj.Stock):
            return 'Cash'

        conf = TradeDataHandler.getStrategyConfig(target)
        if conf is None:
            return 'Cash'

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
        conf = TradeDataHandler.getStrategyConfig(target)
        if conf is None:
            return False

        mode = conf.mode
        if isinstance(contract, sj.Stock):
            quota = TradeDataHandler.getStocksQuota(mode)
            df = self.Order.filterOrderTable('Stocks')
            df = df[df.code.apply(len) == 4]

            price = TradeDataHandler.getQuotesNow(target)['price']
            target_amount = self.Order.get_stock_amount(
                target, price, quantity, mode)
        else:
            quota = TradeDataHandler.getFuturesQuota(self.account_name)
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
        if isinstance(contract, sj.Stock):
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
            db.query(
                SecurityInfo,
                SecurityInfo.account == self.account_name
            ).shape[0]
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

    def _update_runtime_status(self, status: str, message: str = ""):
        try:
            runtime.write_status(
                self.account_name,
                status,
                message,
                mode=TradeData.Account.Mode,
                strategies=list(StrategyList.Config.keys()),
                monitor_targets=list(TradeData.Securities.Monitor.keys()),
                pid=os.getpid(),
            )
        except Exception:
            logging.exception('Update runtime status failed:')

    def _apply_runtime_command(self, stop_flag, pause_flag):
        command = runtime.read_command(self.account_name)
        if not command:
            return

        command_id = command.get("id")
        if runtime.command_is_expired(command):
            runtime.clear_command(self.account_name, command_id)
            return

        action = command.get("command")
        payload = command.get("payload", {})
        message = ""

        try:
            if action == "pause":
                pause_flag.set()
                message = "GUI 暫停監控"
                self.notifier.send.post(
                    f"🛑 [{self.account_name}] 已暫停監控"
                )
            elif action == "resume":
                pause_flag.clear()
                message = "GUI 恢復監控"
                self.notifier.send.post(
                    f"✅ [{self.account_name}] 已恢復監控"
                )
            elif action == "stop":
                stop_flag.set()
                message = "GUI 要求停止監控"
                self.notifier.send.post(
                    f"❌ [{self.account_name}] 程式即將停止"
                )
            elif action == "update_max_qty":
                strategy = payload.get("strategy")
                target = payload.get("target")
                max_qty = int(payload.get("max_qty", 0))

                if max_qty < 0:
                    raise ValueError("max_qty must be >= 0")

                conf = StrategyList.Config.get(strategy)
                if conf is None:
                    raise ValueError(f"Unknown strategy: {strategy}")
                if not hasattr(conf, "max_qty"):
                    raise ValueError(f"Strategy has no max_qty: {strategy}")

                old_qty = conf.max_qty.get(target)
                conf.max_qty[target] = max_qty
                if hasattr(conf, "update_max_qty"):
                    conf.update_max_qty(
                        self.account_name, strategy, target, max_qty)

                message = (
                    f"[Strategy max_qty]Update|{self.account_name}|"
                    f"{strategy}|{target}|{old_qty}->{max_qty}"
                )
                self.notifier.send.post(
                    f"【{self.account_name} 更新部位】最大數量\n"
                    f"{target}: {max_qty}"
                )
            else:
                message = f"Unknown GUI command: {action}"

            logging.info(message)
            self._update_runtime_status(
                "stopping" if stop_flag.is_set()
                else "paused" if pause_flag.is_set()
                else "running",
                message,
            )
        except Exception as exc:
            logging.exception("[GUI Command] failed:")
            self._update_runtime_status(
                "error",
                f"GUI command failed: {exc}",
            )
        finally:
            runtime.clear_command(self.account_name, command_id)

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
            f'[Security position] {db.query(SecurityInfo, SecurityInfo.account == self.account_name).shape[0]}')
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
        logging.info(f"[Data Info] {TradeData.KBars.Freq['1T'].Time.max()}")

        text = f"\n【開始監控】{self.env.ACCOUNT_NAME} 啟動完成({__version__})"
        text += f"\n【操盤模式】{TradeData.Account.Mode}"
        text += f"\n【策略清單】{list(StrategyList.Config.keys())}"
        text += f"\n【數據用量】{usage}MB"
        for target, info in TradeData.Securities.Monitor.items():
            if isinstance(info, dict):
                text += f"\n【庫存部位】{target}: {info.get('action', '')} - {info.get('quantity', 0)}"
        self.notifier.send.post(text)

        def periodic_check():
            # Check if the connection is still alive
            # TODO: delete in the future
            if now.minute % 10 == 0 and now.second == 30:
                balance = self.balance(mode='debug')
                if balance == -1:
                    self._log_and_notify(
                        f"【連線異常】{self.env.ACCOUNT_NAME} 無法查詢餘額")

            if now.minute == 0 and now.second == 0:
                self._log_and_notify('Monitor status: running')

        # 啟動 Telegram 控制 bot
        bot = TelegramBot(self.account_name)

        # 開始監控
        stop_flag, pause_flag = bot.get_flags(self.account_name)
        self._update_runtime_status("running", "監控啟動完成")
        while not stop_flag.is_set():
            self.loop_pause()
            now = datetime.now()
            self._apply_runtime_command(stop_flag, pause_flag)

            if self.is_break_loop(now):
                break

            exec.submit(periodic_check)

            # update K-bar data
            if MonitorFreq <= now.second:
                for freq in [2, 5, 15, 30, 60]:
                    if now.minute % freq == 0:
                        self.updateKBars(f'{freq}T')

            if pause_flag.is_set():
                self._update_runtime_status("paused", "監控暫停中")
                continue

            self._update_runtime_status("running", "監控執行中")

            for target in list(TradeData.Securities.Monitor):
                orders = self._as_order_list(self.monitor_targets(target))
                for order in orders:
                    if not self._should_place_order(order):
                        continue

                    order_data = self.Order.place_order(order)
                    self.update_position_(order, order_data)

        if stop_flag.is_set():
            self._log_and_notify(
                '[Stop Monitoring] Received stop command from Telegram bot')
        else:
            self._log_and_notify(
                '[Stop Monitoring] Stopped due to non-trading hours or no positions')

        for scale in ['2T', '5T', '15T', '30T', '60T']:
            self.updateKBars(scale)

        if self.is_all_zero():
            self._log_and_notify(f"【停止監控】{self.env.ACCOUNT_NAME} 無可監控清單")

        time.sleep(3)
        self.unsubscribe_all(all_targets)
        self._update_runtime_status("stopped", "監控已停止")

    def output_files(self):
        '''停止交易時，輸出庫存資料 & 交易明細'''
        self.StrategySet.export_strategy_data_()
