import time
import logging
import numpy as np
import pandas as pd
import shioaji as sj
from datetime import datetime
from shioaji.account import StockAccount

from ..config import API, PATH, TODAY, TODAY_STR
from . import concat_df
from .time import time_tool
from .crawler import crawler
from .file import file_handler
from .objects import Margin
from .objects.env import UserEnv
from .objects.data import TradeData


class AccountInfo:
    def __init__(self):
        self.filename = f'{TODAY.year}_股票帳務資訊.xlsx'
        self.DEFAULT_TABLE = pd.DataFrame(
            columns=[
                '交易日期',
                'T日交割金額',
                'T日帳戶餘額',
                'T+1日交割金額',
                'T+1日帳戶餘額',
                'T+2日交割金額',
                'T+2日帳戶餘額',
                '庫存總成本',
                '庫存現值',
                '融資金額',
                '融券金額',
                '未實現損益',
                '已實現損益',
                '當日新增庫存未實現損益',
                '總部位現值(含融資金額)',
                '總部位現值(不含融資金額)',
                '結算現值'
            ])
        self.HAS_FUTOPT_ACCOUNT = False

    def login_(self, env):
        self.account_name = env.ACCOUNT_NAME

        n = 0
        while n < 5:
            try:
                API.login(
                    api_key=env.api_key(),
                    secret_key=env.secret_key(),
                    contracts_timeout=10000
                )
                break
            except TimeoutError as e:
                logging.warning(f'{e}')
                n += 1
                time.sleep(5)

        if not self.account_name[-1].isdigit():
            nth_account = 1
        else:
            nth_account = int(self.account_name[-1])

        if nth_account > 1:
            accounts = API.list_accounts()
            accounts = [a for a in accounts if isinstance(a, StockAccount)]
            if len(accounts) > 1:
                API.set_default_account(accounts[nth_account-1])
            else:
                logging.warning('The number of accounts of this ID is 1')

        if API.futopt_account:
            self.HAS_FUTOPT_ACCOUNT = True

        time.sleep(0.05)
        logging.info(f'【{self.account_name}】log-in successful!')

    def _list_settlements(self):
        '''取得交割資訊'''
        n = 0
        while True:
            try:
                return API.settlements(API.stock_account)
            except:
                logging.warning('Cannot get settlement info, retrying...')
                n += 1

                if n >= 60:
                    return 0

                time.sleep(1)

    def _obj_2_df(self, objects: list):
        '''把自API查詢得到的物件轉為DataFrame'''
        try:
            return pd.DataFrame([o.__dict__ for o in objects])
        except:
            return pd.DataFrame([{o[0]: o[1] for o in objects}])

    def create_info_table(self):
        if file_handler.Operate.is_in_dir(self.filename, f'{PATH}/daily_info/'):
            return pd.ExcelFile(f'{PATH}/daily_info/{self.filename}')
        else:
            return self.DEFAULT_TABLE

    def balance(self, mode='info'):
        '''查帳戶餘額'''
        n = 0
        while n < 5:
            try:
                df = self._obj_2_df(API.account_balance())
                df.date = pd.to_datetime(df.date).dt.date.astype(str)
                balance = df[df.date == df.date.max()].acc_balance.values[0]

                if mode == 'info':
                    logging.debug(f'Account balance = {balance}')
            except:
                logging.exception('Catch an exception (balance):')
                balance = None

            if balance != None:
                return balance

            time.sleep(1)
            n += 1

        logging.debug('【Query account balance failed】')
        return -1

    def get_stock_name(self, stockid: str):
        '''以股票代號查詢公司名稱'''
        stockname = API.Contracts.Stocks[stockid]
        if (stockname is not None):
            return stockname.name
        return stockname

    def securityInfo(self, market='Stocks'):
        '''查庫存明細'''

        if market == 'Stocks':
            while True:
                try:
                    stocks = API.list_positions(
                        API.stock_account,
                        unit=sj.constant.Unit.Share
                    )
                    stocks = self._obj_2_df(stocks)
                    break
                except:
                    logging.warning('Cannot get the security info, retrying')
                    time.sleep(1)
            stocks = stocks.rename(columns={
                'cond': 'order_cond',
                'direction': 'action',
                'price': 'cost_price',
            })
            if stocks.shape[0]:
                names = stocks.code.apply(self.get_stock_name)
                stocks.pnl = stocks.pnl.astype(int)  # 未實現損益
                stocks.order_cond = stocks.order_cond.astype(str)  # 交易別
                stocks.insert(1, 'name', names)
                stocks[['account', 'market']] = [self.account_name, 'Stocks']
                return stocks
            return TradeData['Stocks'].InfoDefault

        return self.get_openpositions()

    def get_profit_loss(self, start: str, end: str):
        '''查詢已實現損益'''
        # TODO: delete in the future
        profitloss = API.list_profit_loss(API.stock_account, start, end)
        return self._obj_2_df(profitloss)

    def query_close(self, stockid: str, date: str):
        '''查證券收盤價'''
        ticks = API.ticks(API.Contracts.Stocks[stockid], date)
        df = pd.DataFrame({**ticks})
        df.ts = pd.to_datetime(df.ts)
        try:
            return df.close.values[-1]
        except:
            return -1

    def realized_profit(self, start: str = None, end: str = None):
        '''
        計算已實現損益
        start:開始日期, 預設為最近一個營業日(查詢當日)
        end: 結束日期, 預設為最近一個營業日(查詢當日)
        '''

        i = 0
        while i < 5:
            try:
                day = time_tool._strf_timedelta(TODAY, i)
                if not start:
                    start_ = day
                if not end:
                    end_ = day

                profitloss = self.get_profit_loss(start_, end_)
                if profitloss.shape[0]:
                    return profitloss.pnl.sum()
                return 0
            except:
                # 遇休市則往前一天查詢, 直到查到資料為止
                print(f"查無 {day} 已實現損益, 改查詢前一天\n")
                i += 1
                time.sleep(2)

                if i == 5:
                    return 0

    def settle_info(self, mode='info'):
        '''查詢 T ~ T+2 日的交割金額'''

        df_fail = pd.DataFrame(columns=['date', 'amount', 'T'])

        now = datetime.now()
        if 18 <= now.hour <= 19:
            logging.debug('Settle info temporary not accessable.')
            return df_fail

        n = 0
        while n < 5:
            try:
                settlement = self._list_settlements()
                df = self._obj_2_df(settlement)

                if mode == 'info':
                    logging.debug(f"Settlements:{df.to_dict('records')}")

                return df

            except:
                logging.exception('Catch an exception (settle_info):')

            time.sleep(1)
            n += 1

        return df_fail

    def compute_total_cost(self, stocks: pd.DataFrame):
        '''總成本合計'''
        if stocks.shape[0]:
            return (stocks.cost_price*stocks.quantity).sum()
        return 0

    def compute_total_unrealized_profit(self, stocks: pd.DataFrame):
        '''未實現損益合計'''
        if stocks.shape[0]:
            return stocks.pnl.sum()
        return 0

    def compute_margin_amount(self, stocks: pd.DataFrame):
        '''計算融資/融券金額'''
        if stocks.shape[0]:
            is_leverage = (
                'MarginTrading' == stocks.order_cond.apply(lambda x: x._value_))
            leverages = [
                crawler.FromHTML.Leverage(s)['融資成數']/100 for s in stocks.code]
            return sum(is_leverage*stocks.cost_price*stocks.quantity*leverages)
        return 0

    def compute_today_unrealized_profit(self, stocks: pd.DataFrame):
        '今日新增庫存未實現損益合計'
        if stocks.shape[0]:
            return stocks[stocks.yd_quantity == 0].pnl.sum()
        return 0

    def query_all(self):
        # 庫存明細(股)
        stocks = self.securityInfo()

        # 已實現損益
        profit = self.realized_profit()

        if not stocks.shape[0] and profit == 0:
            logging.info('目前無庫存')
            return None

        if stocks.shape[0]:
            print(f'\n{stocks}\n')

        # 庫存成本 & 現值
        total_cost = self.compute_total_cost(stocks)
        unrealized_profit = self.compute_total_unrealized_profit(stocks)
        total_market_value = total_cost + unrealized_profit

        # 融資金額
        margin_amount = self.compute_margin_amount(stocks)

        # 帳戶餘額
        balance = self.balance()
        if balance < 0:
            logging.error(f'Balance = {balance}, get balance from local')
            balance = pd.read_excel(
                f'{PATH}/daily_info/{self.filename}', sheet_name=self.account_name)
            balance = balance['T+1日帳戶餘額'].values[-1]

        # 今日新增庫存未實現損益
        today_unrealized_profit = self.compute_today_unrealized_profit(stocks)

        # 帳務交割資訊
        settle_info = self.settle_info()
        settles = settle_info.amount[1:].sum()

        # 總現值 = 帳戶餘額 + T+1日交割金額 + T+2日交割金額 + 庫存現值
        total_value = int(balance + total_market_value + settles)

        now = int(total_value - margin_amount - unrealized_profit)
        settle_t1 = settle_info.values[1, 1]
        settle_t2 = settle_info.values[2, 1]
        row = {
            '交易日期': TODAY_STR,
            'T日交割金額': settle_info.values[0, 1],
            'T日帳戶餘額': int(balance),
            'T+1日交割金額': settle_t1,
            'T+1日帳戶餘額': balance + settle_t1,
            'T+2日交割金額': settle_t2,
            'T+2日帳戶餘額': balance + settle_t1 + settle_t2,
            '庫存總成本': int(total_cost),
            '庫存現值': int(total_market_value),
            '融資金額': int(margin_amount),
            '融券金額': 0,
            '未實現損益': int(unrealized_profit),
            '已實現損益': int(profit),
            '當日新增庫存未實現損益': int(today_unrealized_profit),
            '總部位現值(含融資金額)': int(total_value),
            '總部位現值(不含融資金額)': int(total_value - margin_amount),
            '結算現值': now
        }
        return row

    def update_info(self, df, row):
        tb = pd.read_excel(df, sheet_name='dentist_1')
        return concat_df(tb, pd.DataFrame([row]), reset_index=True)

    def get_account_margin(self):
        '''期權保證金資訊'''

        n = 0
        while n < 5 and API.futopt_account.signed:
            try:
                margin = API.margin(API.futopt_account)
            except:
                logging.exception('Catch an exception (get_account_margin):')
                margin = None

            if margin:
                return margin

            time.sleep(1)
            n += 1

        return Margin

    def get_openpositions(self):
        '''查看期權帳戶持有部位'''

        positions = API.list_positions(API.futopt_account)
        if not positions:
            return TradeData['Futures'].InfoDefault

        df = self._obj_2_df(positions)
        if df.shape[0]:
            df = df.rename(columns={
                'direction': 'action',
                'price': 'cost_price',
            })
            df[['account', 'market']] = [self.account_name, 'Futures']
            return df
        return TradeData['Futures'].InfoDefault

    def get_settle_profitloss(self, start_date: str, end_date: str, market='Stocks'):
        '''查詢已實現損益'''

        if start_date:
            start_date = start_date.replace('-', '')

        if end_date:
            end_date = end_date.replace('-', '')

        if market == 'Stocks':
            account = API.stock_account
        else:
            account = API.futopt_account

        profitloss_detail = []
        profitloss = API.list_profit_loss(account, start_date, end_date)
        for pls in profitloss:
            pl_detail = API.list_profit_loss_detail(account, pls.id)
            profitloss_detail += pl_detail

        if not len(profitloss_detail):
            return TradeData.Futures.SettleDefault

        df = self._obj_2_df(profitloss_detail)
        df = df.sort_values('date').drop_duplicates()
        df.date = df.date.apply(lambda x: f'{x[:4]}-{x[4:6]}-{x[6:]}')
        df['profit'] = df.pnl - df.tax - df.fee
        return df

    def dataUsage(self):
        return round(API.usage().bytes/2**20, 2)


class AccountHandler(AccountInfo):
    def __init__(self, account_name: str) -> None:
        super().__init__()

        self.env = UserEnv(account_name)
        self.simulation = self.env.MODE == 'Simulation'
        self.simulate_amount = np.iinfo(np.int64).max

        # Stocks
        self.total_market_value = 0
        self.desposal_money = 0

        # Futures
        self.desposal_margin = 0
        self.ProfitAccCount = 0  # 權益總值

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
            f'[AccountInfo] Desposal amount = {self.desposal_money} (limit: {self.env.POSITION_LIMIT_LONG})')

    def _set_margin_limit(self):
        '''計算可交割的保證金額，不可超過帳戶可下單的保證金額上限'''
        if self.simulation:
            account_balance = 0
            self.desposal_margin = self.simulate_amount
            self.ProfitAccCount = self.simulate_amount
        else:
            account_balance = self.balance()
            margin = self.get_account_margin()
            self.desposal_margin = margin.available_margin
            self.ProfitAccCount = margin.equity  # 權益總值
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

    def activate_ca_(self):
        logging.info(f'[AccountInfo] Activate {self.env.ACCOUNT_NAME} CA')
        id = self.env.account_id()
        API.activate_ca(
            ca_path=f"./lib/ekey/551/{id}/S/Sinopac.pfx",
            ca_passwd=self.env.ca_passwd() if self.env.ca_passwd() else id,
            person_id=id,
        )
