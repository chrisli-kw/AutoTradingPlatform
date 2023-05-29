import os
import time
import logging
import pandas as pd
import shioaji as sj
from datetime import datetime
from shioaji.account import StockAccount

from .. import API, PATH, TODAY, TODAY_STR
from .time import TimeTool
from .crawler import CrawlFromHTML


class AccountInfo(CrawlFromHTML, TimeTool):
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
        self.desposal_margin = 0
        self.ProfitAccCount = 0  # 權益總值
        self.df_securityInfo = pd.DataFrame(
            columns=[
                'code', 'order_cond', 'action', 'pnl',
                'cost_price', 'quantity', 'yd_quantity', 'last_price'
            ]
        )
        self.df_futuresInfo = pd.DataFrame(
            columns=[
                'Account', 'Date', 'Code', 'CodeName', 'OrderNum',
                'OrderBS', 'OrderType', 'Currency', 'paddingByte', 'Volume',
                'ContractAverPrice', 'SettlePrice', 'RealPrice', 'FlowProfitLoss', 'SettleProfitLoss',
                'StartSecurity', 'UpKeepSecurity', 'OTAMT', 'MTAMT'
            ]
        )

    def _login(self, API_KEY, SECRET_KEY, account_name):
        API.login(api_key=API_KEY, secret_key=SECRET_KEY, contracts_timeout=10000)

        nth_account = int(account_name[-1])
        if nth_account > 1:
            accounts = [a for a in API.list_accounts() if isinstance(a, StockAccount)]
            if len(accounts) > 1:
                API.set_default_account(accounts[nth_account-1])
            else:
                logging.warning('此ID只有一個證券戶')

        self.account_name = account_name

        if API.futopt_account:
            self.HAS_FUTOPT_ACCOUNT = True

        time.sleep(0.05)
        logging.info(f'【{account_name}】log-in successful!')

    def _list_settlements(self):
        '''取得交割資訊'''
        n = 0
        while True:
            try:
                return API.settlements(API.stock_account)
            except:
                logging.warning('無法取得交割資訊，重試中')
                n += 1

                if n >= 60:
                    return 0

                time.sleep(1)

    def _obj_2_df(self, objects: list):
        '''把自API查詢得到的物件轉為DataFrame'''
        try:
            return pd.DataFrame([o.__dict__ for o in objects])
        except:
            return pd.DataFrame([{o[0]:o[1] for o in objects}])

    def create_info_table(self):
        if self.filename in os.listdir(f'{PATH}/daily_info/'):
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

    def securityInfo(self):
        '''查庫存明細'''
        while True:
            try:
                stocks = API.list_positions(API.stock_account, unit=sj.constant.Unit.Share)
                stocks = self._obj_2_df(stocks)
                break
            except:
                logging.warning('無法取得庫存，重試中')
                time.sleep(1)
        stocks = stocks.rename(
            columns={
                'cond': 'order_cond',
                'direction': 'action',
                'price': 'cost_price',
            }
        )
        if stocks.shape[0]:
            stocks.pnl = stocks.pnl.astype(int)  # 未實現損益
            stocks.order_cond = stocks.order_cond.astype(str)  # 交易別
            stocks.insert(1, 'name', stocks.code.apply(self.get_stock_name))
            return stocks
        return self.df_securityInfo

    def get_profit_loss(self, start: str, end: str):
        '''查詢已實現損益'''
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
                day = self._strf_timedelta(TODAY, i)
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
            is_leverage = ('MarginTrading' == stocks.order_cond.apply(lambda x: x._value_))
            leverages = [self.get_leverage(s)['融資成數']/100 for s in stocks.code]
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

        # 總現值 = 帳戶餘額 + T+1日交割金額 + T+2日交割金額 + 庫存現值
        total_value = int(balance + total_market_value + settle_info.amount[1:].sum())

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
                self.desposal_margin = margin.available_margin
                self.ProfitAccCount = margin.equity
                break

            time.sleep(1)
            n += 1

    def get_openpositions(self):
        '''查看期權帳戶持有部位'''

        positions = API.list_positions(API.futopt_account)
        if not positions:
            return self.df_futuresInfo

        df = self._obj_2_df(positions)
        if df.shape[0]:
            return self.df_futuresInfo
        return df

    def get_settle_profitloss(self, start_date: str, end_date: str):
        '''查看期權帳戶(已實現)損益'''
        # TODO: 1.0.0 list_profit_loss_detail(*api.futopt_account*), list_profit_loss_summary(*api.futopt_account*)

        if start_date:
            start_date = start_date.replace('-', '')

        if end_date:
            end_date = end_date.replace('-', '')

        settle_profitloss = API.get_account_settle_profitloss(
            summary='Y', start_date=start_date, end_date=end_date)
        df_profitloss = pd.DataFrame(settle_profitloss.data())
        return df_profitloss
