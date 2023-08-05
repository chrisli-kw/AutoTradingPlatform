import time
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from collections import namedtuple

from .base import (
    AccountingNumber,
    compute_profits,
    computeReturn,
    computeWinLoss,
    convert_statement
)
from .. import file_handler
from ..config import PATH, TODAY_STR
from ..utils import progress_bar
from ..utils.time import TimeTool
from ..utils.file import FileHandler
from ..utils.kbar import KBarTool
from ..utils.crawler import readStockList
from ..utils.database import db, KBarTables


class BacktestPerformance(FileHandler):
    def __init__(self, config) -> None:
        self.Market = config.market
        self.scale = config.scale
        self.LEVERAGE_INTEREST = 0.075*(config.leverage != 1)
        self.leverage = 1/config.leverage
        self.margin = config.margin
        self.multipler = config.multipler
        self.mode = config.mode.lower()

        self.DATAPATH = f'{PATH}/backtest'
        self.ResultInfo = namedtuple(
            typename="TestResult",
            field_names=['Configuration', 'Summary', 'Statement', 'DailyInfo'],
            defaults=[None]*4
        )

    def process_daily_info(self, df: pd.DataFrame, **result):
        '''update daily info table'''
        profit = df.groupby('CloseTime').profit.sum().to_dict()
        nOpen = df.groupby('OpenTime').Code.count().to_dict()

        daily_info = result['daily_info'].copy()
        profit = pd.Series(daily_info.index).map(profit).fillna(0).cumsum()
        daily_info['balance'] = result['init_position'] + profit.values
        daily_info['nOpen'] = daily_info.index.map(nOpen).fillna(0)
        daily_info = daily_info.dropna()
        if self.Market != 'Stocks':
            for c in ['TSEopen', 'TSEclose', 'OTCopen', 'OTCclose']:
                daily_info[c] = 1
        return daily_info

    def get_max_profit_loss_days(self, statement: pd.DataFrame):
        '''
        計算最大連續獲利/損失天數
        參數:
        statement - 回測交易明細
        scale - 回測的k棒頻率
        '''

        def count_profits(is_profit):
            condition = profits.is_profit == is_profit
            counts = profits[condition].groupby('labels').profit.count()

            if not counts.shape[0]:
                return {
                    'start': '1900-01-01',
                    'end': '1900-01-01',
                    'n': 0,
                    'amount': 0
                }

            count_max = counts[counts == counts.max()].index[0]
            result = profits[profits.labels == count_max]
            return {
                'start': str(result.CloseTime.min()).split(' ')[0],
                'end': str(result.CloseTime.max()).split(' ')[0],
                'n': result.shape[0],
                'amount': result.profit.sum().astype(int)
            }

        profits = statement.copy()

        if self.scale != '1D':
            profits.CloseTime = profits.CloseTime.dt.date

        profits = profits.groupby('CloseTime').profit.sum().reset_index()
        profits['is_profit'] = (profits.profit > 0).astype(int)

        isprofit = profits.is_profit.values
        labels = []

        n = 1
        for i, p in enumerate(isprofit):
            if i != 0 and p != isprofit[i-1]:
                n = n + 1

            labels.append(n)

        profits['labels'] = labels

        max_positives = count_profits(is_profit=1)
        max_negagives = count_profits(is_profit=0)
        return max_positives, max_negagives

    def get_backtest_result(self, **result):
        '''取得回測報告'''

        configs = {
            '起始資金': AccountingNumber(result['init_position']),
            '做多/做空': self.mode,
            '槓桿倍數': self.leverage,
            '進場順序': result['buyOrder']
        }
        if isinstance(result['statement'], list):
            df = pd.DataFrame(result['statement'])
            df = convert_statement(
                df,
                mode='backtest',
                **result,
                market=self.Market,
                multipler=self.multipler
            )
        else:
            df = result['statement'].copy()
            df['KRun'] = -1  # TODO

        if df.shape[0]:

            start = str(df.OpenTime.min().date())
            end = str(df.CloseTime.max().date())

            win_loss = computeWinLoss(df)
            days_p, days_n = self.get_max_profit_loss_days(df)

            if 'daily_info' in result:
                Kbars = result.get('Kbars')
                result['daily_info'] = self.process_daily_info(df, **result)

                # TSE 漲跌幅
                tse_return = computeReturn(Kbars['1D'], 'TSEopen', 'TSEclose')

                # OTC 漲跌幅
                otc_return = computeReturn(Kbars['1D'], 'OTCopen', 'OTCclose')
            else:
                result['daily_info'] = None
                if db.HAS_DB:
                    table = KBarTables[self.scale]
                    table = db.query(
                        table,
                        table.Time >= df.OpenTime.min(),
                        table.Time <= df.CloseTime.max(),
                    )
                else:
                    dir_path = f'{PATH}/KBars/{self.scale}'
                    table = self.read_tables_in_folder(dir_path)
                    table = table[
                        (table.Time >= df.OpenTime.min()) &
                        (table.Time <= df.CloseTime.max())
                    ]

                df_TSE = table[table.name == '1']
                df_OTC = table[table.name == '101']

                tse_return = computeReturn(df_TSE, 'Open', 'Close')
                otc_return = computeReturn(df_OTC, 'Open', 'Close')

            # 總報酬率
            profits = compute_profits(df)
            balance = result['init_position'] + profits['TotalProfit']
            total_return = balance/result['init_position']

            # 年化報酬率
            days = (df.CloseTime.max() - df.OpenTime.min()).days
            anaualized_return = 100*round(total_return**(365/days) - 1, 2)

            if df.shape[0] >= 5:
                std = df.balance.rolling(5).std().median()
            else:
                std = 0

            if days_p['n']:
                max_win_days = f"{days_p['start']} ~ {days_p['end']}，共{days_p['n']}天"
            else:
                max_win_days = '無'

            if days_n['n']:
                max_loss_days = f"{days_n['start']} ~ {days_n['end']}，共{days_n['n']}天"
            else:
                max_loss_days = '無'

            # 摘要
            summary = pd.DataFrame([{
                '期末資金': AccountingNumber(round(balance)),
                '毛利': AccountingNumber(profits['GrossProfit']),
                '毛損': AccountingNumber(profits['GrossLoss']),
                '淨利': AccountingNumber(profits['TotalProfit']),
                '平均獲利': AccountingNumber(profits['MeanProfit']),
                '平均虧損': AccountingNumber(profits['MeanLoss']),
                '淨值波動度': AccountingNumber(round(std)),
                '總報酬(與大盤比較)': f"{round(100*(total_return - 1), 2)}%",
                '指數報酬(TSE/OTC)': f"{tse_return}%/{otc_return}%",
                '年化報酬率': f"{AccountingNumber(anaualized_return)}%",
                '最大單筆獲利': AccountingNumber(profits['MaxProfit']),
                '最大單筆虧損': AccountingNumber(profits['MaxLoss']),
                "最大區間獲利": max_win_days,
                "最大連續獲利": f"${AccountingNumber(days_p['amount'])}",
                "最大區間虧損": max_loss_days,
                "最大連續虧損": f"${AccountingNumber(days_n['amount'])}",
                '全部平均持倉K線數': round(df.KRun.mean(), 1),
                '獲利平均持倉K線數': profits['KRunProfit'],
                '虧損平均持倉K線數': profits['KRunLoss'],
                '獲利交易筆數': win_loss[True],
                '虧損交易筆數': win_loss[False],
                '總交易筆數': df.shape[0],
                '勝率': f"{round(100*win_loss[True]/df.shape[0], 2)}%",
                '獲利因子': profits['ProfitFactor'],
                '盈虧比': profits['ProfitRatio'],
            }]).T.reset_index()
            summary.columns = ['Content', 'Description']

        else:
            print('無交易紀錄')
            start = str(result['startDate'].date())
            end = str(result['endDate'].date())
            summary, df = None, None

        configs.update({'交易開始日': start, '交易結束日': end})
        configs = pd.DataFrame([configs]).T.reset_index()
        configs.columns = ['Content', 'Description']
        return self.ResultInfo(configs, summary, df, result['daily_info'])

    def generate_tb_reasons(self, statement):
        '''進出場原因統計表'''

        aggs = {'Code': 'count', 'iswin': 'sum', 'profit': ['sum', 'mean']}
        df = statement.groupby(['OpenReason', 'CloseReason']).agg(aggs)
        df = df.unstack(0).reset_index()
        df = df.T.reset_index(level=[0, 1], drop=True).T

        openReasons = list(np.unique(list(df.columns[1:])))
        p = len(openReasons)

        df.insert(p+1, 'total_count', df.fillna(0).iloc[:, 1:p+1].sum(axis=1))
        df.insert(
            2*p+2, 'total_win', df.fillna(0).iloc[:, p+2:2*p+2].sum(axis=1))
        for i in range(p):
            df.insert(
                2*p+3+i, f'win_{openReasons[i]}', 100*df.iloc[:, p+2+i]/df.iloc[:, 1+i])

        df = df.fillna(0)

        sums = df.sum().values
        for i in range(p):
            sums[2*p+3+i] = round(100*sums[p+2+i]/sums[1+i], 2)
            sums[4*p+3+i] = round(sums[3*p+3+i]/sums[1+i])
        df = pd.concat([df, pd.DataFrame(sums, index=df.columns).T])

        for i in range(1, df.shape[1]):
            if 0 < i - 2*(p+1) <= p:
                df.iloc[:, i] = df.iloc[:, i].astype(float).round(2)
            else:
                df.iloc[:, i] = df.iloc[:, i].astype(int)

        columns1 = ['OpenReason'] + (openReasons + ['加總'])*2 + openReasons*3
        columns2 = ['CloseReason'] + ['筆數']*(p+1) + ['獲利筆數'] * \
            (p+1) + ['勝率']*p + ['總獲利']*p + ['平均獲利']*p
        df.columns = pd.MultiIndex.from_arrays([columns1, columns2])
        df.iloc[-1, 0] = 'Total'
        return df

    def generate_win_rates(self, statement):
        '''個股勝率統計'''
        group = statement.groupby('Code')
        win_rates = group.iswin.sum().reset_index()
        win_rates['total_deals'] = group.Code.count().values
        win_rates['win_rate'] = win_rates.iswin/win_rates.total_deals
        return win_rates

    def save_result(self, TestResult):
        '''儲存回測報告'''

        win_rates = self.generate_win_rates(TestResult.Statement)
        reasons = self.generate_tb_reasons(TestResult.Statement)

        writer = pd.ExcelWriter(
            f'{self.DATAPATH}/{TODAY_STR}-backtest.xlsx', engine='xlsxwriter')
        TestResult.Configuration.to_excel(
            writer, index=False, sheet_name='Backtest Settings')
        TestResult.Summary.to_excel(
            writer, index=False, sheet_name='Summary')
        TestResult.Statement.to_excel(
            writer, index=False,  sheet_name='Transaction Detail')
        reasons.to_excel(writer, sheet_name='Transaction Reasons')
        win_rates.to_excel(writer, index=False, sheet_name='Win Rate')
        writer.save()


class BackTester(BacktestPerformance, TimeTool):
    def __init__(self, script):
        self.set_scripts(script)
        BacktestPerformance.__init__(self, script)

        self.isLong = self.mode == 'long'
        self.sign = 1 if self.isLong else -1
        self.FEE_RATE = .001425*.65
        self.TAX_RATE_STOCK = .003
        self.TAX_RATE_FUTURES = .00002
        self.TimeCol = 'Time'

        self.nStocksLimit = 0
        self.nStocks_high = 20
        self.day_trades = []
        self.statements = []
        self.stocks = {}
        self.daily_info = {}
        self.balance = 1000000
        self.init_balance = 1000000
        self.buyOrder = None
        self.raiseQuota = script.raiseQuota  # 加碼參數
        self.market_value = 0
        self.Action = namedtuple(
            typename="Action",
            field_names=['position', 'reason', 'msg', 'price'],
            defaults=[0, '', '', 0]
        )

    def load_datasets(self, backtestScript, start='', end='', **kwargs):
        market = backtestScript.market
        if market == 'Stocks':
            codes = readStockList().code.to_list()
        else:
            codes = ['TX']
        codes += ['1', '101']

        if not start:
            start = '2018-07-01'

        if not end:
            end = TODAY_STR

        Kbars = {scale: None for scale in backtestScript.kbarScales}
        Kbars['1D'] = None
        for scale in Kbars:
            scale_ = scale if market == 'Stocks' else '1T'

            if db.HAS_DB:
                df = db.query(
                    KBarTables[scale_],
                    KBarTables[scale_].Time >= start,
                    KBarTables[scale_].Time <= end,
                    KBarTables[scale_].name.in_(codes)
                )
            else:
                dir_path = f'{PATH}/Kbars/{scale_}'
                df = file_handler.read_tables_in_folder(
                    dir_path,
                    pattern=market.lower()
                )
                df = df[
                    (df.Time >= start) &
                    (df.Time <= end) &
                    df.name.isin(codes)
                ]

            df = df.sort_values(['name', 'Time'])
            if market == 'Futures':
                df = KBarTool().convert_kbar(df, scale=scale)

            df['date'] = df.Time.dt.date.astype(str)
            Kbars[scale] = df

        if kwargs:
            for dataname, func in kwargs.items():
                Kbars[dataname] = func(start=start, end=end)

        return Kbars

    def addFeatures(self, Kbars: dict):
        return Kbars

    def on_addFeatures(self):
        def wrapper(func):
            self.addFeatures = func
        return wrapper

    def selectStocks(self, Kbars: dict):
        '''依照策略選股條件挑出可進場的股票，必須新增一個"isIn"欄位'''
        return Kbars

    def on_selectStocks(self):
        def wrapper(func):
            self.selectStocks = func
        return wrapper

    def examineOpen(self, inputs: dict, **kwargs):
        '''檢查進場條件'''
        return self.Action(100, '進場', 'msg', inputs['Open'])

    def on_examineOpen(self):
        def wrapper(func):
            self.examineOpen = func
        return wrapper

    def examineClose(self, stocks: dict, inputs: dict, **kwargs):
        '''檢查出場條件'''
        if inputs['Low'] < inputs['yLow']:
            closePrice = inputs['yLow']
            return self.Action(100, '出場', 'msg', closePrice)
        return self.Action()

    def on_examineClose(self):
        def wrapper(func):
            self.examineClose = func
        return wrapper

    def computeOpenLimit(self, data: dict, **kwargs):
        '''計算每日買進股票上限(可做幾支)'''
        return 2000

    def on_computeOpenLimit(self):
        def wrapper(func):
            self.computeOpenLimit = func
        return wrapper

    def computeOpenUnit(self, inputs: dict):
        '''
        計算買進股數
        參數:
        inputs - 交易判斷當下的股票資料(開高低收等)
        '''
        return 5

    def on_computeOpenUnit(self):
        def wrapper(func):
            self.computeOpenUnit = func
        return wrapper

    def setVolumeProp(self, market_value: float):
        '''根據成交量比例設定進場張數'''
        return 0.025

    def on_setVolumeProp(self):
        def wrapper(func):
            self.setVolumeProp = func
        return wrapper

    def set_scripts(self, testScript: object):
        '''設定回測腳本'''

        if hasattr(testScript, 'addFeatures'):
            @self.on_addFeatures()
            def add_scale_features(df):
                return testScript.addFeatures(df)

        if hasattr(testScript, 'selectStocks'):
            @self.on_selectStocks()
            def select_stocks_(df):
                return testScript.selectStocks(df)

        if hasattr(testScript, 'examineOpen'):
            @self.on_examineOpen()
            def open_(inputs, **kwargs):
                return testScript.examineOpen(inputs, **kwargs)

        if hasattr(testScript, 'examineClose'):
            @self.on_examineClose()
            def close_(stocks, inputs, **kwargs):
                return testScript.examineClose(stocks, inputs, **kwargs)

        if hasattr(testScript, 'computeOpenLimit'):
            @self.on_computeOpenLimit()
            def compute_limit(Kbars, **kwargs):
                return testScript.computeOpenLimit(Kbars, **kwargs)

        if hasattr(testScript, 'computeOpenUnit'):
            @self.on_computeOpenUnit()
            def open_unit(inputs):
                return testScript.computeOpenUnit(inputs)

        if hasattr(testScript, 'setVolumeProp'):
            @self.on_setVolumeProp()
            def open_unit(market_value):
                return testScript.setVolumeProp(market_value)

        if hasattr(testScript, 'scale'):
            self.scale = testScript.scale

    def updateMarketValue(self):
        '''更新庫存市值'''

        if self.stocks:
            amount = sum([
                self.computeCloseAmount(
                    s['openPrice'], s['price'], s['quantity'])[1]
                for s in self.stocks.values()
            ])
        else:
            amount = 0

        self.market_value = self.balance + amount

    def _updateStatement(self, **kwargs):
        '''更新交易紀錄'''
        price = kwargs['price']
        quantity = kwargs['quantity']
        data = self.stocks[kwargs['stockid']]
        openPrice = data['openPrice']
        openamount, amount = self.computeCloseAmount(
            openPrice, price, quantity)
        openfee = self.computeFee(openamount)
        closefee = self.computeFee(amount)
        interest = self.computeLeverageInterest(
            kwargs['day'], data['day'], openamount)
        tax = self.computeTax(openPrice, price, quantity)

        if self.isLong:
            return_ = data['cum_max_min']/openPrice
        else:
            return_ = openPrice/data['cum_max_min']

        self.balance += (amount - closefee - interest - tax)
        self.statements.append({
            'Code': kwargs['stockid'],
            'OpenTime': data['day'],
            'OpenReason': data['openReason'],
            'OpenPrice': openPrice,
            'OpenQuantity': quantity,
            'OpenAmount': openamount,
            'OpenFee': openfee,
            'CloseTime': kwargs['day'],
            'CloseReason': kwargs['reason'],
            'ClosePrice': price,
            'CloseQuantity': quantity,
            'CloseAmount': amount,
            'CloseFee': closefee,
            'Tax': tax,
            'KRun': data['krun'],
            'PotentialReturn': 100*round(return_ - 1, 4)
        })

    def execute(self, trans_type: str, **kwargs):
        '''
        新增/更新庫存明細
        trans_type: 'Open' or 'Close'
        '''

        stockid = kwargs['stockid']
        day = kwargs['day']
        price = kwargs['price']

        if trans_type == 'Open':
            quantity = kwargs['quantity']
            amount = kwargs['amount']
            self.balance -= amount

            if stockid not in self.stocks:
                self.stocks.update({
                    stockid: {
                        'day': day,
                        'openPrice': price,
                        'quantity': quantity,
                        'position': 100,
                        'price': price,
                        'profit': 1,
                        'amount': amount,
                        'openReason': kwargs['reason'],
                        'krun': 0,
                        'cum_max_min': kwargs['cum_max_min'],
                        'bsh': kwargs['bsh']
                    }
                })
            else:
                self.computeAveragePrice(quantity, price)

        else:
            quantity = self.computeCloseUnit(stockid, kwargs['position'])
            kwargs['quantity'] = quantity
            self._updateStatement(**kwargs)
            if quantity <= self.stocks[stockid]['quantity']:
                self.stocks[stockid]['quantity'] -= quantity
                self.stocks[stockid]['position'] -= kwargs['position']

            if self.stocks[stockid]['quantity'] <= 0:
                self.stocks.pop(stockid, None)
                self.nClose += 1

    def get_tick_delta(self, stock_price: float):
        if stock_price < 10:
            return 0.01
        elif stock_price < 50:
            return 0.05
        elif stock_price < 100:
            return 0.1
        elif stock_price < 500:
            return 0.5
        elif stock_price < 1000:
            return 1
        return 5

    def computePriceByDev(self, stock_price: float, dev: float):
        '''計算真實股價(By tick price)'''

        tick_delta = self.get_tick_delta(stock_price)
        return round(int(stock_price*dev/tick_delta)*tick_delta + tick_delta, 2)

    def computeOpenAmount(self, price: float, quantity: int):
        if self.Market == 'Stocks':
            return quantity*price*self.leverage
        return quantity*self.margin

    def computeCloseAmount(self, openPrice: float, closePrice: float, quantity: int):
        openAmount = self.computeOpenAmount(openPrice, quantity)
        profit = (closePrice - openPrice)*quantity
        if self.Market == 'Stocks' and self.leverage == 1:
            closeAmount = closePrice*quantity
        elif self.Market == 'Stocks' and self.leverage != 1:
            closeAmount = openAmount + profit
        else:
            closeAmount = openAmount + profit*self.multipler*self.sign
        return openAmount, closeAmount

    def computeCloseUnit(self, stockid: str, prop: float):
        '''從出場的比例%推算出場量(張/口)'''
        q_balance = self.stocks[stockid]['quantity']
        position_now = self.stocks[stockid]['position']
        if prop != 100 and position_now != prop and q_balance > 1000:
            q_target = q_balance/1000
            q_target = int(q_target*prop/100)

            if q_target == 0:
                return 1000

            return min(1000*q_target, q_balance)
        return q_balance

    def computeFee(self, amount: float):
        return max(round(amount*self.FEE_RATE), 20)

    def computeAveragePrice(self, stockid: str, quantity: int, price: float):
        # 加碼後更新
        data = self.stocks[stockid]
        total_quantity = data['quantity'] + quantity
        total_amount = data['openPrice']*data['quantity'] + price*quantity
        average_price = round(total_amount/total_quantity, 2)

        self.stocks[stockid].update({
            'openPrice': average_price,
            'quantity': total_quantity,
            'openReason': data['openReason'] + f'<br>{quantity}'
        })

    def computeTax(self, openPrice: float, closePrice: float, quantity: int):
        if self.Market == 'Stocks':
            return round((openPrice*quantity)*self.TAX_RATE_STOCK)
        return round(closePrice*quantity*self.multipler*self.TAX_RATE_FUTURES)

    def computeLeverageInterest(self, day1: Union[str, datetime], day2: Union[str, datetime], amount: float):
        if self.leverage == 1:
            return 0

        d = self.date_diff(day1, day2)
        return amount*(1 - self.leverage)*self.LEVERAGE_INTEREST*d/365

    def checkOpenUnitLimit(self, unit: float, volume_ma: float):
        unit = max(round(unit), 0)
        unit_limit = max(volume_ma*self.volume_prop, 10)
        unit = int(min(unit, unit_limit)/self.leverage)
        return min(unit, 2000)  # 上限 2000 張

    def checkOpen(self, inputs: dict):
        '''
        買進條件判斷
        參數:
        day - 交易日(時間)
        inputs - 交易判斷當下的股票資料(開高低收等)
        '''

        unit = self.computeOpenUnit(inputs)
        # if 'volume_ma' in inputs:
        #     unit = self.checkOpenUnitLimit(unit, inputs['volume_ma'])

        openInfo = self.examineOpen(
            inputs,
            market_value=self.market_value,
            day_trades=self.day_trades
        )

        if openInfo.price > 0 and unit > 0:
            data = inputs[self.scale]
            name = data['name']
            if name in self.stocks:
                # 加碼部位
                quantity = 1000*(self.stocks[name]['quantity']/1000)/3
            elif self.Market == 'Stocks':
                quantity = 1000*unit
            else:
                quantity = unit

            amount = self.computeOpenAmount(openInfo.price, quantity)
            fee = self.computeFee(amount)
            if self.balance >= amount+fee and len(self.stocks) < self.nStocksLimit:
                self.execute(
                    trans_type='Open',
                    day=data['Time'],
                    stockid=name,
                    price=openInfo.price,
                    quantity=quantity,
                    position=None,
                    amount=amount,
                    cum_max_min=data['High'] if self.isLong else data['Low'],
                    reason=openInfo.reason,
                    bsh=data['High']
                )
                self.day_trades.append(name)

    def checkMarginCall(self, name: str, closePrice: float):
        if self.leverage == 1:
            return False

        openPrice = self.stocks[name]['openPrice']
        margin = closePrice/(openPrice*(1 - self.leverage))
        return margin < 1.35

    def checkClose(self, inputs: dict, stocksClosed: dict):
        data = inputs[self.scale]
        name = data['name']
        value = data['High'] if self.isLong else data['Low']
        cum_max_min = min(self.stocks[name]['cum_max_min'], value)
        self.stocks[name].update({
            'price': data['Close'],
            'krun': self.stocks[name]['krun'] + 1,
            'cum_max_min': cum_max_min
        })

        closeInfo = self.examineClose(
            stocks=self.stocks,
            inputs=inputs,
            stocksClosed=stocksClosed
        )

        margin_call = self.checkMarginCall(name, closeInfo.price)
        if closeInfo.position or margin_call:
            self.execute(
                trans_type='Close',
                day=data['Time'],
                stockid=name,
                price=closeInfo.price,
                quantity=None,
                position=100 if margin_call else closeInfo.position,
                reason='維持率<133%' if margin_call else closeInfo.reason
            )

    def set_open_order(self, df: pd.DataFrame):
        # by 成交值
        if self.buyOrder == 'z_totalamount':
            df = df.sort_values('z_totalamount')

        # by 族群＆成交值
        elif self.buyOrder == 'category':
            name_count = df[df.isIn == True].groupby('category').name.count()
            name_count = name_count.to_dict()
            df['n_category'] = df.category.map(name_count).fillna(0)
            df = df.sort_values(['n_category'], ascending=False)
        else:
            df = df.sort_values('Close')
        return df

    def set_params(self, **params):
        '''參數設定'''

        if 'init_position' in params:
            self.balance = params['init_position']
            self.init_balance = params['init_position']
            self.market_value = params['init_position']

        if 'buyOrder' in params:
            self.buyOrder = params['buyOrder']

        self.Kbars = {}

    def run(self, Kbars: pd.DataFrame, **params):
        '''
        回測
        參數:
        Kbars - 歷史資料表
        '''

        self.set_params(**params)
        stocksClosed = {}

        t1 = time.time()

        df = Kbars[self.scale]
        times = np.sort(df.Time.unique())
        N = len(times)
        for i, time_ in enumerate(times):
            self.nClose = 0
            self.volume_prop = self.setVolumeProp(self.market_value)

            # 進場順序
            rows = df[df.Time == time_].copy()
            rows = rows[(rows.isIn == 1) | (rows.name.isin(self.stocks))]
            rows = self.set_open_order(rows)

            # 取出當天(或某小時)所有股票資訊
            chance = rows.isIn.sum()
            if rows.nth_bar.min() == 1:
                day = str(pd.to_datetime(time_).date())
                tb = Kbars['1D'][Kbars['1D'].date == day]
                self.Kbars['1D'] = tb.set_index('name').to_dict('index')
                del tb

                self.nStocksLimit = self.computeOpenLimit(
                    self.Kbars['1D'], day=time_)
                self.day_trades = []

            # 檢查進場 & 出場
            rows = np.array(rows.to_dict('records'))
            for row in rows:
                name = row['name']
                inputs = Kbars.copy()
                inputs[self.scale] = row
                if '1D' in Kbars:
                    inputs['1D'] = self.Kbars['1D'][name]
                    inputs['1D']['name'] = name

                if name in self.stocks:
                    self.checkClose(inputs, stocksClosed)
                elif row['isIn']:
                    self.checkOpen(inputs)

            # 更新交易明細數據
            self.daily_info[time_] = {
                'chance': chance,
                'n_stock_limit': self.nStocksLimit,
                'n_stocks': len(self.stocks),
                'nClose': self.nClose
            }
            self.updateMarketValue()
            progress_bar(N, i)

        t2 = time.time()
        print(f"\nBacktest time: {round(t2-t1, 2)}s")

        # 清空剩餘庫存
        for name in list(self.stocks):
            self.execute(
                trans_type='Close',
                day=time_,
                stockid=name,
                price=self.stocks[name]['price'],
                quantity=None,
                position=100,
                reason='清空庫存'
            )
        self.daily_info[time_].update({'nClose': self.nClose})

        params.update({
            'statement': self.statements,
            'startDate': params['startDate'] if 'startDate' in params else df.Time.min(),
            'endDate': params['endDate'] if 'endDate' in params else df.Time.max(),
            'daily_info': pd.DataFrame(self.daily_info).T,

        })
        del df, rows
        return self.get_backtest_result(
            **params,
            isLong=self.isLong,
            Kbars=Kbars
        )
