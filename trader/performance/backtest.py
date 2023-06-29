import time
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime
from collections import namedtuple

from ..config import PATH, TODAY_STR
from ..utils import progress_bar
from ..utils.kbar import KBarTool
from ..utils.time import TimeTool
from ..utils.database import db
from ..utils.database.tables import PutCallRatioList
from ..utils.select import SelectStock
from .base import AccountingNumber, compute_profits, computeReturn, computeWinLoss
from .charts import BacktestFigures


def merge_pc_ratio(df):
    # merge put call ratio data
    if db.HAS_DB:
        df_pcr = db.query(PutCallRatioList)
    else:
        df_pcr = pd.read_csv(f'{PATH}/put_call_ratio.csv')

    df_pcr = df_pcr.rename(columns={'Date': 'date'})
    df_pcr.date = pd.to_datetime(df_pcr.date)
    df_pcr.PutCallRatio = df_pcr.PutCallRatio.shift(1)
    df = df.merge(df_pcr[['date', 'PutCallRatio']], how='left', on='date')
    return df


class BacktestPerformance:
    def __init__(self) -> None:
        self.DATAPATH = f'{PATH}/backtest'
        self.TestResult = namedtuple(
            typename="TestResult",
            field_names=['Configuration', 'Summary', 'Statement', 'DailyInfo']
        )

    def process_daily_info(self, df: pd.DataFrame, **result):
        '''update daily info table'''
        profit = df.groupby('CloseDate').profit.sum().to_dict()
        nOpen = df.groupby('OpenDate').stock.count().to_dict()

        daily_info = result['daily_info'].copy()
        profit = pd.Series(daily_info.index).map(profit).fillna(0).cumsum()
        daily_info['balance'] = result['init_position'] + profit.values
        daily_info['nOpen'] = daily_info.index.map(nOpen).fillna(0)
        daily_info = daily_info.dropna()
        if result['market'] != 'Stocks':
            for c in ['TSEopen', 'TSEclose', 'OTCopen', 'OTCclose']:
                daily_info[c] = 1
        return daily_info

    def get_max_profit_loss_days(self, statement: pd.DataFrame, scale: str):
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
                'start': str(result.CloseDate.min()).split(' ')[0],
                'end': str(result.CloseDate.max()).split(' ')[0],
                'n': result.shape[0],
                'amount': result.profit.sum()
            }

        profits = statement.copy()

        if scale != '1D':
            profits.CloseDate = profits.CloseDate.dt.date

        profits = profits.groupby('CloseDate').profit.sum().reset_index()
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
            '做多/做空': result['mode'],
            '槓桿倍數': 1/result['leverage'],
            '起始資金': AccountingNumber(result['init_position']),
            '股數因子': result['unit'],
            '進場順序': result['buyOrder']
        }
        if result['statement']:
            # 將交易明細轉為表格
            df = pd.DataFrame(result['statement'])

            df.OpenAmount = df.OpenAmount.astype('int64')
            df.CloseAmount = df.CloseAmount.astype('int64')
            df.ClosePrice = df.ClosePrice.round(2)

            if result['market'] == 'Stocks':
                netOpenAmount = (df.OpenAmount + df.OpenFee)
                netCloseAmount = (df.CloseAmount - df.CloseFee - df.Tax)
                df['profit'] = (netCloseAmount - netOpenAmount).astype('int64')
                df['returns'] = (
                    100*(df.CloseAmount/df.OpenAmount - 1)).round(2)
            else:
                sign = 1 if result['isLong'] else -1

                df['profit'] = (df.ClosePrice - df.OpenPrice)*df.CloseQuantity
                totalExpense = (df.OpenFee + df.CloseFee + df.Tax)*sign
                df.profit = df.profit*result['multipler'] - totalExpense
                df['returns'] = (
                    sign*100*((df.ClosePrice/df.OpenPrice)**sign - 1)).round(2)

            df.profit = df.profit.round()
            df['iswin'] = df.profit > 0

            if not result['isLong']:
                df.profit *= -1
                df.returns *= -1
            df['balance'] = result['init_position'] + df.profit.cumsum()

            configs.update({
                '回測期間': f"{str(df.OpenDate.min().date())} - {str(df.CloseDate.max().date())}",
            })

            win_loss = computeWinLoss(df)
            days_p, days_n = self.get_max_profit_loss_days(df, result['scale'])
            result['daily_info'] = self.process_daily_info(df, **result)

            # TSE 漲跌幅
            tse_return = computeReturn(
                result['daily_info'], 'TSEopen', 'TSEclose')

            # OTC 漲跌幅
            otc_return = computeReturn(
                result['daily_info'], 'OTCopen', 'OTCclose')

            # 總報酬率
            profits = compute_profits(df)
            balance = result['init_position'] + profits['TotalProfit']
            total_return = balance/result['init_position']

            # 年化報酬率
            anaualized_return = total_return**(
                365/(df.CloseDate.max() - df.OpenDate.min()).days)

            # 回測摘要
            summary = pd.DataFrame([{
                '期末資金': AccountingNumber(round(balance)),
                '毛利': AccountingNumber(profits['GrossProfit']),
                '毛損': AccountingNumber(profits['GrossLoss']),
                '單筆最大獲利': AccountingNumber(round(df.profit.max(), 0)),
                '單筆最大虧損': AccountingNumber(round(df.profit.min(), 0)),
                '平均獲利': AccountingNumber(profits['MeanProfit']),
                '平均虧損': AccountingNumber(profits['MeanLoss']),
                "最大區間獲利": f"{days_p['start']} ~ {days_p['end']}，共{days_p['n']}天，${days_p['amount']}",
                "最大區間虧損": f"{days_n['start']} ~ {days_n['end']}，共{days_n['n']}天，${days_n['amount']}",
                '淨利': AccountingNumber(profits['TotalProfit']),
                '全部平均持倉K線數': round(df.KRun.mean(), 1),
                '獲利平均持倉K線數': profits['KRunProfit'],
                '虧損平均持倉K線數': profits['KRunLoss'],
                '淨值波動度': round(df.balance.rolling(5).std().median()),
                '總報酬(與大盤比較)': f"{round(100*(total_return - 1), 2)}% (TSE {tse_return}%; OTC {otc_return}%)",
                '年化報酬率': f"{round(100*(anaualized_return - 1), 2)}%",
                '獲利/虧損/總交易筆數': f"{win_loss[True]}/{win_loss[False]}/{df.shape[0]}",
                '勝率': f"{round(100*win_loss[True]/df.shape[0], 2)}%",
                '獲利因子': profits['ProfitFactor'],
                '盈虧比': profits['ProfitRatio'],
            }]).T.reset_index()
            summary.columns = ['Content', 'Description']

        else:
            print('無回測交易紀錄')
            configs.update({
                '回測期間': f"{str(result['startDate'].date())} - {str(result['endDate'].date())}"
            })
            summary, df = None, None

        configs = pd.DataFrame([configs]).T.reset_index()
        configs.columns = ['Content', 'Description']
        return self.TestResult(configs, summary, df, result['daily_info'])

    def generate_tb_reasons(self, statement):
        '''進出場原因統計表'''

        aggs = {'stock': 'count', 'iswin': 'sum', 'profit': ['sum', 'mean']}
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
        group = statement.groupby('stock')
        win_rates = group.iswin.sum().reset_index()
        win_rates['total_deals'] = group.stock.count().values
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


class BackTester(SelectStock, BacktestFigures, BacktestPerformance, TimeTool):
    def __init__(self, config):
        SelectStock.__init__(
            self,
            dma=5,
            mode=config.mode,
            scale=config.scale
        )
        BacktestPerformance.__init__(self)
        BacktestFigures.__init__(self, config.market)

        self.scale = config.scale
        self.LEVERAGE_INTEREST = 0.075*(config.leverage != 1)
        self.leverage = 1/config.leverage
        self.margin = config.margin
        self.multipler = config.multipler
        self.mode = config.mode.lower()

        self.isLong = self.mode == 'long'
        self.sign = 1 if self.isLong else -1
        self.FEE_RATE = .001425*.65
        self.TAX_RATE_STOCK = .003
        self.TAX_RATE_FUTURES = .00002
        self.TimeCol = 'date' if self.scale == '1D' else 'Time'

        self.nStocksLimit = 0
        self.nStocks_high = 20
        self.day_trades = []
        self.statements = []
        self.stocks = {}
        self.daily_info = {}
        self.unit = 10
        self.balance = 1000000
        self.init_balance = 1000000
        self.buyOrder = None
        self.raiseQuota = False  # 加碼參數
        self.market_value = 0
        self.Action = namedtuple(
            typename="Action",
            field_names=['position', 'reason', 'msg', 'price'],
            defaults=[0, '', '', 0]
        )

    def load_data(self, backtestScript: object):
        print('Loading data...')
        if backtestScript.market == 'Stocks':
            df = self.load_and_merge()
            df = self.preprocess(df)
            if backtestScript.scale == '1D':
                df = merge_pc_ratio(df)
        else:
            kbt = KBarTool()
            df = pd.read_pickle(f'{PATH}/Kbars/futures_data_1T.pkl')
            df = kbt.convert_kbar(df, backtestScript.scale)
            df = merge_pc_ratio(df)

        print('Done')
        return df

    def addFeatures(self, df: pd.DataFrame):
        df = df[df.yClose != 0]
        return df

    def on_addFeatures(self):
        def wrapper(func):
            self.addFeatures = func
        return wrapper

    def selectStocks(self, df: pd.DataFrame):
        '''依照策略選股條件挑出可進場的股票'''
        df['isIn'] = True
        return df

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

    def computeStocksLimit(self, df: pd.DataFrame, **kwargs):
        '''計算每日買進股票上限(可做幾支)'''
        return 2000

    def on_computeStocksLimit(self):
        def wrapper(func):
            self.computeStocksLimit = func
        return wrapper

    def computeOpenUnit(self, inputs: dict, **kwargs):
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

        if hasattr(testScript, 'computeStocksLimit'):
            @self.on_computeStocksLimit()
            def compute_limit(df, chance):
                return testScript.computeStocksLimit(df, chance)

        if hasattr(testScript, 'computeOpenUnit'):
            @self.on_computeOpenUnit()
            def open_unit(inputs):
                return testScript.computeOpenUnit(inputs)

        if hasattr(testScript, 'setVolumeProp'):
            @self.on_setVolumeProp()
            def open_unit(market_value):
                return testScript.setVolumeProp(market_value)

    def filter_data(self, df: pd.DataFrame, **params):
        '''
        過濾要回測的數據，包括起訖日、所需欄位
        參數:
        df - 歷史資料表
        params - 回測參數
        '''

        if 'startDate' in params and params['startDate']:
            df = df[df.date >= params['startDate']]

        if 'endDate' in params and params['endDate']:
            df = df[df.date <= params['endDate']]

        required_cols = [
            'name', 'date', 'isIn', 'PutCallRatio',
            'Open', 'High', 'Low', 'Close', 'Volume'
        ]

        if self.scale != '1D':
            required_cols += ['Time', 'nth_bar']

        if self.Market == 'Stocks':
            required_cols += [
                'TSEopen', 'TSEhigh', 'TSElow', 'TSEclose', 'TSEvolume',
                'OTCopen', 'OTChigh', 'OTClow', 'OTCclose', 'OTCvolume',
            ]

        target_columns = params['target_columns']
        if any(c not in target_columns for c in required_cols):
            cols = [c for c in required_cols if c not in target_columns]
            raise KeyError(
                f'Some required columns are not in the table: {cols}')

        df = df[target_columns]
        df = df[df.Open != 0]
        return df

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
            'stock': kwargs['stockid'],
            'OpenDate': data['day'],
            'OpenReason': data['openReason'],
            'OpenPrice': openPrice,
            'OpenQuantity': quantity,
            'OpenAmount': openamount,
            'OpenFee': openfee,
            'CloseDate': kwargs['day'],
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
        if 'volume_ma' in inputs:
            unit = self.checkOpenUnitLimit(unit, inputs['volume_ma'])

        openInfo = self.examineOpen(
            inputs,
            market_value=self.market_value,
            day_trades=self.day_trades
        )

        if openInfo.price > 0 and unit > 0:

            if inputs['name'] in self.stocks:
                # 加碼部位
                quantity = 1000 * \
                    (self.stocks[inputs['name']]['quantity']/1000)/3
            elif self.Market == 'Stocks':
                quantity = 1000*unit
            else:
                quantity = unit

            amount = self.computeOpenAmount(openInfo.price, quantity)
            fee = self.computeFee(amount)
            if self.balance >= amount+fee and len(self.stocks) < self.nStocksLimit:
                self.execute(
                    trans_type='Open',
                    day=inputs[self.TimeCol],
                    stockid=inputs['name'],
                    price=openInfo.price,
                    quantity=quantity,
                    position=None,
                    amount=amount,
                    cum_max_min=inputs['High'] if self.isLong else inputs['Low'],
                    reason=openInfo.reason,
                    bsh=inputs['High']
                )
                self.day_trades.append(inputs['name'])

    def checkMarginCall(self, name: str, closePrice: float):
        if self.leverage == 1:
            return False

        openPrice = self.stocks[name]['openPrice']
        margin = closePrice/(openPrice*(1 - self.leverage))
        return margin < 1.35

    def checkClose(self, inputs: dict, day: Union[str, datetime], stocksClosed: dict):
        s = inputs['name']
        value = inputs['High'] if self.isLong else inputs['Low']
        cum_max_min = min(self.stocks[s]['cum_max_min'], value)
        self.stocks[s].update({
            'price': inputs['Close'],
            'krun': self.stocks[s]['krun'] + 1,
            'cum_max_min': cum_max_min
        })

        closeInfo = self.examineClose(
            stocks=self.stocks,
            inputs=inputs,
            today=day,
            stocksClosed=stocksClosed
        )

        margin_call = self.checkMarginCall(s, closeInfo.price)
        if closeInfo.position or margin_call:
            self.execute(
                trans_type='Close',
                day=inputs[self.TimeCol],
                stockid=s,
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

        if 'unit' in params:
            self.unit = params['unit']

        if 'init_position' in params:
            self.balance = params['init_position']
            self.init_balance = params['init_position']
            self.market_value = params['init_position']

        if 'buyOrder' in params:
            self.buyOrder = params['buyOrder']

        if 'raiseQuota' in params:
            self.raiseQuota = params['raiseQuota']

    def run(self, df: pd.DataFrame, **params):
        '''
        回測
        參數:
        df - 歷史資料表
        '''

        self.set_params(**params)
        stocksClosed = {}

        t1 = time.time()

        df = self.filter_data(df, **params)

        times = np.sort(df[self.TimeCol].unique())
        N = len(times)
        for i, day in enumerate(times):
            d1 = pd.to_datetime(day).date()
            self.nClose = 0
            self.volume_prop = self.setVolumeProp(self.market_value)

            # 進場順序
            temp = df[df[self.TimeCol] == day].copy()
            temp = self.set_open_order(temp)

            head = temp.head(1).to_dict('records')[0]
            self.daily_info[day] = head

            # 取出當天(或某小時)所有股票資訊
            if self.scale == '1D' or (temp.nth_bar.min() == 1):
                chance = temp.isIn.sum()
                self.nStocksLimit = self.computeStocksLimit(
                    head, chance=chance)
                self.day_trades = []

            temp = temp[(
                (temp.isIn == 1) |
                (temp.name.isin(self.stocks.keys()))
            )]

            # 檢查進場 & 出場
            rows = np.array(temp.to_dict('records'))
            del temp
            for inputs in rows:
                if inputs['name'] in self.stocks:
                    self.checkClose(inputs, d1, stocksClosed)
                elif inputs['isIn']:
                    self.checkOpen(inputs)

            # 更新交易明細數據
            self.daily_info[day].update({
                'chance': chance,
                'n_stock_limit': self.nStocksLimit,
                'n_stocks': len(self.stocks),
                'nClose': self.nClose
            })
            self.updateMarketValue()
            progress_bar(N, i)

        t2 = time.time()
        print(f"\nBacktest time: {round(t2-t1, 2)}s")

        # 清空剩餘庫存
        for name in list(self.stocks):
            self.execute(
                trans_type='Close',
                day=day,
                stockid=name,
                price=self.stocks[name]['price'],
                quantity=None,
                position=100,
                reason='清空庫存'
            )
        self.daily_info[day].update({'nClose': self.nClose})

        params.update({
            'statement': self.statements,
            'startDate': pd.to_datetime(params['startDate']) if 'startDate' in params else df.date.min(),
            'endDate': pd.to_datetime(params['endDate']) if 'endDate' in params else df.date.max(),
            'daily_info': pd.DataFrame(self.daily_info).T,

        })
        del df
        return self.get_backtest_result(
            **params,
            market=self.Market,
            mode=self.mode,
            scale=self.scale,
            leverage=self.leverage,
            isLong=self.isLong,
            multipler=self.multipler
        )
