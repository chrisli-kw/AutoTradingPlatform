import os
import re
import logging
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

from ..config import API, PATH, TODAY, TODAY_STR
from . import progress_bar, get_contract, save_table
from .time import TimeTool
from ..indicators.signals import TechnicalSignals
from ..config import TimeStartStock, TimeEndStock, K15min_feature, K30min_feature, K60min_feature


class KBarTool(TechnicalSignals, TimeTool):
    def __init__(self, kbar_start_day=''):
        self.daysdata = self.__set_daysdata(kbar_start_day)
        self.maps = {
            'name': 'first',
            'date': 'first',
            'hour': 'first',
            'minute': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Amount': 'sum'
        }
        self.kbar_columns = [
            'name', 'date', 'Time', 'hour', 'minute', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        self.KBars = {
            freq: pd.DataFrame(columns=self.kbar_columns) for freq in ['1D', '60T', '30T', '15T', '5T', '1T']
        }

    def __set_daysdata(self, kbar_start_day):
        '''
        設定觀察K棒數(N個交易日)
        若有設定最近崩盤日, 則觀察K棒數 = 上次崩盤日開始算N個交易日
        若沒有設最近崩盤日, 則觀察K棒數 = 35
        參數 - kbar_start_day: 觀察起始日，格式為 yyyy-mm-dd
        '''

        if not kbar_start_day or TODAY < kbar_start_day:
            return 35

        return max((TODAY - kbar_start_day).days, 35)

    def set_kbar_scripts(self, featureScript: object):
        '''設定K線特徵腳本'''

        if featureScript:
            if hasattr(featureScript, 'add_KDay_feature'):
                @self.on_add_KDay_feature()
                def _add_KDay_feature(df):
                    return featureScript.add_KDay_feature(df)

            if hasattr(featureScript, 'add_K15min_feature'):
                @self.on_add_K15min_feature()
                def _add_K15min_feature(df):
                    return featureScript.add_K15min_feature(df)

            if hasattr(featureScript, 'add_K30min_feature'):
                @self.on_add_K30min_feature()
                def _add_K30min_feature(df):
                    return featureScript.add_K30min_feature(df)

            if hasattr(featureScript, 'add_K60min_feature'):
                @self.on_add_K60min_feature()
                def _add_K60min_feature(df):
                    return featureScript.add_K60min_feature(df)

    def load_kbar_file(self, filename: str):
        '''讀取k棒歷史資料csv'''

        try:
            df = pd.read_csv(f'{PATH}/{filename}')
            df.Time = pd.to_datetime(df.Time)
            df.name = df.name.astype(str)
            return df
        except:
            return pd.DataFrame(columns=self.kbar_columns)

    def tbKBar(self, stockid: str, start: str, end: str = None):
        '''取得k棒資料'''

        if not end:
            end = TODAY_STR

        contract = get_contract(stockid)
        # TODO: fix AttributeError
        try:
            kbars = API.kbars(contract, start=start, end=end, timeout=60000)
        except AttributeError:
            logging.exception(f'tbKBar({stockid}) Catch an Exception:')

        tb = pd.DataFrame({**kbars})
        tb.ts = pd.to_datetime(tb.ts)
        tb.insert(0, 'name', stockid)
        tb.insert(1, 'date', tb.ts.apply(self.datetime_to_str))
        tb.name = tb.name.replace('OTC101', '101').replace('TSE001', '001')
        tb['hour'] = tb.ts.dt.hour
        tb['minute'] = tb.ts.dt.minute
        tb = tb.rename(columns={'ts': 'Time'})
        return tb

    def get_kbars(self, stockid: str, daysdata: int):
        '''取得N日的日K資料'''
        tb = self.tbKBar(stockid, self._strf_timedelta(TODAY, daysdata))

        # 日K
        tb1 = self.convert_kbar(tb, '1D')[self.kbar_columns].dropna()

        # 60分K
        tb2 = self.convert_kbar(tb, '60T')[self.kbar_columns].dropna()

        # 30分K
        tb3 = self.convert_kbar(tb, '30T')[self.kbar_columns].dropna()

        # 15分K
        tb4 = self.convert_kbar(tb, '15T')[self.kbar_columns].dropna()

        return tb1, tb2, tb3, tb4

    def add_KDay_feature(self, df: pd.DataFrame):
        return df

    def on_add_KDay_feature(self):
        def wrapper(func):
            self.add_KDay_feature = func
        return wrapper

    def add_K60min_feature(self, df: pd.DataFrame):
        return df

    def on_add_K60min_feature(self):
        def wrapper(func):
            self.add_K60min_feature = func
        return wrapper

    def add_K30min_feature(self, df: pd.DataFrame):
        return df

    def on_add_K30min_feature(self):
        def wrapper(func):
            self.add_K30min_feature = func
        return wrapper

    def add_K15min_feature(self, df: pd.DataFrame):
        return df

    def on_add_K15min_feature(self):
        def wrapper(func):
            self.add_K15min_feature = func
        return wrapper

    def history_kbars(self, stockids: List[str], daysdata: int = 0):
        '''取得歷史k棒資料(1D/60T/30T/15T)'''
        ndays = daysdata if daysdata else self.daysdata
        KDay = self.KBars['1D'].copy()
        K60min = self.KBars['60T'].copy()
        K30min = self.KBars['30T'].copy()
        K15min = self.KBars['15T'].copy()
        for stockid in stockids:
            tb1, tb2, tb3, tb4 = self.get_kbars(stockid, ndays)
            KDay = pd.concat([KDay, tb1]).reset_index(drop=True)
            K60min = pd.concat([K60min, tb2]).reset_index(drop=True)
            K30min = pd.concat([K30min, tb3]).reset_index(drop=True)
            K15min = pd.concat([K15min, tb4]).reset_index(drop=True)

        if KDay.shape[0]:
            KDay = self.add_KDay_feature(KDay)
            self.KBars['1D'] = KDay[KDay.date != TODAY_STR]

        if K60min.shape[0]:
            K60min = self.add_K60min_feature(K60min)

            now = datetime.now()
            current_time = pd.to_datetime(
                TODAY_STR) + timedelta(hours=now.hour)
            K60min = K60min[(K60min.Time < current_time)]
            self.KBars['60T'] = K60min

        if K30min.shape[0]:
            self.KBars['30T'] = self.add_K30min_feature(K30min)

        if K15min.shape[0]:
            self.KBars['15T'] = self.add_K15min_feature(K15min)

    def convert_kbar(self, tb: pd.DataFrame, scale='60T'):
        '''將1分K轉換成其他週期K線資料'''
        if tb.shape[0]:
            return (
                tb.set_index('Time')
                .groupby('name')
                .resample(scale, closed='left', label='left')
                .apply(self.maps)
                .reset_index(level='Time')
                .reset_index(drop=True)
                .dropna()
            )
        return tb

    def revert_dividend_price(self, df: pd.DataFrame, dividends: Dict[str, float]):
        '''還原除權息股價'''

        if df.shape[0]:
            has_dividend = df.name.isin(dividends.keys())
            if has_dividend.sum():
                _dividends = df[has_dividend].name.map(dividends)
                for col in ['Open', 'High', 'Low', 'Close']:
                    df.loc[has_dividend, col] += _dividends
        return df

    def tick_to_df_targets(self, q_all: dict, q_now: dict):
        '''將個股tick資料轉為K棒'''

        if not q_all and not q_now:
            return pd.DataFrame()

        if any([] in q_all[s].values() for s in q_all):
            q_all = {s: q_all[s] for s in q_all if [] not in q_all[s].values()}
            q_all.update(
                {s: {k: [v] for k, v in q_now[s].items()} for s in q_now if s not in q_all})

        tb = pd.DataFrame(q_all).T

        if not tb.shape[1]:
            return pd.DataFrame()

        tb['date'] = TODAY_STR
        tb['Time'] = pd.to_datetime(datetime.now())
        tb['hour'] = tb.Time.dt.hour
        tb['minute'] = tb.Time.dt.minute
        tb['Open'] = tb.price.apply(lambda x: x[0])
        tb['High'] = tb.price.apply(max)
        tb['Low'] = tb.price.apply(min)
        tb['Close'] = tb.price.apply(lambda x: x[-1])
        tb.volume = tb.volume.apply(sum)
        tb['Amount'] = tb.amount.apply(lambda x: x[-1])
        tb = tb.reset_index().rename(
            columns={'index': 'name', 'volume': 'Volume'})
        return tb[self.kbar_columns]

    def tick_to_df_index(self, quotes: list):
        '''將指數tick資料轉為K棒'''

        if not quotes:
            return pd.DataFrame()

        try:
            tb = pd.DataFrame(quotes)
            tb = tb.rename(columns={'Code': 'name', 'Date': 'date'})

            tb['Time'] = pd.to_datetime(datetime.now())
            tb['hour'] = tb.Time.dt.hour
            tb['minute'] = tb.Time.dt.minute
            tb['Open'] = tb.Close.values[0]
            tb['High'] = tb.Close.max()
            tb['Low'] = tb.Close.min()
            tb['Close'] = tb.Close.values[-1]
            tb.Volume = tb.Volume.sum()
            tb['Amount'] = tb.Amount.sum()
            tb = tb[self.kbar_columns]
            return tb.drop_duplicates(['name', 'date', 'hour', 'minute'])
        except:
            return pd.DataFrame()

    def _concat_kbar(self, df1: pd.DataFrame, df2: pd.DataFrame):
        '''合併K棒資料表'''
        return pd.concat([df1, df2]).sort_values(['name', 'date', 'hour', 'minute']).reset_index(drop=True)

    def _update_kbar(self, scale: str):
        '''檢查並更新K棒資料表'''

        _scale = int(re.findall('\d+', scale)[0])
        t1 = datetime.now()
        t2 = t1 - timedelta(minutes=_scale + .5)
        if not self.KBars[scale][self.KBars[scale].Time >= t2].shape[0]:
            tb = self.KBars['1T'].copy()
            tb = tb[(tb.Time >= t2) & (tb.Time < t1)]
            if tb.shape[0]:
                return self.convert_kbar(tb, scale=scale)

    def _update_K1(self, dividends: dict, quotes):
        '''每隔1分鐘更新1分K'''

        def concat_df(df):
            if df.shape[0]:
                self.KBars['1T'] = pd.concat(
                    [self.KBars['1T'], df]).sort_values(['name', 'date']).reset_index(drop=True)

        if TimeStartStock <= datetime.now() <= TimeEndStock:
            for i in quotes.AllIndex:
                tb = self.tick_to_df_index(quotes.AllIndex[i])
                concat_df(tb)

        df = self.tick_to_df_targets(quotes.AllTargets, quotes.NowTargets)
        df = self.revert_dividend_price(df, dividends)
        concat_df(df)

    def _update_K5(self):
        '''每隔5分鐘更新5分K'''

        tb = self._update_kbar('5T')
        if isinstance(tb, pd.DataFrame) and tb.shape[0]:
            self.KBars['5T'] = self._concat_kbar(self.KBars['5T'], tb)

    def _update_K15(self):
        '''每隔15分鐘更新15分K'''

        tb = self._update_kbar('15T')
        if isinstance(tb, pd.DataFrame) and tb.shape[0]:
            for col in K15min_feature:
                tb[col] = None

            K15min = self._concat_kbar(self.KBars['15T'], tb)
            self.KBars['15T'] = self.add_K15min_feature(K15min)

    def _update_K30(self):
        '''每隔30分鐘更新30分K'''

        tb = self._update_kbar('30T')
        if isinstance(tb, pd.DataFrame) and tb.shape[0]:
            for col in K30min_feature:
                tb[col] = None

            K30min = self._concat_kbar(self.KBars['30T'], tb)
            self.KBars['30T'] = self.add_K30min_feature(K30min)

    def _update_K60(self, hour: int):
        '''每隔60分鐘更新60分K'''

        K60min = self.KBars['60T'].copy()
        if not K60min[(K60min.date == TODAY_STR) & (K60min.hour == hour)].shape[0]:
            tb = self.KBars['1T'][self.KBars['1T'].hour == hour].copy()

            if tb.shape[0]:
                tb = self.convert_kbar(tb, scale='60T')

                for col in K60min_feature:
                    tb[col] = None

                K60min = self._concat_kbar(K60min, tb[K60min.columns])
                self.KBars['60T'] = self.add_K60min_feature(K60min)

    def update_today_previous_kbar(self, stockids: List[str], dividends: dict = {}):
        '''盤中執行程式時，更新當天啟動前的資料'''
        if isinstance(stockids, str):
            stockids = stockids.replace(' ', '').split(',')
        elif isinstance(stockids, np.ndarray):
            stockids = stockids.tolist()

        if 'OTC101' not in stockids:
            stockids += ['OTC101']

        now = datetime.now()
        t1 = now-timedelta(minutes=1)
        t2 = now-timedelta(minutes=(now.minute % 5))

        for s in stockids:
            df = self.tbKBar(s, TODAY_STR)
            df = self.revert_dividend_price(df, dividends)

            k5 = df[df.Time <= t2]
            k5 = self.convert_kbar(k5, '5T')[self.kbar_columns]

            self.KBars['5T'] = self._concat_kbar(self.KBars['5T'], k5)
            self.KBars['1T'] = pd.concat([
                self.KBars['1T'], df[df.Time <= t1][self.kbar_columns]
            ]).sort_values(['name', 'date']).reset_index(drop=True)

    def save_OTC_5k(self):
        filename = f'{PATH}/Kbars/OTC/{TODAY_STR}_OTC_5k.csv'
        otc = self.KBars['5T'][self.KBars['5T'].name == '101'].copy()
        save_table(otc, filename)


class TickDataProcesser(TimeTool):
    '''轉換期貨逐筆交易'''

    def convert_daily_tick(self, date: str, scale: str):
        ymd = date.split('-')
        m = f'Daily_{ymd[0]}_{ymd[1]}'
        folder = f'{PATH}/ticks/futures/{ymd[0]}/{m}'
        filename = f'{m}_{ymd[2]}.csv'

        if filename not in os.listdir(folder):
            return pd.DataFrame()

        filename = f'{folder}/{filename}'
        try:
            df = pd.read_csv(filename, low_memory=False)
        except:
            df = pd.read_csv(filename, low_memory=False, encoding='big5')
        df = self.preprocess_futures_tick(df)
        df = self.convert_tick_2_kbar(df, scale, period='all')
        return df

    def merge_futures_tick_data(self, year: list, months: list = None):
        '''
        合併期貨逐筆交易明細表
        以year為主，合併該年度的資料，可另外指定要合併的月份(該年度)
        '''

        path = f'{PATH}/ticks/futures/{year}'

        if not months:
            Months = os.listdir(path)
            Months = [m for m in Months if 'Daily' in m]
        else:
            Months = [f'Daily_{year}_{str(m).zfill(2)}' for m in months]

        df = []
        for m in Months:
            files = os.listdir(f'{path}/{m}')
            files = [f for f in files if 'Daily' in f]

            N = len(files)
            for i, f in enumerate(files):
                filename = f'{path}/{m}/{f}'
                try:
                    temp = pd.read_csv(filename, low_memory=False)
                except:
                    temp = pd.read_csv(
                        filename, low_memory=False, encoding='big5')
                df.append(temp)
                progress_bar(N, i, status=f'[{f}]')

        return pd.concat(df)

    def preprocess_futures_tick(self, df, underlying='TX'):
        df.商品代號 = df.商品代號.apply(lambda x: x.replace(' ', ''))
        df['到期月份(週別)'] = df['到期月份(週別)'].apply(lambda x: x.replace(' ', ''))
        df.近月價格 = df.近月價格.replace('-', 0).astype(float)
        df.遠月價格 = df.遠月價格.replace('-', 0).astype(float)

        if underlying != 'all':
            df = df[df.商品代號 == underlying].reset_index()

        df['Time'] = pd.to_datetime(df.成交日期.astype(
            str) + df.成交時間.astype(str).str.zfill(6))
        df['date'] = pd.to_datetime(df.Time.dt.date)
        df['hour'] = df.Time.dt.hour
        df['minute'] = df.Time.dt.minute

        df['due_month'] = df.date.apply(self.GetDueMonth)

        df['is_spread'] = df['到期月份(週別)'].apply(lambda x: 1 if '/' in x else 0)
        df['period'] = 2
        df.loc[df.hour.isin(range(8, 14)), 'period'] = 1

        # 處理跨月委託交易(轉倉)
        # df['到期月份(週別)'] = df['到期月份(週別)'].apply(lambda x: x.split('/'))
        # df.成交價格 += df.近月價格
        # df = df.explode(['到期月份(週別)'])

        df = df.rename(columns={'商品代號': 'name'})
        return df

    def convert_tick_2_kbar(self, df, scale, period='day_only'):
        '''將逐筆資料轉為K線資料。period = day_only(日盤), night_only(夜盤), all(日盤+夜盤)'''
        df['Open'] = df['High'] = df['Low'] = df['Close'] = df.成交價格
        df['Volume'] = df['成交數量(B+S)']/2
        df['Amount'] = df.Close*df.Volume

        df = df[(df.is_spread == False) & (df['到期月份(週別)'] == df.due_month)]
        if period == 'day_only':
            df = df[df.period == 1]

        df = KBarTool().convert_kbar(df, scale=scale)
        return df
