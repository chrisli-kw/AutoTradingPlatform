import re
import logging
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

from ..config import API, PATH, TODAY, TODAY_STR, TimeStartStock, TimeEndStock
from ..config import KbarFeatures
from ..indicators.signals import TechnicalSignals
from . import get_contract
from .time import TimeTool
from .file import FileHandler
try:
    from ..scripts.features import KBarFeatureTool
except:
    logging.warning('Cannot import KBar Scripts from package.')
    KBarFeatureTool = None


class KBarTool(TechnicalSignals, TimeTool, FileHandler):
    def __init__(self, kbar_start_day=''):
        self.set_kbar_scripts(KBarFeatureTool)
        self.daysdata = self.__set_daysdata(kbar_start_day)
        self.maps = {
            'name': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Amount': 'sum'
        }
        self.kbar_columns = [
            'name', 'Time',
            'Open', 'High', 'Low', 'Close', 'Volume', 'Amount'
        ]
        self.featureFuncs = {
            '1T': self.add_K1min_feature,
            '2T': self.add_K2min_feature,
            '5T': self.add_K5min_feature,
            '15T': self.add_K15min_feature,
            '30T': self.add_K30min_feature,
            '60T': self.add_K60min_feature,
            '1D': self.add_KDay_feature,
        }
        self.KBars = {
            freq: pd.DataFrame(columns=self.kbar_columns) for freq in self.featureFuncs
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

    def set_kbar_scripts(self, kbar_scripts: object = None):
        '''設定K線特徵腳本'''

        if kbar_scripts:
            kbar_scripts = kbar_scripts()
            if hasattr(kbar_scripts, 'add_KDay_feature'):
                @self.on_add_KDay_feature()
                def _add_KDay_feature(df):
                    return kbar_scripts.add_KDay_feature(df)

            if hasattr(kbar_scripts, 'add_K1min_feature'):
                @self.on_add_K1min_feature()
                def _add_K1min_feature(df):
                    return kbar_scripts.add_K1min_feature(df)

            if hasattr(kbar_scripts, 'add_K2min_feature'):
                @self.on_add_K2min_feature()
                def _add_K2min_feature(df):
                    return kbar_scripts.add_K2min_feature(df)

            if hasattr(kbar_scripts, 'add_K5min_feature'):
                @self.on_add_K5min_feature()
                def _add_K5min_feature(df):
                    return kbar_scripts.add_K5min_feature(df)

            if hasattr(kbar_scripts, 'add_K15min_feature'):
                @self.on_add_K15min_feature()
                def _add_K15min_feature(df):
                    return kbar_scripts.add_K15min_feature(df)

            if hasattr(kbar_scripts, 'add_K30min_feature'):
                @self.on_add_K30min_feature()
                def _add_K30min_feature(df):
                    return kbar_scripts.add_K30min_feature(df)

            if hasattr(kbar_scripts, 'add_K60min_feature'):
                @self.on_add_K60min_feature()
                def _add_K60min_feature(df):
                    return kbar_scripts.add_K60min_feature(df)

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

    def add_K5min_feature(self, df: pd.DataFrame):
        return df

    def on_add_K5min_feature(self):
        def wrapper(func):
            self.add_K5min_feature = func
        return wrapper

    def add_K2min_feature(self, df: pd.DataFrame):
        return df

    def on_add_K2min_feature(self):
        def wrapper(func):
            self.add_K2min_feature = func
        return wrapper

    def add_K1min_feature(self, df: pd.DataFrame):
        return df

    def on_add_K1min_feature(self):
        def wrapper(func):
            self.add_K1min_feature = func
        return wrapper

    def _scale_converter(self, scale: str):
        '''Convert scale format from str to int'''
        return int(re.findall('\d+', scale)[0])

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
            kbars = {'ts': []}

        tb = pd.DataFrame({**kbars})
        tb.ts = pd.to_datetime(tb.ts)
        tb.insert(0, 'name', stockid)
        tb.name = tb.name.replace('OTC101', '101').replace('TSE001', '001')
        tb = tb.rename(columns={'ts': 'Time'})
        return tb

    def history_kbars(self, stockids: List[str], daysdata: int = 0):
        '''Get history kbar data'''

        now = datetime.now()
        ndays = daysdata if daysdata else self.daysdata
        for stockid in stockids:
            tb = self.tbKBar(stockid, self._strf_timedelta(TODAY, ndays))
            for scale in self.featureFuncs:
                kbar = self.convert_kbar(tb, scale)
                if scale == '1D':
                    kbar = kbar[kbar.Time.dt.date.astype(str) != TODAY_STR]
                else:
                    scale_ = self._scale_converter(scale)
                    n = self.count_n_kbars(TimeStartStock, now, scale_)
                    time_ = TimeStartStock + timedelta(minutes=scale_*n)
                    kbar = kbar[kbar.Time < time_]

                self.KBars[scale] = self.concatKBars(self.KBars[scale], kbar)

        for scale, kbar in self.KBars.items():
            kbar = self.featureFuncs[scale](kbar)
            self.KBars[scale] = kbar

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

        tb['Time'] = pd.to_datetime(datetime.now())
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
            tb['Open'] = tb.Close.values[0]
            tb['High'] = tb.Close.max()
            tb['Low'] = tb.Close.min()
            tb['Close'] = tb.Close.values[-1]
            tb.Volume = tb.Volume.sum()
            tb['Amount'] = tb.Amount.sum()
            tb = tb[self.kbar_columns]
            return tb.drop_duplicates(['name', 'Time'])
        except:
            return pd.DataFrame()

    def concatKBars(self, df1: pd.DataFrame, df2: pd.DataFrame):
        '''合併K棒資料表'''
        return pd.concat([df1, df2]).sort_values(['name', 'Time']).reset_index(drop=True)

    def updateKBars(self, scale: str):
        '''檢查並更新K棒資料表'''

        _scale = self._scale_converter(scale)
        t1 = datetime.now()
        t2 = t1 - timedelta(minutes=_scale + .5)
        if not self.KBars[scale][self.KBars[scale].Time >= t2].shape[0]:
            tb = self.KBars['1T'].copy()
            tb = tb[(tb.Time >= t2) & (tb.Time < t1)]
            if tb.shape[0]:
                tb = self.convert_kbar(tb, scale=scale)

                for col in KbarFeatures[scale]:
                    tb[col] = None

                kbar = self.concatKBars(self.KBars[scale], tb)
                self.KBars[scale] = self.featureFuncs[scale](kbar)

    def _update_K1(self, dividends: dict, quotes):
        '''每隔1分鐘更新1分K'''

        def concat_df(df):
            if df.shape[0]:
                self.KBars['1T'] = pd.concat(
                    [self.KBars['1T'], df]).sort_values(['name', 'Time']).reset_index(drop=True)

        if TimeStartStock <= datetime.now() <= TimeEndStock:
            for i in quotes.AllIndex:
                tb = self.tick_to_df_index(quotes.AllIndex[i])
                concat_df(tb)

        df = self.tick_to_df_targets(quotes.AllTargets, quotes.NowTargets)
        df = self.revert_dividend_price(df, dividends)
        concat_df(df)


class TickDataProcesser(TimeTool, FileHandler):
    '''轉換期貨逐筆交易'''

    def convert_daily_tick(self, date: str, scale: str):
        ymd = date.split('-')
        m = f'Daily_{ymd[0]}_{ymd[1]}'
        folder = f'{PATH}/ticks/futures/{ymd[0]}/{m}'
        df = self.read_table(f'{folder}/{m}_{ymd[2]}.csv')
        if df.shape[0]:
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
            Months = self.listdir(path, pattern='Daily')
        else:
            Months = [f'Daily_{year}_{str(m).zfill(2)}' for m in months]

        df = []
        for m in Months:
            temp = self.read_tables_in_folder(f'{path}/{m}', pattern='Daily')
            df.append(temp)
        df = pd.concat(df)
        df = df.reset_index(drop=True)
        return df

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
