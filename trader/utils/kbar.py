import re
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from datetime import datetime, timedelta

from ..config import (
    API,
    PATH,
    TODAY,
    TODAY_STR,
    TimeStartStock,
    StrategyList
)
from . import get_contract, concat_df
from .time import time_tool
from .objects.data import TradeData
from .. import file_handler
from ..indicators.signals import TechnicalSignals


class KBarTool(TechnicalSignals):
    def __init__(self):
        self.daysdata = self.__set_daysdata()
        self.maps = {
            'name': 'first',
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum',
            'Amount': 'sum'
        }
        self.featureFuncs = {
            '1T': self.add_K1min_feature,
            '2T': self.add_K2min_feature,
            '5T': self.add_K5min_feature,
            '15T': self.add_K15min_feature,
            '30T': self.add_K30min_feature,
            '60T': self.add_K60min_feature,
            '1D': self.add_KDay_feature,
        }
        self.is_kbar_1t_updated = {
            '2T': False,
            '5T': False,
            '15T': False,
            '30T': False,
            '60T': False
        }

    def __set_daysdata(self, kbar_start_day=''):
        '''
        設定觀察K棒數(N個交易日)
        參數 - kbar_start_day: 觀察起始日，格式為 yyyy-mm-dd
        '''

        if not kbar_start_day or TODAY < kbar_start_day:
            return 35

        return max((TODAY - kbar_start_day).days, 35)

    def _apply_feature_by_scale(self, df: pd.DataFrame, scale: str):
        for conf in StrategyList.Config.values():
            if scale in getattr(conf, 'kbarScales', []):
                df = conf.add_features(df)
        return df

    def add_KDay_feature(self, df: pd.DataFrame):
        return self._apply_feature_by_scale(df, '1D')

    def add_K60min_feature(self, df: pd.DataFrame):
        return self._apply_feature_by_scale(df, '60T')

    def add_K30min_feature(self, df: pd.DataFrame):
        return self._apply_feature_by_scale(df, '30T')

    def add_K15min_feature(self, df: pd.DataFrame):
        return self._apply_feature_by_scale(df, '15T')

    def add_K5min_feature(self, df: pd.DataFrame):
        return self._apply_feature_by_scale(df, '5T')

    def add_K2min_feature(self, df: pd.DataFrame):
        return self._apply_feature_by_scale(df, '2T')

    def add_K1min_feature(self, df: pd.DataFrame):
        return self._apply_feature_by_scale(df, '1T')

    def _scale_converter(self, scale: str):
        '''Convert scale format from str to int'''
        return int(re.findall('\d+', scale)[0])

    def tbKBar(self, stockid: str, start: str, end: str = None):
        '''取得k棒資料'''

        if not end:
            end = TODAY_STR

        contract = get_contract(stockid)
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
            td = time_tool._strf_timedelta(TODAY, ndays)
            tb = self.tbKBar(stockid, td)
            for scale in self.featureFuncs:
                kbar = self.convert_kbar(tb, scale)
                if scale == '1D':
                    kbar = kbar[kbar.Time.dt.date.astype(str) != TODAY_STR]
                else:
                    scale_ = self._scale_converter(scale)
                    n = time_tool.count_n_kbars(TimeStartStock, now, scale_)
                    time_ = TimeStartStock + timedelta(minutes=scale_*n)
                    kbar = kbar[kbar.Time < time_]

                TradeData.KBars.Freq[scale] = self.concatKBars(scale, kbar)

        for scale, kbar in TradeData.KBars.Freq.items():
            kbar = self.featureFuncs[scale](kbar)
            TradeData.KBars.Freq[scale] = kbar

    def convert_kbar(self, tb: pd.DataFrame, scale='60T'):
        '''將1分K轉換成其他週期K線資料'''
        if tb.shape[0]:
            return (
                tb.set_index('Time')
                .groupby('name')
                .resample(scale, closed='right', label='right')
                .apply(self.maps)
                .reset_index(level='Time')
                .reset_index(drop=True)
                .dropna()
            )
        return tb

    def revert_dividend_price(self, df: pd.DataFrame):
        '''還原除權息股價'''

        if df.shape[0]:
            dividends = TradeData.Stocks.Dividends
            has_dividend = df.name.isin(dividends.keys())
            if has_dividend.sum():
                _dividends = df[has_dividend].name.map(dividends)
                for col in ['Open', 'High', 'Low', 'Close']:
                    df.loc[has_dividend, col] += _dividends
        return df

    def tick_to_df_targets(self, target: str):
        '''將個股tick資料轉為K棒'''

        q_all = TradeData.Quotes.AllTargets[target]
        q_now = TradeData.Quotes.NowTargets

        if not q_all:
            return pd.DataFrame()

        if None in q_all.values():
            q_all[target].update({
                'Open': q_now[target]['price'],
                'High': q_now[target]['price'],
                'Low': q_now[target]['price'],
                'Close': q_now[target]['price']
            })

        now = datetime.now().replace(microsecond=0)
        tb = pd.DataFrame([q_all])
        tb['Time'] = pd.to_datetime(now)
        tb['name'] = target

        if not tb.shape[0] or tb.shape[1] == 1:
            return pd.DataFrame()

        return tb[TradeData.KBars.kbar_columns]

    def tick_to_df_index(self, target: str):
        '''將指數tick資料轉為K棒'''

        quotes = TradeData.Quotes.AllIndex[target]
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
            tb = tb[TradeData.KBars.kbar_columns]
            return tb.drop_duplicates(['name', 'Time'])
        except:
            return pd.DataFrame()

    def concatKBars(self, freq: str, df2: pd.DataFrame):
        '''合併K棒資料表'''
        df1 = TradeData.KBars.Freq[freq]
        return concat_df(df1, df2, sort_by=['name', 'Time'], reset_index=True)

    def updateKBars(self, scale: str):
        '''檢查並更新K棒資料表'''

        _scale = self._scale_converter(scale)
        now = time_tool.round_time(datetime.now())
        t1 = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        t2 = now - timedelta(minutes=_scale - .5)
        tb = TradeData.KBars.Freq['1T'].tail(
            _scale)[TradeData.KBars.kbar_columns].copy()

        if (
            not self.is_kbar_1t_updated[scale] or
            not TradeData.KBars.Freq[scale][TradeData.KBars.Freq[scale].Time >= t2].empty or
            tb.isnull().values.any()
        ):
            return

        tb = tb[(tb.Time >= t2) & (tb.Time < t1)]
        tb.Time = tb.Time.astype(str).apply(time_tool.round_time)
        if tb.shape[0]:
            logging.debug(
                f'Update {scale} kbar data| from {tb.Time.min()} to {tb.Time.max()}')
            tb = self.convert_kbar(tb, scale=scale)
            kbar = self.concatKBars(scale, tb)
            TradeData.KBars.Freq[scale] = self.featureFuncs[scale](kbar)
            self.is_kbar_1t_updated[scale] = False

    def _update_K1(self, target, quote_type='Securities'):
        '''每隔1分鐘更新1分K'''

        now = datetime.now()
        if quote_type == 'Index':
            # if TimeStartStock <= now <= TimeEndStock:
            df = self.tick_to_df_index(target)
        else:
            df = self.tick_to_df_targets(target)
            df = self.revert_dividend_price(df)
        df = self.concatKBars('1T', df)
        TradeData.KBars.Freq['1T'] = self.featureFuncs['1T'](df)

        now = time_tool.round_time(now)
        for freq in [2, 5, 15, 30, 60]:
            if now.minute % freq == 0:
                self.is_kbar_1t_updated[f'{freq}T'] = True


class TickDataProcesser:
    '''轉換期貨逐筆交易'''

    def filter_file_dates(self, market: str, **kwargs):
        '''
        取得逐筆交易明細表的日期清單。以year為主，取得該年度的資料日期，可另外指定要合併的區間
        '''

        dir_path = f'{PATH}/ticks/{market.lower()}'
        files = file_handler.Operate.list_files(dir_path, pattern='.csv')

        # Filter files by time interval
        start = kwargs.get('start')
        if start:
            if not isinstance(start, pd.Timestamp):
                start = pd.to_datetime(start)
        else:
            start = pd.to_datetime('1970-01-01')

        end = kwargs.get('end')
        if end:
            if not isinstance(end, pd.Timestamp):
                end = pd.to_datetime(end)
        else:
            end = TODAY

        dates = []
        for f in files:
            date = f.split('Daily_')[-1][:-4]
            if start <= datetime(*(int(d) for d in date.split('_')[-3:])) <= end:
                dates.append(date.replace('Daily_', '').replace('_', '-'))

        return dates

    def convert_daily_tick(self, date: str, scale: str):
        ymd = date.split('-')
        m = f'Daily_{ymd[0]}_{ymd[1]}'
        folder = f'{PATH}/ticks/futures/{ymd[0]}/{m}'
        df = file_handler.Process.read_table(f'{folder}/{m}_{ymd[2]}.csv')
        if df.shape[0]:
            df = self.preprocess_futures_tick(df)
            df = self.tick_2_kbar(df, scale, period='all')
        return df

    def convert_period_tick(self, start: str, end: str, scale='1T'):
        dates = self.filter_file_dates('Futures', start=start, end=end)
        dates = sorted(dates)

        df = np.array([None]*len(dates))
        iterator = tqdm(dates)
        for i, date in enumerate(iterator):
            temp = self.convert_daily_tick(date, scale=scale)
            df[i] = temp
            iterator.set_description(f'[{date}]')
        df = pd.concat(df)
        df = df.sort_values('Time').drop_duplicates('Time')
        return df

    def add_period(self, df: pd.DataFrame):
        '''period: 日盤(1)/夜盤(2)/隔日夜盤(3, 凌晨0:00後)'''

        if isinstance(df, pd.DataFrame):
            df['period'] = 2
            df.loc[df.Time.dt.hour.isin(range(8, 14)), 'period'] = 1
            df.loc[df.Time.dt.hour.isin(range(0, 8)), 'period'] = 3
        elif isinstance(df, dict):
            hour = df['datetime'].hour
            if hour in range(8, 14):
                df['period'] = 1
            elif hour in range(0, 8):
                df['period'] = 3
            else:
                df['period'] = 2
        return df

    def preprocess_futures_tick(self, df, underlying='TX'):
        df = df.rename(columns={
            '商品代號': 'name',
            '成交價格': 'Price',
            '成交數量(B+S)': 'Quantity',
            '開盤集合競價 ': 'Simtrade',
            '開盤集合競價': 'Simtrade',
            '近月價格': 'PriceOld',
            '遠月價格': 'PriceNew',
            '到期月份(週別)': 'DueMonth'
        })

        df.name = df.name.str.replace(' ', '')
        if underlying != 'all':
            df = df[df.name == underlying].reset_index(drop=True)

        df = df[df.成交時間.notnull()]
        df.PriceOld = df.PriceOld.replace('-', 0).astype(float)
        df.PriceNew = df.PriceNew.replace('-', 0).astype(float)

        date_name = '成交日期' if '成交日期' in df.columns else '交易日期'
        df['Time'] = pd.to_datetime(
            df[date_name].astype(
                str) + df.成交時間.astype(int).astype(str).str.zfill(6),
            format="%Y%m%d%H%M%S"
        )
        df.Simtrade = df.Simtrade == '*'
        df.DueMonth = df.DueMonth.apply(
            lambda x: x.replace(' ', '').split('/'))
        df['DueMonthOld'] = df.DueMonth.str[0]
        df['DueMonthNew'] = df.DueMonth.str[-1]
        df = self.add_period(df)

        df = df.drop([date_name, '成交時間', 'DueMonth'], axis=1)
        df = df.sort_values('Time').reset_index(drop=True)
        return df

    def expand_columns(self, df: pd.DataFrame):
        df['Open'] = df['High'] = df['Low'] = df['Close'] = df.Price
        df['Volume'] = df['Quantity']/2
        df['Amount'] = df.Close*df.Volume
        return df

    def tick_2_kbar(self, df, scale, period='day_only'):
        '''
        將逐筆資料轉為K線資料(近月)。
        period = day_only(日盤), night_only(夜盤), all(日盤+夜盤)
        '''

        # 留下近月交割 & 非時間價差交易
        df['due'] = df.Time.apply(
            lambda x: time_tool.GetDueMonth(x, is_backtest=True))
        df = df[
            (df.Simtrade == False) &
            (df.DueMonthOld == df.DueMonthNew) &
            (df.DueMonthOld == df.due)
        ]
        if period == 'day_only':
            df = df[df.period == 1]

        if scale != 'tick':
            df = self.expand_columns(df.copy())
            df = KBarTool().convert_kbar(df, scale=scale)

        return df
