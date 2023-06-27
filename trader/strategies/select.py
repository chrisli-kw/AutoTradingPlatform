import os
import numpy as np
import pandas as pd
from datetime import timedelta

from ..config import PATH, TODAY_STR, TODAY
from .conditions import SelectConditions
from ..utils import save_table
from ..utils.time import TimeTool
from ..utils.database import db
from ..utils.database.tables import KBarData1D, KBarData1T, KBarData30T, KBarData60T, SelectedStocks


def map_BKD(OTCclose, OTChigh, add_days=10):
    for d in range(80, 0, -10):
        highs = OTChigh[-d:]
        if len(highs) >= d and OTCclose >= max(highs):
            return d + add_days
    return None


class SelectStock(SelectConditions, TimeTool):
    def __init__(self, d_shift=0, dma=5, mode='select', scale='1D'):
        self.d_shift = d_shift
        self.dma = dma
        self.mode = mode
        self.scale = scale
        self.tables = {
            '1D': KBarData1D,
            '1T': KBarData1T,
            '30T': KBarData30T,
            '60T': KBarData60T
        }
        self.categories = {
            1: '水泥工業',
            2: '食品工業',
            3: '塑膠工業',
            4: '紡織纖維',
            5: '電機機械',
            6: '電器電纜',
            21: '化學工業',
            22: '生技醫療業',
            8: '玻璃陶瓷',
            9: '造紙工業',
            10: '鋼鐵工業',
            11: '橡膠工業',
            12: '汽車工業',
            24: '半導體業',
            25: '電腦及週邊設備業',
            26: '光電業',
            27: '通信網路業',
            28: '電子零組件業',
            29: '電子通路業',
            30: '資訊服務業',
            31: '其他電子業',
            14: '建材營造',
            15: '航運業',
            16: '觀光事業',
            17: '金融業',
            18: '貿易百貨',
            23: '油電燃氣業',
            19: '綜合',
            20: '其他',
            32: '文化創意業',
            33: '農業科技業',
            34: '電子商務',
            80: '管理股票'
        }

    def set_select_methods(self, methods):
        self.Preprocess = {
            m: getattr(self, f'preprocess_{m}') for m in methods}
        self.METHODS = {m: getattr(self, f'condition_{m}') for m in methods}

    def load_and_merge(self):
        # TODO: 加入籌碼資料、自選資料
        if db.HAS_DB:
            if self.mode == 'select':
                start = TODAY - timedelta(days=365*2)
                condition = self.tables[self.scale].date >= start
                df = db.query(self.tables[self.scale], condition)
            else:
                df = db.query(self.tables[self.scale])
        else:
            dir_path = f'{PATH}/Kbars/{self.scale}'
            files = os.listdir(dir_path)
            df = pd.DataFrame()
            for f in files:
                tb = pd.read_pickle(f'{dir_path}/{f}')
                df = pd.concat([df, tb])
            df = df.reset_index(drop=True)

        df = df.drop(['date', 'hour', 'minute'], axis=1)
        df = df.drop_duplicates(['name', 'Time'], keep='last')
        df = df.sort_values(['name', 'Time'])
        return df

    def preprocess(self, df: pd.DataFrame):
        df.name = df.name.astype(int).astype(str)
        if self.mode == 'select':
            df = df.groupby('name').tail(365).reset_index(drop=True)

        df.Close = df.Close.replace(0, np.nan).fillna(method='ffill')

        # process OTC data
        ohlcva = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
        otc = df[df.name == '101'].rename(
            columns={c: f'OTC{c.lower()}' for c in ohlcva}).drop(['name'], axis=1)
        otc['Time'] = pd.to_datetime(otc['Time'])
        otc[f'otc_{self.dma}ma'] = otc.OTCclose.shift(
            self.d_shift).rolling(self.dma).mean()
        otc['otcbkd30'] = [window.to_list()
                           for window in otc.OTChigh.rolling(window=80)]
        otc['otcbkd30'] = otc.apply(lambda x: map_BKD(
            x.OTCclose, x.otcbkd30, 30), axis=1)
        otc.otcbkd30 = otc.otcbkd30.fillna(method='ffill')
        
        # process TSE data
        tse = df[df.name == '1'].rename(
            columns={c: f'TSE{c.lower()}' for c in ohlcva}).drop(['name'], axis=1)
        tse['Time'] = pd.to_datetime(tse['Time'])
        tse[f'tse_{self.dma}ma'] = tse.TSEclose.shift(
            self.d_shift).rolling(self.dma).mean()
        
        df = df.merge(otc, how='left', on='Time')
        df = df.merge(tse, how='left', on='Time')

        if self.scale == '1D':
            for col in ['Open', 'High', 'Low', 'Close']:
                df.loc[
                    (df.name == '8070') & (df.Time < '2020-08-17'), col] /= 10
                df.loc[
                    (df.name == '6548') & (df.Time < '2019-09-09'), col] /= 10
                
            group = df.groupby('name')
            df['volume_ma'] = group.Volume.transform(
                lambda x: x.shift(1).rolling(22).mean())
            df['volume_std'] = group.Volume.transform(
                lambda x: x.shift(1).rolling(22).std())
            df['yClose'] = group.Close.transform('shift')
            df['y2Close'] = group.Close.transform(lambda x: x.shift(2))
            df['ma_1_20'] = group.Close.transform(
                lambda x: x.shift(1).rolling(20).mean())

        if self.scale == '1D':
            df = df.rename(columns={'Time':'date'})

        return df

    def pick(self, *args):
        df = self.load_and_merge()
        df = self.preprocess(df)

        for m, func in self.Preprocess.items():
            df = func(df)

        for i, (m, func) in enumerate(self.METHODS.items()):
            df.insert(i+2, m, func(df, *args))

        # insert columns
        stockids = pd.read_excel(f'{PATH}/selections/stock_list.xlsx')
        stockids.code = stockids.code.astype(int).astype(str)
        stockids.category = stockids.category.astype(int)
        df.insert(1, 'company_name', df.name.map(
            stockids.set_index('code').name.to_dict()))
        df.insert(2, 'category', df.name.map(stockids.set_index(
            'code').category.to_dict()).map(self.categories))

        return df

    def melt_table(self, df: pd.DataFrame, columns=[]):
        '''Melt the "strategy" columns into values of a table'''

        if not columns:
            columns = [
                'name', 'company_name', 'category', 'date',
                'Open', 'High', 'Low', 'Close', 'Volume', 'Amount',
            ]
        select_methods = list(self.METHODS)

        df = df[columns + select_methods]
        df = df.melt(
            id_vars=columns,
            value_vars=select_methods,
            var_name='Strategy',
            value_name='isMatch'
        )
        df.Strategy *= df.isMatch
        df = df[df.Strategy != '']

        df = df.reset_index(drop=True).drop('isMatch', axis=1)
        df = df.rename(columns={'name': 'code'})
        df = df.sort_values(['Strategy', 'code'])

        return df

    def export(self, df: pd.DataFrame):
        if db.HAS_DB:
            db.dataframe_to_DB(df, SelectedStocks)
        else:
            save_table(df, f'{PATH}/selections/all.csv')
            save_table(df, f'{PATH}/selections/history/{TODAY_STR}-all.csv')

    def get_selection_files(self):
        '''取得選股清單'''

        filename = f'{PATH}/selections/all.csv'
        day = self.last_business_day()

        if db.HAS_DB:
            df = db.query(SelectedStocks, SelectedStocks.date == day)
        elif os.path.exists(filename):
            df = pd.read_csv(filename)
            df.date = pd.to_datetime(df.date)
            df.code = df.code.astype(str)
            df = df[df.date == day]
        else:
            df = pd.DataFrame()

        return df
