import logging
import numpy as np
import pandas as pd
from datetime import timedelta

from ..config import PATH, TODAY_STR, TODAY, SelectMethods
from .time import TimeTool
from .file import file_handler
from .crawler import readStockList
from .database import db, KBarTables
from .database.tables import SelectedStocks
try:
    from ..scripts.conditions import SelectConditions
except:
    logging.warning('Cannot import select scripts from package.')
    SelectConditions = None
try:
    from ..scripts.features import FeaturesSelect
except:
    logging.warning('Cannot import feature scripts from package.')
    FeaturesSelect = None


class SelectStock(TimeTool):
    def __init__(self, scale='1D'):
        self.set_select_scripts(SelectConditions, FeaturesSelect)
        self.scale = scale
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

    def set_select_scripts(self, select_scripts: object = None, feature_scripts: object = None):
        '''Set preprocess & stock selection scripts'''

        if select_scripts and feature_scripts:
            feature_scripts = feature_scripts()
            self.Preprocess = {
                m: getattr(feature_scripts, f'preprocess_{m}') for m in SelectMethods}

            if hasattr(feature_scripts, 'preprocess_common'):
                self.Preprocess.update({
                    'preprocess_common': feature_scripts.preprocess_common})

            select_scripts = select_scripts()
            self.METHODS = {
                m: getattr(select_scripts, f'condition_{m}') for m in SelectMethods}
        else:
            self.Preprocess = {}
            self.METHODS = {}

    def load_and_merge(self, targets):
        if db.HAS_DB:
            start = TODAY - timedelta(days=365*2)
            condition1 = KBarTables[self.scale].Time >= start
            condition2 = KBarTables[self.scale].name.in_(targets)
            df = db.query(KBarTables[self.scale], condition1, condition2)
        else:
            dir_path = f'{PATH}/Kbars/{self.scale}'
            df = file_handler.read_tables_in_folder(dir_path, pattern='stocks')

        df = df.drop_duplicates(['name', 'Time'], keep='last')
        df = df.sort_values(['name', 'Time'])
        return df

    def preprocess(self, df: pd.DataFrame):
        df = df.groupby('name').tail(365).reset_index(drop=True)
        df.name = df.name.astype(int).astype(str)
        df.Close = df.Close.replace(0, np.nan).ffill()

        for col in ['Open', 'High', 'Low', 'Close']:
            df.loc[
                (df.name == '8070') & (df.Time < '2020-08-17'), col] /= 10
            df.loc[
                (df.name == '6548') & (df.Time < '2019-09-09'), col] /= 10

        if 'preprocess_common' in self.Preprocess:
            df = self.Preprocess['preprocess_common'](df)

        return df

    def pick(self, *args):
        stockids = readStockList()
        df = self.load_and_merge(stockids.code.tolist() + ['1', '101'])
        df = self.preprocess(df)

        for m, func in self.Preprocess.items():
            if m != 'preprocess_common':
                df = func(df)

        for i, (m, func) in enumerate(self.METHODS.items()):
            df.insert(i+2, m, func(df, *args))

        # insert columns
        stockids.category = stockids.category.astype(int)
        stockids = stockids.set_index('code')

        df.insert(1, 'company_name', df.name.map(stockids.name.to_dict()))
        df.insert(2, 'category', df.name.map(
            stockids.category.to_dict()).map(self.categories))

        return df

    def melt_table(self, df: pd.DataFrame, columns=[]):
        '''Melt the "strategy" columns into values of a table'''

        if not columns:
            columns = [
                'name', 'company_name', 'category', 'Time',
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
            file_handler.Process.save_table(df, f'{PATH}/selections/all.csv')
            file_handler.Process.save_table(
                df, f'{PATH}/selections/history/{TODAY_STR}-all.csv')

    def get_selection_files(self):
        '''取得選股清單'''

        day = self.last_business_day()

        if db.HAS_DB:
            df = db.query(SelectedStocks, SelectedStocks.Time == day)
        else:
            df = file_handler.Process.read_table(
                filename=f'{PATH}/selections/all.csv',
                df_default=pd.DataFrame(columns=[
                    'code', 'company_name', 'category', 'Time',
                    'Open', 'High', 'Low', 'Close',
                    'Volume', 'Amount', 'Strategy'
                ])
            )
            df.Time = pd.to_datetime(df.Time)
            df.code = df.code.astype(str)
            df = df[df.Time == day]

        return df
