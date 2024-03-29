import os
import numpy as np
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime

from . import progress_bar, concat_df
from ..config import PATH, TODAY


class FileHandler:
    def create_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def listdir(self, dir_path: str, pattern: str = '', filter_out: list = ['desktop.ini']):
        if pattern:
            result = [f for f in os.listdir(dir_path) if pattern in f]
        else:
            result = os.listdir(dir_path)
        return [f for f in result if f not in filter_out]

    def list_files(self, dir_path: str, pattern: str = '', filter_out: list = ['desktop.ini']):
        file_list = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if pattern in file and file not in filter_out:
                    file_list.append(os.path.join(root, file))
        return file_list

    def is_in_dir(self, filename: str, dir_path: str):
        '''Check if a filename is in dir_path.'''
        return filename in os.listdir(dir_path)

    def rpt_2_df(self, file: str):
        '''Conver .rpt file to Pandas DataFrame'''

        f = file.readlines()
        f = [
            t.decode('big5').rstrip('\r\n') if isinstance(t, bytes) else t.rstrip('\n') for t in f]
        f = [t.replace(' ', '').split(',') for t in f]
        return pd.DataFrame(f[1:], columns=f[0])

    def unzip_file(self, filename: str, filters: list = [], filepath=''):
        '''解壓縮檔案並匯出'''

        if isinstance(filename, BytesIO):
            folders = ZipFile(filename)
        else:
            folders = ZipFile(f'{filename}.zip')
        names = folders.namelist()
        N = len(names)

        if N > 1 and isinstance(filename, str):
            self.create_folder(filename)

        for i, folder in enumerate(names):
            if any(f in filters for f in [folder, folder.rstrip('/'), folder[:4]]):
                continue

            if '.zip' in folder:
                folderPath = f"{filename}/{folder.replace('.zip', '')}"
                print(f"\n{folder.replace('.zip', '')}")
                self.create_folder(folderPath)
                zfiledata = BytesIO(folders.read(folder))
                self.unzip_file(zfiledata, filepath=folderPath)

            elif '.rpt' in folder:
                file = BytesIO(folders.read(folder))
                file = self.rpt_2_df(file)
                folderPath = filepath if filepath else (
                    filename if N > 1 else '.')
                self.save_table(
                    file, f"{folderPath}/{folder.replace('.rpt', '.csv')}")
                progress_bar(N, i)

    def remove_files(self, dirpath: str, files: list = None, pattern: str = ''):
        if not files:
            files = self.listdir(dirpath, pattern=pattern)

        for f in files:
            os.remove(f'{dirpath}/{f}')

    def save_table(self, df: pd.DataFrame, filename: str, saveEmpty=False):
        if df.shape[0] or saveEmpty:
            if '.csv' in filename:
                df.to_csv(filename, index=False, encoding='utf-8-sig')
            elif '.xlsx' in filename:
                df.to_excel(filename, index=False)  # , encoding='utf-8-sig')
            else:
                df.to_pickle(filename)

    def read_table(self, filename: str, df_default: pd.DataFrame = None):
        if os.path.exists(filename):
            if '.pkl' in filename:
                tb = pd.read_pickle(filename)
            elif '.xlsx' in filename:
                tb = pd.read_excel(filename)
            else:
                try:
                    tb = pd.read_csv(filename, low_memory=False)
                except:
                    tb = pd.read_csv(
                        filename, low_memory=False, encoding='big5')
        elif isinstance(df_default, pd.DataFrame):
            tb = df_default
        else:
            tb = pd.DataFrame()

        return tb

    def read_and_concat(self, filename: str, df: pd.DataFrame):
        tb = self.read_table(filename)
        tb = concat_df(tb, df, reset_index=True)
        return tb

    def read_tables_in_folder(self, dir_path: str, pattern: str = None, **kwargs):
        files = self.listdir(dir_path, pattern=pattern)

        # filter files by time interval
        start = kwargs.get('start')
        if start:
            if not isinstance(start, pd.Timestamp):
                start = pd.to_datetime(start)
            y1, m1 = start.year, start.month
            files = [
                f for f in files if int(f[:4]) >= y1 and int(f[5:7]) >= m1]

        end = kwargs.get('end')
        if end:
            if not isinstance(end, pd.Timestamp):
                end = pd.to_datetime(end)
            y2, m2 = end.year, end.month
            files = [
                f for f in files if int(f[:4]) <= y2 and int(f[5:7]) <= m2]

        # read tables
        N = len(files)
        if N:
            df = np.array([None]*N)
            for i, f in enumerate(files):
                df[i] = self.read_table(f'{dir_path}/{f}')
                progress_bar(N, i, status=f'[{f}]')

            df = pd.concat(df).reset_index(drop=True)
            return df
        return pd.DataFrame()

    def filter_file_dates(self, market: str, **kwargs):
        '''
        取得逐筆交易明細表的日期清單。以year為主，取得該年度的資料日期，可另外指定要合併的區間
        '''

        dir_path = f'{PATH}/ticks/{market.lower()}'
        files = self.list_files(dir_path, pattern='.csv')

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
