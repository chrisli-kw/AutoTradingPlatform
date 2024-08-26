import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile

from . import concat_df


class FileOperations:
    @staticmethod
    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def listdir(dir_path: str, pattern: str = '', filter_out: list = ['desktop.ini']):
        if pattern:
            result = [f for f in os.listdir(dir_path) if pattern in f]
        else:
            result = os.listdir(dir_path)
        return [f for f in result if f not in filter_out]

    @staticmethod
    def list_files(dir_path: str, pattern: str = '', filter_out: list = ['desktop.ini']):
        file_list = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if pattern in file and file not in filter_out:
                    file_list.append(os.path.join(root, file))
        return file_list

    @staticmethod
    def is_in_dir(filename: str, dir_path: str):
        '''Check if a filename is in dir_path.'''
        return filename in os.listdir(dir_path)

    def remove_files(self, dirpath: str, files: list = None, pattern: str = ''):
        if not files:
            files = self.listdir(dirpath, pattern=pattern)

        for f in files:
            os.remove(f'{dirpath}/{f}')


class DataProcessing:
    @staticmethod
    def save_table(df: pd.DataFrame, filename: str, saveEmpty=False):
        if df.shape[0] or saveEmpty:
            if '.csv' in filename:
                df.to_csv(filename, index=False, encoding='utf-8-sig')
            elif '.xlsx' in filename:
                df.to_excel(filename, index=False)
            else:
                df.to_pickle(filename)

    @staticmethod
    def read_table(filename: str, df_default: pd.DataFrame = None):
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


class FileConversion:
    @staticmethod
    def rpt_2_df(file: str):
        '''Conver .rpt file to Pandas DataFrame'''

        f = file.readlines()
        f = [
            t.decode('big5').rstrip('\r\n') if isinstance(t, bytes) else t.rstrip('\n') for t in f]
        f = [t.replace(' ', '').split(',') for t in f]
        return pd.DataFrame(f[1:], columns=f[0])


class FileHandler:
    def __init__(self) -> None:
        self.Operate = FileOperations()
        self.Process = DataProcessing()
        self.Convert = FileConversion()

    def read_tables_in_folder(self, dir_path: str, pattern: str = None, **kwargs):
        files = self.Operate.listdir(dir_path, pattern=pattern)

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
            iterator = files
            for i, f in enumerate(iterator):
                df[i] = self.Process.read_table(f'{dir_path}/{f}')
                iterator.set_description(f'[{f}]')

            df = pd.concat(df).reset_index(drop=True)
            return df
        return pd.DataFrame()

    def unzip_file(self, filename: str, filters: list = [], filepath=''):
        '''解壓縮檔案並匯出'''

        if isinstance(filename, BytesIO):
            folders = ZipFile(filename)
        else:
            folders = ZipFile(f'{filename}.zip')
        names = folders.namelist()
        N = len(names)

        if N > 1 and isinstance(filename, str):
            self.Operate.create_folder(filename)

        for folder in tqdm(names):
            if any(f in filters for f in [folder, folder.rstrip('/'), folder[:4]]):
                continue

            if '.zip' in folder:
                folderPath = f"{filename}/{folder.replace('.zip', '')}"
                print(f"\n{folder.replace('.zip', '')}")
                self.Operate.create_folder(folderPath)
                zfiledata = BytesIO(folders.read(folder))
                self.unzip_file(zfiledata, filepath=folderPath)

            elif '.rpt' in folder:
                file = BytesIO(folders.read(folder))
                file = self.Convert.rpt_2_df(file)
                folderPath = filepath if filepath else (
                    filename if N > 1 else '.')
                self.Process.save_table(
                    file, f"{folderPath}/{folder.replace('.rpt', '.csv')}")


file_handler = FileHandler()
