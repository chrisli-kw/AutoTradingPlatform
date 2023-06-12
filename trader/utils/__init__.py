import os
import sys
import queue
import traceback
import numpy as np
import pandas as pd
from io import BytesIO
from zipfile import ZipFile

from .. import API, create_folder


def print_error_msg(excp, time_str=None):
    error_class = excp.__class__.__name__
    _, _, tb = sys.exc_info()
    file, line, func, code = traceback.extract_tb(tb)[-1]
    print(
        f'{time_str}[File "{file}"][{func}][line {line}][{code}][{error_class}: {excp.args[0]}]')
    traceback.print_exc()


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def progress_bar(N, i, status=''):
    n = i+1
    progress = f"\r{status} |{'█'*int(n*50/N)}{' '*(50-int(n*50/N))} | {n}/{N} ({round(n/N*100, 2)}%)"
    print(progress, end='')


def create_queue(target_list, crawled_list=[]):
    '''put candidate companies not crawled into queue'''
    print("Put targets in queue...", flush=True)
    q = queue.Queue(0)

    target_list = list(set(target_list)-set(crawled_list))
    N = len(target_list)

    for i, company in enumerate(target_list):
        q.put(company)
        progress_bar(N, i)

    print("\nTargets in queue:", q.qsize(), flush=True)

    return (q)


def delete_selection_files(dirpath: str, files=None):
    if not files:
        files = [f for f in os.listdir(dirpath) if '.csv' in f]

    if files:
        for f in files:
            os.remove(f'{dirpath}/{f}')


def get_contract(target: str):
    if target[:3] in API.Contracts.Indexs.__dict__:
        return API.Contracts.Indexs[target[:3]][target]
    elif target[:3] in API.Contracts.Futures.__dict__:
        return API.Contracts.Futures[target[:3]][target]
    elif target[:3] in API.Contracts.Options.__dict__:
        return API.Contracts.Options[target[:3]][target]
    return API.Contracts.Stocks[target]


def save_table(df: pd.DataFrame, filename: str, saveEmpty=False):
    if df.shape[0] or saveEmpty:
        if '.csv' in filename:
            df.to_csv(filename, index=False, encoding='utf-8-sig')
        elif '.xlsx' in filename:
            df.to_excel(filename, index=False, encoding='utf-8-sig')
        else:
            df.to_pickle(filename)


def rpt_2_df(file):
    '''RPT轉DataFrame'''

    f = file.readlines()
    f = [t.decode('big5').rstrip('\r\n') if isinstance(t, bytes) else t.rstrip('\n') for t in f]
    f = [t.replace(' ', '').split(',') for t in f]
    return pd.DataFrame(f[1:], columns=f[0])


def unzip_file(filename: str, filters: list = [], filepath=''):
    '''解壓縮檔案並匯出'''

    folders = ZipFile(filename) if isinstance(filename, BytesIO) else ZipFile(f'{filename}.zip')
    names = folders.namelist()
    N = len(names)

    if N > 1 and isinstance(filename, str):
        create_folder(filename)

    for i, folder in enumerate(names):
        if folder in filters or folder.rstrip('/') in filters or folder[:4] in filters:
            continue

        if '.zip' in folder:
            folderPath = f"{filename}/{folder.replace('.zip', '')}"
            print(f"\n{folder.replace('.zip', '')}")
            create_folder(folderPath)
            zfiledata = BytesIO(folders.read(folder))
            unzip_file(zfiledata, filepath=folderPath)

        elif '.rpt' in folder:
            file = BytesIO(folders.read(folder))
            file = rpt_2_df(file)
            folderPath = filepath if filepath else (filename if N > 1 else '.')
            save_table(file, f"{folderPath}/{folder.replace('.rpt', '.csv')}")
            progress_bar(N, i)
