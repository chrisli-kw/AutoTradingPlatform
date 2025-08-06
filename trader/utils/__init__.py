import queue
import logging
import numpy as np
import pandas as pd

from ..config import API
from .notify import notifier


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
    logging.info("Put targets in queue...")
    q = queue.Queue(0)

    target_list = list(set(target_list)-set(crawled_list))
    N = len(target_list)

    for i, company in enumerate(target_list):
        q.put(company)
        progress_bar(N, i)

    logging.info(f"Targets in queue: {q.qsize()}")
    return (q)


def get_contract(target: str, api=None):
    if api is None:
        api = API

    if target[:3] in api.Contracts.Indexs.__dict__:
        return api.Contracts.Indexs[target[:3]][target]
    elif target[:3] in api.Contracts.Futures.__dict__:
        return api.Contracts.Futures[target[:3]][target]
    elif target[:3] in api.Contracts.Options.__dict__:
        return api.Contracts.Options[target[:3]][target]
    return api.Contracts.Stocks[target]


def concat_df(df1: pd.DataFrame, df2: pd.DataFrame, sort_by=[], reset_index=False):
    if df1.empty:
        return df2
    elif df2.empty:
        return df1

    df = pd.concat([df1, df2])

    if sort_by:
        df = df.sort_values(sort_by)

    if reset_index:
        df = df.reset_index(drop=True)
    return df


def tasker(func):
    def wrapper(**kwargs):
        name = func.__name__
        try:
            func(**kwargs)
        except KeyboardInterrupt:
            notifier.send.post(f"\n【Interrupt】【{name}】已手動關閉")
        except:
            logging.exception('Catch an exception:')
            notifier.send.post(f"\n【Error】【{name}】發生異常")
        finally:
            logging.info(f'API log out: {API.logout()}')
    return wrapper
