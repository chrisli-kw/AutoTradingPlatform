import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import shioaji as sj

__version__ = '0.1.4'
API = sj.Shioaji()
TODAY = datetime.today()
TODAY_STR = TODAY.strftime("%Y-%m-%d")
PATH = './data'


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_holidays():
    try:
        df = pd.read_csv('./lib/政府行政機關辦公日曆表.csv')
        df.date = pd.to_datetime(df.date)
        df.name = df.name.fillna(df.holidayCategory)
        holidays = df.set_index('date').name.to_dict()

        eves = {k: v for k, v in holidays.items() if v == '農曆除夕'}
        for i in range(2):
            days = {d - timedelta(days=i+1) if d - timedelta(days=i+1)
                    not in holidays else d - timedelta(days=i+2): '年前封關' for d in eves}
            holidays.update(days)
        return holidays
    except:
        logging.warning('Run trader without holiday data.')


for f in [PATH, './logs']:
    create_folder(f)

for f in ['daily_info', 'Kbars', 'ticks', 'selections', 'stock_pool']:
    create_folder(f'{PATH}/{f}')

create_folder(f'{PATH}/Kbars/1min')
create_folder(f'{PATH}/ticks/stocks')
create_folder(f'{PATH}/ticks/futures')
create_folder(f'{PATH}/selections/history')


holidays = get_holidays()
