import os
import logging
import pandas as pd
from argparse import ArgumentParser
from logging.handlers import RotatingFileHandler

from trader import __version__ as ver
from trader.config import TODAY_STR, holidays, LOG_LEVEL
from trader.tasker import Tasks


def parse_args():
    """
    執行指令:$ python run.py -TASK auto_trader -ACCT account_name

    參數說明:
    1. task(執行的目標程式): auto_trader, account_info, update_and_select_stock
    2. account(代號): 若 task == 'auto_trader', 需指定要執行的帳戶代號

    """
    parser = ArgumentParser()
    parser.add_argument(
        '--task', '-TASK', type=str, default='auto_trader', help='執行的目標程式')
    parser.add_argument(
        '--account', '-ACCT', type=str, default='chrisli_1', help='代號')
    args = parser.parse_args()
    return (args)


args = parse_args()
task = args.task
account = args.account
filename = account if task == 'auto_trader' else task

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s.%(msecs)03d|%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %a %H:%M:%S',
    handlers=[
            RotatingFileHandler(
                f'./logs/{filename}.log', 'a', maxBytes=1*1024*1024, backupCount=5, encoding='utf-8'),
            logging.StreamHandler()
    ]
)
logging.info('—'*100)
logging.info(f'Current trader version is {ver}')

if __name__ == "__main__":
    date = pd.to_datetime(TODAY_STR)
    if task in [
        'account_info', 'update_and_select_stock',
        'select_stock', 'crawl_html',
        'auto_trader', 'subscribe'
    ] and date in holidays:
        logging.warning(f'{holidays[date]}不開盤')
    elif task in Tasks:
        functions = Tasks[task]
        for func in functions:
            if func.__name__ in ['runAutoTrader', 'runCrawlStockData']:
                func(account)
            else:
                func()
    else:
        logging.warning(f"The input task 【{task}】 does not exist.")

    logging.debug('End of tasker')
    os._exit(0)
