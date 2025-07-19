import os
import logging
import pandas as pd
from argparse import ArgumentParser
from logging.handlers import RotatingFileHandler

from trader import __version__ as ver
from trader.config import TODAY_STR, holidays, LOG_LEVEL
from trader.tasker import get_tasks
from trader.utils.bot import TelegramBot


def parse_args():
    """
    Command: $ python run.py -TASK auto_trader -ACCT account_name

    Parameters:
    1. task: The name of tasks. 
       EX: auto_trader, account_info, update_and_select_stock
    2. account: An account name is required if task == 'auto_trader'
    """

    parser = ArgumentParser()
    parser.add_argument(
        '--task', '-TASK', type=str, default='auto_trader', help='Target task name')
    parser.add_argument(
        '--account', '-ACCT', type=str, default='chrisli_1', help='account name')
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
            f'./logs/{filename}.log',
            'a',
            maxBytes=1*1024*1024,
            backupCount=5,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logging.info('—'*100)
logging.info(f'Current trader version is {ver}')

if __name__ == "__main__":
    # 啟動 Telegram 控制 bot
    bot = TelegramBot(account)

    Tasks = get_tasks()
    date = pd.to_datetime(TODAY_STR)
    if task in [
        'account_info',
        'auto_trader',
        'crawl_html',
        'select_stock',
        'subscribe',
        'update_and_select_stock'
    ] and date in holidays:
        logging.warning(f'{holidays[date]}不開盤')
    elif task in Tasks:
        functions = Tasks[task]
        for func in functions:
            if func.__name__ in ['runAutoTrader', 'runCrawlStockData']:
                func(account=account)
            else:
                func()
    else:
        logging.warning(f"The input task 【{task}】 does not exist.")

    logging.debug('End of tasker')
    os._exit(0)
