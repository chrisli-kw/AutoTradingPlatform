# coding: utf-8
import os
import time
import logging
import pandas as pd
from datetime import datetime
from dotenv import dotenv_values
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from trader import __version__ as ver
from trader import API, PATH, TODAY_STR, holidays
from trader.config import ACCOUNTS, TEnd, SelectMethods, ConvertScales
from trader.strategies.select import SelectStock
from trader.utils import save_csv, save_excel
from trader.utils.notify import Notification
from trader.utils.database import RedisTools
from trader.utils.subscribe import Subscriber
from trader.utils.kbar import TickDataProcesser
from trader.utils.crawler import CrawlStockData, CrawlFromHTML
from trader.utils.accounts import AccountInfo
from trader.executor import StrategyExecutor
from trader.backtest import convert_statement
from trader.strategies.features import KBarFeatureTool


def parse_args():
    """
    執行指令:$ python tasker.py -TASK auto_trader -ACCT account_name

    參數說明:
    1. task(執行的目標程式): auto_trader, account_info, update_and_select_stock
    2. account(代號): 若 task == 'auto_trader', 需指定要執行的帳戶代號

    """
    parser = ArgumentParser()
    parser.add_argument('--task', '-TASK', type=str, default='auto_trader', help='執行的目標程式')
    parser.add_argument('--account', '-ACCT', type=str, default='chrisli_1', help='代號')
    args = parser.parse_args()
    return (args)


def runAccountInfo():
    account = AccountInfo()
    df = account.create_info_table()
    tables = {}

    try:
        logging.debug(f'ACCOUNTS: {ACCOUNTS}')
        for e in ACCOUNTS:
            logging.debug(f'Load 【{e}】 config')
            config = dotenv_values(f'./lib/envs/{e}.env')

            API_KEY = config['API_KEY']
            SECRET_KEY = config['SECRET_KEY']
            acct = config['ACCOUNT_NAME']
            account._login(API_KEY, SECRET_KEY, acct)
            time.sleep(1)

            row = account.query_all()
            if row:
                for i, data in enumerate(row):
                    if i < 3:
                        logging.info(f"{data}： {row[data]}")
                    else:
                        logging.info(f"{data}： NT$ {'{:,}'.format(row[data])}")

                if hasattr(df, 'sheet_names') and acct in df.sheet_names:
                    tb = account.update_info(df, row)
                else:
                    tb = pd.DataFrame([row])

                tables[acct] = tb

                # 推播訊息
                account_id = API.stock_account.account_id
                notifier.post_account_info(account_id, row)

            elif hasattr(df, 'sheet_names') and account.account_name in df.sheet_names:
                tables[acct] = pd.read_excel(df, sheet_name=account.account_name)

            else:
                tables[acct] = account.DEFAULT_TABLE

            # 登出
            time.sleep(5)
            logging.info(f'登出系統: {API.logout()}')
            time.sleep(10)

        logging.info('儲存資訊')
        writer = pd.ExcelWriter(f'{PATH}/daily_info/{account.filename}', engine='xlsxwriter')

        for sheet in tables:
            try:
                tables[sheet].to_excel(writer, encoding='utf-8-sig',
                                       index=False, sheet_name=sheet)
            except:
                logging.exception('Catch an exception:')
                tables[sheet].to_excel(sheet+'.csv', encoding='utf-8-sig', index=False)
        writer.save()
    except:
        logging.exception('Catch an exception:')
        notifier.post('\n【Error】【帳務資訊查詢】發生異常', msgType='Tasker')
        API.logout()


def runAutoTrader():
    try:
        config = dotenv_values(f'./lib/envs/{account}.env')
        se = StrategyExecutor(config=config, kbar_script=KBarFeatureTool())
        se.login_and_activate()
        se.run()
    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【下單機監控】{se.ACCOUNT_NAME}已手動關閉", msgType='Tasker')
        se.output_files()
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【下單機監控】{se.ACCOUNT_NAME}發生異常", msgType='Tasker')
        se.output_files()
    finally:
        logging.info(f'登出系統: {API.logout()}')
        notifier.post(f"\n【停止監控】{se.ACCOUNT_NAME}關閉程式並登出", msgType='Tasker')

    del se


def runCrawlStockData():
    target = pd.to_datetime('15:05:00')
    config = dotenv_values(f'./lib/envs/{account}.env')
    aInfo = AccountInfo()
    crawler = CrawlStockData()

    try:
        now = datetime.now()
        if now < target:
            logging.info(f'Current time is still early, will start to crawl after {target}')
            aInfo.CountDown(target)

        logging.info('開始爬蟲')

        # 登入
        API_KEY = config['API_KEY']
        SECRET_KEY = config['SECRET_KEY']
        acct = config['ACCOUNT_NAME']
        aInfo._login(API_KEY, SECRET_KEY, acct)
        time.sleep(30)

        # 更新股票清單
        logging.info('Updating stock list')
        stock_list = crawler.get_stock_list(stock_only=True)
        stock_list.to_excel(f'{PATH}/selections/stock_list.xlsx', index=False)

        # 爬當天股價資料
        crawler.crawl_from_sinopac(stockids='all', update=True)
        crawler.merge_stockinfo()

        # 更新歷史資料
        for scale in ConvertScales:
            crawler.add_new_data(scale, save=True, start=TODAY_STR)

    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】股價爬蟲發生異常", msgType='Tasker')
        if len(crawler.StockData):
            pd.concat(crawler.StockData).to_pickle(f'{crawler.folder_path}/stock_data_1T.pkl')
    finally:
        logging.info(f'登出系統: {API.logout()}')


def runSelectStock():
    picker = SelectStock()
    try:
        picker.set_select_methods(SelectMethods)
        df = picker.pick(3, 1.8, 3)
        tb = df[df.date == TODAY_STR].copy()
        exists = tb[SelectMethods].isin([True]).any(axis=1)
        tb1 = tb[exists == True]
        # TODO: 加入籌碼資料
        picker.export(tb1)
        notifier.post_stock_selection(tb1[['name', 'company_name']+SelectMethods].copy())

        # 更新前一日資料
        d = str(df[(df.date != TODAY_STR)].date.max().date())
        filename = f'{PATH}/selections/history/{d}-all.csv'
        if os.path.exists(filename):
            tb2 = pd.read_csv(filename)
            tb2.name = tb2.name.astype(str)

            tb = tb.set_index('name')
            for i, c in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
                if f'{TODAY_STR}-Open' in tb2.columns:
                    tb2[f'{TODAY_STR}-{c}'] = tb2.name.map(tb[c].to_dict())
                else:
                    tb2.insert(11+i, f'{TODAY_STR}-{c}',  tb2.name.map(tb[c].to_dict()))
            tb2.to_csv(filename, index=False, encoding='utf-8-sig')

    except FileNotFoundError as e:
        logging.warning(f'{e} No stock is selected.')
    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【選股程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【選股程式】選股發生異常", msgType='Tasker')


def runCrawlFromHTML():
    crawler2 = CrawlFromHTML()
    tdp = TickDataProcesser()

    try:
        # update PutCallRatio
        step = 'PutCallRatio'
        try:
            df_pcr = pd.read_csv(f'{PATH}/put_call_ratio.csv')
        except FileNotFoundError:
            df_pcr = pd.DataFrame()
        df_pcr_new = crawler2.put_call_ratio()
        df_pcr = pd.concat([df_pcr, df_pcr_new])
        save_csv(df_pcr, f'{PATH}/put_call_ratio.csv')
        notifier.post_put_call_ratio(df_pcr_new)

        # 爬除權息資料
        step = '爬除權息資料'
        dividends = crawler2.ex_dividend_list()
        save_csv(dividends, f'{PATH}/exdividends.csv')

        # 期貨逐筆成交資料
        step = '期貨逐筆成交資料'
        crawler2.get_FuturesTickData(TODAY_STR)

        # 轉換&更新期貨逐筆成交資料
        df = tdp.convert_daily_tick(TODAY_STR, '1T')
        if isinstance(df, pd.DataFrame):
            filename = f'{PATH}/期貨日夜盤1T.pkl'
            if os.path.exists(filename):
                tick_old = pd.read_pickle(filename)
                df = pd.concat([tick_old, df])
            df.to_pickle(filename)

    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】{step}發生異常", msgType='Tasker')


def thread_subscribe(user, targets):
    import shioaji as sj

    redis = RedisTools(redisKey='TickData')
    subscriber = Subscriber()
    api = sj.Shioaji()

    @api.quote.on_event
    def event_callback(resp_code: int, event_code: int, info: str, event: str):
        if 'Subscription Not Found' in info:
            logging.warning(info)

        else:
            logging.info(
                f'Response code: {resp_code} | Event code: {event_code} | info: {info} | Event: {event}')

    @api.on_quote_stk_v1()
    def stk_quote_callback_v1(exchange, tick):
        if tick.intraday_odd == 0 and tick.simtrade == 0:
            tick_data = subscriber.stk_quote_v1(tick)
            redis.to_redis({tick.code: tick_data})

    config = dotenv_values(f'./lib/envs/{user}.env')
    API_KEY = config['API_KEY']
    SECRET_KEY = config['SECRET_KEY']
    api.login(API_KEY, SECRET_KEY)
    time.sleep(2)

    try:
        logging.info('subscribe_targets')
        for t in targets:
            if t[:3] in api.Contracts.Indexs.__dict__:
                target = api.Contracts.Indexs[t[:3]][t]
            elif t[:3] in api.Contracts.Futures.__dict__:
                target = api.Contracts.Futures[t[:3]][t]
            elif t[:3] in api.Contracts.Options.__dict__:
                target = api.Contracts.Options[t[:3]][t]
            else:
                target = api.Contracts.Stocks[t]
            api.quote.subscribe(target, quote_type='tick', version='v1')

        logging.info(f'Done subscribe {len(targets)} targets')

        now = datetime.now()
        time.sleep(max((TEnd - now).total_seconds(), 0))
    except:
        logging.exception('Catch an exception:')
    finally:
        logging.info(f'{datetime.now()} is log-out: {api.logout()}')
        time.sleep(10)
    return "Task completed"


def runShioajiSubscriber():
    #TODO: 讀取要盤中選股的股票池
    df = pd.read_excel('./data/selections/stock_list.xlsx')
    codes = df[df.exchange.isin(['TSE', 'OTC'])].code.astype(str).values

    N = 200
    futures = []
    for i, user in enumerate(ACCOUNTS):
        targets = codes[N*i:N*(i+1)]
        future = executor.submit(thread_subscribe, user, targets)
        futures.append(future)

    for future in as_completed(futures):
        logging.info(future.result())


def runSimulationChecker():
    try:
        filepath = f'{PATH}/stock_pool'
        for account in ACCOUNTS:
            config = dotenv_values(f'./lib/envs/{account}.env')

            if config['MODE'] == 'Simulation':
                # check stock pool size
                watchlist = pd.read_csv(f'{filepath}/watchlist_{account}.csv')
                stocks = pd.read_pickle(f'{filepath}/simulation_stocks_{account}.pkl')

                watchlist.buyday = pd.to_datetime(watchlist.buyday).dt.date
                watchlist.code = watchlist.code.astype(str)
                watchlist = watchlist[
                    ~watchlist.code.str.contains('MXF') &
                    ~watchlist.code.str.contains('TXF')
                ]

                is_same_shape = watchlist.shape[0] == stocks.shape[0]
                if not is_same_shape:
                    text = f'\n【{account} 庫存不一致】'
                    text += f'\nSize: watchlist {watchlist.shape[0]}; stocks: {stocks.shape[0]}'
                    text += f'\nwatchlist day start: {watchlist.buyday.min()}'
                    text += f'\nwatchlist day end: {watchlist.buyday.max()}'
                    text += f'\nwatchlist - stocks: {set(watchlist.code) - set(stocks.code)}'
                    text += f'\nstocks - watchlist: {set(stocks.code) - set(watchlist.code)}'

                    notifier.post(text, msgType='Monitor')

                # update performance statement
                df = pd.read_csv(f'{filepath}/statement_stocks_{account}.csv')
                df = convert_statement(df)
                save_excel(df, f'{filepath}/simulation_performance_{account}.xlsx', saveEmpty=True)
    except FileNotFoundError as e:
        logging.warning(e)
        notifier.post(f'\n{e}', msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post('\n【Error】【模擬帳戶檢查】發生異常', msgType='Tasker')


args = parse_args()
task = args.task
account = args.account
filename = account if task == 'auto_trader' else task
notifier = Notification()
executor = ThreadPoolExecutor(max_workers=5)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d|%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %a %H:%M:%S',
    handlers=[
            logging.FileHandler(f'./logs/{filename}.log', 'a'),
            logging.StreamHandler()
    ]
)
logging.info('—'*250)
logging.info(f'Current trader version is {ver}')

if __name__ == "__main__":
    date = pd.to_datetime(TODAY_STR)
    if date in holidays:
        logging.warning(f'{holidays[date]}不開盤')
    else:
        if task == 'account_info':
            # runAccountInfo()
            runSimulationChecker()
        elif task == 'update_and_select_stock':
            runCrawlStockData()
            runSelectStock()
            runCrawlFromHTML()
        elif task == 'crawl_stock_data':
            runCrawlStockData()
        elif task == 'select_stock':
            runSelectStock()
        elif task == 'crawl_html':
            runCrawlFromHTML()
        elif task == 'auto_trader':
            runAutoTrader()
        elif task == 'subscribe':
            runShioajiSubscriber()
        else:
            logging.warning(f"The input task 【{task}】 does not exist.")

    logging.debug('End of tasker')
    os._exit(0)
