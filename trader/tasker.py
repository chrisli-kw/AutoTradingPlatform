import time
import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import as_completed

from . import exec, notifier, picker, tdp
from .config import (
    create_api,
    API,
    PATH,
    TODAY_STR,
    ACCOUNTS,
    TEnd,
    ConvertScales
)
from .create_env import app
from .utils.time import time_tool
from .utils.crawler import crawler
from .utils.file import file_handler
from .utils.database import redis_tick
from .utils.objects.env import UserEnv
from .utils.subscribe import Subscriber
from .utils.accounts import AccountInfo
from .executor import StrategyExecutor

try:
    from .scripts.TaskList import customTasks
except:
    customTasks = {}


def runCreateENV():
    file_handler.Operate.create_folder('./lib')
    file_handler.Operate.create_folder('./lib/envs')
    file_handler.Operate.create_folder('./lib/schedules')
    app.run()


def runAccountInfo():
    account = AccountInfo()
    df = account.create_info_table()
    tables = {}

    try:
        logging.debug(f'ACCOUNTS: {ACCOUNTS}')
        for env in ACCOUNTS:
            logging.debug(f'Load 【{env}】 config')
            config = UserEnv(env)

            acct = config.ACCOUNT_NAME
            account.login_(config)
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
                tables[acct] = pd.read_excel(df, sheet_name=env)
            else:
                tables[acct] = account.DEFAULT_TABLE

            # 登出
            time.sleep(5)
            logging.info(f'API log out: {API.logout()}')
            time.sleep(10)

        logging.info('Export data')
        writer = pd.ExcelWriter(
            f'{PATH}/daily_info/{account.filename}', engine='xlsxwriter')

        for sheet in tables:
            try:
                tables[sheet].to_excel(writer, index=False, sheet_name=sheet)
            except:
                logging.exception('Catch an exception:')
                tables[sheet].to_excel(sheet+'.csv', index=False)
        writer.close()
    except:
        logging.exception('Catch an exception:')
        notifier.post('\n【Error】【帳務資訊查詢】發生異常', msgType='Tasker')
        API.logout()


def runAutoTrader(account: str):
    try:
        se = StrategyExecutor(account)
        se.login_and_activate()
        se.run()
    except KeyboardInterrupt:
        notifier.post(
            f"\n【Interrupt】【下單機監控】{se.account_name}已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(
            f"\n【Error】【下單機監控】{se.account_name}發生異常", msgType='Tasker')
    finally:
        try:
            se.output_files()
        except:
            logging.exception('Catch an exception (output_files):')
            notifier.post(
                f"\n【Error】【下單機監控】{se.account_name}資料儲存失敗", msgType='Tasker')

        logging.info(f'API log out: {API.logout()}')
        notifier.post(f"\n【停止監控】{se.account_name}關閉程式並登出", msgType='Tasker')

    del se


def runCrawlStockData(account: str, start=None, end=None):
    target = pd.to_datetime('15:05:00')
    config = UserEnv(account)
    aInfo = AccountInfo()

    try:
        now = datetime.now()
        if now < target and not start:
            logging.info(
                f'Current time is still early, will start to crawl after {target}')
            time_tool.CountDown(target)

        logging.info('Start the crawler')

        # Log-in account
        aInfo.login_(config)
        time.sleep(30)

        # Update stock list
        logging.info('Updating stock list')
        stock_list = crawler.FromSJ.get_security_list(stock_only=True)
        crawler.FromSJ.export_security_list(stock_list)

        # Crawler daily stock data from Sinopac
        crawler.FromSJ.run(stockids='all', update=True, start=start, end=end)
        crawler.FromSJ.merge_stockinfo()

        # Update historical data
        for scale in ConvertScales:
            crawler.FromSJ.add_new_data(scale, save=True, start=TODAY_STR)
            crawler.FromSJ.merge_daily_data(TODAY_STR, scale, save=True)
        crawler.FromSJ.merge_daily_data(TODAY_STR, '1T', save=True)

    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】股價爬蟲發生異常", msgType='Tasker')
        if len(crawler.FromSJ.StockData):
            df = pd.concat(crawler.FromSJ.StockData)
            filename = f'{crawler.FromSJ.folder_path}/stock_data_1T.pkl'
            file_handler.Process.save_table(df, filename)
    finally:
        logging.info(f'API log out: {API.logout()}')


def runSelectStock():
    try:
        df = picker.pick(3, 1.8, 3)
        df = picker.melt_table(df)
        tb = df[df.Time == TODAY_STR].reset_index(drop=True)
        picker.export(tb)
        notifier.post_stock_selection(tb)

    except FileNotFoundError as e:
        logging.warning(f'{e} No stock is selected.')
    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【選股程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【選股程式】選股發生異常", msgType='Tasker')


def runCrawlPutCallRatio():
    try:
        df_pcr_new = crawler.FromHTML.PutCallRatio()
        crawler.FromHTML.export_put_call_ratio(df_pcr_new)
    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】PutCallRatio發生異常", msgType='Tasker')


def runCrawlExDividendList():
    try:
        dividends = crawler.FromHTML.ex_dividend_list()
        crawler.FromHTML.export_ex_dividend_list(dividends)
    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】爬除權息資料發生異常", msgType='Tasker')


def runCrawlFuturesTickData(date=TODAY_STR):
    try:
        crawler.FromHTML.get_FuturesTickData(date)

        # 轉換&更新期貨逐筆成交資料
        df = tdp.convert_daily_tick(date, '1T')
        crawler.FromHTML.export_futures_kbar(df)
    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】期貨逐筆成交資料發生異常", msgType='Tasker')


def runCrawlIndexMargin():
    try:
        df = crawler.FromHTML.get_IndexMargin()
        file_handler.Process.save_table(df, './lib/indexMarging.csv')
    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】期貨股價指數類保證金發生異常", msgType='Tasker')


def thread_subscribe(user: str, targets: list):
    subscriber = Subscriber()
    api = create_api()

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
            tick_data = subscriber.update_quote_v1(tick)
            redis_tick.to_redis({tick.code: tick_data})

    config = UserEnv(user)
    api.login(config.api_key(), config.secret_key())
    time.sleep(2)

    try:
        logging.info('Subscribe targets')
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
    # TODO: 讀取要盤中選股的股票池
    df = file_handler.Process.read_table(f'{PATH}/selections/stock_list.xlsx')
    codes = df[df.exchange.isin(['TSE', 'OTC'])].code.astype(str).values

    N = 200
    futures = []
    for i, user in enumerate(ACCOUNTS):
        targets = codes[N*i:N*(i+1)]
        future = exec.submit(thread_subscribe, user, targets)
        futures.append(future)

    for future in as_completed(futures):
        logging.info(future.result())


def runSimulationChecker():
    try:
        for account in ACCOUNTS:
            config = UserEnv(account)

            if config.MODE == 'Simulation':
                se = StrategyExecutor(account)

                # check stock pool size
                watchlist = se.watchlist[se.watchlist.market == 'Stocks']
                stocks = se.get_securityInfo('Stocks')
                is_same_shape = watchlist.shape[0] == stocks.shape[0]
                if not is_same_shape:
                    text = f'\n【{account} 庫存不一致】'
                    text += f'\nSize: watchlist {watchlist.shape[0]}; stocks: {stocks.shape[0]}'
                    text += f'\nwatchlist day start: {watchlist.buyday.min()}'
                    text += f'\nwatchlist day end: {watchlist.buyday.max()}'
                    text += f'\nwatchlist - stocks: {set(watchlist.code) - set(stocks.code)}'
                    text += f'\nstocks - watchlist: {set(stocks.code) - set(watchlist.code)}'

                    notifier.post(text, msgType='Monitor')

    except FileNotFoundError as e:
        logging.warning(e)
        notifier.post(f'\n{e}', msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post('\n【Error】【模擬帳戶檢查】發生異常', msgType='Tasker')


Tasks = {
    'create_env': [runCreateENV],
    'account_info': [runAccountInfo, runSimulationChecker],
    'update_and_select_stock': [
        runCrawlStockData,
        runSelectStock,
        runCrawlPutCallRatio,
        runCrawlExDividendList,
        runCrawlFuturesTickData,
        runCrawlIndexMargin
    ],
    'crawl_stock_data': [runCrawlStockData],
    'select_stock': [runSelectStock],
    'crawl_put_call_ratio': [runCrawlPutCallRatio],
    'crawl_ex_dividend_list': [runCrawlExDividendList],
    'crawl_futures_tick_data': [runCrawlFuturesTickData],
    'crawl_index_margin': [runCrawlIndexMargin],
    'auto_trader': [runAutoTrader],
    'subscribe': [runShioajiSubscriber],
}

for taskName, tasks in customTasks.items():
    if taskName in Tasks:
        Tasks[taskName] += tasks
    else:
        Tasks[taskName] = tasks
