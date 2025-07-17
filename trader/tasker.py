import time
import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import as_completed

from . import exec, picker, tdp
from .config import (
    create_api,
    API,
    PATH,
    TODAY_STR,
    ACCOUNTS,
    TimeEndStock,
    ConvertScales
)
from .create_env import app
from .utils import tasker, get_contract
from .utils.time import time_tool
from .utils.crawler import crawler
from .utils.notify import notifier
from .utils.file import file_handler
from .utils.database import db, redis_tick
from .utils.database.tables import SecurityList
from .utils.objects.env import UserEnv
from .utils.subscribe import Subscriber
from .utils.accounts import AccountInfo
from .utils.callback import CallbackHandler
from .executor import StrategyExecutor

try:
    from .scripts.TaskList import customTasks
except:
    logging.exception(f'Importing customTasks failed:')
    customTasks = {}


@tasker
def runCreateENV(**kwargs):
    file_handler.Operate.create_folder('./lib')
    file_handler.Operate.create_folder('./lib/envs')
    file_handler.Operate.create_folder('./lib/schedules')
    app.run(host='0.0.0.0', port=8090)


@tasker
def runAccountInfo(**kwargs):
    account = AccountInfo()
    df = account.create_info_table()
    tables = {}

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


@tasker
def runSelectStock(**kwargs):
    df = picker.pick(3, 1.8, 3)
    df = picker.melt_table(df)
    tb = df[df.Time == TODAY_STR].reset_index(drop=True)
    picker.export(tb)
    notifier.post_stock_selection(tb)


@tasker
def runCrawlPutCallRatio(**kwargs):
    df_pcr_new = crawler.FromHTML.PutCallRatio()
    crawler.FromHTML.export_put_call_ratio(df_pcr_new)


@tasker
def runCrawlExDividendList(**kwargs):
    dividends = crawler.FromHTML.ex_dividend_list()
    crawler.FromHTML.export_ex_dividend_list(dividends)


@tasker
def runCrawlFuturesTickData(date=TODAY_STR, **kwargs):
    crawler.FromHTML.get_FuturesTickData(date)

    # 轉換&更新期貨逐筆成交資料
    df = tdp.convert_daily_tick(date, '1T')
    crawler.FromHTML.export_futures_kbar(df)


@tasker
def runCrawlIndexMargin(**kwargs):
    df = crawler.FromHTML.get_IndexMargin()
    file_handler.Process.save_table(df, './lib/indexMarging.csv')


@tasker
def runShioajiSubscriber(**kwargs):
    # TODO: 讀取要盤中選股的股票池
    df = db.query(SecurityList)
    codes = df[df.exchange.isin(['TSE', 'OTC'])].code.astype(str).values

    N = 200
    futures = []
    for i, user in enumerate(ACCOUNTS):
        targets = codes[N*i:N*(i+1)]
        future = exec.submit(thread_subscribe, user, targets)
        futures.append(future)

    for future in as_completed(futures):
        logging.info(future.result())


def runAutoTrader(account: str):
    try:
        se = StrategyExecutor(account)
        se.init_account()
        se.run()
    except KeyboardInterrupt:
        notifier.send.post(
            f"\n【Interrupt】【下單機監控】{se.account_name}已手動關閉")
    except:
        logging.exception('Catch an exception:')
        notifier.send.post(
            f"\n【Error】【下單機監控】{se.account_name}發生異常")
    finally:
        try:
            se.output_files()
        except:
            logging.exception('Catch an exception (output_files):')
            notifier.send.post(
                f"\n【Error】【下單機監控】{se.account_name}資料儲存失敗")

        logging.info(f'API log out: {API.logout()}')
        notifier.send.post(f"\n【停止監控】{se.account_name}關閉程式並登出")

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
        notifier.send.post(f"\n【Interrupt】【爬蟲程式】已手動關閉")
    except:
        logging.exception('Catch an exception:')
        notifier.send.post(f"\n【Error】【爬蟲程式】股價爬蟲發生異常")
        if len(crawler.FromSJ.StockData):
            df = pd.concat(crawler.FromSJ.StockData)
            filename = f'{crawler.FromSJ.folder_path}/stock_data_1T.pkl'
            file_handler.Process.save_table(df, filename)
    finally:
        logging.info(f'API log out: {API.logout()}')


def thread_subscribe(user: str, targets: list):
    config = UserEnv(user)
    subscriber = Subscriber()
    api = create_api()

    @api.quote.on_event
    def event_callback(resp_code: int, event_code: int, info: str, event: str):
        CallbackHandler.events(resp_code, event_code, info, event, config)

    @api.on_quote_stk_v1()
    def stk_quote_callback_v1(exchange, tick):
        if tick.intraday_odd == 0 and tick.simtrade == 0:
            tick_data = subscriber.update_quote_v1(tick)
            redis_tick.to_redis({tick.code: tick_data})

    api.login(config.api_key(), config.secret_key())
    time.sleep(2)

    try:
        logging.info('Subscribe targets')
        for t in targets:
            target = get_contract(t, api=api)
            api.quote.subscribe(target, quote_type='tick', version='v1')

        logging.info(f'Done subscribe {len(targets)} targets')
        time.sleep(max((TimeEndStock - datetime.now()).total_seconds(), 0))
    except:
        logging.exception('Catch an exception:')
    finally:
        logging.info(f'{user} log-out: {api.logout()}')
    return "Task completed"


Tasks = {
    'create_env': [runCreateENV],
    'account_info': [runAccountInfo],
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
