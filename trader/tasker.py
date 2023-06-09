import time
import logging
import pandas as pd
from datetime import datetime
from dotenv import dotenv_values
from concurrent.futures import as_completed

from . import executor, notifier, picker, crawler1, crawler2, tdp, file_handler
from .create_env import app
from .config import API, PATH, TODAY_STR
from .config import ACCOUNTS, TEnd, SelectMethods, ConvertScales
from .utils.database import redis_tick
from .utils.subscribe import Subscriber
from .utils.accounts import AccountInfo
from .executor import StrategyExecutor
from .performance.reports import PerformanceReport


def runCreateENV():
    file_handler.create_folder('./lib')
    file_handler.create_folder('./lib/envs')
    file_handler.create_folder('./lib/schedules')
    app.run()


def runAccountInfo():
    # account = AccountInfo()
    # df = account.create_info_table()
    # tables = {}

    try:
        logging.debug(f'ACCOUNTS: {ACCOUNTS}')
        for env in ACCOUNTS:
            logging.debug(f'Load 【{env}】 config')
            config = dotenv_values(f'./lib/envs/{env}.env')

            pr = PerformanceReport(env)
            Tables = pr.getTables(config)
            pr.save_tables(Tables)
            pr.plot_performance_report(Tables, save=True)

            notifier.post(
                pr.TablesFile.split('/')[-1][:-5],
                image_name=pr.TablesFile.replace('xlsx', 'jpg'),
                msgType='AccountInfo'
            )
            break

        #     API_KEY = config['API_KEY']
        #     SECRET_KEY = config['SECRET_KEY']
        #     acct = config['ACCOUNT_NAME']
        #     account._login(API_KEY, SECRET_KEY, acct)
        #     time.sleep(1)

        #     row = account.query_all()
        #     if row:
        #         for i, data in enumerate(row):
        #             if i < 3:
        #                 logging.info(f"{data}： {row[data]}")
        #             else:
        #                 logging.info(f"{data}： NT$ {'{:,}'.format(row[data])}")

        #         if hasattr(df, 'sheet_names') and acct in df.sheet_names:
        #             tb = account.update_info(df, row)
        #         else:
        #             tb = pd.DataFrame([row])

        #         tables[acct] = tb

        #         # 推播訊息
        #         account_id = API.stock_account.account_id
        #         notifier.post_account_info(account_id, row)

        #     elif hasattr(df, 'sheet_names') and account.account_name in df.sheet_names:
        #         tables[acct] = pd.read_excel(
        #             df, sheet_name=account.account_name)

        #     else:
        #         tables[acct] = account.DEFAULT_TABLE

        #     # 登出
        #     time.sleep(5)
        #     logging.info(f'登出系統: {API.logout()}')
        #     time.sleep(10)

        # logging.info('儲存資訊')
        # writer = pd.ExcelWriter(
        #     f'{PATH}/daily_info/{account.filename}', engine='xlsxwriter')

        # for sheet in tables:
        #     try:
        #         tables[sheet].to_excel(
        #             writer, encoding='utf-8-sig', index=False, sheet_name=sheet)
        #     except:
        #         logging.exception('Catch an exception:')
        #         tables[sheet].to_excel(
        #             sheet+'.csv', encoding='utf-8-sig', index=False)
        # writer.save()
    except:
        logging.exception('Catch an exception:')
        notifier.post('\n【Error】【帳務資訊查詢】發生異常', msgType='Tasker')
        API.logout()


def runAutoTrader(account):
    try:
        config = dotenv_values(f'./lib/envs/{account}.env')
        se = StrategyExecutor(config=config)
        se.login_and_activate()
        se.run()
    except KeyboardInterrupt:
        notifier.post(
            f"\n【Interrupt】【下單機監控】{se.ACCOUNT_NAME}已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(
            f"\n【Error】【下單機監控】{se.ACCOUNT_NAME}發生異常", msgType='Tasker')
    finally:
        try:
            se.output_files()
        except:
            logging.exception('Catch an exception (output_files):')
            notifier.post(
                f"\n【Error】【下單機監控】{se.ACCOUNT_NAME}資料儲存失敗", msgType='Tasker')

        logging.info(f'登出系統: {API.logout()}')
        notifier.post(f"\n【停止監控】{se.ACCOUNT_NAME}關閉程式並登出", msgType='Tasker')

    del se


def runCrawlStockData(account):
    target = pd.to_datetime('15:05:00')
    config = dotenv_values(f'./lib/envs/{account}.env')
    aInfo = AccountInfo()

    try:
        now = datetime.now()
        if now < target:
            logging.info(
                f'Current time is still early, will start to crawl after {target}')
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
        stock_list = crawler1.get_security_list(stock_only=True)
        crawler1.export_security_list(stock_list)

        # 爬當天股價資料
        crawler1.crawl_from_sinopac(stockids='all', update=True)
        crawler1.merge_stockinfo()

        # 更新歷史資料
        for scale in ConvertScales:
            crawler1.add_new_data(scale, save=True, start=TODAY_STR)
            crawler1.merge_daily_data(TODAY_STR, scale, save=True)

    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】股價爬蟲發生異常", msgType='Tasker')
        if len(crawler1.StockData):
            df = pd.concat(crawler1.StockData)
            df.to_pickle(f'{crawler1.folder_path}/stock_data_1T.pkl')
    finally:
        logging.info(f'登出系統: {API.logout()}')


def runSelectStock():
    try:
        picker.setScripts(SelectMethods)
        df = picker.pick(3, 1.8, 3)
        df = picker.melt_table(df)
        tb = df[df.date == TODAY_STR].reset_index(drop=True)
        picker.export(tb)
        notifier.post_stock_selection(tb)

    except FileNotFoundError as e:
        logging.warning(f'{e} No stock is selected.')
    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【選股程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【選股程式】選股發生異常", msgType='Tasker')


def runCrawlFromHTML():
    try:
        # update PutCallRatio
        step = 'PutCallRatio'
        df_pcr_new = crawler2.put_call_ratio()
        crawler2.export_put_call_ratio(df_pcr_new)
        notifier.post_put_call_ratio(df_pcr_new)

        # 爬除權息資料
        step = '爬除權息資料'
        dividends = crawler2.ex_dividend_list()
        crawler2.export_ex_dividend_list(dividends)

        # 期貨逐筆成交資料
        step = '期貨逐筆成交資料'
        crawler2.get_FuturesTickData(TODAY_STR)

        # 轉換&更新期貨逐筆成交資料
        df = tdp.convert_daily_tick(TODAY_STR, '1T')
        crawler2.export_futures_kbar(df)

    except KeyboardInterrupt:
        notifier.post(f"\n【Interrupt】【爬蟲程式】已手動關閉", msgType='Tasker')
    except:
        logging.exception('Catch an exception:')
        notifier.post(f"\n【Error】【爬蟲程式】{step}發生異常", msgType='Tasker')


def thread_subscribe(user, targets):
    import shioaji as sj

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
            redis_tick.to_redis({tick.code: tick_data})

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
    # TODO: 讀取要盤中選股的股票池
    df = pd.read_excel(f'{PATH}/selections/stock_list.xlsx')
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
        for account in ACCOUNTS:
            config = dotenv_values(f'./lib/envs/{account}.env')

            if config['MODE'] == 'Simulation':
                se = StrategyExecutor(config=config)

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
    'update_and_select_stock': [runCrawlStockData, runSelectStock, runCrawlFromHTML],
    'crawl_stock_data': [runCrawlStockData],
    'select_stock': [runSelectStock],
    'crawl_html': [runCrawlFromHTML],
    'auto_trader': [runAutoTrader],
    'subscribe': [runShioajiSubscriber],
}
