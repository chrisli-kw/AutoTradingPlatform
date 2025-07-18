import io
import os
import re
import time
import json
import logging
import zipfile
import requests
import numpy as np
import pandas as pd
from io import StringIO
from typing import Union
from sqlalchemy import text
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from ..config import API, PATH, TODAY_STR, TODAY
from . import progress_bar, create_queue, concat_df
from .kbar import KBarTool
from .time import time_tool
from .file import file_handler
from .database import db, KBarTables
from .database.tables import KBarData1T, SecurityList, PutCallRatioList, ExDividendTable


def readStockList(markets=['OTC', 'TSE']):
    if db.HAS_DB:
        df = db.query(
            SecurityList,
            SecurityList.exchange.in_(markets)
        )
    else:
        df = pd.DataFrame(columns=['code', 'exchange'])

    return df


class CrawlFromSJ:
    def __init__(self, folder_path: str = f'{PATH}/Kbars/1T/{TODAY_STR}', scale='1D'):
        self.folder_path = folder_path
        self.kbartool = KBarTool()
        self.filename = 'company_stock_data'
        self.tempFile = f'{self.folder_path}/crawled_list.pkl'
        self.scale = scale
        self.StockData = []

    def get_security_list(self, market='Stocks', stock_only: bool = True):
        '''
        自 Shioaji 取得商品清單
        當 market='Stocks' 且只保留普通股股票且不需要權證: stock_only = True 
        '''
        id_list = [
            {**id} for exchange in API.Contracts.get(market) for id in exchange]
        df = pd.DataFrame(id_list)
        df = df[(df.update_date == df.update_date.max())]

        if market == 'Stocks' and stock_only:
            df.update_date = pd.to_datetime(df.update_date)
            df = df[
                ~(df.category.isin(['00', '  '])) & (df.code.apply(len) == 4)]

        return df

    def export_security_list(self, df: pd.DataFrame):
        '''Export security data either to local or to DB'''

        if not db.HAS_DB:
            return

        df.reference = df.reference.fillna(0)
        df.limit_up = df.limit_up.fillna(0)
        df.limit_down = df.limit_down.fillna(0)
        df.margin_trading_balance = df.margin_trading_balance.fillna(0)
        df.day_trade = df.day_trade.fillna(0)
        df.short_selling_balance = df.short_selling_balance.fillna(0)

        codes = db.query(SecurityList.code).code.values

        # add new data
        tb = df[~df.code.isin(codes)].copy()
        db.dataframe_to_DB(tb, SecurityList)

        # delete old data
        code1 = list(set(codes) - set(df.code))
        condition = SecurityList.code.in_(code1)
        db.delete(SecurityList, condition)

    def export_kbar_data(self, df: pd.DataFrame, scale: str):
        '''Export kbar data either to local or to DB'''

        if db.HAS_DB:
            db.dataframe_to_DB(df, KBarTables[scale])

        df.Volume = df.Volume.astype('int32')
        filename = f'{PATH}/Kbars/{scale}/{TODAY_STR}-stocks-{scale}.pkl'
        file_handler.Process.save_table(df, filename)
        return df

    def _load_data_into_queue(self, stockids: list):
        '''創建股票待爬清單(queue)'''

        # get stock id list
        if type(stockids) == str and stockids == 'all':
            stockids = readStockList().code.values
        elif type(stockids) == str:
            stockids = stockids.split(',')
        logging.info(f"Target list size: {len(stockids)}")

        # get crawled list
        df_default = pd.DataFrame({'stockid': []})
        self.crawled_list = file_handler.Process.read_table(
            self.tempFile, df_default)
        logging.info(f"Crawled size: {self.crawled_list.shape[0]}")

        return create_queue(stockids, self.crawled_list.stockid.values)

    def run(self, stockids: Union[str, list] = 'all', update=False, start=None, end=None):
        file_handler.Operate.create_folder(self.folder_path)
        q = self._load_data_into_queue(stockids)
        if isinstance(stockids, str) and stockids == 'all':
            q.put('TSE001')
            q.put('OTC101')
        N = q.qsize()

        if not start:
            start = TODAY_STR if update else '2017-01-01'

        self.StockData = np.array([None]*N)
        logging.info(f'Strat crawling, from {start} to {end}')
        while True:
            if q.empty():
                os.remove(self.tempFile)
                logging.info("Queue is deleted successfully")
                break

            i = N-q.qsize()
            stockid = q.get()
            tStart = time.time()

            try:
                df = self.kbartool.tbKBar(stockid, start, end)

                if df is not None:
                    self.StockData[i] = df

                    # back-up queue
                    self.crawled_list = concat_df(
                        self.crawled_list,
                        pd.DataFrame([{'stockid': stockid}])
                    )
                    file_handler.Process.save_table(
                        self.crawled_list, filename=self.tempFile)
            except:
                logging.exception(f"Put back into queue: {stockid}")
                q.put(stockid)

            progress_bar(
                N,
                N-q.qsize()-1,
                status=f'Getting {stockid} done ({round(time.time() - tStart, 2)}s)'
            )
            time.sleep(0.1)

    def merge_stockinfo(self):
        '''
        爬蟲爬下來的資料是按公司股票代號分別存成CSV檔
        此程式用來將全部的公司股票資料合併成一個
        '''

        logging.info(f"Merge {len(self.StockData)} stockinfo")
        if len(self.StockData):
            df = pd.concat(self.StockData)
            df.Time = pd.to_datetime(df.Time)
            df.name = df.name.astype(int).astype(str)
            self.StockData = df
        else:
            df = file_handler.read_tables_in_folder(
                self.folder_path, pattern='.csv')
            if df.shape[0]:
                if 'Time' in df.columns:
                    df['Time'] = pd.to_datetime(df['Time'])

                df.name = df.name.astype(int)

            file_handler.Operate.remove_files(self.folder_path, pattern='.csv')

        logging.info(f'Done, shape = {df.shape}')
        df = df.sort_values(['name', 'Time']).reset_index(drop=True)
        df = self.export_kbar_data(df, '1T')
        os.rmdir(self.folder_path)

    def add_new_data(self, scale: str, save=True, start=None, end=None):
        '''加入新資料到舊K棒資料中'''

        if isinstance(self.StockData, pd.DataFrame) and self.StockData.shape[0]:
            df = self.StockData.copy()
        elif db.HAS_DB:
            condition1 = KBarData1T.Time >= text(
                start) if start else text('1901-01-01')
            condition2 = KBarData1T.Time <= text(
                end) if end else text(TODAY_STR)
            df = db.query(KBarData1T, condition1, condition2)
        else:
            folders = file_handler.Operate.listdir(f'{PATH}/Kbars/1T')
            folders = [fd for fd in folders if '.' not in fd]

            if start:
                folders = [fd for fd in folders if fd >= start]

            if end:
                folders = [fd for fd in folders if fd <= end]

            N = len(folders)
            df = np.array([None]*N)
            for i, fd in enumerate(folders):
                filename = f'{PATH}/Kbars/1T/{fd}-stock_data_1T.pkl'
                tb = file_handler.Process.read_table(filename)
                tb = tb.sort_values(['name', 'Time'])

                if tb.shape[0]:
                    if 'Time' in tb.columns:
                        tb['Time'] = pd.to_datetime(tb['Time'])

                    tb.name = tb.name.astype(int)

                df[i] = tb

            df = pd.concat(df)
            df = df.sort_values(['name', 'Time'])
            df = df.reset_index(drop=True)
            df.name = df.name.astype(int).astype(str)

        if scale != '1T':
            logging.info(f'Converting data scale to {scale}...')
            df = self.kbartool.convert_kbar(df, scale=scale).dropna()

        if save:
            df = self.export_kbar_data(df, scale)

        return df

    def merge_daily_data(self, day: datetime, scale: str, save=True):
        '''
        This function merges daily kbar data at the 1st trading-day of 
        each month
        '''

        if not isinstance(day, datetime):
            day = pd.to_datetime(day)

        last_day = time_tool.last_business_day(day)
        if last_day.month != day.month:
            dir_path = f'{PATH}/Kbars/{scale}'
            year_month = time_tool.datetime_to_str(last_day)[:-3]
            df = file_handler.read_tables_in_folder(
                dir_path, pattern=year_month)
            df = df.sort_values(['name', 'Time'])
            df = df.reset_index(drop=True)

            if save:
                filename = f'{dir_path}/{year_month}-stocks-{scale}.pkl'
                file_handler.Operate.remove_files(dir_path, pattern=year_month)
                file_handler.Process.save_table(df, filename)


class CrawlFromHTML:

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url_pc_ratio = 'https://www.taifex.com.tw/cht/3/pcRatio'
    url_ex_dividend = 'https://www.twse.com.tw/rwd/zh/exRight/TWT48U?response=csv'
    url_futures_tick = 'https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV'
    url_cash_settle = 'https://www.sinotrade.com.tw/Stock/Stock_3_8_3'
    url_futures_margin = 'https://www.taifex.com.tw/cht/5/indexMarging'

    def above_one_url(self, pageName: str, date: str, stockid: str):
        '''產生豹投資網站資料的URL'''

        domain1 = 'https://www.above.one/cdn/TProStk'
        domain2 = 'https://www.above.one/cdn/TProStkChips'
        token = 'HVtjdOMP4f'
        URLs = {
            # 1. K線Header資訊
            'HeaderInfo': f'{domain1}/{pageName}/HVtjdOMP4f.json?day={date}&sym={stockid}&',

            # 2. 股懂券商排行榜
            'SymbolWinBroker': f'{domain2}/{pageName}/{token}.json?dateRange=20&day={date}&sym={stockid}&s=ProfitLoss:d',

            # 3. 分價量表
            'SymbolPeriodPriceAccumVol': f'{domain2}/{pageName}/{token}.json?day={date}&pastNDays=1&sym={stockid}&',

            # 4. 主力透視圖
            'DayMajorPriceNetVol': f'{domain2}/{pageName}/{token}.json?day={date}&sym={stockid}&',

            # 5. 外本比與投本比
            'FgnItrustRatio': f'{domain2}/{pageName}/{token}.json?ed={date}&kfreq=day&sd=2022-07-22&sym={stockid}&',

            # 6. 三大法人成交比
            'ExternalRatio': f'{domain2}/{pageName}/{token}.json?ed={date}&kfreq=day&sd=2022-07-22&sym={stockid}&s=Date:d',

            # 7. 融資融券
            'MarginOffsetRatio': f'{domain2}/{pageName}/{token}.json?ed={date}&kfreq=day&sd=2022-07-22&sym={stockid}&s=Date:d',

            # 8. 借券賣出餘額
            'MgnSBLRatio': f'{domain2}/{pageName}/{token}.json?ed={date}&kfreq=day&sd=2022-07-22&sym={stockid}&s=Date:d',

            # 9. 區間週轉率
            'RangeTurnover': f'{domain2}/{pageName}/{token}.json?ed={date}&pastNDays=5&sd=2022-07-15&sym={stockid}&',

            # 10. 當日沖銷
            'DayTrade': f'{domain2}/{pageName}/{token}.json?ed={date}&kfreq=day&sd=2022-07-22&sym={stockid}&s=Date:d',

            # 11. 大戶與羊群持股
            'LambsAndMajorHold': f'{domain2}/{pageName}/{token}.json?ed={date}&holdVolMajor=15&kfreq=day&sd=2021-11-23&sym={stockid}&s=Date:d',

            # 12. 集保戶數
            'TdccCount': f'{domain2}/{pageName}/{token}.json?day={date}&sym={stockid}&',

            # 13. 大股東持股變動
            'ShareBalChg': f'{domain2}/{pageName}/{token}.json?holdVolMajor=12&sym={stockid}&s=Date:d',

            # 14. 券商買超統計
            'BrkBuyNet': f'{domain2}/{pageName}/{token}.html?ed={date}&sd={date}&sym={stockid}&s=BuyNetVol:d&sno=0',

            # 15. 券商賣超統計
            'BrkSellNet': f'{domain2}/{pageName}/{token}.html?ed={date}&sd={date}&sym={stockid}&s=SellNetVol:d&sno=0',

            # 16. 內部人持股比率
            'ShareHoldRatio': f'{domain2}/{pageName}/{token}.json?ed=2022-08-31&sd=2015-03-31&sym={stockid}&s=YearMonth:d',

            # 17. 董監大股東持股
            'ShareBal': f'{domain2}/{pageName}/{token}.html?sym={stockid}&ym=2022%2F8&sno=0',

            # 18. 券商買超統計
            'BrkBuyNet_InputRadio': f'{domain2}/{pageName}/{token}.html?ed={date}&sd={date}&sym={stockid}&s=BuyNetVol:d&sno=0',

            # 19. 股懂券商
            'SymbolWinBroker_InputRadio': f'{domain2}/{pageName}/{token}.html?dateRange=20&day={date}&sym={stockid}&s=ProfitLoss:d&sno=0'
        }
        return URLs[pageName]

    def Leverage(self, stockid: str):
        '''取得個股融資成數'''

        url = f'https://www.sinotrade.com.tw/Stock/Stock_3_8_6?code={stockid}'
        tb = pd.read_html(url)[-1]

        if tb.shape[0] == 0:
            return {'融資成數': 0, '融券成數': 100}
        return tb.to_dict('records')[0]

    def PunishList(self, format='List'):
        '''取得上市櫃處置股票清單(證券代號、名稱、處置起訖日)'''

        def get(url, period):
            df_default = pd.DataFrame(columns=['證券代號', '證券名稱', 'period'])

            n = 0
            while n < 10:
                try:
                    df = requests.get(url, headers=self.headers).text
                    df = pd.read_html(StringIO(df))[0]
                    df = df.dropna().rename(
                        columns={period: 'period'}).iloc[:-1, :]
                    return df
                except:
                    logging.exception('Catch an exception (PunishList):')
                    n += 1
                    time.sleep(1)
            return df_default

        # 上市處置公告
        start = time_tool._strf_timedelta(TODAY, 30).replace('-', '')
        url = f'https://www.twse.com.tw/announcement/punish?response=html&startDate={start}&endDate='
        df1 = get(url, '處置起迄時間')
        df1 = df1[['證券代號', '證券名稱', 'period']]

        # 上櫃處置公告
        start = time_tool._strf_timedelta(TODAY, 30).replace('-', '/')
        start = start.replace(start[:4], str(int(start[:4])-1911))
        url = f'https://www.tpex.org.tw/web/bulletin/disposal_information/disposal_information_print.php?l=zh-tw&sd={start}'
        df2 = get(url, '處置起訖時間')
        df2 = df2[['證券代號', '證券名稱', 'period']]

        # 合併
        df = concat_df(df1, df2)
        df.period = df.period.apply(lambda x: re.findall('[\d+/]+', x))
        df.證券代號 = df.證券代號.astype(str)
        df['startDate'] = df.period.apply(
            lambda x: x[0].replace(x[0][:3], str(int(x[0][:3])+1911)))
        df['startDate'] = pd.to_datetime(df['startDate'])
        df['endDate'] = df.period.apply(
            lambda x: x[1].replace(x[1][:3], str(int(x[1][:3])+1911)))
        df['endDate'] = pd.to_datetime(df['endDate'])
        df = df.sort_values('startDate')
        df = df.drop_duplicates('證券代號', keep='last').drop('period', axis=1)
        df = df[df.endDate >= TODAY_STR]

        if format == 'List':
            return df.證券代號.to_list()
        return df

    def PutCallRatio(self, start: str = '', end: str = ''):
        if not start:
            start = TODAY_STR.replace('-', '/')

        if not end:
            end = TODAY_STR.replace('-', '/')

        df = pd.DataFrame()

        s = start
        while True:
            e = str(
                (pd.to_datetime(s) + timedelta(days=30)).date()).replace('-', '/')
            logging.info(f'Period: {s} - {e}')

            url = f'{self.url_pc_ratio}?queryStartDate={s}&queryEndDate={e}'
            try:
                tb = pd.read_html(url)[3]
            except:
                tb = pd.read_html(url, encoding='utf-8')[0]

            df = concat_df(df, tb)

            if pd.to_datetime(e) >= pd.to_datetime(end):
                break

            s = str(
                (pd.to_datetime(e) + timedelta(days=1)).date()).replace('-', '/')
            time.sleep(10)

        df = df.rename(columns={
            '日期': 'Date',
            '賣權成交量': 'PutVolume',
            '買權成交量': 'CallVolume',
            '買賣權成交量比率%': 'PutCallVolumeRatio',
            '賣權未平倉量': 'PutOpenInterest',
            '買權未平倉量': 'CallOpenInterest',
            '買賣權未平倉量比率%': 'PutCallRatio'
        })
        df.Date = pd.to_datetime(df.Date)

        return df.sort_values('Date').reset_index(drop=True)

    def export_put_call_ratio(self, df: pd.DataFrame):
        if not db.HAS_DB:
            return

        dates = db.query(PutCallRatioList.Date).Date
        db.dataframe_to_DB(df[~df.Date.isin(dates)], PutCallRatioList)

    def export_futures_kbar(self, df: pd.DataFrame):
        if db.HAS_DB:
            db.dataframe_to_DB(df, KBarData1T)

        if df.empty:
            return

        year_month = TODAY_STR[:-3]
        filename = f'{PATH}/Kbars/1T/{year_month}-futures-1T.pkl'
        df.Volume = df.Volume.astype('int32')
        df = file_handler.Process.read_and_concat(filename, df)
        file_handler.Process.save_table(df, filename)

    def export_ex_dividend_list(self, df: pd.DataFrame):
        if not db.HAS_DB:
            return

        df_old = db.query(ExDividendTable)
        if not df_old.empty:
            df = df[df.Date > df_old.Date.max()].copy()

        db.dataframe_to_DB(df, ExDividendTable)

    def ex_dividend_list(self):
        '''爬蟲:證交所除權息公告表'''

        try:
            df = pd.read_csv(
                self.url_ex_dividend, encoding='cp950', on_bad_lines='skip')
        except:
            df = pd.read_csv(
                self.url_ex_dividend, encoding='big5', on_bad_lines='skip')

        df = df.reset_index()
        df.columns = df.iloc[0, :]
        df = df.rename(columns={
            '除權除息日期': 'Date',
            '股票代號': 'Code',
            '名稱': 'Name',
            '除權息': 'DividendType',
            '無償配股率': 'DividendRate',
            '現金增資配股率': 'CashCapitalRate',
            '現金增資認購價': 'CashCapitalPrice',
            '現金股利': 'CashDividend',
            '詳細資料': 'Details',
            '參考價試算': 'Reference',
            '最近一次申報資料 季別/日期': 'Quarter',
            '最近一次申報每股 (單位)淨值': 'NetValue',
            '最近一次申報每股 (單位)盈餘': 'EPS'
        })
        df = df.iloc[1:, :-1]
        df = df[df.Code.notnull()]
        df = df[(df.Code.apply(len) == 4)]
        df.Date = df.Date.apply(time_tool.convert_date_format)
        df.Date = pd.to_datetime(df.Date)
        df.CashDividend = df.CashDividend.replace('尚未公告', -1).astype(float)
        df.CashCapitalPrice = df.CashCapitalPrice.replace('尚未公告', -1)
        return df.sort_values('Date')

    def DowJones(self, start: str, end: str):
        '''鉅亨網道瓊報價'''

        url_dow_jones = 'https://ws.api.cnyes.com/ws/api/v1/charting/history'
        start = time_tool.date_2_mktime(start)
        end = time_tool.date_2_mktime(end)
        params = f'?resolution=D&symbol=GI:DJI:INDEX&from={end}&to={start}'

        try:
            result = requests.get(url_dow_jones+params)
            result = json.loads(result.text)
            if 'data' not in result:
                result['data'] = {}
        except requests.exceptions.ConnectionError as e:
            logging.warning(e)
            result = {'data': {}}
        except:
            logging.exception('【Error】DowJones:')
            result = {'data': {}}

        return result['data']

    def DowJones_pct_chg(self):
        '''取得道瓊指數前一天的漲跌幅'''

        start = time_tool._strf_timedelta(TODAY, 30)
        dj = self.DowJones(start, TODAY_STR)
        if 'c' in dj and len(dj['c']):
            dj = dj['c']
            return 100*round(dj[0]/dj[1] - 1, 4)
        return 0

    def get_SymbolWinBroker_InputRadio(self, date: str, stockid: str):
        '''股懂券商'''

        url = self.above_one_url('SymbolWinBroker_InputRadio', date, stockid)
        content = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(content.text, "lxml")
        soup = soup.find(
            "table", {"class": "TProStkChips/SymbolWinBroker_InputRadio_css"})
        soup = soup.find_all('td')
        soup = [td.text.replace('▼➀', '').replace('%', '')
                for td in soup if len(td.text)]

        columns = soup[:3]
        soup = np.reshape(soup[3:], (int(len(soup[3:])/3), 3))
        df = pd.DataFrame(soup, columns=columns)
        df['損益(百萬)'] = df['損益(百萬)'].astype(float)
        df.分點報酬率 = df.分點報酬率.astype(float)
        df = df.sort_values('分點報酬率', ascending=False).reset_index(drop=True)
        return df

    def get_SymbolWinBroker(self, date: str, stockid: str):
        '''股懂券商排行榜'''

        url = self.above_one_url('SymbolWinBroker', date, stockid)
        result = requests.get(url, headers=self.headers)
        result = json.loads(result.text)['result']
        df = pd.DataFrame(result['data'], columns=result['cheaders'])
        df = df.sort_values('分點報酬率', ascending=False).reset_index(drop=True)
        return df

    def get_SymbolBrkNetVol(self, df_brokers: pd.DataFrame, stockid: str, topN=2):
        '''查分點報酬前N高的券商買賣超'''

        brokers = df_brokers.head(topN).券商代號.values

        domain = 'https://www.above.one/cdn/TProStk'
        pageName = 'SymbolBrkNetVol'
        krange2 = 'OneYear'
        tb = pd.DataFrame(columns=['日期', 'Code'])
        for brkId in brokers:
            url = f'{domain}/AuxData/RpNpJBcMvT.json?s=Date:a&aux={pageName}&auxParams=brkId_{brkId}&kfreq=day&krange2={krange2}&sym={stockid}&country=Tw'
            result = requests.get(url, headers=self.headers)
            result = json.loads(result.text)['result']
            df = pd.DataFrame(result['data'], columns=result['cheaders'])
            df.日期 = pd.to_datetime(df.日期)
            df = df.sort_values('日期').rename(columns={'買超股數': f'{brkId}'})
            df.insert(1, 'Code', stockid)
            tb = tb.merge(df, how='outer', on=['日期', 'Code'])
            time.sleep(.5)

        tb['TotalBuy'] = tb.iloc[:, 2:].sum(axis=1)  # .cumsum()
        return tb

    def get_FuturesTickData(self, date: str):
        '''前30個交易日期貨每筆成交資料'''

        try:
            year = date.split('-')[0]
            month = date.split('-')[1]
            date = date.replace('-', '_')

            result = requests.get(f"{self.url_futures_tick}/Daily_{date}.zip")
            z = zipfile.ZipFile(io.BytesIO(result.content))
            z.extractall(f'{PATH}/ticks/futures/{year}/Daily_{year}_{month}')
        except zipfile.BadZipFile:
            logging.error('輸入的日期非交易日')
        except:
            logging.exception('Catch an exception (get_FuturesTickData):')

    def get_CashSettle(self):
        '''取得交易當日全額交割股清單'''

        try:
            df = pd.read_html(self.url_cash_settle)
            return df[0]
        except:
            logging.warning('查無全額交割股清單')
            return pd.DataFrame(columns=['股票代碼'])

    def get_IndexMargin(self):
        '''取得期貨股價指數類保證金'''

        try:
            df = pd.read_html(self.url_futures_margin, encoding='utf8')
            return df[0]
        except:
            logging.exception('查詢失敗：')
            return pd.DataFrame(columns=['商品別', '結算保證金', '維持保證金', '原始保證金'])


class Crawler:
    def __init__(self) -> None:
        self.FromSJ = CrawlFromSJ()
        self.FromHTML = CrawlFromHTML()


crawler = Crawler()
