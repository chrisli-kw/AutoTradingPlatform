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
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Union

from . import progress_bar, create_queue, delete_selection_files, save_csv
from .notify import Notification
from .kbar import KBarTool
from .time import TimeTool
from .. import API, PATH, TODAY_STR, TODAY


class CrawlStockData:
    def __init__(self, folder_path: str = '', scale='1D'):
        if not folder_path:
            self.folder_path = f'{PATH}/Kbars/1min/{TODAY_STR}'
        else:
            self.folder_path = folder_path

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        self.notifier = Notification()
        self.timetool = TimeTool()
        self.kbartool = KBarTool()
        self.filename = 'company_stock_data'
        self.scale = scale
        self.StockData = []

    def get_stock_list(self, stock_only: bool = True):
        '''
        自 Shioaji 取得股票清單
        只保留普通股股票且不需要權證: stock_only = True 
        '''
        stock_list = [{**stock} for exchange in API.Contracts.Stocks for stock in exchange]
        df = pd.DataFrame(stock_list)
        df = df[(df.update_date == df.update_date.max())]

        if stock_only:
            df.update_date = pd.to_datetime(df.update_date)
            df = df[~(df.category.isin(['00', '  '])) & (df.code.apply(len) == 4)]

        return df

    def _load_data_into_queue(self, stockids: list):
        '''創建股票待爬清單(queue)'''

        # get stock id list
        if type(stockids) == str and stockids == 'all':
            stockids = pd.read_excel(f'{PATH}/selections/stock_list.xlsx')
            stockids.code = stockids.code.astype(int).astype(str)
            stockids = stockids[stockids.exchange.isin(['OTC', 'TSE'])].code.values

        elif type(stockids) == str:
            stockids = stockids.split(',')
        logging.info(f"Target list size: {len(stockids)}")

        # get crawled list
        if 'crawled_list.pkl' in os.listdir(self.folder_path):
            self.crawled_list = pd.read_pickle(f'{self.folder_path}/crawled_list.pkl')
        else:
            self.crawled_list = pd.DataFrame({'stockid': []})
            self.crawled_list.to_pickle(f'{self.folder_path}/crawled_list.pkl')
        logging.info(f"Crawled size: {self.crawled_list.shape[0]}")

        return create_queue(stockids, self.crawled_list.stockid.values)

    def _listfiles(self, folder_path: str):
        files = os.listdir(folder_path)
        return [f for f in files if '.csv' in f]

    def crawl_from_sinopac(self, stockids: Union[str, list] = 'all', update=False, start=None, end=None):

        q = self._load_data_into_queue(stockids)
        if isinstance(stockids, str) and stockids == 'all':
            q.put('TSE001')
            q.put('OTC101')
        N = q.qsize()

        if not start:
            if update:
                filename = f'{PATH}/Kbars/{self.filename}_{self.scale}.pkl'
                if os.path.exists(filename):
                    last_end = pd.read_pickle(filename).date.max()
                else:
                    last_end = self.timetool.last_business_day()
                start = self.timetool._strf_timedelta(last_end, -1)

                del old
            else:
                start = '2017-01-01'

        self.StockData = np.array([None]*N)
        while True:
            if q.empty():
                os.remove(f'{self.folder_path}/crawled_list.pkl')
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
                    self.crawled_list = pd.concat([
                        self.crawled_list,
                        pd.DataFrame([{'stockid': stockid}])
                    ])
                    self.crawled_list.to_pickle(f'{self.folder_path}/crawled_list.pkl')

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

        if len(self.StockData):
            df = pd.concat(self.StockData)
        else:
            files = self._listfiles(self.folder_path)
            N = len(files)
            df = np.array([None]*N)

            logging.info('Merging stockinfo...')
            for i, f in enumerate(files):
                try:
                    tb = pd.read_csv(f'{self.folder_path}/{f}').sort_values('date')
                except:
                    logging.exception('Catch an exception:')
                    self.notifier.post(f"\n【Error】讀取檔案發生異常:{f}", msgType='Crawler')
                    tb = pd.DataFrame()

                if tb.shape[0]:
                    for col in ['date', 'Time']:
                        if col in tb.columns:
                            tb[col] = pd.to_datetime(tb[col])

                    tb.name = tb.name.astype(int)

                    df[i] = tb

                n = i+1
                progress = f"\r|{'█'*int(n*50/N)}{' '*(50-int(n*50/N))} | [{files[i]}] {n}/{N} ({round(n/N*100, 2)}%)"
                print(progress, end='')
            print('')

            df = pd.concat(df)

            delete_selection_files(self.folder_path, files=files)

        df = df.sort_values(['name', 'date', 'Time']).reset_index(drop=True)
        day = self.folder_path.split('/')[-1]
        save_csv(df, f'{self.folder_path}/{day}-stock_data_1T.csv')

    def add_new_data(self, scale: str, save=True, start=None, end=None):
        '''加入新資料到舊K棒資料中'''

        folders = os.listdir(f'{PATH}/Kbars/1min')
        folders = [fd for fd in folders if '.' not in fd]

        if start:
            folders = [fd for fd in folders if fd >= start]

        if end:
            folders = [fd for fd in folders if fd <= end]

        N = len(folders)
        temp = np.array([None]*N)
        for i, fd in enumerate(folders):
            tb = pd.read_csv(
                f'{PATH}/Kbars/1min/{fd}/{fd}-stock_data_1T.csv').sort_values('date')

            if tb.shape[0]:
                for col in ['date', 'Time']:
                    if col in tb.columns:
                        tb[col] = pd.to_datetime(tb[col])

                tb.name = tb.name.astype(int)

            temp[i] = tb

        temp = pd.concat(temp).sort_values(['name', 'date', 'Time']).reset_index(drop=True)
        temp.name = temp.name.astype(int).astype(str)

        if scale != '1T':
            logging.info(f'Converting data scale to {scale}...')
            temp = self.kbartool.convert_kbar(temp, scale=scale).dropna()

        filename = f'{PATH}/Kbars/{self.filename}_{scale}.pkl'
        if os.path.exists(filename):
            df = pd.read_pickle(filename)
        else:
            df = pd.DataFrame()

        df = pd.concat([df, temp]).reset_index(drop=True)
        if save:
            df.to_pickle(filename)

        return df


class CrawlFromHTML(TimeTool):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url_pc_ratio = 'https://www.taifex.com.tw/cht/3/pcRatio'
    url_ex_dividend = 'https://www.twse.com.tw/rwd/zh/exRight/TWT48U?response=csv'
    url_futures_tick = 'https://www.taifex.com.tw/file/taifex/Dailydownload/DailydownloadCSV'

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

    def get_leverage(self, stockid: str):
        '''取得個股融資成數'''

        url = f'https://www.sinotrade.com.tw/Stock/Stock_3_8_6?code={stockid}'
        tb = pd.read_html(url)[-1]

        if tb.shape[0] == 0:
            return {'融資成數': 0, '融券成數': 100}
        return tb.to_dict('records')[0]

    def get_punish_list(self):
        '''取得上市櫃處置股票清單(證券代號、名稱、處置起訖日)'''

        def get(url, period):
            df_default = pd.DataFrame(columns=['證券代號', '證券名稱', 'period'])

            n = 0
            while n < 10:
                try:
                    df = pd.read_html(url)[0]
                    df = df.dropna().rename(columns={period: 'period'}).iloc[:-1, :]
                    return df
                except:
                    logging.exception('Catch an exception:')
                    n += 1
                    time.sleep(1)
            return df_default

        # 上市處置公告
        start = self._strf_timedelta(TODAY, 30).replace('-', '')
        url = f'https://www.twse.com.tw/announcement/punish?response=html&startDate={start}&endDate='
        df1 = get(url, '處置起迄時間')
        df1 = df1[['證券代號', '證券名稱', 'period']]

        # 上櫃處置公告
        start = self._strf_timedelta(TODAY, 30).replace('-', '/')
        start = start.replace(start[:4], str(int(start[:4])-1911))
        url = f'https://www.tpex.org.tw/web/bulletin/disposal_information/disposal_information_print.php?l=zh-tw&sd={start}'
        df2 = get(url, '處置起訖時間')
        df2 = df2[['證券代號', '證券名稱', 'period']]

        # 合併
        df = pd.concat([df1, df2])
        df.period = df.period.apply(lambda x: re.findall('[\d+/]+', x))
        df.證券代號 = df.證券代號.astype(str)
        df['startDate'] = df.period.apply(lambda x: x[0].replace(x[0][:3], str(int(x[0][:3])+1911)))
        df['startDate'] = pd.to_datetime(df['startDate'])
        df['endDate'] = df.period.apply(lambda x: x[1].replace(x[1][:3], str(int(x[1][:3])+1911)))
        df['endDate'] = pd.to_datetime(df['endDate'])
        df = df.sort_values('startDate').drop_duplicates('證券代號', keep='last').drop('period', axis=1)
        df = df[df.endDate >= TODAY_STR]

        return df

    def put_call_ratio(self, start: str = '', end: str = ''):
        if not start:
            start = TODAY_STR.replace('-', '/')

        if not end:
            end = TODAY_STR.replace('-', '/')

        df = pd.DataFrame()

        s = start
        while True:
            e = str((pd.to_datetime(s) + timedelta(days=30)).date()).replace('-', '/')
            logging.info(f'Period: {s} - {e}')

            url = f'{self.url_pc_ratio}?queryStartDate={s}&queryEndDate={e}'
            tb = pd.read_html(url)[3]
            df = pd.concat([df, tb])

            if pd.to_datetime(e) >= pd.to_datetime(end):
                break

            s = str((pd.to_datetime(e) + timedelta(days=1)).date()).replace('-', '/')
            time.sleep(10)

        df = df.rename(
            columns={'買賣權成交量比率%': '買賣權成交量比率', '買賣權未平倉量比率%': '買賣權未平倉量比率'}
        )
        df.日期 = pd.to_datetime(df.日期)

        return df.sort_values('日期').reset_index(drop=True)

    def ex_dividend_list(self):
        '''爬蟲:證交所除權息公告表'''

        df = pd.read_csv(self.url_ex_dividend, encoding='big5', error_bad_lines=False).reset_index()
        df.columns = df.iloc[0, :]
        df = df.iloc[1:, :8]
        df = df[df.股票代號.notnull()]
        df = df[(df.股票代號.apply(len) == 4)]
        df.除權除息日期 = df.除權除息日期.apply(self.convert_date_format)
        df.現金股利 = df.現金股利.astype(float)
        return df.sort_values('除權除息日期')

    def DowJones(self, start: datetime, end: datetime):
        '''鉅亨網道瓊報價'''

        url_dow_jones = 'https://ws.api.cnyes.com/ws/api/v1/charting/history'
        start = self.date_2_mktime(start)
        end = self.date_2_mktime(end)
        params = f'?resolution=D&symbol=GI:DJI:INDEX&from={end}&to={start}'

        try:
            result = requests.get(url_dow_jones+params)
            result = json.loads(result.text)
            if 'data' not in result:
                result['data'] = {'c': []}
        except:
            logging.exception('【Error】DowJones:')
            result['data'] = {'c': []}

        return result['data']['c']

    def get_SymbolWinBroker_InputRadio(self, date: str, stockid: str):
        '''股懂券商'''

        url = self.above_one_url('SymbolWinBroker_InputRadio', date, stockid)
        content = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(content.text, "lxml")
        soup = soup.find("table", {"class": "TProStkChips/SymbolWinBroker_InputRadio_css"})
        soup = soup.find_all('td')
        soup = [td.text.replace('▼➀', '').replace('%', '') for td in soup if len(td.text)]

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
            logging.exception('Catch an exception:')
