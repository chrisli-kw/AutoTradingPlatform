import re
import time
import pandas as pd
from typing import Union
from datetime import datetime, timedelta

from ..config import TODAY_STR, holidays, TimeTransferFutures


class TimeTool:

    DueDays = {
        (d.year, d.month): d for d in pd.date_range('2001-01-01', '2030-12-31', freq='WOM-3WED')}

    def datetime_to_str(self, date: datetime):
        '''將datetime時間轉為字串, 輸出格式:YYYY-MM-DD'''
        return date.strftime("%Y-%m-%d")

    def str_to_datetime(self, date: str):
        '''將字串時間轉為datetime, 輸入格式:YYYY-MM-DD'''
        return datetime.strptime(date, "%Y-%m-%d")

    @staticmethod
    def now_str(format='%Y-%m-%d %a %H:%M:%S.%f'):
        '''取得當下時間戳的字串格式'''
        return datetime.now().strftime(format)

    def now_for_log(self):
        return f"【{self.now_str()}】"

    def _strf_timedelta(self, date: Union[str, datetime], delta: int):
        '''計算前N日的日期(str)'''

        if isinstance(date, str):
            date = self.str_to_datetime(date)

        return self.datetime_to_str(date - timedelta(days=delta))

    def get_buyday(self, day: datetime):
        '''計算進場日'''
        n = self.nday_diff(day) + 1
        return self._strf_timedelta(day, n)

    def nday_diff(self, date: datetime):
        '''
        遇假日的日數判斷
        若今日為星期一, 往前推算2天
        若今日為星期天, 往前推算1天
        其餘日子只要前一天為交易日則不須往前推算
        '''
        if (date.weekday() == 0):
            return 2
        elif (date.weekday() == 6):
            return 1
        return 0

    def date_diff(self, day1: Union[str, datetime], day2: Union[str, datetime]):
        '''計算兩個日期之間的天數'''
        if isinstance(day1, str):
            day1 = pd.to_datetime(day1)

        if isinstance(day2, str):
            day2 = pd.to_datetime(day2)

        return (day1 - day2).days

    def is_pass_time(self, time_target: Union[str, datetime]):
        '''檢查當下時間是否已過想檢查的目標時間'''
        if isinstance(time_target, str):
            time_target = pd.to_datetime(time_target)

        return datetime.now() >= time_target

    def count_n_kbars(self, start: datetime, end: datetime, scale: int):
        '''
        計算K棒數量
        scale單位: min
        '''
        # TODO: 扣除國定假日不開盤 & 收盤期間

        if isinstance(start, str):
            start = pd.to_datetime(start)

        if isinstance(end, str):
            end = pd.to_datetime(end)

        m1 = start.minute - (start.minute % scale)
        start = start.replace(minute=m1, second=0, microsecond=0)

        m2 = end.minute - (end.minute % scale)
        end = end.replace(minute=m2, second=0, microsecond=0)

        n_kbar = (end - start).total_seconds()/60/scale

        if start.weekday() == 4 and end.weekday() == 0:
            n_kbar -= 1440*2/scale

        return n_kbar

    def convert_date_format(self, x: str):
        '''轉換民國格式為西元格式'''

        x = re.findall('\d+', x)
        year = int(x[0])
        if year < 1911:
            x[0] = str(year + 1911)
        return '-'.join(x)

    def date_2_mktime(self, date: str):
        '''將日期轉為浮點數格式'''

        date = date.replace('-', '').replace('/', '')
        date = date.replace(' ', '').replace(':', '')
        date = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), 8)
        return int(time.mktime(date.timetuple()))

    def utcfromtimestamp_(self, timestamp):
        '''將浮點格式時間戳轉為datetime格式'''
        return datetime.utcfromtimestamp(timestamp)

    def last_business_day(self, date: datetime = None):
        '''取得最近一個交易日，遇假日, 連假, or補班日則往前推算'''

        if not date:
            date = TODAY_STR

        d = 1
        while True:
            day = pd.to_datetime(date) - timedelta(days=d)
            if day not in holidays and day.weekday() not in [5, 6]:
                return day

            d += 1

    def GetDueMonth(self, sourcedate: datetime, months: int = 1, refer_time=None):
        '''推算交割月份，在交割日之前的日期，交割月為當月，交割日之後為次月'''

        sourcedate = pd.to_datetime(sourcedate)
        dueday = self.DueDays[(sourcedate.year, sourcedate.month)]

        if refer_time is None:
            refer_time = datetime.now() <= TimeTransferFutures
        else:
            refer_time = sourcedate < dueday.replace(
                hour=13, minute=30, second=0)

        if sourcedate.date() <= dueday.date() and refer_time:
            return str(sourcedate.year) + str(sourcedate.month).zfill(2)

        month = sourcedate.month - 1 + months
        year = sourcedate.year + month // 12
        month = month % 12 + 1
        return str(year) + str(month).zfill(2)

    def CountDown(self, target: Union[int, datetime]):
        '''倒數計時'''

        if isinstance(target, int):
            N = target
        else:
            now = datetime.now()
            N = int((target - now).total_seconds())

        for i in range(N, 0, -1):
            print(
                f"Time remaining: {int(i/60)}min {i % 60}s", end="\r", flush=True)
            time.sleep(1)

        print("Time remaining: 0min 0s", end="\r", flush=True)
