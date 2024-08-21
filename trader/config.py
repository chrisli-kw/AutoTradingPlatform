import logging
import pandas as pd
import configparser
import shioaji as sj
from dotenv import dotenv_values
from datetime import datetime, timedelta


API = sj.Shioaji()
TODAY = datetime.today()
SystemConfig = configparser.ConfigParser()
SystemConfig.read('./lib/config.ini', encoding='utf8')


def get_env(account_name: str):
    return dotenv_values(f'./lib/envs/{account_name}.env')


def getList(section, option):
    content = SystemConfig.get(section, option)
    if len(content) > 0:
        return content.replace(' ', '').split(',')
    return []


def get_settings(section, option, dataType='str'):
    funcs = {
        'str': SystemConfig.get,
        'int': SystemConfig.getint,
        'float': SystemConfig.getfloat,
        'bool': SystemConfig.getboolean,
        'list': getList
    }
    if section in SystemConfig.sections():
        options = SystemConfig.options(section)
        if options and option.lower() in options:
            try:
                return funcs[dataType](section, option)
            except:
                return SystemConfig.get(section, option)

    if dataType == 'list':
        return []
    return ''


def get_holidays():
    try:
        filename = './lib/政府行政機關辦公日曆表.csv'
        df = pd.read_csv(filename, low_memory=False, encoding='big5')
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
        return {}


# 使用者相關
ACCOUNTS = get_settings("ACCOUNT", "USERS", dataType='list')
LOG_LEVEL = get_settings("ACCOUNT", "LOG_LEVEL")

# 資料庫相關
REDIS_HOST = get_settings('DB', 'REDIS_HOST')
REDIS_PORT = get_settings('DB', 'REDIS_PORT', dataType='int')
REDIS_PWD = get_settings('DB', 'REDIS_PWD')
HAS_REDIS = all(x for x in [REDIS_HOST, REDIS_PORT, REDIS_PWD])

DB_HOST = get_settings('DB', 'DB_HOST')
DB_PORT = get_settings('DB', 'DB_PORT', dataType='int')
DB_USER = get_settings('DB', 'DB_USER')
DB_PWD = get_settings('DB', 'DB_PWD')
DB_NAME = get_settings('DB', 'DB_NAME')
HAS_DB = all(x for x in [DB_HOST, DB_NAME, DB_PORT, DB_PWD, DB_USER])
DB_URL = f'{DB_USER}:{DB_PWD}@{DB_HOST}:{DB_PORT}' if HAS_DB else ''

PATH = get_settings('DATA', 'DATA_PATH')

# LINE notify
TOKEN_MONITOR = get_settings('LINENOTIFY', 'TOKEN_MONITOR')
TOKEN_INFO = get_settings('LINENOTIFY', 'TOKEN_INFO')


class StrategyNameList:
    StrategyLongNDT = get_settings('STRATEGY', 'Long', dataType='list')
    StrategyShortNDT = get_settings('STRATEGY', 'Short', dataType='list')
    StrategyLongDT = get_settings('STRATEGY', 'LongDT', dataType='list')
    StrategyShortDT = get_settings('STRATEGY', 'ShortDT', dataType='list')

    All = StrategyLongNDT + StrategyLongDT + StrategyShortNDT + StrategyShortDT
    Long = StrategyLongNDT + StrategyLongDT
    Short = StrategyShortNDT + StrategyShortDT
    DayTrade = StrategyLongDT + StrategyShortDT
    Code = {stra: f'Strategy{i+1}' for i, stra in enumerate(All)}


# 策略相關
StrategyList = StrategyNameList()
MonitorFreq = get_settings('STRATEGY', 'MonitorFreq', 'int')

# 交易相關
FEE_RATE = .01

# 時間相關
T15K = pd.to_datetime('09:15:00')
T30K = pd.to_datetime('09:30:00')
TStart = pd.to_datetime('09:05:00')
TEnd = pd.to_datetime('13:30:05')
TCheck1 = pd.to_datetime('09:03:00')
TBuy1 = pd.to_datetime('09:00:00')
TBuy2 = pd.to_datetime('13:15:00')
TTry = pd.to_datetime('13:25:00')
TimeStartStock = pd.to_datetime('09:00:00')
TimeEndStock = pd.to_datetime('13:30:00')

now = datetime.now()
if pd.to_datetime('00:00:00') < now < pd.to_datetime('05:00:00'):
    TODAY = TODAY - timedelta(days=1)
    TimeStartFuturesDay = pd.to_datetime('08:45:00') - timedelta(days=1)
    TimeEndFuturesDay = pd.to_datetime('13:45:00') - timedelta(days=1)
    TimeStartFuturesNight = pd.to_datetime('15:00:00') - timedelta(days=1)
    TimeEndFuturesNight = pd.to_datetime('05:00:00')
    TimeTransferFutures = pd.to_datetime('13:00:00') - timedelta(days=1)
else:
    TimeStartFuturesDay = pd.to_datetime('08:45:00')
    TimeEndFuturesDay = pd.to_datetime('13:45:00')
    TimeStartFuturesNight = pd.to_datetime('15:00:00')
    TimeEndFuturesNight = pd.to_datetime('05:00:00') + timedelta(days=1)
    TimeTransferFutures = pd.to_datetime('13:00:00')

TODAY_STR = TODAY.strftime("%Y-%m-%d")
holidays = get_holidays()

# 選股相關
SelectMethods = get_settings('SELECT', 'METHODS', dataType='list')

# 爬蟲相關
ConvertScales = get_settings('CRAWLER', 'SCALES', dataType='list')

# K棒特徵
KbarFeatures = {
    '2T': get_settings('KBARFEATURE', 'K2min', dataType='list'),
    '5T': get_settings('KBARFEATURE', 'K5min', dataType='list'),
    '15T': get_settings('KBARFEATURE', 'K15min', dataType='list'),
    '30T': get_settings('KBARFEATURE', 'K30min', dataType='list'),
    '60T': get_settings('KBARFEATURE', 'K60min', dataType='list'),
}
