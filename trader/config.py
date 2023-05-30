import pandas as pd
import configparser
from datetime import timedelta

config = configparser.ConfigParser()
config.read('./lib/config.ini', encoding='utf8')


def getList(section, option):
    content = config.get(section, option)
    if len(content) > 0:
        return content.replace(' ', '').split(',')
    return []

def get_settings(section, option, dataType='str'):
    funcs = {
        'str': config.get,
        'int': config.getint,
        'float': config.getfloat,
        'bool': config.getboolean,
        'list':getList
    }
    if section in config.sections():
        options = config.options(section)
        if options and option.lower() in options:
            return funcs[dataType](section, option)
    return ''


# 使用者相關
ACCOUNTS = get_settings("ACCOUNT", "USERS", dataType='list')

# 資料庫相關
REDIS_HOST = get_settings('DB', 'REDIS_HOST')
REDIS_PORT = get_settings('DB', 'REDIS_PORT', dataType='int')
REDIS_PWD = get_settings('DB', 'REDIS_PWD')
DB_HOST = get_settings('DB', 'DB_HOST')
DB_PORT = get_settings('DB', 'DB_PORT', dataType='int')
DB_USER = get_settings('DB', 'DB_USER')
DB_PWD = get_settings('DB', 'DB_PWD')
DB_NAME = get_settings('DB', 'DB_NAME')
HAS_DB = all(x for x in [DB_HOST, DB_NAME, DB_PORT, DB_PWD, DB_USER])
DB_URL = f'{DB_USER}:{DB_PWD}@{DB_HOST}:{DB_PORT}' if HAS_DB else ''

# LINE notify
TOKEN_MONITOR = get_settings('LINENOTIFY', 'TOKEN_MONITOR')
TOKEN_INFO = get_settings('LINENOTIFY', 'TOKEN_INFO')

# 策略相關
StrategyLong = get_settings('STRATEGY', 'Long', dataType='list')
StrategyShort = get_settings('STRATEGY', 'Short', dataType='list')
StrategyLongDT = get_settings('STRATEGY', 'LongDT', dataType='list')
StrategyShortDT = get_settings('STRATEGY', 'ShortDT', dataType='list')

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
TimeStartFuturesDay = pd.to_datetime('08:45:00')
TimeEndFuturesDay = pd.to_datetime('13:45:00')
TimeStartFuturesNight = pd.to_datetime('15:00:00')
TimeEndFuturesNight = pd.to_datetime('05:00:00') + timedelta(days=1)

# 選股相關
SelectMethods = get_settings('SELECT', 'methods', dataType='list')

# 爬蟲相關
ConvertScales = get_settings('CRAWLER', 'SCALES', dataType='list')

# K棒特徵
K60min_feature = get_settings('KBARFEATURE', 'K60min', dataType='list')
K30min_feature = get_settings('KBARFEATURE', 'K30min', dataType='list')
K15min_feature = get_settings('KBARFEATURE', 'K15min', dataType='list')
