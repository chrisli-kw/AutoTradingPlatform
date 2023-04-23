import pandas as pd
import configparser
from datetime import timedelta

config = configparser.ConfigParser()
config.read('./lib/config.ini', encoding='utf8')


def getlist(section, option):
    content = config.get(section, option)
    if len(content) > 0:
        return content.replace(' ', '').split(',')
    return []


# 使用者相關
ACCOUNTS = getlist("ACCOUNT", "USERS")

# 資料庫相關
if 'DB' in config.sections() and config.options('DB'):
    REDIS_HOST = config.get('DB', 'REDIS_HOST')
    REDIS_PORT = config.getint('DB', 'REDIS_PORT')
    REDIS_PWD = config.get('DB', 'REDIS_PWD')
else:
    REDIS_HOST = ''
    REDIS_PORT = ''
    REDIS_PWD = ''

# LINE notify
if 'LINENOTIFY' in config.sections() and config.options('LINENOTIFY'):
    TOKEN_MONITOR = config.get('LINENOTIFY', 'TOKEN_MONITOR')
    TOKEN_INFO = config.get('LINENOTIFY', 'TOKEN_INFO')
else:
    TOKEN_MONITOR = ''
    TOKEN_INFO = ''

# 策略相關
StrategyLong = getlist('STRATEGY', 'Long')
StrategyShort = getlist('STRATEGY', 'Short')
StrategyLongDT = getlist('STRATEGY', 'LongDT')
StrategyShortDT = getlist('STRATEGY', 'ShortDT')

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
SelectMethods = getlist('SELECT', 'methods')

# 爬蟲相關
ConvertScales = getlist('CRAWLER', 'SCALES')

# K棒特徵
K60min_feature = getlist('KBARFEATURE', 'K60min')
K30min_feature = getlist('KBARFEATURE', 'K30min')
K15min_feature = getlist('KBARFEATURE', 'K15min')
