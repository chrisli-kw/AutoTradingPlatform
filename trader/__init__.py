from concurrent.futures import ThreadPoolExecutor

from .config import PATH
from .strategies.select import SelectStock
from .utils import create_folder
from .utils.kbar import TickDataProcesser
from .utils.notify import Notification
from .utils.crawler import CrawlStockData, CrawlFromHTML


__version__ = '1.4.0'

for f in [PATH, './logs']:
    create_folder(f)

for f in ['daily_info', 'Kbars', 'ticks', 'selections', 'stock_pool']:
    create_folder(f'{PATH}/{f}')

create_folder(f'{PATH}/Kbars/1min')
create_folder(f'{PATH}/ticks/stocks')
create_folder(f'{PATH}/ticks/futures')
create_folder(f'{PATH}/selections/history')


executor = ThreadPoolExecutor(max_workers=5)
notifier = Notification()
picker = SelectStock()
crawler1 = CrawlStockData()
crawler2 = CrawlFromHTML()
tdp = TickDataProcesser()
