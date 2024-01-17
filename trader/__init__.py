from concurrent.futures import ThreadPoolExecutor

from .config import PATH
from .utils.file import FileHandler
from .utils.select import SelectStock
from .utils.kbar import TickDataProcesser
from .utils.notify import Notification
from .utils.crawler import CrawlStockData, CrawlFromHTML


__version__ = '1.11.12'

executor = ThreadPoolExecutor(max_workers=5)
file_handler = FileHandler()
notifier = Notification()
picker = SelectStock()
crawler1 = CrawlStockData()
crawler2 = CrawlFromHTML()
tdp = TickDataProcesser()


for f in [PATH, './logs']:
    file_handler.create_folder(f)

for f in ['daily_info', 'Kbars', 'ticks', 'selections', 'stock_pool']:
    file_handler.create_folder(f'{PATH}/{f}')

for f in ['1D', '60T', '30T', '1T']:
    file_handler.create_folder(f'{PATH}/Kbars/{f}')

file_handler.create_folder(f'{PATH}/ticks/stocks')
file_handler.create_folder(f'{PATH}/ticks/futures')
file_handler.create_folder(f'{PATH}/selections/history')
