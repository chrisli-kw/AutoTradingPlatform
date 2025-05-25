from concurrent.futures import ThreadPoolExecutor

from .config import PATH
from .utils.file import file_handler
from .utils.select import SelectStock
from .utils.kbar import TickDataProcesser


__version__ = '1.17.0'

exec = ThreadPoolExecutor(max_workers=5)
picker = SelectStock()
tdp = TickDataProcesser()


for f in [PATH, './logs']:
    file_handler.Operate.create_folder(f)

for f in ['daily_info', 'Kbars', 'ticks', 'selections', 'stock_pool']:
    file_handler.Operate.create_folder(f'{PATH}/{f}')

for f in ['1D', '60T', '30T', '1T']:
    file_handler.Operate.create_folder(f'{PATH}/Kbars/{f}')

file_handler.Operate.create_folder(f'{PATH}/ticks/stocks')
file_handler.Operate.create_folder(f'{PATH}/ticks/futures')
