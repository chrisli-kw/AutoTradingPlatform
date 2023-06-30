import os
import sys

from ..config import PATH
from .. import file_handler


sys.path.append(os.path.abspath(os.path.join('./', os.pardir)))
file_handler.create_folder(f'{PATH}/backtest')
