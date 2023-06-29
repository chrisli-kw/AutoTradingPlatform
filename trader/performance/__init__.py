import os
import sys

from ..config import PATH
from ..utils import create_folder


sys.path.append(os.path.abspath(os.path.join('./', os.pardir)))
create_folder(f'{PATH}/backtest')
