import pandas as pd
from dotenv import dotenv_values

from ..cipher import CipherTool


class UserEnv:
    def __init__(self, account_name: str):
        self.CONFIG = dotenv_values(f'./lib/envs/{account_name}.env')

        # 交易帳戶設定
        self.ACCOUNT_NAME = self.get('ACCOUNT_NAME', default='unknown')
        self.__API_KEY__ = self.get('API_KEY')
        self.__SECRET_KEY__ = self.get('SECRET_KEY')
        self.__ACCOUNT_ID__ = self.get('ACCOUNT_ID', 'decrypt')
        self.__CA_PASSWD__ = self.get('CA_PASSWD', 'decrypt')

        self.MODE = self.get('MODE')
        self.can_sell = self.MODE not in ['LongBuy', 'ShortBuy']
        self.can_buy = self.MODE not in ['LongSell', 'ShortSell']

        # 股票使用者設定
        self.INIT_POSITION = self.get('INIT_POSITION', 'int')
        self.POSITION_LIMIT_LONG = self.get('POSITION_LIMIT_LONG', 'int')
        self.POSITION_LIMIT_SHORT = self.get('POSITION_LIMIT_SHORT', 'int')
        self.ORDER_COND1 = self.get('ORDER_COND1')
        self.ORDER_COND2 = self.get('ORDER_COND2')
        self.STOCK_MODEL_VERSION = self.get(
            'STOCK_MODEL_VERSION', default='1.0.0')

        # 期貨使用者設定
        self.TRADING_PERIOD = self.get('TRADING_PERIOD')
        self.MARGIN_LIMIT = self.get('MARGIN_LIMIT', 'int')
        self.FUTURES_MODEL_VERSION = self.get(
            'FUTURES_MODEL_VERSION', default='1.0.0')

    def get(self, key: str, type_: str = 'text', default=None):
        if self.CONFIG:
            env = self.CONFIG.get(key, default)

            if type_ == 'int':
                return int(env)
            elif type_ == 'list':
                if 'none' in env.lower():
                    return []
                return env.replace(' ', '').split(',')
            elif type_ == 'dict':
                envs = {}
                if env:
                    for e in env.split(','):
                        e = e.split(':')
                        envs.update({e[0]: e[1]})
                return envs
            elif type_ == 'date' and env:
                return pd.to_datetime(env)
            elif type_ == 'decrypt':
                if not env or (not env[0].isdigit() and env[1:].isdigit()):
                    return env
                ct = CipherTool(decrypt=True, encrypt=False)
                return ct.decrypt(env)
            return env
        elif type_ == 'int':
            return 0
        elif type_ == 'list':
            return []
        elif type_ == 'dict':
            return {}
        return None

    def api_key(self) -> str:
        return self.__API_KEY__

    def secret_key(self) -> str:
        return self.__SECRET_KEY__

    def account_id(self) -> str:
        return self.__ACCOUNT_ID__

    def ca_passwd(self) -> str:
        return self.__CA_PASSWD__
