import logging
import pandas as pd
from dotenv import dotenv_values

from ..cipher import CipherTool
from ..database import db
from ..database.tables import UserSettings


class UserEnv:
    def __init__(self, account_name: str):
        config = db.query(UserSettings, UserSettings.account == account_name)

        upload_env = False
        if config.empty:
            logging.warning(
                f'No config found for {account_name}, using local settings.')
            self.CONFIG = dotenv_values(f'./lib/envs/{account_name}.env')
            upload_env = True
        else:
            self.CONFIG = config.iloc[0].to_dict()

        # 交易帳戶設定
        self.ACCOUNT_NAME = self.get('ACCOUNT_NAME', default='unknown')
        self.__API_KEY__ = self.get('API_KEY')
        self.__SECRET_KEY__ = self.get('SECRET_KEY')
        self.__ACCOUNT_ID__ = self.get('ACCOUNT_ID', 'decrypt')
        self.__CA_PASSWD__ = self.get('CA_PASSWD', 'decrypt')
        self.MODE = self.get('MODE')

        # 股票使用者設定
        self.INIT_BALANCE = self.get('INIT_BALANCE', 'int')
        self.MARGING_TRADING_AMOUNT = self.get('MARGING_TRADING_AMOUNT', 'int')
        self.SHORT_SELLING_AMOUNT = self.get('SHORT_SELLING_AMOUNT', 'int')

        # 期貨使用者設定
        self.TRADING_PERIOD = self.get('TRADING_PERIOD')
        self.MARGIN_AMOUNT = self.get('MARGIN_AMOUNT', 'int')

        if upload_env:
            self.env_to_db()

    def get(self, key: str, type_: str = 'text', default=None):
        if type(self.CONFIG) == dict:
            key = key.lower()

        env = self.CONFIG.get(key, default) if self.CONFIG else None

        if type_ == 'int':
            try:
                return int(env or 0)
            except:
                return 0

        if type_ == 'list':
            if not env or 'none' in env.lower():
                return []
            return [item.strip() for item in env.split(',')]

        if type_ == 'dict':
            if not env:
                return {}
            return {k.strip(): v.strip() for k, v in (item.split(':') for item in env.split(','))}

        if type_ == 'date':
            return pd.to_datetime(env) if env else None

        if type_ == 'decrypt':
            if not env or (not env[0].isdigit() and env[1:].isdigit()):
                return env
            return CipherTool(decrypt=True, encrypt=False).decrypt(env)

        return env

    def env_to_db(self, **env):
        """將環境變數存入資料庫"""

        if not env:

            env = {
                'account': self.ACCOUNT_NAME,
                'api_key': self.__API_KEY__,
                'secret_key': self.__SECRET_KEY__,
                'account_id': self.__ACCOUNT_ID__,
                'ca_passwd': self.__CA_PASSWD__,
                'mode': self.MODE,
                'init_balance': self.INIT_BALANCE,
                'marging_trading_amount': self.MARGING_TRADING_AMOUNT,
                'short_selling_amount': self.SHORT_SELLING_AMOUNT,
                'trading_period': self.TRADING_PERIOD,
                'margin_amount': self.MARGIN_AMOUNT
            }

        db.add_data(UserSettings, **env)
        logging.warning(f'User settings saved for {self.ACCOUNT_NAME}')

    def api_key(self) -> str:
        return self.__API_KEY__

    def secret_key(self) -> str:
        return self.__SECRET_KEY__

    def account_id(self) -> str:
        return self.__ACCOUNT_ID__

    def ca_passwd(self) -> str:
        return self.__CA_PASSWD__
