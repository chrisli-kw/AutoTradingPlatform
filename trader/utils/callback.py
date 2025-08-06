import time
import logging

from ..config import API
from .objects.data import TradeData
from .positions import TradeDataHandler


class CallbackHandler:
    @staticmethod
    def FuturesDeal(msg: dict):
        pass

    @staticmethod
    def update_stock_msg(msg: dict):
        if msg['order_lot'] == 'Common':
            msg['quantity'] *= 1000
        return msg

    def update_futures_msg(self, msg: dict):
        symbol = self.fut_symbol(msg)
        price = msg['order']['price']
        if price == 0:
            price = TradeDataHandler.getQuotesNow(symbol)['price']
            msg['order']['price'] = price
        msg.update({
            'symbol': symbol,
            'code': symbol,
        })
        return msg

    @staticmethod
    def fut_symbol(msg: dict):
        symbol = msg['contract']['code'] + msg['contract']['delivery_month']
        if symbol not in TradeData.Quotes.NowTargets:
            for k in TradeData.Quotes.NowTargets:
                if symbol in k:
                    symbol = k
        return symbol

    @staticmethod
    def events(resp_code: int, event_code: int, info: str, event: str, env):
        if 'Subscription Not Found' in info:
            logging.warning(info)

        else:
            logging.info(
                f'Response code: {resp_code} | Event code: {event_code} | info: {info} | Event: {event}')

            if info == 'Session connect timeout' or event_code == 1:
                time.sleep(5)
                logging.warning(f'API log out: {API.logout()}')
                logging.warning('Re-login')

                time.sleep(5)
                API.login(
                    api_key=env.api_key(),
                    secret_key=env.secret_key(),
                    contracts_timeout=10000
                )
