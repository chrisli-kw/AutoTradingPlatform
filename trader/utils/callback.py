import time
import logging
from datetime import datetime

from ..config import API
from .objs import TradeData


class CallbackHandler:
    @staticmethod
    def fDeal(msg: dict):
        code = msg['code']
        delivery_month = msg['delivery_month']
        symbol = code + delivery_month
        if symbol in TradeData.Futures.Monitor and TradeData.Futures.Monitor[symbol] is not None:
            price = msg['price']
            TradeData.Futures.Monitor[symbol]['cost_price'] = price

    @staticmethod
    def update_stock_msg(msg: dict):
        msg.update({
            'position': 100,
            'yd_quantity': 0,
            'bst': datetime.now(),
            'cost_price': msg['price']
        })

        if msg['order_lot'] == 'Common':
            msg['quantity'] *= 1000
        return msg

    @staticmethod
    def fut_symbol(msg: dict):
        return msg['contract']['code'] + msg['contract']['delivery_month']

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
