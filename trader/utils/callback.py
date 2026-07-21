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
            price = TradeDataHandler.getQuotesNow(symbol).get('price', 0)
            msg['order']['price'] = price
        msg.update({
            'symbol': symbol,
            'code': symbol,
        })
        return msg

    @staticmethod
    def _futures_symbol(msg: dict):
        """Build the internal symbol from either an order or deal callback."""
        contract = msg.get('contract', msg)
        symbol = contract['code'] + contract['delivery_month']
        is_option = (
            contract.get('security_type') == 'OPT' or
            contract.get('option_right') not in [None, '', 'Future']
        )
        if is_option:
            option_right = str(contract.get('option_right', ''))
            option = 'C' if 'Call' in option_right else 'P'
            strike = float(contract.get('strike_price', 0) or 0)
            strike = int(strike) if strike.is_integer() else strike
            symbol = f'{symbol}{strike}{option}'

        return symbol

    @staticmethod
    def _match_monitored_symbol(symbol: str):
        if symbol in TradeData.Quotes.NowTargets:
            return symbol

        for target in TradeData.Quotes.NowTargets:
            if symbol in target:
                return target
        return symbol

    @classmethod
    def fut_symbol(cls, msg: dict):
        return cls._match_monitored_symbol(cls._futures_symbol(msg))

    @classmethod
    def fut_deal_symbol(cls, msg: dict):
        return cls._match_monitored_symbol(cls._futures_symbol(msg))

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
