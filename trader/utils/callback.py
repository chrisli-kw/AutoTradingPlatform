from datetime import datetime

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
