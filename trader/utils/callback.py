from datetime import datetime


class CallbackHandler:
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
