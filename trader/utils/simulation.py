import logging
import pandas as pd
from datetime import datetime
from collections import namedtuple

from .database import db
from .database.tables import SecurityInfo
from .orders import OrderTool
from .positions import TradeDataHandler
from .objects.data import TradeData


class Simulator:
    def __init__(self, account_name: str) -> None:
        self.account_name = account_name
        self.order_tool = OrderTool(account_name)

    def securityInfo(self, account: str):
        try:
            df = db.query(SecurityInfo, SecurityInfo.account == account)
        except:
            df = TradeData.Securities.InfoDefault

        df['account_id'] = 'simulate'
        return df

    def update_monitor(self, order: namedtuple, order_data: dict):
        target = order.target

        # update monitor list position
        action = order.octype
        order_data.update({
            'code': target,
            'order': {
                'action': order.action,
                'quantity': self.order_tool.get_sell_quantity(order),
                'price': abs(order_data['price']),
                'order_cond': order_data.get('order_cond', 'Cash'),
            }
        })

        self.order_tool.WatchListTool.update_monitor(action, order_data)
        TradeDataHandler.update_deal_list(target, action)
