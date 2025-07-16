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

    def monitor_list_to_df(self, account: str, market='Stocks'):
        # TODO: delete
        '''Convert stocks/futures monitor list to dataframe'''

        data = TradeData.Securities.Monitor
        logging.debug(f'targets_to_monitor: {data}')

        df = {k: v for k, v in data.items() if v}
        df = pd.DataFrame(df).T
        if df.shape[0]:
            # df = df[df.account_id.str.contains('simulate')]
            df['account'] = account
            df['market'] = market

            if market == 'Stocks':
                df = df.sort_values('code').reset_index()
                df.yd_quantity = df.quantity
                df['pnl'] = df.action.apply(lambda x: 1 if x == 'Buy' else -1)
            else:
                df = df.reset_index(drop=True)

                if 'order' in df.columns:
                    df['direction'] = df.order.apply(lambda x: x['action'])
                else:
                    df['direction'] = df.action
                df['pnl'] = df.direction.apply(
                    lambda x: 1 if x == 'Buy' else -1)

            df['last_price'] = df.code.map(
                {s: TradeDataHandler.getQuotesNow(s).get('price', 0) for s in df.code})
            df['pnl'] = df.pnl*(df.last_price - df.cost_price)*df.quantity
            df['timestamp'] = datetime.now()
            df['strategy'] = df.code.map(TradeData.Securities.Strategy)
            df = df[TradeData[market].InfoDefault.columns]
        else:
            df = TradeData[market].InfoDefault

        return df

    def update_securityInfo(self, account: str, df: pd.DataFrame):
        # TODO: delete
        if df.empty:
            return

        if db.HAS_DB:
            match_account = SecurityInfo.account == account
            codes = db.query(SecurityInfo.code, match_account).code.values
            tb = df[~df.code.isin(codes)]
            update_values = df[df.code.isin(codes)].set_index('code')

            # add new stocks
            db.dataframe_to_DB(tb, SecurityInfo)

            # update in-stocks
            update_values = update_values.to_dict('index')
            for target, values in update_values.items():
                condition = SecurityInfo.code == target, match_account
                db.update(SecurityInfo, values, *condition)

    def save_securityInfo(self, env, market='Stocks'):
        # TODO: delete
        '''Save the security info table if running under simulation mode.'''

        if (
            (market == 'Stocks' and not TradeData.Stocks.CanTrade) or
            (market == 'Futures' and not TradeData.Futures.CanTrade)
        ):
            return

        df = self.monitor_list_to_df(env.ACCOUNT_NAME, market=market)
        self.update_securityInfo(env.ACCOUNT_NAME, df)

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
