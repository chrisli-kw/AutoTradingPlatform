import logging
import pandas as pd

from .. import PATH
from .database import db
from .database.tables import SecurityInfoStocks, SecurityInfoFutures
from .file import FileHandler
from .objs import TradeData


class Simulator(FileHandler):
    def securityInfo(self, account: str, market='Stocks'):
        try:
            if db.HAS_DB:
                table = SecurityInfoStocks if market == 'Stocks' else SecurityInfoFutures
                df = db.query(table, table.account == account)
            else:
                df = self.read_table(
                    f'{PATH}/stock_pool/simulation_{market.lower()}_{account}.pkl',
                    df_default=TradeData[market].InfoDefault
                )
        except:
            df = TradeData[market].InfoDefault

        df['account_id'] = 'simulate'
        return df

    def monitor_list_to_df(
            self,
            account: str,
            data: dict,
            quotes: dict,
            market='Stocks',
            is_trading_time=True
    ):
        '''Convert stocks/futures monitor list to dataframe'''
        logging.debug(f'{market.lower()}_to_monitor: {data}')

        df = {k: v for k, v in data.items() if v}
        df = pd.DataFrame(df).T
        if df.shape[0]:
            df = df[df.account_id.str.contains('simulate')]
            df['account'] = account

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

            if is_trading_time:
                df['last_price'] = df.code.map(
                    {s: quotes.NowTargets[s]['price'] for s in df.code})
            else:
                df['last_price'] = 0

            df['pnl'] = df.pnl*(df.last_price - df.cost_price)*df.quantity
            df = df[TradeData[market].InfoDefault.columns]
        else:
            df = TradeData[market].InfoDefault

    def update_securityInfo(self, account: str, df: pd.DataFrame, market='Stocks'):
        if df.empty:
            return

        table = SecurityInfoStocks if market == 'Stocks' else SecurityInfoFutures
        if db.HAS_DB:
            match_account = table.account == account
            codes = db.query(table.code, match_account).code.values
            tb = df[~df.code.isin(codes)]
            update_values = df[df.code.isin(codes)].set_index('code')

            # add new stocks
            db.dataframe_to_DB(tb, table)

            # update in-stocks
            update_values = update_values.to_dict('index')
            for target, values in update_values.items():
                condition = table.code == target, match_account
                db.update(table, values, *condition)
        else:
            self.save_table(
                df=df,
                filename=f'{PATH}/stock_pool/simulation_{market.lower()}_{account}.pkl'
            )
