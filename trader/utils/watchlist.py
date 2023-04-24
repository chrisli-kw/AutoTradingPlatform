import pandas as pd
from collections import namedtuple

from .. import PATH, TODAY
from . import save_csv, get_contract
from .time import TimeTool


class WatchListTool(TimeTool):

    def __init__(self, account_name):
        self.watchlist_file = f'watchlist_{account_name}'
        self.watchlist = self._open_watchlist()

    def _open_watchlist(self):
        """讀取庫存股 & 關注股票清單CSV"""
        try:
            df = pd.read_csv(f'{PATH}/stock_pool/{self.watchlist_file}.csv')
        except:
            columns = ['market', 'code', 'buyday', 'bsh', 'position', 'strategy']
            df = pd.DataFrame(columns=columns)
            self.save_watchlist(df)

        df.code = df.code.astype(str)
        df.buyday = pd.to_datetime(df.buyday.astype(str))
        df.position = df.position.fillna(100)
        df.strategy = df.strategy.fillna('Unknown')
        return df

    def _append_watchlist(self, market: str, orderinfo: namedtuple, quotes: dict, strategy_pool: dict = None):
        '''新增一筆商品(進場後使用)'''
        if isinstance(orderinfo, str):
            # 手操
            target = orderinfo
            position = 100
            cost_price = quotes
            check_quantity = True
        else:
            # 自動交易
            target = orderinfo.target
            position = abs(orderinfo.pos_target)
            cost_price = quotes[target]['price']
            check_quantity = orderinfo.quantity != 0

        if target not in self.watchlist.code.values and check_quantity:
            data = {
                'market': market,
                'code': target,
                'buyday': pd.to_datetime(self.now_str()),
                'bsh': cost_price,
                'position': position,
                'strategy': strategy_pool[target] if strategy_pool and target in strategy_pool else '手操'
            }
            self.watchlist = pd.concat([self.watchlist, pd.DataFrame([data])])

    def init_watchlist(self, stocks: pd.DataFrame, strategy_pool: dict):
        stocks = stocks[~stocks.code.isin(self.watchlist.code)].code.values

        # 若 watching list 為空，先將庫存股票加入watching list
        condition1 = (not self.watchlist.shape[0])
        condition2 = stocks.shape[0]
        if condition1 or condition2:

            for stock in stocks:
                cost_price = get_contract(stock).reference
                self._append_watchlist('Stocks', stock, cost_price, strategy_pool)

    # def get_cost_price(self, target: str, price: float, order_cond: str):
    #     '''取得股票的進場價'''

    #     if order_cond == 'ShortSelling':
    #         return price

    #     cost_price = self.watchlist.set_index('code').cost_price.to_dict()
    #     if target in cost_price:
    #         return cost_price[target]
    #     return 0

    def update_watchlist(self, df_stocks: pd.DataFrame):
        '''程式結束時，更新本地庫存表'''

        df1 = self.watchlist[self.watchlist.market == 'Stocks']
        df2 = self.watchlist[self.watchlist.market == 'Futures']

        # update stock watchlist
        condi1 = (df_stocks.yd_quantity == 0) & (df_stocks.quantity > 0)
        condi2 = ~df_stocks.code.isin(df1.code)
        _stocks = df_stocks[condi1 | condi2]

        for stock in _stocks.code.values:
            if stock in df1.code.values:
                # 新增手動買進的股票，自動買進不需更新
                condition = (df1.code == stock) & (df1.strategy.isnull())
                df1.loc[condition, ['buyday', 'position']] = [TODAY, 100]

        # 更新部位 (已全出 or 持有部位 <= 0 的股票)
        condi1 = df1.code.isin(df_stocks.code)
        condi2 = (df1.position <= 0)
        df1.loc[~condi1 | condi2, 'position'] = 0
        df1.loc[condi1 & condi2, 'position'] = 100

        self.watchlist = pd.concat([df1, df2])
        self.watchlist = self.watchlist[self.watchlist.position > 0]

    def save_watchlist(self, df: pd.DataFrame):
        save_csv(df, f'{PATH}/stock_pool/{self.watchlist_file}.csv', saveEmpty=True)
