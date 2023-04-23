import pandas as pd


class ChipAnalysis:
    '''籌碼分析'''

    def __init__(self, df, target):
        self.df = df
        self.target = target
        self.group = df.groupby('stockid')[self.target]

    def change(self):
        # 漲跌
        return self.group.transform('diff')

    def changepercent(self):
        # 漲跌幅
        return 100*self.df.quotechange/self.group.transform('shift')

    def dailychange(self):
        # 當日高低價差
        return self.df.high - self.df.low

    def dailychangepercent(self):
        # 當日震幅
        return 100*self.df.dailychange/self.group.transform('shift')

    def MA(self, d):
        # 移動平均
        return self.group.transform(lambda x: x.rolling(d).mean())

    def short_margin(self):
        ## 券資比 = 融券餘額 / 融資餘額
        shortMarginPercent = self.df.shortRemaining/self.df.marginRemaining
        return shortMarginPercent

    def close_short_index(self):
        # 券補力道 = 融券餘額 / 5 日成交均量
        volume_ma5 = self.df.groupby('stockid').volume.transform(
            lambda x: x.rolling(5).mean())
        closeShortIndex = self.df.shortRemaining/volume_ma5
        return closeShortIndex


class TechnicalSignals:
    def _MA(self, df: pd.DataFrame, col: str, n=7, shift=0):
        return df.groupby('name')[col].transform(lambda x: x.shift(shift).rolling(n).mean())

    def _STD(self, df: pd.DataFrame, col: str, n=7, shift=0):
        return df.groupby('name')[col].transform(lambda x: x.shift(shift).rolling(n).std())

    def MACD(self, tb, d1=12, d2=26, dma=9):
        group = tb.groupby('name').Close

        # # 股價與交易量/MACD背離
        # cond1 = (tb.quotechange > 0) & (tb.ema_diff < 0)
        # cond2 = (tb.quotechange < 0) & (tb.ema_diff > 0)
        # tb['diverge_MACD'] = (cond1 | cond2).astype(int)

        # DIFF (快線) = EMA (收盤價, 12) - EMA (收盤價, 26)
        tb[f'ema_{d1}'] = group.transform(lambda x: x.ewm(span=d1, adjust=False).mean())
        tb[f'ema_{d2}'] = group.transform(lambda x: x.ewm(span=d2, adjust=False).mean())
        tb['ema_diff'] = tb[f'ema_{d1}'] - tb[f'ema_{d2}']

        # DEA(慢線) = EMA (DIFF, 9)
        tb['MACD'] = tb.groupby('name').ema_diff.transform(
            lambda x: x.ewm(span=dma, adjust=False).mean())

        # MACD紅綠柱狀體 = DIFF - DEA
        tb['diff_MACD'] = tb.ema_diff - tb.MACD
        return tb

    def check_MACD_dev(self, df: pd.DataFrame):
        '''背離(MACD)'''

        close_diff = df.groupby('name').Close.transform('diff').fillna(0)
        diff_MACD_diff = df.groupby('name').diff_MACD.transform('diff').fillna(0)
        return ((close_diff >= 0) & (diff_MACD_diff <= 0)) | ((close_diff <= 0) & (diff_MACD_diff >= 0))

    def RSI(self, change, period=12):
        '''
        change: stock close price daily quote change

        RSI = RS/(1+RS), where RS = Up/Down
        # Up = d day mean of previous change ups
        # Down = d day mean of previous change downs
        '''
        ups = pd.Series(index=change.index, data=change[change > 0])
        downs = pd.Series(index=change.index, data=-change[change < 0])

        # 計算d日平均漲跌
        mean_u = ups.fillna(0).rolling(period).mean()
        mean_d = downs.fillna(0).rolling(period).mean()

        # 計算 RSI
        rsi = 100*mean_u/(mean_u + mean_d)
        return rsi

    def RSV(self, tb, d=9):
        d_min = tb.groupby('name').Close.transform(lambda x: x.rolling(d).min())
        d_max = tb.groupby('name').Close.transform(lambda x: x.rolling(d).max())

        try:
            (100*(tb.Close - d_min)/(d_max - d_min)).fillna(-1)
        except:
            tb['d_min'] = d_min
            tb['d_max'] = d_max

        return (100*(tb.Close - d_min)/(d_max - d_min)).fillna(-1)

    def KD(self, tb):
        '''
        Reference: https://medium.com/%E5%8F%B0%E8%82%A1etf%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8-%E7%A8%8B%E5%BC%8F%E9%A1%9E/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80-%E8%87%AA%E5%BB%BAkd%E5%80%BC-819d6fd707c8
        一、 要算出KD值，必先算出RSV強弱值，以下以 9 天為計算基準。
            RSV=( 收盤 - 9日內的最低 ) / ( 9日內的最高 - 9日內的最低 ) * 100
        二、 再以平滑移動平均法，來計算KD值。
            期初:K= 50，D= 50
            當日K值=前一日K值 * 2/3 + 當日RSV * 1/3
            當日D值=前一日D值 * 2/3 + 當日K值 * 1/3
        '''
        def _getK(rsv):
            K = []
            for r in rsv:
                if r == -1:
                    K.append(50)
                else:
                    K.append(K[-1]*2/3 + r/3)
            return K

        def _getD(k):
            D = []
            for i, k in enumerate(k):
                if i == 0:
                    D.append(50)
                else:
                    D.append(D[-1] * 2/3 + k/3)
            return D

        tb['K'] = tb.groupby('name').RSV.transform(_getK)
        tb['D'] = tb.groupby('name').K.transform(_getD)
        return tb
