import numpy as np
import pandas as pd
from sys import float_info as sflt


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
        # 券資比 = 融券餘額 / 融資餘額
        shortMarginPercent = self.df.shortRemaining/self.df.marginRemaining
        return shortMarginPercent

    def close_short_index(self):
        # 券補力道 = 融券餘額 / 5 日成交均量
        V_1D_5_MA = self.df.groupby('stockid').volume.transform(
            lambda x: x.rolling(5).mean())
        closeShortIndex = self.df.shortRemaining/V_1D_5_MA
        return closeShortIndex


class TechnicalSignals:
    @staticmethod
    def SMA(data: pd.DataFrame, col: str, n=7, shift=0):

        def SMA_(x):
            return x.shift(shift).rolling(n).mean()

        if isinstance(data, pd.Series):
            return SMA_(data)

        return data.groupby('name')[col].transform(lambda x: SMA_(x))

    @staticmethod
    def STD(data: pd.DataFrame, col: str, n=7, shift=0):

        def STD_(x):
            return x.shift(shift).rolling(n).std()

        if isinstance(data, pd.Series):
            return STD_(data)

        return data.groupby('name')[col].transform(lambda x: STD_(x))

    @staticmethod
    def MAX(data: pd.DataFrame, col: str, n=7, shift=0):

        def MAX_(x):
            return x.shift(shift).rolling(n).max()

        if isinstance(data, pd.Series):
            return MAX_(data)

        return data.groupby('name')[col].transform(lambda x: MAX_(x))

    @staticmethod
    def MIN(data: pd.DataFrame, col: str, n=7, shift=0):

        def MIN_(x):
            return x.shift(shift).rolling(n).min()

        if isinstance(data, pd.Series):
            return MIN_(data)

        return data.groupby('name')[col].transform(lambda x: MIN_(x))

    @classmethod
    def MACD(cls, tb, d1=12, d2=26, dma=9):
        # DIFF (快線) = EMA (收盤價, 12) - EMA (收盤價, 26)
        tb['MACD_fast'] = cls.EMA(tb, 'Close', d1, 0)
        tb['MACD_slow'] = cls.EMA(tb, 'Close', d2, 0)
        tb['MACD_ema_diff'] = tb['MACD_fast'] - tb['MACD_slow']

        # DEA(慢線) = EMA (DIFF, 9)
        tb['MACD'] = cls.EMA(tb, 'MACD_ema_diff', dma, 0)

        # MACD紅綠柱狀體 = DIFF - DEA
        tb['MACD_hist'] = tb.MACD_ema_diff - tb.MACD
        return tb

    @staticmethod
    def check_MACD_dev(df: pd.DataFrame):
        '''背離(MACD)'''
        # # 股價與交易量/MACD背離
        # cond1 = (tb.quotechange > 0) & (tb.ema_diff < 0)
        # cond2 = (tb.quotechange < 0) & (tb.ema_diff > 0)
        # tb['diverge_MACD'] = (cond1 | cond2).astype(int)

        close_diff = df.groupby('name').Close.transform('diff').fillna(0)
        diff_MACD_diff = df.groupby(
            'name').diff_MACD.transform('diff').fillna(0)
        return ((close_diff >= 0) & (diff_MACD_diff <= 0)) | ((close_diff <= 0) & (diff_MACD_diff >= 0))

    @classmethod
    def RSI(cls, change, period=12):
        '''
        Range: [0, 100]
        change: stock close price daily quote change

        RSI = RS/(1+RS), where RS = Up/Down
        # Up = d day mean of previous change ups
        # Down = d day mean of previous change downs
        '''
        ups = pd.Series(index=change.index, data=change[change > 0])
        downs = pd.Series(index=change.index, data=-change[change < 0])

        # 計算d日平均漲跌
        mean_u = cls.SMA(ups.fillna(0), '', period, shift=0)
        mean_d = cls.SMA(downs.fillna(0), '', period, shift=0)

        # 計算 RSI
        rsi = 100*mean_u/(mean_u + mean_d)
        return rsi

    @classmethod
    def RSV(cls, tb, d=9, target='Close'):
        '''
        Range: [0, 100]
        '''
        d_min = cls.MIN(tb, target, d, shift=0)
        d_max = cls.MAX(tb, target, d, shift=0)

        try:
            (100*(tb[target] - d_min)/(d_max - d_min)).fillna(-1)
        except:
            tb['d_min'] = d_min
            tb['d_max'] = d_max

        return (100*(tb[target] - d_min)/(d_max - d_min)).fillna(-1)

    @staticmethod
    def KD(tb, d=9):
        '''
        Range: [0, 100]
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
                    K.append(K[-1]*(d-1)/d + r/d)
            return K

        def _getD(k):
            D = []
            for i, k in enumerate(k):
                if i == 0:
                    D.append(50)
                else:
                    D.append(D[-1]*(d-1)/d + k/d)
            return D

        tb['K'] = tb.groupby('name').RSV.transform(_getK)
        tb['D'] = tb.groupby('name').K.transform(_getD)
        tb.RSV = tb.RSV.replace(-1, np.nan)
        return tb

    @classmethod
    def Williams_Indicator(cls, df, window_size=14):
        '''
        Williams %R
        Range: [-100, 0]
        R = (max(high) - close)/(max(high) - min(low))*(-100)
        '''

        # 計算n日內最高價
        highest_high = cls.MAX(df, 'High', window_size, shift=0)

        # 計算n日內最低價
        lowest_low = cls.MIN(df, 'Low', window_size, shift=0)

        # 計算威廉指標倒傳遞(William %R Echo)
        Williams = (highest_high - df.Close)/(highest_high - lowest_low) * -100

        return Williams

    @staticmethod
    def EMA(data: pd.DataFrame, col: str = '', window_size: int = 5, shift: int = 0):
        '''Exponential Moving Average'''

        def EMA_(x):
            return x.shift(shift).ewm(span=window_size, adjust=False).mean()

        if isinstance(data, pd.Series):
            return EMA_(data)

        return data.groupby('name')[col].transform(lambda x: EMA_(x))

    @staticmethod
    def WMA(data: pd.DataFrame, col: str = '', window_size: int = 5, shift: int = 0):
        '''Weighted Moving Average'''

        weights = np.arange(window_size)
        weights = weights/weights.sum()

        def WMA_(x: pd.Series):
            return x.shift(shift).rolling(window_size).apply(lambda x: np.sum(weights*x.values))

        if isinstance(data, pd.Series):
            return WMA_(data)

        return data.groupby('name')[col].transform(lambda x: WMA_(x))

    @classmethod
    def HMA(cls, df: pd.DataFrame, col: str = '', window_size: int = 5, shift: int = 0):
        '''
        Hull Moving Average
        WMA(M; n) = WMA(2*WMA(n/2) - WMA(n)); sqrt(n)
        '''

        wma1 = 2*cls.WMA(df, col, window_size//2, shift)
        wma2 = cls.WMA(df, col, window_size, shift)
        sqrt_window = int(np.sqrt(window_size))
        return cls.WMA(wma1-wma2, col, sqrt_window, shift)

    @classmethod
    def TMA(cls, df: pd.DataFrame, col: str = '', window_size: int = 5, shift: int = 0):
        '''
        Triple Exponential Moving Average
        (3*EMA - 3*EMA(EMA)) + EMA(EMA(EMA))
        '''

        ema1 = cls.EMA(df, col, window_size, shift)
        ema2 = cls.EMA(ema1, col, window_size, shift)
        ema3 = cls.EMA(ema2, col, window_size, shift)
        return 3*ema1 - 3*ema2 + ema3

    @classmethod
    def CCI(cls, df: pd.DataFrame, window_size: int = 5, shift: int = 0):
        '''
        Commodity Channel Index
        Range: [-100, 100]

        1. TypicalPrice(TP) = (High + Low + Close)/3
        2. MeanDeviation = Mean(Abs(TypicalPrice - SMA(TP)))
        3. CCI = (TypicalPrice - SMA(TP))/.015*MeanDeviation
        '''

        tp = (df.High + df.Low + df.Close)/3
        sma = cls.SMA(tp, '', window_size, shift=shift)
        md = cls.SMA((tp - sma).abs(), '', window_size, shift=shift)
        return (tp - sma)/(.015*md)

    @staticmethod
    def CMO(data: pd.Series, window_size: int = 12):
        '''
        Chande Momentum Oscilator Indicator
        Range: [-100, 100]
        CMO = 100*(Su - Sd)/(Su + Sd)
        # Su = the sum of the momentum of up days
        # Sd = the sum of the momentum of down days
        '''

        change = data.diff(1)
        ups = change.copy().clip(lower=0)
        downs = change.copy().clip(upper=0).abs()

        Su = ups.rolling(window_size).sum()
        Sd = downs.rolling(window_size).sum()

        cmo = 100*(Su - Sd)/(Su + Sd)
        return cmo

    @classmethod
    def PPO(cls, data: pd.Series, fast: int = 12, slow: int = 26, signal_day: int = 9):
        '''
        Percentage Price Oscillator
        PPO = 100*(EMA(12) - EMA(26))/EMA(26)
        Signal Line : EMA(9) of PPO
        '''

        ema_fast = cls.EMA(data, '', fast, 0)
        ema_slow = cls.EMA(data, '', slow, 0)
        ppo = 100*(ema_fast - ema_slow)/ema_slow
        ema_signal = cls.EMA(ppo, '', signal_day, 0)

        return ppo, ema_signal

    @staticmethod
    def CMF(df: pd.DataFrame, window_size: int = 21):
        '''
        Chaikin Money Flow Indicator
        Range: [-1, 1]
        Multiplier = ((Close - Low) - (High - Close))/(High - Low)
        Money Flow Volume (MFV) = Volume * Multiplier
        21 Period CMF = 21 Period Sum of MFV / 21 Period Sum of V olume
        '''

        multiplier = (2*df.Close - df.High - df.Low)/(df.High - df.Low)
        mfv = df.Volume*multiplier.fillna(sflt.epsilon)

        cmf = mfv.rolling(window_size).sum()
        cmf /= df.Volume.rolling(window_size).sum()

        return cmf  # TODO: NaN

    @staticmethod
    def ATR(df: pd.DataFrame, window_size: int = 14):
        tb = df[['High', 'Low', 'Close']].copy()
        tb['tr0'] = abs(tb.High - tb.Low)
        tb['tr1'] = abs(tb.High - tb.Close.shift())
        tb['tr2'] = abs(tb.Low - tb.Close.shift())
        tr = tb[['tr0', 'tr1', 'tr2']].max(axis=1)
        atr = tr.ewm(alpha=1/window_size, adjust=False).mean()
        return atr

    @classmethod
    def ADX(cls, df: pd.DataFrame, window_size: int = 21, atr_window: int = 10, shift: int = 0):
        '''
        Average Directional Index
        Range: [0, 100]
        Procedure of Calculating DMI:
        * UpMove = CurrentHigh - PreviousHigh
        * DownMove = CurrentLow - PreviousLow

        If (UpMove > DownMove and UpMove > 0):
            +DMI = UpMove
        else:
            +DMI = 0

        If (DownMove > Upmove and DownMove > 0):
            -DMI = DownMove
        else
            -DMI = 0

        +DI = 100*EMA(+DMI/ATR)
        -DI = 100*EMA(-DMI/ATR)
        ADX = 100*EMA(Abs((+DI - -DI)/(+DI + -DI)))
        '''

        up = df.High.diff(1)
        down = df.Low.diff(1)

        pos_dmi = ((up > down) & (up > 0)) * up
        neg_dmi = ((down > up) & (down > 0)) * down

        pos_dm = 100*cls.EMA(
            pos_dmi/cls.ATR(df, atr_window), '', window_size, shift)
        neg_dm = 100*cls.EMA(
            neg_dmi/cls.ATR(df, atr_window), '', window_size, shift)

        adx = abs((pos_dm - neg_dm)/(pos_dm + neg_dm))
        adx = 100*cls.EMA(adx, '', window_size, shift)

        return adx
