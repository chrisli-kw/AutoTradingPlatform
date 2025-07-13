import pandas as pd


class Leverage:
    Long = {}
    Short = {}


class Account:
    Simulate = False
    DesposalMoney = 0
    DesposalMargin = 0


class Securities:
    InfoDefault = pd.DataFrame(
        columns=[
            'account', 'market',
            'code', 'action', 'quantity', 'cost_price',
            'last_price', 'pnl',
            'yd_quantity', 'order_cond', 'order',
            'timestamp', 'position', 'strategy'
        ]
    )
    Strategy = {}
    Monitor = {}


class Stocks:
    N_Long = 0
    N_Short = 0
    Leverage = Leverage()
    Bought = []
    Sold = []
    Dividends = {}
    LimitLong = 0
    LimitShort = 0
    CanTrade = False
    Punish = []


class Futures:
    SettleDefault = pd.DataFrame(
        columns=[
            'date', 'code', 'quantity', 'dseq', 'fee', 'tax', 'currency',
            'direction', 'entry_price', 'cover_price', 'pnl', 'profit'
        ]
    )
    Opened = []
    Closed = []
    Transferred = {}
    CodeList = {}
    Limit = 0
    CanTrade = False


class Quotes:
    AllIndex = {'TSE001': [], 'OTC101': []}
    NowIndex = {}
    AllTargets = {}
    NowTargets = {}
    TempKbars = {}


class KBars:
    kbar_columns = [
        'name', 'Time',
        'Open', 'High', 'Low', 'Close', 'Volume', 'Amount'
    ]
    Freq = {
        '1T': pd.DataFrame(columns=kbar_columns),
        '2T': pd.DataFrame(columns=kbar_columns),
        '5T': pd.DataFrame(columns=kbar_columns),
        '15T': pd.DataFrame(columns=kbar_columns),
        '30T': pd.DataFrame(columns=kbar_columns),
        '60T': pd.DataFrame(columns=kbar_columns),
        '1D': pd.DataFrame(columns=kbar_columns)
    }


class DefaultTableMeta(type):
    _data = {
        'Account': Account,
        'Securities': Securities,
        'Contracts': {},
        'Stocks': Stocks,
        'Futures': Futures,
        'BidAsk': {},
        'Quotes': Quotes(),
        'KBars': KBars(),
    }

    def __getitem__(cls, key):
        return cls._data[key]

    def __getattr__(cls, name):
        if name in cls._data:
            return cls._data[name]
        raise AttributeError(
            f"'{cls.__name__}' object has no attribute '{name}'")


class TradeData(metaclass=DefaultTableMeta):
    pass
