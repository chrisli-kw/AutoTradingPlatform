from sqlalchemy import Column, Integer, FLOAT, String, text
from sqlalchemy.dialects.mysql import TIMESTAMP

from .sql import Base


time_default = text("CURRENT_TIMESTAMP(6)")
collation = 'utf8mb4_unicode_ci'


class SecurityInfoStocks(Base):
    __tablename__ = 'security_info_stocks'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    account = Column(String(50, collation), nullable=False, comment='帳戶代號')
    market = Column(String(10, collation), nullable=False, comment='市場別')
    code = Column(String(10, collation), nullable=False, comment='證券代號')
    order_cond = Column(String(10, collation), nullable=False, comment='委託類型')
    action = Column(String(10, collation), nullable=False, comment='買賣別')
    pnl = Column(Integer, comment='未實現損益')
    cost_price = Column(FLOAT(2), nullable=False, comment='成本價')
    quantity = Column(Integer, nullable=False, comment='今日庫存量')
    yd_quantity = Column(Integer, comment='昨日庫存量')
    last_price = Column(FLOAT(2), nullable=False, comment='前一日收盤價')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class SecurityInfoFutures(Base):
    __tablename__ = 'security_info_futures'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    Account = Column(String(50, collation), nullable=False, comment='帳戶代號')
    Market = Column(String(10, collation), nullable=False, comment='市場別')
    Date = Column(
        TIMESTAMP(fsp=6), server_default=time_default, nullable=False, comment='日期')
    Code = Column(String(10, collation), nullable=False, comment='證券代號')
    CodeName = Column(String(50, collation), comment='標的名稱')
    OrderNum = Column(String(50, collation), comment='委託編號')
    OrderBS = Column(String(50, collation), comment='買賣別')
    OrderType = Column(String(50, collation), comment='委託類型')
    Currency = Column(String(50, collation), comment='幣別')
    paddingByte = Column(String(50, collation), comment='unknown')
    Volume = Column(FLOAT(2), comment='成交量')
    ContractAverPrice = Column(FLOAT(2), comment='平均合約價')
    SettlePrice = Column(FLOAT(2), comment='履約價')
    RealPrice = Column(FLOAT(2), comment='unknown')
    FlowProfitLoss = Column(FLOAT(2), comment='unknown')
    SettleProfitLoss = Column(FLOAT(2), comment='unknown')
    StartSecurity = Column(String(50, collation), comment='unknown')
    UpKeepSecurity = Column(String(50, collation), comment='unknown')
    OTAMT = Column(FLOAT(2), comment='unknown')
    MTAMT = Column(FLOAT(2), comment='unknown')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class Watchlist(Base):
    __tablename__ = 'watchlist'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    account = Column(String(50, collation), nullable=False)
    market = Column(String(10, collation), nullable=False)
    code = Column(String(10, collation), nullable=False)
    buyday = Column(TIMESTAMP(fsp=6), server_default=time_default)
    bsh = Column(FLOAT(2), nullable=False)
    position = Column(Integer, nullable=False)
    strategy = Column(String(50, collation), server_default='unknown')
    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class SecurityList(Base):
    __tablename__ = 'security_list'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    underlying_kind = Column(String(50, collation))
    update_date = Column(TIMESTAMP(fsp=6), server_default=time_default)
    target_code = Column(String(10, collation))
    reference = Column(FLOAT(2), nullable=False)
    delivery_date = Column(String(10, collation))
    exchange = Column(String(3, collation))
    delivery_month = Column(String(6, collation))
    name = Column(String(50, collation), default='unknown')
    short_selling_balance = Column(Integer)
    option_right = Column(String(50, collation))
    strike_price = Column(FLOAT(2))
    underlying_code = Column(String(10, collation))
    margin_trading_balance = Column(Integer)
    limit_up = Column(FLOAT(2), nullable=False)
    limit_down = Column(FLOAT(2), nullable=False)
    symbol = Column(String(10, collation))
    category = Column(String(3, collation))
    multiplier = Column(FLOAT(2), default=0)
    currency = Column(String(3, collation), default='TWD')
    day_trade = Column(String(7, collation), default='No')
    code = Column(String(10, collation), nullable=False)
    unit = Column(Integer)
    security_type = Column(String(3, collation))

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData1T(Base):
    __tablename__ = 'kbar_data_1T'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10, collation), nullable=False)
    date = Column(TIMESTAMP(fsp=6), server_default=time_default)
    Time = Column(TIMESTAMP(fsp=6), server_default=time_default)
    hour = Column(Integer)
    minute = Column(Integer)
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData30T(Base):
    __tablename__ = 'kbar_data_30T'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10, collation), nullable=False)
    date = Column(TIMESTAMP(fsp=6), server_default=time_default)
    Time = Column(TIMESTAMP(fsp=6), server_default=time_default)
    hour = Column(Integer)
    minute = Column(Integer)
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData60T(Base):
    __tablename__ = 'kbar_data_60T'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10, collation), nullable=False)
    date = Column(TIMESTAMP(fsp=6), server_default=time_default)
    Time = Column(TIMESTAMP(fsp=6), server_default=time_default)
    hour = Column(Integer)
    minute = Column(Integer)
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class KBarData1D(Base):
    __tablename__ = 'kbar_data_1D'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    name = Column(String(10, collation), nullable=False)
    date = Column(TIMESTAMP(fsp=6), server_default=time_default)
    Time = Column(TIMESTAMP(fsp=6), server_default=time_default)
    hour = Column(Integer)
    minute = Column(Integer)
    Open = Column(FLOAT(2), default=0)
    High = Column(FLOAT(2), default=0)
    Low = Column(FLOAT(2), default=0)
    Close = Column(FLOAT(2), default=0)
    Volume = Column(FLOAT(2), default=0)
    Amount = Column(FLOAT(2), default=0)

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class PutCallRatioList(Base):
    __tablename__ = 'put_call_ratio'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    Date = Column(TIMESTAMP(fsp=6), server_default=time_default, comment='日期')
    PutVolume = Column(FLOAT(2), default=0, comment='賣權成交量')
    CallVolume = Column(FLOAT(2), default=0, comment='買權成交量')
    PutCallVolumeRatio = Column(FLOAT(2), default=0, comment='買賣權成交量比率%')
    PutOpenInterest = Column(FLOAT(2), default=0, comment='賣權未平倉')
    CallOpenInterest = Column(FLOAT(2), default=0, comment='買權未平倉')
    PutCallRatio = Column(FLOAT(2), default=0, comment='買賣權未平倉量比率%')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class ExDividendTable(Base):
    __tablename__ = 'ex_dividend_table'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    Date = Column(
        TIMESTAMP(fsp=6), server_default=time_default, comment='除權除息日期')
    Code = Column(String(10, collation), nullable=False, comment='股票代號')
    Name = Column(String(50, collation), default='unknown', comment='名稱')
    DividendType = Column(String(5, collation), comment='除權息')
    DividendRate = Column(FLOAT(2), default=0, comment='無償配股率')
    CashCapitalRate = Column(FLOAT(2), default=0, comment='現金增資配股率')
    CashCapitalPrice = Column(FLOAT(2), default=0, comment='現金增資認購價')
    CashDividend = Column(FLOAT(2), default=0, comment='現金股利')
    Details = Column(String(10, collation), comment='詳細資料')
    Reference = Column(String(10, collation), comment='參考價試算')
    Quarter = Column(String(10, collation), comment='最近一次申報資料 季別/日期')
    NetValue = Column(FLOAT(2), default=0, comment='最近一次申報每股 (單位)淨值')
    EPS = Column(FLOAT(2), default=0, comment='最近一次申報每股 (單位)盈餘')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class TradingStatement(Base):
    __tablename__ = 'trading_statement'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    Time = Column(
        TIMESTAMP(fsp=6), server_default=time_default, comment='除權除息日期')
    market = Column(String(10, collation), nullable=False, comment='市場別')
    account_id = Column(
        String(50, collation), default='unknown', comment='帳號別')
    code = Column(String(10, collation), nullable=False, comment='證券代號')
    action = Column(String(50, collation), default='unknown', comment='買賣別')
    price = Column(FLOAT(2), default=0, comment='成交價')
    quantity = Column(FLOAT(2), default=0, comment='成交數量')
    amount = Column(FLOAT(2), default=0, comment='成交金額')
    order_cond = Column(
        String(50, collation), default='unknown', comment='現股或融資')
    order_lot = Column(
        String(50, collation), default='unknown', comment='整張或零股')
    op_type = Column(
        String(50, collation), default='unknown', comment='期權委託類型')
    leverage = Column(FLOAT(2), default=1, comment='槓桿比例')
    msg = Column(
        String(200, collation), default='unknown', comment='進出場訊息')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)


class SelectedStocks(Base):
    __tablename__ = 'selected_stocks'

    pk_id = Column(
        Integer, primary_key=True, autoincrement=True, nullable=False)

    code = Column(String(10, collation), nullable=False, comment='證券代號')
    company_name = Column(String(10, collation), comment='證券名稱')
    category = Column(String(50, collation), comment='產業類別')
    date = Column(
        TIMESTAMP(fsp=6), server_default=time_default, comment='選股日期')
    Open = Column(FLOAT(2), nullable=False, comment='開盤價')
    High = Column(FLOAT(2), nullable=False, comment='最高價')
    Low = Column(FLOAT(2), nullable=False, comment='最低價')
    Close = Column(FLOAT(2), nullable=False, comment='收盤價')
    Volume = Column(Integer, comment='成交量')
    Amount = Column(Integer, comment='成交額')
    Strategy = Column(String(10, collation), nullable=False, comment='選股策略')

    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=time_default)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
