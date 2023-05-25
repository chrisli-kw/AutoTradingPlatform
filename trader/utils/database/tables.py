from sqlalchemy import Column, Integer, FLOAT, String, text, DateTime, Numeric, JSON, BOOLEAN, Text
from sqlalchemy.dialects.mysql import TIMESTAMP

from .sql import Base


class Watchlist(Base):
    __tablename__ = 'watchlist'

    pk_id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    account = Column(String(50, 'utf8mb4_unicode_ci'), nullable=False)
    market = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    code = Column(String(10, 'utf8mb4_unicode_ci'), nullable=False)
    buyday = Column(TIMESTAMP(fsp=6), server_default=text("CURRENT_TIMESTAMP(6)"))
    bsh = Column(FLOAT(2), nullable=False)
    position = Column(Integer, nullable=False)
    strategy = Column(String(50, 'utf8mb4_unicode_ci'), server_default='unknown')
    create_time = Column(
        TIMESTAMP(fsp=6), nullable=False, server_default=text("CURRENT_TIMESTAMP(6)"))

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
