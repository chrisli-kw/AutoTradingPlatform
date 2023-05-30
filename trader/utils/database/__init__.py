import logging
from sqlalchemy import create_engine

from .sql import SQLDatabase
from .tables import *
from ...config import DB_NAME, DB_URL


def create_schema(schemaName):
    engine = create_engine(f'mysql+pymysql://{DB_URL}')

    conn = engine.connect()
    if schemaName.lower() not in conn.dialect.get_schema_names(conn):
        logging.warning(f'Schema {schemaName} not exist, create {schemaName}.')
        engine.execute(f"CREATE SCHEMA {schemaName}")
        conn.close()
        logging.warning(f'Done creating schema {schemaName}')

    # disconnect database
    engine.dispose()


db = SQLDatabase()

if db.HAS_DB:
    create_schema(DB_NAME)

    for table in [
        SecurityInfoStocks, SecurityInfoFutures,
        TradingStatement, Watchlist,
        SecurityList, PutCallRatioList, ExDividendTable,
        KBarData1D, KBarData1T, KBarData30T, KBarData60T,

    ]:
        db.create_table(table)
