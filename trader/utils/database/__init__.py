import logging
from sqlalchemy import create_engine

from .sql import SQLDatabase
from .redis import RedisTools
from .tables import *
from ...config import DBConfig


def create_schema(schemaName):
    engine = create_engine(f'{DBConfig.ENGINE}://{DBConfig.URL}')

    conn = engine.connect()
    if schemaName.lower() not in conn.dialect.get_schema_names(conn):
        logging.warning(f'Schema {schemaName} not exist, create {schemaName}.')
        engine.execute(f"CREATE SCHEMA {schemaName}")
        conn.close()
        logging.warning(f'Done creating schema {schemaName}')

    # disconnect database
    engine.dispose()


db = SQLDatabase()
redis_tick = RedisTools(redisKey='TickData')

if db.HAS_DB:
    create_schema(DBConfig.NAME)
    Base.metadata.create_all(db.engine)

    KBarTables = {
        '1D': KBarData1D,
        '1T': KBarData1T,
        '30T': KBarData30T,
        '60T': KBarData60T
    }
else:
    KBarTables = {}
