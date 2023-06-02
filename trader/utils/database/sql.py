import logging
import numpy as np
import pandas as pd
from sqlalchemy import update
from sqlalchemy.pool import QueuePool
from sqlalchemy import asc, desc, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, load_only

from ...config import HAS_DB, DB_NAME, DB_URL


Base = declarative_base()


class SQLDatabase:
    def __init__(self):
        self.HAS_DB = HAS_DB
        if self.HAS_DB:
            self.sql_connect = f"mysql+pymysql://{DB_URL}/{DB_NAME}?charset=utf8mb4&binary_prefix=true"
            self.engine = create_engine(
                self.sql_connect,
                pool_size=50,
                # max_overflow=max_overflow,
                pool_recycle=10,
                pool_timeout=10,
                pool_pre_ping=True,
                poolclass=QueuePool,
                pool_use_lifo=True,
                echo=False
            )
            self.sessionmaker_ = sessionmaker(bind=self.engine)
        
    def get_session(self):
        return scoped_session(self.sessionmaker_)

    def create_table(self, table):
        engine = create_engine(self.sql_connect)
        table
        Base.metadata.create_all(engine)

    def query(self, table, *filterBy, **conditions):
        '''Get data from DB'''

        session = self.get_session()
        query = session.query(table).filter(*filterBy)

        if conditions.get('orderBy'):
            if conditions['orderBy'] == 'asc':
                query = query.order_by(asc(conditions['orderTarget']))
            else:
                query = query.order_by(desc(conditions['orderTarget']))

        if conditions.get('limit'):
            query = query.limit(conditions['limit'])

        if conditions.get('fields'):
            query = query.options(load_only(*conditions['fields']))

        result = pd.read_sql(query.statement, session.bind)
        session.close()

        for c in ['pk_id', 'create_time']:
            if c in result.columns:
                result = result.drop(['pk_id', 'create_time'], axis=1)
        return result
    
    def queryAll(self, table):
        '''Query all data from DB'''

        session = self.get_session()
        sql = f"SELECT * FROM {DB_NAME}.{table.__tablename__}"
        result = pd.read_sql(sql, session.bind)
        session.close()
        return result

    def update(self, table, update_content:dict, *filterBy):
        '''Update data in table'''
        
        session = self.get_session()
        session.execute(
            update(table).where(*filterBy).values(update_content)
        )
        session.commit()
        session.close()

    def delete(self, table, *args):
        '''Delete data in table'''

        session = self.get_session()
        query_data = session.query(table).filter(*args).all()
        for row in query_data:
            session.delete(row)
            session.commit()
        session.close()

    def check_exist(self, table, **kwargs):
        '''Check if data exists in table'''

        session = self.get_session()
        q = session.query(table).filter_by(**kwargs)
        check_result = session.query(q.exists()).scalar()
        session.close()
        return check_result

    def dataframe_to_DB(self, df: pd.DataFrame, table):
        '''Import dataframe to DB'''

        # 轉換時間格式
        for col in df.columns:
            if df[col].dtype == pd._libs.tslibs.timestamps.Timestamp:
                df[col] = df[col].astype(str).apply(
                    lambda x: None if x == 'NaT' else x)

        # 補空值
        if df.isnull().sum().sum():
            df = df.where(pd.notnull(df), '')

        datarows = np.array(df.to_dict('records'))
        alldata = np.array([table(**row) for row in datarows])

        # upload data
        session = self.get_session()
        session.add_all(alldata)
        session.commit()
        session.close()

    def add_data(self, table, **input_data):
        try:
            session = self.get_session()
            data = table(**input_data)
            session.add(data)
            session.commit()
        except:
            session.rollback()
            logging.error(
                f"Save data into {table.__tablename__} failed")
        finally:
            session.close()
