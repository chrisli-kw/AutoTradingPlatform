import time
import redis
import pickle
import logging
from typing import Iterable

from ...config import HAS_REDIS, REDIS_HOST, REDIS_PORT, REDIS_PWD
from .. import progress_bar


class RedisTools:
    def __init__(self, redisKey='Trader', ttl=86400):
        self.HAS_REDIS = HAS_REDIS
        self.REDIS_KEY = redisKey
        self.ttl = ttl
        self.redis_client = self.init_client()

    def init_client(self):
        '''Initialize Redis by config settings'''

        if self.HAS_REDIS:
            return redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PWD,
                decode_responses=False
            )
        return None

    def _data2byte(self, data: Iterable):
        return pickle.dumps(data)

    def to_redis(self, data: dict):
        '''insert data to Redis at a time'''

        if not self.HAS_REDIS:
            return

        N = len(data)
        if N > 1:
            pipe = self.redis_client.pipeline()

            for i, c in enumerate(data):
                key = f'{self.REDIS_KEY}:{c}'
                pipe.set(key, self._data2byte(data[c]))
                pipe.expire(key, self.ttl)
                progress_bar(N, i)

            pipe.execute()

        else:
            for k, v in data.items():
                key = f'{self.REDIS_KEY}:{k}'
                self.redis_client.set(key, self._data2byte(v))
                self.redis_client.expire(key, self.ttl)

    def query(self, key: str):
        '''query data from redis'''

        if not self.HAS_REDIS:
            return

        n = 0
        while n < 5:
            try:
                data = self.redis_client.get(f"{self.REDIS_KEY}:{key}")
                break
            except:
                n += 1
                logging.error(
                    f"Cannot connect to {REDIS_HOST}, reconnect ({n}/5).")
                self.redis_client = self.init_client()
                time.sleep(1)

        if n == 5:
            return 'Redis ConnectionError'

        try:
            return pickle.loads(data)
        except TypeError:
            return None

    def query_keys(self, keys: str = None, match: str = None):

        if not self.HAS_REDIS:
            return

        if not keys and not match:
            keys = self.redis_client.keys()
        elif match:
            _, keys = self.redis_client.scan(match=match)

        data = self.redis_client.mget(keys)
        return [pickle.loads(d) for d in data if d]

    def delete_keys(self, keys: list):
        '''delete data stored in Redis by key'''

        if not self.HAS_REDIS:
            return

        for k in keys:
            self.redis_client.delete(f'{self.REDIS_KEY}:{k}')

    def clear_all(self):
        '''delete all data stored in Redis'''

        if not self.HAS_REDIS:
            return

        self.redis_client.delete(*self.redis_client.keys())

    def memory_usage(self):
        '''check Redis memory usage'''

        if not self.HAS_REDIS:
            return

        used_memory = self.redis_client.info()['total_system_memory_human']

        print(f'Total keys = {len(self.redis_client.keys())}')
        print(f"Used_memory of {REDIS_HOST}:{REDIS_PORT} = {used_memory}")
