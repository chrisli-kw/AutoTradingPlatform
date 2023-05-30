import time
import redis
import pickle
import logging
from typing import Iterable

from ...config import REDIS_HOST, REDIS_PORT, REDIS_PWD


class RedisTools:
    def __init__(self, redisKey='TickData'):
        self.REDIS_KEY = redisKey

        self.redis_client = self.init_client()
        self.redis_data = {}

    def init_client(self):
        '''Initialize  Redis/Codis by config settings'''

        return redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PWD,
            decode_responses=False
        )

    def _progress_bar(self, N: int, i: int):
        n = i+1
        progress = f"\r|{'█'*int(n*50/N)}{' '*(50-int(n*50/N))} | {n}/{N} ({round(n/N*100, 2)})%"
        print(progress, end='')

    def _data2byte(self, data: Iterable):
        return pickle.dumps(data)

    def to_redis(self, data: dict):
        '''insert data to Redis at a time'''

        N = len(data)
        if N > 1:
            pipe = self.redis_client.pipeline()

            for i, c in enumerate(data):
                pipe.set(f'{self.REDIS_KEY}:{c}', self._data2byte(data[c]))
                self._progress_bar(N, i)

            pipe.execute()

        else:
            for k, v in data.items():
                self.redis_client.set(f'{self.REDIS_KEY}:{k}', self._data2byte(v))

    def query(self, key: str):
        '''query data from redis'''

        n = 0
        while n < 5:
            try:
                data = self.redis_client.get(f"{self.REDIS_KEY}:{key}")
                break
            except:
                n += 1
                logging.error(f"Cannot connect to {REDIS_HOST}, reconnect ({n}/5).")
                self.redis_client = self.init_client()
                time.sleep(1)

        if n == 5:
            return 'Redis ConnectionError'

        try:
            return pickle.loads(data)
        except TypeError:
            return None

    def query_keys(self, keys: str = None, match: str = None):
        if not keys and not match:
            keys = self.redis_client.keys()
        elif match:
            _, keys = self.redis_client.scan(match=match)

        data = self.redis_client.mget(keys)
        return [pickle.loads(d) for d in data if d]

    def delete_keys(self, keys: list):
        '''delete data stored in Redis by key'''

        for k in keys:
            self.redis_client.delete(f'{self.REDIS_KEY}:{k}')

    def clear_all(self):
        '''delete all data stored in Redis'''

        self.redis_client.delete(*self.redis_client.keys())

    def memory_usage(self):
        '''check Redis memory usage'''
        used_memory = self.redis_client.info()['total_system_memory_human']

        print(f'Total keys = {len(self.redis_client.keys())}')
        print(f"Used_memory of {REDIS_HOST}:{REDIS_PORT} = {used_memory}")
