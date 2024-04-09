import redis, pickle
from dotenv import load_dotenv
import os

from typing import Union, Dict, Optional


class RedisUtil():
    load_dotenv()
    __host = os.getenv("REDIS_HOST")
    __port = int(os.getenv("REDIS_PORT_INT"))
    __database = int(os.getenv("REDIS_DB_INT"))
    __username = os.getenv("REDIS_USERNAME")
    __password = os.getenv("REDIS_PASSWORD")

    @classmethod
    def put(cls, key: Union[str, int], value: Union[Dict[str, int], str, int], ex: int = 8640000):
        with cls.__get_client() as r:
            # str, int, float이 아닌 경우 bytes로 변환해서 입출력해야 한다.
            # 근데 str, int도 변환해서 입출력해도 딱히 문제가 없다(시간이 더 걸릴 뿐)
            if value:
                print(f"put {key}: {value}")
                r.set(key, pickle.dumps(value), ex=ex)


    @classmethod
    def delete(cls, key: Union[str, int]):
        with cls.__get_client() as r:
            if key in r.keys():
                r.delete(key)


    @classmethod
    def get(cls, key: Union[str, int]) -> Optional[Union[Dict[str, int], str, int]]:
        result = None
        with cls.__get_client() as r:
            
            serialized_value = r.get(key)
            if serialized_value:
                result = pickle.loads(serialized_value)

        return result

    @classmethod
    def get_serialized(cls, key: Union[str, int]) -> Optional[Union[Dict[str, int], str, int]]:
        with cls.__get_client() as r:
            return r.get(key)


    @classmethod
    def clear(cls):
        with cls.__get_client() as r:
            for key in r.keys():
                r.delete(key)


    @classmethod
    def check(cls):
        with cls.__get_client() as r:
            for key in r.keys():
                print(f"{key}: {pickle.loads(r.get(key))}")


    @classmethod
    def __get_client(cls):
        return redis.Redis(
            host=cls.__host,
            port=cls.__port,
            db=cls.__database,
            username=cls.__username,
            password=cls.__password
        )
