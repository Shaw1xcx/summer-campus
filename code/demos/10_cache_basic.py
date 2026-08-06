from functools import lru_cache
from time import perf_counter
import duckdb

# 装饰器在函数外面增加一层内存缓存。
# 调用时它先用全部函数参数组成 key，再决定是直接返回还是进入函数体。
# maxsize=128 表示最多保留 128 组不同 key 对应的结果。
@lru_cache(maxsize=128)
def top_amount(date: str, data_version: str):
    # 只有 miss 时才会进入这个函数体并执行 SQL。
    # data_version 不用于 SQL 过滤，但它仍是函数参数，因此参与 key。
    # 这样 v2 不会命中 v1 时算出的旧结果。
    return duckdb.execute("""
        SELECT symbol, sum(Price * Volume) / 100 AS amount
        FROM read_parquet('data/lake/**/data_*.parquet', hive_partitioning=true)
        -- 问号是参数占位符，不把外部字符串直接拼进 SQL。
        WHERE date = ?
        GROUP BY symbol ORDER BY amount DESC LIMIT 10
    """, [date]).fetchall()

# 三次调用的 key 依次是 (20260630,v1)、(20260630,v1)、(20260630,v2)。
# cache_info() 中 hits 是直接复用次数，misses 是实际进入函数体的次数。
for version in ["v1", "v1", "v2"]:
    t0 = perf_counter()
    top_amount("20260630", version)
    print(version, perf_counter() - t0, top_amount.cache_info())
