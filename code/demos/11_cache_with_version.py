from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import duckdb

DATA_ROOT = Path("data/lake")

# 指纹只读取文件元数据，速度快；它不等同于逐字节内容哈希。
def data_fingerprint(root: Path) -> str:
    records = []
    for path in sorted(root.rglob("*.parquet")):
        stat = path.stat()
        # 相对路径、大小或修改时间任一变化，最终 SHA-256 都会变化。
        records.append(
            f"{path.relative_to(root)}|{stat.st_size}|{stat.st_mtime_ns}"
        )
    return sha256("\n".join(records).encode()).hexdigest()[:16]

@lru_cache(maxsize=128)
def cached_top_amount(date: str, fingerprint: str):
    # fingerprint 虽未出现在 SQL 中，但它是函数参数，因此进入缓存键。
    return duckdb.execute("""
        SELECT symbol, sum(Price * Volume) / 100 AS amount
        FROM read_parquet('data/lake/**/data_*.parquet', hive_partitioning=true)
        WHERE date = ? GROUP BY symbol
        ORDER BY amount DESC LIMIT 10
    """, [date]).fetchall()

# 同时显示耗时与 cache_info，区分“很快”和“确实命中缓存”。
def run(label, version):
    start = perf_counter()
    cached_top_amount("20260630", version)
    print(label, round((perf_counter() - start) * 1000, 3), "ms",
          cached_top_amount.cache_info())

version_1 = data_fingerprint(DATA_ROOT)
run("第一次：冷查询", version_1)
run("第二次：缓存命中", version_1)

# 只改变演示文件的修改时间，模拟一次新的数据发布。
target = next(DATA_ROOT.rglob("*.parquet"))
target.touch()
version_2 = data_fingerprint(DATA_ROOT)
run("数据版本变化后", version_2)

# 通过断言把课堂观察变成可自动检查的结论。
assert version_1 != version_2
assert cached_top_amount.cache_info().misses == 2
