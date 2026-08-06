from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import hashlib
import shutil

import duckdb
import numpy as np
import polars as pl
import zarr


SOURCE = Path("data/trades.csv")
WORK = Path("data/end-to-end")
PARQUET = WORK / "trades-zstd.parquet"
LAKE = WORK / "by-date"
ZARR_STORE = WORK / "daily-vwap.zarr"

# 每次从同一份虚构 CSV 重建结果，确保课堂上可以重复演示。
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

# T1 · 数据表示：统一类型，再把文本 CSV 写成带统计信息的 ZSTD Parquet。
trades = pl.read_csv(SOURCE).with_columns(
    pl.col("date").cast(pl.String),
    pl.col("Price").cast(pl.Int32),
    pl.col("Volume").cast(pl.Int32),
    pl.col("BSFlag").cast(pl.Int8),
)
trades.write_parquet(PARQUET, compression="zstd", statistics=True)

# T2 + T4 · 文件布局与 SQL：DuckDB 把总文件拆成 6 个日期目录。
connection = duckdb.connect()
connection.execute(f"""
COPY (SELECT * FROM read_parquet('{PARQUET.as_posix()}'))
TO '{LAKE.as_posix()}'
(FORMAT parquet, PARTITION_BY (date), OVERWRITE);
""")
daily_files = sorted(LAKE.rglob("data_*.parquet"))
daily_file_by_date = {
    path.parent.name.removeprefix("date="): path.resolve()
    for path in daily_files
}
available_dates = sorted(daily_file_by_date)
available_symbols = sorted(trades["symbol"].unique().to_list())

# T5 · 缓存一致性：路径、大小和修改时间共同形成数据版本。
def lake_version():
    parts = []
    for path in daily_files:
        stat = path.stat()
        parts.append(f"{path.relative_to(LAKE)}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]

# T3 · Polars Lazy：只有 collect() 才真正读取某一天的 Parquet。
@lru_cache(maxsize=32)
def cached_daily_summary(date_text, data_version):
    path = daily_file_by_date[date_text]
    return (
        pl.scan_parquet(path)
        .with_columns((pl.col("Price") * pl.col("Volume")).alias("turnover"))
        .group_by("symbol")
        .agg(
            pl.len().alias("trades"),
            pl.col("Volume").sum().alias("volume"),
            pl.col("turnover").sum(),
        )
        .with_columns((pl.col("turnover") / pl.col("volume")).round(2).alias("vwap"))
        .sort(["turnover", "symbol"], descending=[True, False])
        .collect()
    )

def daily_summary(date_text):
    # 数据文件变化时，lake_version 变化，自然形成一个新的缓存键。
    return cached_daily_summary(date_text, lake_version())

selected_date = "20260630"
selected_path = daily_file_by_date[selected_date]
polars_first = daily_summary(selected_date)   # 冷查询：miss
cache_after_first = cached_daily_summary.cache_info()
polars_second = daily_summary(selected_date)  # 相同日期和版本：hit
cache_after_second = cached_daily_summary.cache_info()

# T4 · DuckDB：独立用 SQL 计算同一结果，作为交叉核对。
duckdb_top3 = connection.execute("""
SELECT
    symbol,
    count(*) AS trades,
    sum(Volume) AS volume,
    sum(Price * Volume) AS turnover,
    round(sum(Price * Volume)::DOUBLE / sum(Volume), 2) AS vwap
FROM read_parquet(?)
GROUP BY symbol
ORDER BY turnover DESC, symbol
LIMIT 3
""", [str(selected_path)]).pl()

columns = ["symbol", "trades", "volume", "turnover", "vwap"]
polars_top3_rows = polars_second.select(columns).head(3).rows()
duckdb_top3_rows = duckdb_top3.select(columns).rows()

# T7 · 并发执行：六个互不依赖的日期文件可以并发读取和聚合。
with ThreadPoolExecutor(max_workers=4) as pool:
    summaries = dict(zip(available_dates, pool.map(daily_summary, available_dates)))

# T6 · Zarr：把各日各股票的 VWAP 变成“日期 × 股票”二维数组。
vwap_cube = np.empty((len(available_dates), len(available_symbols)), dtype="float32")
for date_index, date_text in enumerate(available_dates):
    values_by_symbol = dict(summaries[date_text].select("symbol", "vwap").iter_rows())
    for symbol_index, symbol in enumerate(available_symbols):
        vwap_cube[date_index, symbol_index] = values_by_symbol[symbol]

vwap_array = zarr.create_array(
    store=str(ZARR_STORE),
    shape=vwap_cube.shape,
    chunks=(1, 4),
    shards=(2, 8),
    dtype="float32",
    zarr_format=3,
)
vwap_array[:] = vwap_cube
vwap_array.attrs["dates"] = available_dates
vwap_array.attrs["symbols"] = available_symbols
reopened = zarr.open_array(str(ZARR_STORE), mode="r")

# 最终只打印课堂上需要核对的事实，避免用不稳定的耗时作结论。
print("输入:", trades.height, "行 /", len(available_dates), "天 /", len(available_symbols), "只股票")
print("Parquet:", round(PARQUET.stat().st_size / 1024, 1), "KiB / 日期文件:", len(daily_files), "个")
print("第一次查询:", cache_after_first)
print("第二次查询:", cache_after_second)
print("两次结果一致:", polars_first.equals(polars_second))
print("Polars 与 DuckDB Top 3 一致:", polars_top3_rows == duckdb_top3_rows)
print("20260630 按成交额 Top 3（价格单位：分）:")
for symbol, trade_count, volume, turnover, vwap in duckdb_top3_rows:
    print(f"  {symbol}: trades={trade_count}, volume={volume}, turnover={turnover}, vwap={vwap:.2f}")
print("并发汇总:", len(summaries), "天，每天", summaries[selected_date].height, "只股票")
print("Zarr shape/chunks/shards:", reopened.shape, reopened.chunks, reopened.shards)
date_index = available_dates.index(selected_date)
symbol_index = available_symbols.index("S002")
print("Zarr[20260630, S002] VWAP:", round(float(reopened[date_index, symbol_index]), 2))
print("最终缓存:", cached_daily_summary.cache_info())

assert np.allclose(reopened[:], vwap_cube)
assert polars_top3_rows == duckdb_top3_rows
