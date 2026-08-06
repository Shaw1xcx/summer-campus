from time import perf_counter
import polars as pl

# 收紧文件名模式，不读取 macOS 可能生成的 ._ 辅助文件。
path = "data/lake/**/data_*.parquet"

# 这段业务逻辑既能接收 DataFrame，也能接收 LazyFrame。
# 因而两条路径的过滤、聚合和排序完全一致，比较才公平。
def summarize(frame):
    return (
        frame
        .filter(pl.col("Volume") >= 100)
        .group_by("symbol")
        .agg(
            pl.col("Volume").sum().alias("volume"),
            (pl.col("Price") * pl.col("Volume") / 100)
                .sum().alias("amount_yuan"),
        )
        .sort("amount_yuan", descending=True)
        .head(10)
    )

start = perf_counter()
eager_source = pl.read_parquet(path)       # 先物化全部可读列和行
eager_result = summarize(eager_source)
eager_seconds = perf_counter() - start

# Lazy 路径先得到计划；此时还没有真正读取全部数据。
lazy_query = summarize(pl.scan_parquet(path))
print("优化后的 Lazy 计划：")
print(lazy_query.explain(optimized=True))

start = perf_counter()
lazy_result = lazy_query.collect()         # 优化后再读取和执行
lazy_seconds = perf_counter() - start

# 耗时受缓存和机器状态影响，课堂上关注执行计划与数量级。
print("Eager:", round(eager_seconds, 4), "s")
print("Lazy: ", round(lazy_seconds, 4), "s")
print(lazy_result)
# 性能优化不能改变业务结果，因此最后必须做一致性校验。
assert eager_result.equals(lazy_result)
