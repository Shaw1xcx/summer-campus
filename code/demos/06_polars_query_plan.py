import polars as pl

# scan_parquet 只登记数据源；下面的操作先组合成查询计划。
query = (
    # 指定 data_*.parquet，避免 macOS 外置盘的 ._ 辅助文件被当成数据。
    pl.scan_parquet("data/lake/**/data_*.parquet")
    # 过滤条件会尽量下推到扫描阶段，减少后续处理行数。
    .filter(pl.col("Volume") >= 100)
    .group_by("symbol")
    .agg(
        pl.len().alias("trades"),
        pl.col("Volume").sum().alias("volume"),
        (pl.col("Price") * pl.col("Volume") / 100)
            .sum().alias("amount_yuan"),
    )
    # 先按成交额降序排序，再只保留前 10 名。
    .sort("amount_yuan", descending=True)
    .head(10)
)

print(query.explain())  # 先看优化后的计划
result = query.collect() # 到这里才执行
print(result)
