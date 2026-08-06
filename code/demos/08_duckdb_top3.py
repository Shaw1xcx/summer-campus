import duckdb

# SQL 单独保存为字符串，既可以执行，也可以拼到 EXPLAIN ANALYZE 后面。
sql = """
SELECT
    symbol,
    count(*) AS trades,
    sum(Volume) AS volume,
    sum(Price * Volume) / 100 AS amount_yuan
FROM read_parquet(
    'data/lake/**/data_*.parquet',
    -- 从 year=.../date=... 等目录名恢复分区列。
    hive_partitioning = true
)
WHERE date = '20260630'
GROUP BY symbol
ORDER BY amount_yuan DESC
LIMIT 10
"""

# 第一行输出 Polars 格式的业务结果，无需额外安装 pandas。
# 第二行输出每个物理算子的实际运行信息。
print(duckdb.sql(sql).pl())
print(duckdb.sql("EXPLAIN ANALYZE " + sql).fetchall()[0][1])
