import polars as pl

df = pl.read_parquet("data/trades_zstd.parquet")

print(df.head())     # 前5行
print(df.shape)      # 行数和列数
print(df.schema)     # 每列的数据类型
print(df.describe()) # 基本统计信息
