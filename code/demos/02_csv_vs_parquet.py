from pathlib import Path
import polars as pl

# 输入和输出保存的是同一张逻辑表，只是物理格式不同。
csv = Path("data/trades.csv")
parquet = Path("data/trades_zstd.parquet")

# CSV 没有可靠的列类型元数据，因此读取时明确指定整数宽度。
df = pl.read_csv(csv, schema_overrides={
    "Price": pl.Int32,
    "Volume": pl.Int32,
    "BSFlag": pl.Int8,
})
# statistics=True 会写入行组统计，查询引擎可据此跳过部分数据。
df.write_parquet(parquet, compression="zstd", statistics=True)

# 先比较磁盘大小，再查看 Parquet 中保留下来的列类型。
print("CSV:    ", round(csv.stat().st_size / 1024, 1), "KiB")
print("Parquet:", round(parquet.stat().st_size / 1024, 1), "KiB")
print(df.schema)
