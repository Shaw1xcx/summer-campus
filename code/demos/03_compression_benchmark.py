from pathlib import Path
from time import perf_counter
import gzip
import io
import polars as pl

source = Path("data/trades.csv")
df = pl.read_csv(source)
out = Path("data/format-benchmark")
out.mkdir(parents=True, exist_ok=True)

# 统一计时函数：既返回函数结果，也返回实际经过的秒数。
def timed(callable_):
    start = perf_counter()
    value = callable_()
    return value, perf_counter() - start

# CSV.gz 仍然是文本 CSV，只是在文件外层增加 gzip 压缩。
def write_gzip(path):
    with gzip.open(path, "wb", compresslevel=6) as stream:
        stream.write(df.write_csv().encode("utf-8"))

# 五种输出使用不同文件名，防止互相覆盖。
files = {
    "CSV": out / "trades.csv",
    "CSV.gz": out / "trades.csv.gz",
    "Parquet-ZSTD": out / "trades_zstd.parquet",
    "Parquet-Snappy": out / "trades_snappy.parquet",
    "Parquet-LZ4_RAW": out / "trades_lz4.parquet",
}

# 每种格式分别定义写入和读取动作，便于使用同一套计时逻辑。
writers = {
    "CSV": lambda: df.write_csv(files["CSV"]),
    "CSV.gz": lambda: write_gzip(files["CSV.gz"]),
    "Parquet-ZSTD": lambda: df.write_parquet(files["Parquet-ZSTD"], compression="zstd"),
    "Parquet-Snappy": lambda: df.write_parquet(files["Parquet-Snappy"], compression="snappy"),
    # Polars 参数名为 "lz4"，写入的是 Parquet 支持的 LZ4_RAW codec。
    "Parquet-LZ4_RAW": lambda: df.write_parquet(files["Parquet-LZ4_RAW"], compression="lz4"),
}
readers = {
    "CSV": lambda: pl.read_csv(files["CSV"]),
    "CSV.gz": lambda: pl.read_csv(io.BytesIO(gzip.decompress(files["CSV.gz"].read_bytes()))),
    "Parquet-ZSTD": lambda: pl.read_parquet(files["Parquet-ZSTD"]),
    "Parquet-Snappy": lambda: pl.read_parquet(files["Parquet-Snappy"]),
    "Parquet-LZ4_RAW": lambda: pl.read_parquet(files["Parquet-LZ4_RAW"]),
}

# 每次演示都重新写入、重新读取，并检查读取行数是否一致。
rows = []
for name in files:
    _, write_seconds = timed(writers[name])
    loaded, read_seconds = timed(readers[name])
    assert loaded.height == df.height
    rows.append({
        "format": name,
        "size_KiB": round(files[name].stat().st_size / 1024, 1),
        "write_ms": round(write_seconds * 1000, 2),
        "read_ms": round(read_seconds * 1000, 2),
    })

# 按文件大小排序，再讨论“更小”和“更快”是否为同一件事。
print(pl.DataFrame(rows).sort("size_KiB"))
