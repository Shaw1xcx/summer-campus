"""预处理: 加载分钟数据并生成特征"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import polars as pl
from pathlib import Path

ROOT = Path(r"C:\Users\28247\Desktop\nju\兆信")
M_DIR = ROOT / "任务" / "output" / "minute"
OUT = ROOT / "任务" / "output" / "lstm_v2"
OUT.mkdir(exist_ok=True)

date_files = sorted((M_DIR / "adj_close").iterdir())
codes = sorted(pl.read_csv(date_files[0])['code'].unique().to_list())[:20]
dates = sorted([f.stem for f in date_files])[:40]
print(f"Stocks: {len(codes)}, Dates: {len(dates)}")

frames = []
for d in dates:
    di = int(d)
    ac = pl.read_csv(M_DIR / "adj_close" / f"{d}.csv").with_columns(pl.lit(di).alias("date"))
    ac = ac.filter(pl.col("code").is_in(codes))
    if ac.is_empty(): continue
    for f in ["adj_high", "adj_low", "volume", "turnover"]:
        tmp = pl.read_csv(M_DIR / f / f"{d}.csv").filter(pl.col("code").is_in(codes))
        ac = ac.join(tmp, on=["code", "minute"], how="left")
    frames.append(ac)

df = pl.concat(frames).sort(["code", "date", "minute"])
print(f"Data shape: {df.shape}")

df = df.with_columns([
    (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1).over("code", "date").alias("ret"),
    pl.col("volume").log1p().alias("log_vol"),
    (pl.col("volume") / pl.col("volume").shift(1) - 1).over("code", "date").alias("vol_chg"),
    ((pl.col("adj_high") - pl.col("adj_low")) / (pl.col("adj_close") + 1e-8)).alias("amplitude"),
    (pl.col("turnover") / (pl.col("volume") + 1e-8)).alias("vwap"),
    ((pl.col("adj_close") - pl.col("turnover") / (pl.col("volume") + 1e-8)) / 
     (pl.col("turnover") / (pl.col("volume") + 1e-8) + 1e-8)).alias("vwap_dev"),
])

df = df.with_columns([
    pl.col("ret").rolling_mean(30, min_samples=5).over("code", "date").alias("ret_ma"),
    pl.col("ret").rolling_std(30, min_samples=5).over("code", "date").alias("ret_std"),
    pl.col("vol_chg").rolling_mean(30, min_samples=5).over("code", "date").alias("vol_chg_ma"),
    pl.col("amplitude").rolling_mean(30, min_samples=5).over("code", "date").alias("amp_ma"),
    pl.col("vwap_dev").rolling_mean(30, min_samples=5).over("code", "date").alias("vwap_dev_ma"),
    pl.col("log_vol").rolling_mean(30, min_samples=5).over("code", "date").alias("vol_ma"),
])

df = df.with_columns(pl.col("ret").shift(-1).over("code", "date").alias("next_ret"))
df = df.with_columns(
    pl.when(pl.col("next_ret") > 0.0005).then(pl.lit(2))
    .when(pl.col("next_ret") < -0.0005).then(pl.lit(0))
    .otherwise(pl.lit(1))
    .alias("target")
)
df = df.drop_nulls()
print(f"Valid: {df.shape}")

keep_cols = ["code","date","minute","ret","log_vol","vol_chg","amplitude","vwap_dev",
             "ret_ma","ret_std","vol_chg_ma","amp_ma","vwap_dev_ma","vol_ma","target"]
df.select(keep_cols).write_parquet(OUT / "features.parquet")
print(f"Saved to {OUT / 'features.parquet'}")

cd = df["target"].value_counts().sort("target")
print(f"Class dist: down={cd[0,1]}, flat={cd[1,1]}, up={cd[2,1]}")
print(f"Dates: {len(df['date'].unique())}")