import sys; sys.stdout.reconfigure(encoding='utf-8')
import polars as pl, time
from pathlib import Path

t0 = time.time()
df = pl.read_parquet(r"C:\Users\28247\Desktop\nju\兆信\任务\output\lstm_v2\features.parquet")
print(f"Loaded: {df.shape} in {time.time()-t0:.1f}s")
print(f"Dates: {len(df['date'].unique())}")
print(f"Cols: {df.columns}")
print(f"Head: {df.head(2)}")
print(f"Target counts: {df['target'].value_counts().sort('target')}")