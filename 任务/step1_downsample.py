"""
数据处理任务 - 完整实现
数据源: 任务/data/TRADE/YYYYMMDD/STOCK.csv
复权因子: 任务/adjfactor.pkl
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pickle
from pathlib import Path
import polars as pl
import gc

# === 配置 ===
ROOT = Path(r"C:\Users\28247\Desktop\nju\兆信")
TRADE_DIR = ROOT / "任务" / "data" / "TRADE"
ADJ_PATH = ROOT / "任务" / "adjfactor.pkl"
OUT_DIR = ROOT / "任务" / "output"
OUT_DIR.mkdir(exist_ok=True)
DAILY_DIR = OUT_DIR / "daily"
MINUTE_DIR = OUT_DIR / "minute"
DAILY_DIR.mkdir(exist_ok=True)
MINUTE_DIR.mkdir(exist_ok=True)

# === 1. 读取复权因子 ===
print("=== 读取复权因子 ===")
with open(ADJ_PATH, 'rb') as f:
    adj_raw = pickle.load(f)  # pandas DataFrame, index=date, columns=stock.SH/SZ
adj_dates = adj_raw.index.tolist()
adj_cols = adj_raw.columns.tolist()

# 构建 code -> adj map: {date: {code: adj_factor}}
code_adj = {}
for i, date in enumerate(adj_dates):
    date_str = str(date).replace('-', '')[:8]
    row = adj_raw.iloc[i]
    code_adj[date_str] = {col.split('.')[0]: float(row[col]) for col in adj_cols}

# 获取股票代码列表和日期列表
all_dates = sorted([d.name for d in TRADE_DIR.iterdir() if d.is_dir()])
all_codes = sorted(set(f.stem for d in TRADE_DIR.iterdir() if d.is_dir() for f in d.iterdir() if f.suffix == '.csv'))
print(f"日期数: {len(all_dates)}, 股票数: {len(all_codes)}")

# === 2. 逐笔数据读取并降采样 ===
print("\n=== 数据降采样 ===")

# 定义 BSFlag 含义：0=主买, 1=主卖, 2=集合竞价
# 主买量 = BSFlag==0 的成交量求和
# 主卖量 = BSFlag==1 的成交量求和
# 主买额 = BSFlag==0 的 Price*Volume 求和
# 主卖额 = BSFlag==1 的 Price*Volume 求和

all_daily = []
all_minute = []

for di, date_str in enumerate(all_dates):
    date_dir = TRADE_DIR / date_str
    files = sorted(date_dir.iterdir())
    
    date_dfs = []
    for fp in files:
        code = fp.stem
        try:
            df = pl.read_csv(fp, schema_overrides={
                'Time': pl.Int64, 'Price': pl.Int64, 'Volume': pl.Int64, 'BSFlag': pl.Int64
            })
        except Exception:
            continue
        if df.is_empty():
            continue
        
        df = df.with_columns(pl.lit(code).alias('code'))
        date_dfs.append(df)
    
    if not date_dfs:
        continue
    
    tick = pl.concat(date_dfs)
    
    # 计算成交额 (Price/100 * Volume，因为 Price 乘了 100)
    tick = tick.with_columns(
        ((pl.col('Price') / 100.0) * pl.col('Volume')).alias('amount'),
        pl.col('Time').cast(pl.Int64).alias('_time'),
    )
    
    # 分钟标签：Time // 100000 得到 HHMM (如 930, 931, ...)
    # 注意：915-925 是集合竞价，930-1130, 1300-1457 是连续竞价
    tick = tick.with_columns(
        (pl.col('Time') // 100_000).cast(pl.Int32).alias('minute'),
    )
    
    # --- 日频聚合 ---
    daily = tick.group_by('code').agg([
        pl.col('Price').first().alias('open'),
        pl.col('Price').max().alias('high'),
        pl.col('Price').min().alias('low'),
        pl.col('Price').last().alias('close'),
        pl.col('Volume').sum().alias('volume'),
        pl.len().alias('tick_count'),
        pl.col('amount').sum().alias('turnover'),
        pl.col('Volume').filter(pl.col('BSFlag') == 0).sum().alias('buy_volume'),
        pl.col('Volume').filter(pl.col('BSFlag') == 1).sum().alias('sell_volume'),
        pl.col('amount').filter(pl.col('BSFlag') == 0).sum().alias('buy_turnover'),
        pl.col('amount').filter(pl.col('BSFlag') == 1).sum().alias('sell_turnover'),
    ]).with_columns(pl.lit(date_str).alias('date'))
    
    # 应用复权因子
    adj_map = code_adj.get(date_str, {})
    daily = daily.with_columns(
        pl.col('code').replace_strict(adj_map, default=1.0).alias('adj_factor')
    )
    for col in ['open', 'high', 'low', 'close']:
        daily = daily.with_columns(
            (pl.col(col) * pl.col('adj_factor') / 100.0).alias(f'adj_{col}')
        )
    # 原始价格除以100恢复
    for col in ['open', 'high', 'low', 'close']:
        daily = daily.with_columns((pl.col(col) / 100.0).alias(col))
    
    # 填充规则：日频价格用前一日收盘价填充（后面统一处理）
    all_daily.append(daily)
    
    # --- 分钟频聚合 ---
    # 过滤连续竞价阶段: 930-1129, 1300-1456
    minute = tick.filter(
        ((pl.col('minute') >= 930) & (pl.col('minute') <= 1129)) |
        ((pl.col('minute') >= 1300) & (pl.col('minute') <= 1456))
    ).group_by(['code', 'minute']).agg([
        pl.col('Price').first().alias('open'),
        pl.col('Price').max().alias('high'),
        pl.col('Price').min().alias('low'),
        pl.col('Price').last().alias('close'),
        pl.col('Volume').sum().alias('volume'),
        pl.len().alias('tick_count'),
        pl.col('amount').sum().alias('turnover'),
        pl.col('Volume').filter(pl.col('BSFlag') == 0).sum().alias('buy_volume'),
        pl.col('Volume').filter(pl.col('BSFlag') == 1).sum().alias('sell_volume'),
        pl.col('amount').filter(pl.col('BSFlag') == 0).sum().alias('buy_turnover'),
        pl.col('amount').filter(pl.col('BSFlag') == 1).sum().alias('sell_turnover'),
    ]).with_columns(pl.lit(date_str).alias('date'))
    
    minute = minute.with_columns(
        pl.col('code').replace_strict(adj_map, default=1.0).alias('adj_factor')
    )
    for col in ['open', 'high', 'low', 'close']:
        minute = minute.with_columns(
            (pl.col(col) * pl.col('adj_factor') / 100.0).alias(f'adj_{col}')
        )
    for col in ['open', 'high', 'low', 'close']:
        minute = minute.with_columns((pl.col(col) / 100.0).alias(col))
    
    # 填充规则：分钟频用前一分钟收盘价填充（后面统一处理）
    all_minute.append(minute)
    
    if (di + 1) % 50 == 0:
        print(f"  已处理 {di+1}/{len(all_dates)} 天")

# 合并所有天
daily_df = pl.concat(all_daily).sort(['code', 'date'])
minute_df = pl.concat(all_minute).sort(['code', 'date', 'minute'])

print(f"日频数据: {daily_df.shape}")
print(f"分钟频数据: {minute_df.shape}")

# === 填充规则 ===
# 日频：用前一日收盘价填充缺失的 OHLC
daily_df = daily_df.with_columns([
    pl.col('open').fill_null(pl.col('close').shift(1)).over('code'),
    pl.col('high').fill_null(pl.col('close').shift(1)).over('code'),
    pl.col('low').fill_null(pl.col('close').shift(1)).over('code'),
    pl.col('close').fill_null(pl.col('close').shift(1)).over('code'),
    pl.col('adj_open').fill_null(pl.col('adj_close').shift(1)).over('code'),
    pl.col('adj_high').fill_null(pl.col('adj_close').shift(1)).over('code'),
    pl.col('adj_low').fill_null(pl.col('adj_close').shift(1)).over('code'),
    pl.col('adj_close').fill_null(pl.col('adj_close').shift(1)).over('code'),
])

# 分钟频：用前一分钟收盘价填充
minute_df = minute_df.with_columns([
    pl.col('open').fill_null(pl.col('close').shift(1)).over('code'),
    pl.col('high').fill_null(pl.col('close').shift(1)).over('code'),
    pl.col('low').fill_null(pl.col('close').shift(1)).over('code'),
    pl.col('close').fill_null(pl.col('close').shift(1)).over('code'),
    pl.col('adj_open').fill_null(pl.col('adj_close').shift(1)).over('code'),
    pl.col('adj_high').fill_null(pl.col('adj_close').shift(1)).over('code'),
    pl.col('adj_low').fill_null(pl.col('adj_close').shift(1)).over('code'),
    pl.col('adj_close').fill_null(pl.col('adj_close').shift(1)).over('code'),
])

# === 保存日频数据：每个字段一张表 ===
print("\n=== 保存日频数据 ===")
daily_fields = ['open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close',
                'volume', 'tick_count', 'turnover', 'buy_volume', 'sell_volume', 'buy_turnover', 'sell_turnover']
for field in daily_fields:
    tbl = daily_df.select(['date', 'code', field]).sort(['date', 'code'])
    tbl.write_csv(DAILY_DIR / f"daily_{field}.csv")
    print(f"  daily_{field}.csv: {tbl.shape}")

# === 保存分钟频数据：每个字段一个文件夹，文件夹内每日一张表 ===
print("\n=== 保存分钟频数据 ===")
minute_fields = ['open', 'high', 'low', 'close', 'adj_open', 'adj_high', 'adj_low', 'adj_close',
                 'volume', 'tick_count', 'turnover', 'buy_volume', 'sell_volume', 'buy_turnover', 'sell_turnover']
for field in minute_fields:
    field_dir = MINUTE_DIR / field
    field_dir.mkdir(exist_ok=True)
    for date_str in minute_df['date'].unique().to_list():
        tbl = minute_df.filter(pl.col('date') == date_str).select(['code', 'minute', field]).sort(['code', 'minute'])
        tbl.write_csv(field_dir / f"{date_str}.csv")
    print(f"  minute/{field}/: {len(list(field_dir.iterdir()))} 天")

print("\n=== 降采样完成 ===")
print(f"日频数据: {DAILY_DIR}")
print(f"分钟频数据: {MINUTE_DIR}")

# 清理内存
del all_daily, all_minute, tick, daily_df, minute_df
gc.collect()