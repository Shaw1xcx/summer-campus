from pathlib import Path
import polars as pl

# 所有数字都是为了课堂演示自行构造的，不读取课程作业源文件。
root = Path("data")
root.mkdir(exist_ok=True)

# 一、交易表：6 天 × 8 只虚构股票 × 每天每只 250 笔 = 12,000 行。
dates = ["20260625", "20260626", "20260627",
         "20260628", "20260629", "20260630"]
symbols = [f"S{i:03d}" for i in range(1, 9)]
rows = []
for day_no, day in enumerate(dates):
    for symbol_no, symbol in enumerate(symbols):
        base_price = 1000 + symbol_no * 25 + day_no * 3
        for trade_no in range(250):
            rows.append({
                "Time": 93_000_000 + trade_no * 1_000 + symbol_no,
                "Price": base_price + trade_no % 11 - 5,
                "Volume": (trade_no % 8 + 1) * 100,
                "BSFlag": trade_no % 3,
                "date": day,
                "symbol": symbol,
            })

trades = pl.DataFrame(rows).cast({
    "Time": pl.Int64, "Price": pl.Int32,
    "Volume": pl.Int32, "BSFlag": pl.Int8,
})
trades.write_csv(root / "trades.csv")
trades.write_parquet(
    root / "trades.parquet", compression="zstd", statistics=True
)

# 二、因子表：4 天 × 6 只虚构股票，每个坐标有 3 个手工公式生成的因子。
factor_rows = []
for day_no, day in enumerate(dates[-4:]):
    for symbol_no, symbol in enumerate(symbols[:6]):
        factor_rows.append({
            "date": day,
            "symbol": symbol,
            "momentum": round((day_no * 6 + symbol_no + 1) / 100, 2),
            "volatility": round(0.20 + day_no * 0.02 + symbol_no * 0.01, 2),
            "quality": round(1.00 + symbol_no * 0.10 - day_no * 0.05, 2),
        })
factors = pl.DataFrame(factor_rows)
factors.write_csv(root / "factor-values.csv")

# 三、线程任务表：4 个 CPU 任务和 4 个模拟 I/O 等待任务。
tasks = pl.DataFrame({
    "task": ["CPU-A", "CPU-B", "CPU-C", "CPU-D",
             "IO-A", "IO-B", "IO-C", "IO-D"],
    "kind": ["cpu"] * 4 + ["io"] * 4,
    "seed": [1, 2, 3, 4, 0, 0, 0, 0],
    "rounds": [1_500_000] * 4 + [0] * 4,
    "delay_seconds": [0.0] * 4 + [0.12] * 4,
})
tasks.write_csv(root / "thread-tasks.csv")

# 四、竞态条件订单：库存只有 10，两张订单都想购买 7。
orders = pl.DataFrame({
    "order": ["Order-A", "Order-B"],
    "initial_stock": [10, 10],
    "quantity": [7, 7],
})
orders.write_csv(root / "orders.csv")

print("trades:", trades.shape, "→", root / "trades.csv")
print("factors:", factors.shape, "→", root / "factor-values.csv")
print("thread tasks:", tasks.shape, "→", root / "thread-tasks.csv")
print("orders:", orders.shape, "→", root / "orders.csv")
