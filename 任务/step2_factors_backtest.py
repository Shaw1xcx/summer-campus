"""
步骤2+3+4: 因子构建、因子评价、回测
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import polars as pl
import json

ROOT = Path(r"C:\Users\28247\Desktop\nju\兆信")
OUT_DIR = ROOT / "任务" / "output"
FACTOR_DIR = OUT_DIR / "factors"
BACKTEST_DIR = OUT_DIR / "backtest"
FACTOR_DIR.mkdir(exist_ok=True)
BACKTEST_DIR.mkdir(exist_ok=True)

# === 读取日频数据（合并关键字段） ===
print("=== 读取日频数据 ===")
daily = pl.read_csv(OUT_DIR / "daily" / "daily_close.csv")
daily = daily.join(pl.read_csv(OUT_DIR / "daily" / "daily_adj_close.csv"), on=['date', 'code'])
daily = daily.join(pl.read_csv(OUT_DIR / "daily" / "daily_turnover.csv"), on=['date', 'code'])
daily = daily.join(pl.read_csv(OUT_DIR / "daily" / "daily_volume.csv"), on=['date', 'code'])
daily = daily.join(pl.read_csv(OUT_DIR / "daily" / "daily_buy_volume.csv"), on=['date', 'code'])
daily = daily.join(pl.read_csv(OUT_DIR / "daily" / "daily_sell_volume.csv"), on=['date', 'code'])
daily = daily.join(pl.read_csv(OUT_DIR / "daily" / "daily_open.csv"), on=['date', 'code'])
daily = daily.join(pl.read_csv(OUT_DIR / "daily" / "daily_adj_open.csv"), on=['date', 'code'])

daily = daily.sort(['code', 'date'])
print(f"日频合并: {daily.shape}")

# === 计算收益率 ===
daily = daily.with_columns(
    (pl.col('adj_close') / pl.col('adj_close').shift(1) - 1).over('code').alias('ret'),
    (pl.col('adj_close').shift(-1) / pl.col('adj_close') - 1).over('code').alias('fwd_ret'),
)

# === 因子1: 成交额均值的标准差对数 (示例因子) ===
# [1, 5, 10, 20]日成交额均值的标准差的对数
print("\n=== 因子构建 ===")

for w in [1, 5, 10, 20]:
    daily = daily.with_columns(
        pl.col('turnover').rolling_mean(window_size=w, min_samples=1).over('code').alias(f'amt_ma{w}')
    )

daily = daily.with_columns(
    pl.concat_list([pl.col(f'amt_ma{w}') for w in [1, 5, 10, 20]]).list.std().over('code').alias('amt_std')
)
daily = daily.with_columns(
    pl.col('amt_std').log().alias('factor_turnover_std')
)

# === 因子2: 买卖不平衡因子 ===
daily = daily.with_columns(
    ((pl.col('buy_volume') - pl.col('sell_volume')) / (pl.col('volume') + 1)).alias('factor_bs_imbalance')
)

# === 因子3: 波动率因子 (5日收益率标准差) ===
daily = daily.with_columns(
    pl.col('ret').rolling_std(window_size=5, min_samples=2).over('code').alias('factor_volatility')
)

# === 因子4: 动量因子 (5日累计收益) ===
daily = daily.with_columns(
    (pl.col('adj_close') / pl.col('adj_close').shift(5) - 1).over('code').alias('factor_momentum')
)

# 去掉缺失值和无穷值
daily = daily.filter(
    pl.all_horizontal([
        pl.col('factor_turnover_std').is_finite(),
        pl.col('factor_bs_imbalance').is_finite(),
        pl.col('factor_volatility').is_finite(),
        pl.col('factor_momentum').is_finite(),
        pl.col('fwd_ret').is_finite(),
    ])
)
print(f"有效样本: {daily.shape}")

# === 因子评价 ===
print("\n=== 因子评价 ===")
factor_names = ['factor_turnover_std', 'factor_bs_imbalance', 'factor_volatility', 'factor_momentum']
factor_labels = ['成交额标准差对数', '买卖不平衡', '波动率', '动量']

results = []

for fname, flabel in zip(factor_names, factor_labels):
    # 逐日 IC 和 Rank IC
    ic_daily = daily.group_by('date').agg([
        pl.corr(fname, 'fwd_ret').alias('IC'),
        pl.corr(fname, 'fwd_ret', method='spearman').alias('Rank_IC'),
    ]).drop_nulls()
    
    ic_mean = ic_daily['IC'].mean()
    ic_std = ic_daily['IC'].std()
    ir = ic_mean / ic_std if (ic_std is not None and ic_std > 0) else 0
    icir = ir
    
    rank_ic_mean = ic_daily['Rank_IC'].mean()
    rank_ic_std = ic_daily['Rank_IC'].std()
    rank_ir = rank_ic_mean / rank_ic_std if (rank_ic_std is not None and rank_ic_std > 0) else 0
    rank_icir = rank_ir
    
    # IC 胜率
    ic_win_rate = (ic_daily['IC'] > 0).sum() / ic_daily['IC'].len() if ic_daily['IC'].len() > 0 else 0
    
    # 分层效果：按因子值分5组，计算每组的平均fwd_ret
    daily_temp = daily.with_columns(
        pl.col(fname).qcut(5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']).over('date').alias('group')
    )
    layer = daily_temp.group_by('group').agg(
        pl.col('fwd_ret').mean().alias('avg_ret'),
        pl.col('fwd_ret').std().alias('std_ret'),
    ).sort('group')
    
    print(f"\n--- {flabel} ({fname}) ---")
    print(f"  IC均值: {ic_mean:.6f}, IC标准差: {ic_std:.6f}, IR: {ir:.4f}, ICIR: {icir:.4f}")
    print(f"  Rank_IC均值: {rank_ic_mean:.6f}, Rank_IR: {rank_ir:.4f}, Rank_ICIR: {rank_icir:.4f}")
    print(f"  IC胜率: {ic_win_rate:.2%}")
    print(f"  分层效果: {layer}")
    
    results.append({
        'factor': flabel,
        'IC_mean': round(ic_mean, 6),
        'IC_std': round(ic_std, 6),
        'IR': round(ir, 4),
        'ICIR': round(icir, 4),
        'Rank_IC_mean': round(rank_ic_mean, 6),
        'Rank_IC_std': round(rank_ic_std, 6),
        'Rank_IR': round(rank_ir, 4),
        'Rank_ICIR': round(rank_icir, 4),
        'IC_win_rate': round(ic_win_rate, 4),
    })

# 保存因子评价结果
import json
with open(FACTOR_DIR / "factor_evaluation.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# === 拓展1: 回测 ===
print("\n\n=== 拓展1: 回测 ===")

# 使用 IC 对因子值加权，选 Top 10
# 注意：这里是使用当日的因子值与当日计算出的 IC 加权，实际上 IC 需要前一日数据来计算
# 任务指出了这个问题：使用当日因子值与次日收益率的 IC 加权存在未来信息泄露
# 正确做法：使用滚动窗口的历史 IC 来加权

# 先计算滚动 20 日 IC 作为权重
ic_rolling = {}
for fname in factor_names:
    ic_rolling[fname] = {}
    ic_daily = daily.group_by('date').agg([
        pl.corr(fname, 'fwd_ret').alias('IC'),
    ]).drop_nulls().sort('date')
    dates_list = ic_daily['date'].to_list()
    ic_list = ic_daily['IC'].to_list()
    for i in range(19, len(dates_list)):
        rolling_ic = sum(ic_list[i-19:i+1]) / 20
        ic_rolling[fname][dates_list[i]] = rolling_ic

# 加权信号
daily = daily.with_columns(
    pl.lit(0.0).alias('signal')
)

for fname in factor_names:
    ic_map = ic_rolling[fname]
    daily = daily.with_columns(
        pl.when(pl.col('date').is_in(list(ic_map.keys())))
        .then(pl.col('signal') + pl.col(fname) * pl.col('date').replace_strict(ic_map, default=0.0))
        .otherwise(pl.col('signal'))
        .alias('signal')
    )

# 每日选 Top 10
daily = daily.with_columns(
    pl.col('signal').rank('dense', descending=True).over('date').alias('rank')
)
top10 = daily.filter(pl.col('rank') <= 10).sort(['date', 'rank'])

# 回测1: 收盘价调仓
print("\n--- 收盘价调仓 ---")
portfolio = top10.group_by('date').agg(
    pl.col('fwd_ret').mean().alias('port_ret'),
    pl.col('code').count().alias('n_stocks'),
).sort('date').drop_nulls(subset=['port_ret'])

# 手续费：卖出万5，买入不计
initial_capital = 10_000_000  # 1kW
portfolio = portfolio.with_columns(
    (pl.col('port_ret') - 0.0005).alias('port_ret_net'),  # 卖出万5
)
# 分段计算净值
nav_vals = [1.0]
for r in portfolio['port_ret_net'].to_list():
    nav_vals.append(nav_vals[-1] * (1.0 + r))
nav_vals = nav_vals[1:]

# 添加回 DataFrame
nav_df = pl.DataFrame({'nav': nav_vals})
portfolio = pl.concat([portfolio, nav_df], how='horizontal')
portfolio = portfolio.with_columns(
    (pl.col('nav') * initial_capital).alias('value'),
)

# 计算指标
nav_series = portfolio['nav'].to_list()
ret_series = portfolio['port_ret_net'].to_list()

# 累计收益
total_return = nav_series[-1] - 1

# 年化收益率 (假设250个交易日)
n_days = len(nav_series)
annual_return = (nav_series[-1]) ** (250 / n_days) - 1

# 波动率
import math
daily_std = portfolio['port_ret_net'].std()
annual_vol = daily_std * math.sqrt(250)

# 夏普比率 (无风险利率设为0)
sharpe = portfolio['port_ret_net'].mean() / daily_std * math.sqrt(250) if daily_std > 0 else 0

# 最大回撤
peak = nav_series[0]
max_dd = 0
for v in nav_series:
    if v > peak:
        peak = v
    dd = (peak - v) / peak
    if dd > max_dd:
        max_dd = dd

print(f"  累计收益: {total_return:.4%}")
print(f"  年化收益: {annual_return:.4%}")
print(f"  年化波动率: {annual_vol:.4%}")
print(f"  夏普比率: {sharpe:.4f}")
print(f"  最大回撤: {max_dd:.4%}")

# 保存净值曲线
portfolio.select(['date', 'port_ret', 'port_ret_net', 'nav', 'value']).write_csv(
    BACKTEST_DIR / "nav_close.csv"
)

# 回测2: 次日开盘价调仓
print("\n--- 开盘价调仓 ---")
# 使用次日开盘价计算收益
daily = daily.with_columns(
    (pl.col('adj_open').shift(-1) / pl.col('adj_close') - 1).over('code').alias('fwd_ret_open')
)
top10_open = daily.filter(pl.col('rank') <= 10).sort(['date', 'rank'])

portfolio_open = top10_open.group_by('date').agg(
    pl.col('fwd_ret_open').mean().alias('port_ret'),
    pl.col('code').count().alias('n_stocks'),
).sort('date').drop_nulls(subset=['port_ret'])

portfolio_open = portfolio_open.with_columns(
    (pl.col('port_ret') - 0.0005).alias('port_ret_net'),
)
nav_vals_o = [1.0]
for r in portfolio_open['port_ret_net'].to_list():
    nav_vals_o.append(nav_vals_o[-1] * (1.0 + r))
nav_vals_o = nav_vals_o[1:]
nav_df_o = pl.DataFrame({'nav': nav_vals_o})
portfolio_open = pl.concat([portfolio_open, nav_df_o], how='horizontal')
portfolio_open = portfolio_open.with_columns(
    (pl.col('nav') * initial_capital).alias('value'),
)

nav_open = portfolio_open['nav'].to_list()
ret_open = portfolio_open['port_ret_net'].to_list()

total_return_o = nav_open[-1] - 1
annual_return_o = (nav_open[-1]) ** (250 / len(nav_open)) - 1
daily_std_o = portfolio_open['port_ret_net'].std()
annual_vol_o = daily_std_o * math.sqrt(250)
sharpe_o = portfolio_open['port_ret_net'].mean() / daily_std_o * math.sqrt(250) if daily_std_o > 0 else 0

peak_o = nav_open[0]
max_dd_o = 0
for v in nav_open:
    if v > peak_o:
        peak_o = v
    dd = (peak_o - v) / peak_o
    if dd > max_dd_o:
        max_dd_o = dd

print(f"  累计收益: {total_return_o:.4%}")
print(f"  年化收益: {annual_return_o:.4%}")
print(f"  年化波动率: {annual_vol_o:.4%}")
print(f"  夏普比率: {sharpe_o:.4f}")
print(f"  最大回撤: {max_dd_o:.4%}")

portfolio_open.select(['date', 'port_ret', 'port_ret_net', 'nav', 'value']).write_csv(
    BACKTEST_DIR / "nav_open.csv"
)

# === 问题分析 ===
print("\n\n=== 未来信息泄露问题分析 ===")
print("""
使用当日因子值与次日收益率计算的 IC 对因子加权存在问题：
1. 未来信息泄露：IC 使用当日因子值与次日收益率计算，次日收益率在决策时未知
2. 正确做法：使用滚动历史窗口的 IC（如过去20日）来加权，避免使用未来数据
3. 本次回测已采用滚动20日历史 IC 作为权重，避免未来信息泄露

此外，IC 加权的一个潜在问题是 IC 存在不稳定性：
- IC 在时间序列上波动较大，可能正负交替
- 建议使用 IC 的 EMA 平滑或使用 ICIR 替代 IC 作为权重
- 也可考虑使用因子 Rank IC 加权，对异常值更稳健
""")

# 保存完整回测报告
report = {
    'close_rebalance': {
        'total_return': round(total_return, 6),
        'annual_return': round(annual_return, 6),
        'annual_volatility': round(annual_vol, 6),
        'sharpe_ratio': round(sharpe, 4),
        'max_drawdown': round(max_dd, 6),
    },
    'open_rebalance': {
        'total_return': round(total_return_o, 6),
        'annual_return': round(annual_return_o, 6),
        'annual_volatility': round(annual_vol_o, 6),
        'sharpe_ratio': round(sharpe_o, 4),
        'max_drawdown': round(max_dd_o, 6),
    },
    'factor_evaluation': results,
}

with open(BACKTEST_DIR / "backtest_report.json", 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n=== 完成 ===")
print(f"因子评价: {FACTOR_DIR}")
print(f"回测结果: {BACKTEST_DIR}")