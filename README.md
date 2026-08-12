# 数据处理任务

本仓库包含两个部分：
1. **课堂讲义**：`数据处理技术补充.html` 及配套的 15 个演示代码
2. **量化任务**：基于 300 只 A 股逐笔成交数据的降采样、因子挖掘、回测与 LSTM 预测

---

## 一、文件结构

### 根目录

| 文件 | 说明 |
|------|------|
| `数据处理技术补充.html` | 课堂讲义，8 个主题（编码压缩、Parquet、Polars、DuckDB、缓存、Zarr、Free-threaded Python、端到端示例） |
| `README.md` | 本文件 |
| `AGENTS.md` | AI 助手的约束规则 |

### `code/` — 课堂演示代码

| 路径 | 说明 |
|------|------|
| `run_all_demos.py` | 从 HTML 讲义中提取 15 段 Python 代码并依次执行 |
| `demos/01_prepare_data.py` | 生成虚构数据（交易表、因子表、线程任务表、订单表） |
| `demos/02_csv_vs_parquet.py` | 对比 CSV 与 Parquet 的存储大小和类型信息 |
| `demos/03_compression_benchmark.py` | 测试不同压缩算法（ZSTD/Snappy/LZ4）的效果 |
| `demos/04_partition_by_date.py` | 按日期分区写入 Parquet |
| `demos/05_layout_candidates.py` | 对比不同分区方案对查询性能的影响 |
| `demos/06_polars_query_plan.py` | Polars Lazy API 查询计划分析 |
| `demos/07_polars_eager_vs_lazy.py` | 对比 Eager 和 Lazy 执行模式的性能 |
| `demos/08_duckdb_top3.py` | DuckDB SQL 查询与 Top N 优化 |
| `demos/09_duckdb_explain_cases.py` | DuckDB 执行计划分析 |
| `demos/10_cache_basic.py` | 基础缓存设计与 LRU 策略 |
| `demos/11_cache_with_version.py` | 带版本控制的缓存失效策略 |
| `demos/12_zarr_chunks.py` | Zarr 3 多维数组的分块与 Shard 存储 |
| `demos/13_threads_cpu_vs_io.py` | 线程在 CPU 密集与 I/O 密集任务中的表现 |
| `demos/14_thread_lock.py` | GIL 与线程锁竞态条件演示 |
| `demos/15_end_to_end.py` | 端到端：从 CSV 到分区、查询、缓存与 Zarr |
| `demos/data/` | 演示代码使用的虚构数据副本 |
| `results/` | 15 个演示脚本的实际运行输出 + `run_summary.json` |

### `data/` — 课堂演示数据

| 文件 | 说明 |
|------|------|
| `trades.csv` / `trades.parquet` | 虚构交易表，12,000 行 |
| `trades_zstd.parquet` | ZSTD 压缩的 Parquet 副本 |
| `factor-values.csv` | 虚构因子表，4 天 × 6 只股票 |
| `thread-tasks.csv` | 线程任务表（4 CPU + 4 I/O） |
| `orders.csv` | 竞态条件订单表 |
| `kline_minute.csv` | 分钟 K 线（含复权价格） |
| `backtest_nav.csv` | 回测净值曲线 |
| `end-to-end/` | 端到端演示产物：分区 Parquet、Zarr 3 |
| `format-benchmark/` | 格式对比：CSV/CSV.gz/LZ4/Snappy/ZSTD |
| `lake/` | 分区数据湖 `year=2026/month=06/date=*/` |
| `factors.zarr/` | Zarr 3 多维数组（因子数据） |
| `生成的data/` | 数据准备脚本的副本输出 |

### `py/` — PyCharm 项目

| 文件 | 说明 |
|------|------|
| `main.py` | PyCharm 默认模板 |
| `1.py` | 读取 Parquet 文件并打印基本信息 |

### `任务/` — 量化任务（核心）

#### 输入数据

| 文件 | 说明 |
|------|------|
| `readme.pdf` / `readme (1).pdf` | 任务说明文档（含补丁说明） |
| `adjfactor.pkl` | 复权因子，302 天 × 300 只股票 |
| `data/TRADE/` | 原始逐笔成交数据（302 天 × 300 只股票，未上传到 GitHub） |

#### 处理脚本

| 文件 | 说明 |
|------|------|
| `step1_downsample.py` | **步骤 1**：逐笔 → 日频 + 分钟频降采样，应用复权因子，按指定格式存储 |
| `step2_factors_backtest.py` | **步骤 2-4**：因子构建、因子评价、IC 加权 Top 10 回测 |
| `step3a_prepare_features.py` | **步骤 5a**：分钟数据特征工程（30 分钟窗口，11 维特征） |
| `step3b_train_lstm.py` | **步骤 5b**：LSTM 三分类滚动时间测试 |
| `test_load.py` | 数据加载测试脚本 |

#### 输出结果

| 文件 | 说明 |
|------|------|
| `output/daily/` | 日频数据（15 个字段，每个字段一张表） |
| `output/minute/` | 分钟频数据（15 个字段，每个字段一个文件夹，每日一张表） |
| `output/factors/factor_evaluation.json` | 因子评价结果（IC/IR/Rank_IC/分层效果） |
| `output/backtest/` | 回测净值曲线（收盘价/开盘价调仓）和回测报告 |
| `output/lstm/` | LSTM v1 模型和预测结果 |
| `output/lstm_v2/` | LSTM v2 改进版滚动测试结果 |

---

## 二、任务执行流程

### 环境准备

```bash
pip install polars pandas pyarrow numpy torch
```

### 步骤 1：数据降采样

```bash
python 任务/step1_downsample.py
```

从 302 天 × 300 只股票的逐笔成交数据生成：
- **日频**：90,029 条，每字段一张 CSV
- **分钟频**：20,982,472 条，每字段一个文件夹，每日一张 CSV

字段包括：OHLCV、复权 OHLC、成交额、成交笔数、主买/主卖量、主买/主卖额。

### 步骤 2-4：因子构建、评价与回测

```bash
python 任务/step2_factors_backtest.py
```

构建 4 个因子，计算 IC/IR/Rank_IC，IC 加权选 Top 10 回测。

### 步骤 5：LSTM 分钟涨跌预测

```bash
python 任务/step3a_prepare_features.py  # 特征工程
python 任务/step3b_train_lstm.py        # 滚动训练
```

---

## 三、关键结果

### 因子评价

| 因子 | IC 均值 | Rank_IC | IR | 分层效果 |
|------|---------|---------|-----|----------|
| 成交额标准差对数 | -0.078 | -0.108 | -0.56 | Q1→Q5 单调递减 |
| 买卖不平衡 | -0.040 | -0.019 | -0.40 | 分层较弱 |
| 波动率 | -0.058 | -0.076 | -0.41 | Q1→Q5 递减 |
| 动量 | -0.027 | -0.056 | -0.16 | Q1→Q5 递减 |

### 回测（初始资金 1 千万，Top 10 等权）

| 调仓方式 | 累计收益 | 夏普比率 | 最大回撤 |
|----------|----------|----------|----------|
| 收盘价调仓 | +142.1% | 3.42 | 10.1% |
| 开盘价调仓 | +12.7% | 1.17 | 10.3% |

### LSTM 分钟涨跌预测

- 双层 64 维双向 LSTM，输入 11 维特征（30 分钟窗口）
- 严格按时间滚动测试（walk-forward）
- 平均准确率 50.07%，vs 多数类基准 44.6%（+5.4%）

---

## 四、依赖

```
polars>=1.21
pandas
pyarrow
numpy
torch
```