"""
拓展2 改进版: LSTM 分钟涨跌三分类预测
- 过去30分钟特征: 收益率、成交量变化、振幅、VWAP偏离
- 双层64维 LSTM
- 三分类: 涨/跌/平
- 严格按时间滚动测试 (walk-forward)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import polars as pl
from pathlib import Path
import torch, torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

ROOT = Path(r"C:\Users\28247\Desktop\nju\兆信")
M_DIR = ROOT / "任务" / "output" / "minute"
OUT = ROOT / "任务" / "output" / "lstm_v2"
OUT.mkdir(exist_ok=True)

# === 1. 加载数据 ===
print("=== 加载分钟数据 ===")
# 读取关键字段: adj_close, adj_high, adj_low, volume, turnover
date_files = sorted((M_DIR / "adj_close").iterdir())
dates_all = sorted([f.stem for f in date_files])
print(f"总日期: {len(dates_all)}")

# 取前 30 只股票, 前 60 天 (足够做滚动测试)
sample_day = pl.read_csv(date_files[0])
all_codes = sorted(sample_day['code'].unique().to_list())[:30]
sample_dates = dates_all[:60]
print(f"股票: {len(all_codes)}, 日期: {len(sample_dates)}")

# 加载所有需要的字段
fields = ['adj_close', 'adj_high', 'adj_low', 'volume', 'turnover']
data_dict = {f: [] for f in fields}

for d in sample_dates:
    date_int = int(d)
    base = None
    for field in fields:
        df = pl.read_csv(M_DIR / field / f"{d}.csv").with_columns(pl.lit(date_int).alias('date'))
        df = df.filter(pl.col('code').is_in(all_codes))
        if base is None:
            base = df
        else:
            base = base.join(df.select(['code', 'minute', field]), on=['code', 'minute'], how='left')
    if not base.is_empty():
        data_dict['_merged'] = data_dict.get('_merged', []) + [base]

df = pl.concat(data_dict['_merged']).sort(['code', 'date', 'minute'])
print(f"原始数据: {df.shape}")
print(f"字段: {df.columns}")

# === 2. 特征工程 ===
print("\n=== 特征工程 ===")
LOOKBACK = 30

# 基础指标
df = df.with_columns([
    # 收益率
    (pl.col('adj_close') / pl.col('adj_close').shift(1) - 1).over('code', 'date').alias('ret'),
    # 对数成交量
    pl.col('volume').log1p().over('code', 'date').alias('log_vol'),
    # 成交量变化率
    (pl.col('volume') / pl.col('volume').shift(1) - 1).over('code', 'date').alias('vol_chg'),
    # 振幅
    ((pl.col('adj_high') - pl.col('adj_low')) / (pl.col('adj_close') + 1e-8)).over('code', 'date').alias('amplitude'),
    # VWAP = turnover / volume
    (pl.col('turnover') / (pl.col('volume') + 1e-8)).over('code', 'date').alias('vwap'),
    # VWAP 偏离
    ((pl.col('adj_close') - pl.col('turnover') / (pl.col('volume') + 1e-8)) / 
     (pl.col('turnover') / (pl.col('volume') + 1e-8) + 1e-8)).over('code', 'date').alias('vwap_dev'),
])

# 滚动特征 (30分钟窗口)
df = df.with_columns([
    pl.col('ret').rolling_mean(LOOKBACK, min_samples=5).over('code', 'date').alias('ret_ma'),
    pl.col('ret').rolling_std(LOOKBACK, min_samples=5).over('code', 'date').alias('ret_std'),
    pl.col('vol_chg').rolling_mean(LOOKBACK, min_samples=5).over('code', 'date').alias('vol_chg_ma'),
    pl.col('amplitude').rolling_mean(LOOKBACK, min_samples=5).over('code', 'date').alias('amp_ma'),
    pl.col('vwap_dev').rolling_mean(LOOKBACK, min_samples=5).over('code', 'date').alias('vwap_dev_ma'),
    pl.col('log_vol').rolling_mean(LOOKBACK, min_samples=5).over('code', 'date').alias('vol_ma'),
])

# 三分类目标: 下一分钟
# 涨: ret > 0.001 (0.1%), 跌: ret < -0.001, 平: 居中
df = df.with_columns(
    pl.col('ret').shift(-1).over('code', 'date').alias('next_ret')
)
df = df.with_columns(
    pl.when(pl.col('next_ret') > 0.0005).then(pl.lit(2))
    .when(pl.col('next_ret') < -0.0005).then(pl.lit(0))
    .otherwise(pl.lit(1))
    .alias('target')
)

# 去除缺失值
df = df.drop_nulls()
print(f"有效样本: {df.shape}")

# 类别分布
class_dist = df['target'].value_counts().sort('target')
print(f"类别分布: 跌={class_dist[0,1]}, 平={class_dist[1,1]}, 涨={class_dist[2,1]}")

# === 3. 特征选择与标准化 ===
BASE_FEATURES = ['ret', 'log_vol', 'vol_chg', 'amplitude', 'vwap_dev']
ROLLING_FEATURES = ['ret_ma', 'ret_std', 'vol_chg_ma', 'amp_ma', 'vwap_dev_ma', 'vol_ma']

# 所有特征的组合作为输入
ALL_FEATURES = BASE_FEATURES + ROLLING_FEATURES
N_FEATURES = len(ALL_FEATURES)

# === 4. 严格按时间滚动测试 (Walk-Forward) ===
print("\n=== 滚动测试 (Walk-Forward) ===")
dates_sorted = sorted(df['date'].unique().to_list())

# 参数: 每 10 天训练一次, 滚动预测后 5 天
TRAIN_WINDOW = 30  # 初始训练窗口天数
TEST_WINDOW = 5    # 每次预测天数
STEP = 5           # 滚动步长

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])

class RollingDataset(Dataset):
    def __init__(self, X, y, seq_len=LOOKBACK):
        self.X = X
        self.y = y
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.X) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]
        y = self.y[idx + self.seq_len - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# 遍历所有滚动窗口
all_results = []
current_train_end = TRAIN_WINDOW

while current_train_end + TEST_WINDOW <= len(dates_sorted):
    train_dates_set = set(dates_sorted[:current_train_end])
    test_dates_set = set(dates_sorted[current_train_end:current_train_end + TEST_WINDOW])
    
    train_df = df.filter(pl.col('date').is_in(train_dates_set))
    test_df = df.filter(pl.col('date').is_in(test_dates_set))
    
    if train_df.is_empty() or test_df.is_empty():
        current_train_end += STEP
        continue
    
    # 标准化 (用训练集统计量)
    stats = {}
    for c in ALL_FEATURES:
        m = train_df[c].mean()
        s = train_df[c].std()
        if s is None or s == 0:
            s = 1.0
        stats[c] = (m, s)
        train_df = train_df.with_columns(((pl.col(c) - m) / s).alias(c))
        test_df = test_df.with_columns(((pl.col(c) - m) / s).alias(c))
    
    X_train = train_df.select(ALL_FEATURES).to_numpy().astype(np.float32)
    y_train = train_df['target'].to_numpy().astype(np.int64)
    X_test = test_df.select(ALL_FEATURES).to_numpy().astype(np.float32)
    y_test = test_df['target'].to_numpy().astype(np.int64)
    
    train_ds = RollingDataset(X_train, y_train)
    test_ds = RollingDataset(X_test, y_test)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=512, shuffle=False)
    
    # 模型训练
    device = 'cpu'
    model = LSTMModel(N_FEATURES, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    for epoch in range(10):
        model.train()
        train_loss = 0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step(train_loss)
    
    # 测试
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in test_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    # 指标
    acc = (preds == targets).mean()
    # 各类精确率
    precisions = []
    recalls = []
    for cls in range(3):
        tp = ((preds == cls) & (targets == cls)).sum()
        fp = ((preds == cls) & (targets != cls)).sum()
        fn = ((preds != cls) & (targets == cls)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        precisions.append(prec)
        recalls.append(rec)
    
    test_dates_list = sorted(test_dates_set)
    all_results.append({
        'train_end': dates_sorted[current_train_end - 1],
        'test_start': test_dates_list[0],
        'test_end': test_dates_list[-1],
        'accuracy': acc,
        'prec_down': precisions[0], 'rec_down': recalls[0],
        'prec_flat': precisions[1], 'rec_flat': recalls[1],
        'prec_up': precisions[2], 'rec_up': recalls[2],
    })
    
    print(f"窗口 {len(all_results)}: 训练至 {dates_sorted[current_train_end-1]}, "
          f"测试 {test_dates_list[0]}-{test_dates_list[-1]}, "
          f"Acc={acc:.4f}, Prec_up={precisions[2]:.3f}, Rec_up={recalls[2]:.3f}")
    
    current_train_end += STEP

# === 5. 汇总结果 ===
print("\n=== 滚动测试汇总 ===")
result_df = pl.DataFrame(all_results)
print(result_df)

acc_mean = result_df['accuracy'].mean()
acc_std = result_df['accuracy'].std()
prec_up_mean = result_df['prec_up'].mean()
rec_up_mean = result_df['rec_up'].mean()

print(f"\n平均准确率: {acc_mean:.4f} ± {acc_std:.4f}")
print(f"平均涨精确率: {prec_up_mean:.4f}, 平均涨召回率: {rec_up_mean:.4f}")

# 基准: 多数类比例
majority = df['target'].value_counts().sort('count', descending=True)
majority_ratio = majority[0, 'count'] / df.shape[0]
print(f"多数类基准: {majority_ratio:.4f}")
print(f"相对提升: {acc_mean - majority_ratio:+.4f}")

# 保存
result_df.write_csv(OUT / "rolling_results.csv")
torch.save({
    'feature_names': ALL_FEATURES,
    'lookback': LOOKBACK,
    'results': all_results,
}, OUT / "model_meta.pt")

print(f"\n结果已保存至: {OUT}")
print("\n=== 模型说明 ===")
print(f"输入特征 ({N_FEATURES}维): {ALL_FEATURES}")
print(f"回顾窗口: {LOOKBACK} 分钟")
print(f"模型: 2层双向LSTM, hidden=64, 输出3分类")
print(f"测试方式: 严格滚动时间窗口 (训练{ TRAIN_WINDOW }天, 每次预测{ TEST_WINDOW }天, 步长{ STEP }天)")
print(f"窗口总数: {len(all_results)}")