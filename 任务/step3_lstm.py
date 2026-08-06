"""
拓展2: LSTM 分钟涨跌预测
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import polars as pl
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

ROOT = Path(r"C:\Users\28247\Desktop\nju\兆信")
OUT_DIR = ROOT / "任务" / "output"
LSTM_DIR = OUT_DIR / "lstm"
LSTM_DIR.mkdir(exist_ok=True)

# === 1. 读取分钟数据并提取特征 ===
print("=== 读取分钟数据 ===")
# 读取 adj_close 和 volume
minute_dir = OUT_DIR / "minute"
adj_close_files = sorted((minute_dir / "adj_close").iterdir())
volume_files = sorted((minute_dir / "volume").iterdir())

all_data = []
# 只取前50只股票和部分日期以减少计算量
sample_codes = sorted(set(
    int(pl.read_csv(f)['code'][0]) 
    for f in sorted((minute_dir / "adj_close").iterdir())[:1]
    for _ in range(1)
))
# 取第一天的所有股票代码，取前50个
first_day = sorted((minute_dir / "adj_close").iterdir())[0]
sample_codes = sorted(pl.read_csv(first_day)['code'].unique().to_list())[:50]
sample_dates = [f.stem for f in sorted((minute_dir / "adj_close").iterdir())[:50]]

print(f"采样股票数: {len(sample_codes)}, 采样日期: {len(sample_dates)}")

for acf, vf in zip(adj_close_files, volume_files):
    date_str = acf.stem
    if date_str not in sample_dates:
        continue
    ac = pl.read_csv(acf).with_columns(pl.lit(date_str).cast(pl.Int64).alias('date'))
    ac = ac.filter(pl.col('code').is_in(sample_codes))
    if ac.is_empty():
        continue
    vo = pl.read_csv(vf).with_columns(pl.lit(date_str).cast(pl.Int64).alias('date'))
    vo = vo.filter(pl.col('code').is_in(sample_codes))
    merged = ac.join(vo, on=['code', 'minute', 'date'])
    all_data.append(merged)

minute = pl.concat(all_data).sort(['code', 'date', 'minute'])
print(f"分钟数据: {minute.shape}")

# === 2. 特征工程 ===
print("\n=== 特征工程 ===")
# 每只股票按分钟排序
minute = minute.sort(['code', 'minute'])

# 计算分钟收益率
minute = minute.with_columns(
    (pl.col('adj_close') / pl.col('adj_close').shift(1) - 1).over('code').alias('ret'),
    pl.col('volume').log1p().over('code').alias('log_volume'),
)

# 目标：下一分钟涨跌 (1 涨, 0 跌)
minute = minute.with_columns(
    (pl.col('adj_close').shift(-1) > pl.col('adj_close')).over('code').cast(pl.Int32).alias('target')
)

# 滚动特征
minute = minute.with_columns([
    pl.col('ret').rolling_mean(5, min_samples=2).over('code').alias('ret_ma5'),
    pl.col('ret').rolling_std(5, min_samples=2).over('code').alias('ret_std5'),
    pl.col('volume').rolling_mean(5, min_samples=2).over('code').alias('vol_ma5'),
    pl.col('ret').shift(1).over('code').alias('ret_lag1'),
    pl.col('ret').shift(2).over('code').alias('ret_lag2'),
    pl.col('ret').shift(3).over('code').alias('ret_lag3'),
])

# 去掉早期没有完整特征的样本
minute = minute.drop_nulls(subset=['ret_ma5', 'ret_std5', 'vol_ma5', 'ret_lag1', 'ret_lag2', 'ret_lag3', 'target'])
print(f"有效样本: {minute.shape}")

# === 3. 准备训练数据 ===
feature_cols = ['ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_ma5', 'ret_std5', 'log_volume', 'vol_ma5']

# 按日期划分训练/测试集 (前80%训练, 后20%测试)
dates = sorted(minute['date'].unique().to_list())
split_idx = int(len(dates) * 0.8)
train_dates = set(dates[:split_idx])
test_dates = set(dates[split_idx:])

train_df = minute.filter(pl.col('date').is_in(train_dates))
test_df = minute.filter(pl.col('date').is_in(test_dates))

print(f"训练日期: {len(train_dates)} 天, 测试日期: {len(test_dates)} 天")
print(f"训练样本: {train_df.shape[0]}, 测试样本: {test_df.shape[0]}")

# 标准化
train_mean = train_df.select([pl.col(c).mean() for c in feature_cols]).to_dict(as_series=False)
train_std = train_df.select([pl.col(c).std() for c in feature_cols]).to_dict(as_series=False)

for c in feature_cols:
    train_df = train_df.with_columns(
        ((pl.col(c) - train_mean[c][0]) / (train_std[c][0] + 1e-8)).alias(c)
    )
    test_df = test_df.with_columns(
        ((pl.col(c) - train_mean[c][0]) / (train_std[c][0] + 1e-8)).alias(c)
    )

# 采样 (数据量太大，随机采样)
sample_size = 200000
if train_df.shape[0] > sample_size:
    train_df = train_df.sample(n=sample_size, seed=42)
if test_df.shape[0] > sample_size // 4:
    test_df = test_df.sample(n=sample_size // 4, seed=42)

X_train = train_df.select(feature_cols).to_numpy().astype(np.float32)
y_train = train_df['target'].to_numpy().astype(np.float32)
X_test = test_df.select(feature_cols).to_numpy().astype(np.float32)
y_test = test_df['target'].to_numpy().astype(np.float32)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"正样本比例 - 训练: {y_train.mean():.3f}, 测试: {y_test.mean():.3f}")

# === 4. LSTM 模型 ===
class MinuteLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        return self.fc(out).squeeze()


class MinuteDataset(Dataset):
    def __init__(self, X, y, seq_len=10):
        self.X = X
        self.y = y
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.X) - self.seq_len)
    
    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]
        y = self.y[idx + self.seq_len - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# === 5. 训练 ===
print("\n=== 训练 LSTM 模型 ===")
SEQ_LEN = 10
BATCH_SIZE = 256
EPOCHS = 20
LR = 0.001

train_dataset = MinuteDataset(X_train, y_train, SEQ_LEN)
test_dataset = MinuteDataset(X_test, y_test, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

model = MinuteLSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.3).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        predicted = (pred > 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    train_acc = correct / total if total > 0 else 0
    
    # 验证
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item()
            predicted = (pred > 0.5).float()
            val_correct += (predicted == y_batch).sum().item()
            val_total += y_batch.size(0)
    
    val_acc = val_correct / val_total if val_total > 0 else 0
    scheduler.step(val_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# === 6. 最终评估 ===
print("\n=== 最终评估 ===")
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
binary_preds = (all_preds > 0.5).astype(int)

accuracy = (binary_preds == all_targets).mean()
precision = (binary_preds * all_targets).sum() / (binary_preds.sum() + 1e-8)
recall = (binary_preds * all_targets).sum() / (all_targets.sum() + 1e-8)

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 Score: {2 * precision * recall / (precision + recall + 1e-8):.4f}")

# 保存模型
torch.save(model.state_dict(), LSTM_DIR / "lstm_model.pt")
print(f"\n模型已保存至: {LSTM_DIR / 'lstm_model.pt'}")

# 保存预测结果
result_df = pl.DataFrame({
    'pred_prob': all_preds.tolist(),
    'pred_label': binary_preds.tolist(),
    'target': all_targets.tolist(),
})
result_df.write_csv(LSTM_DIR / "predictions.csv")
print(f"预测结果已保存至: {LSTM_DIR / 'predictions.csv'}")

# 基准：始终预测涨的准确率
baseline = all_targets.mean()
print(f"\n基准准确率 (始终预测涨): {baseline:.4f}")
print(f"模型相对提升: {accuracy - baseline:+.4f}")