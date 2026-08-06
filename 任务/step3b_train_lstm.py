"""LSTM 三分类 - 快速版"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import polars as pl
from pathlib import Path
import torch, torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

ROOT = Path(r"C:\Users\28247\Desktop\nju\兆信")
OUT = ROOT / "任务" / "output" / "lstm_v2"

df = pl.read_parquet(OUT / "features.parquet")
FEATURES = ["ret","log_vol","vol_chg","amplitude","vwap_dev",
            "ret_ma","ret_std","vol_chg_ma","amp_ma","vwap_dev_ma","vol_ma"]
N_FEAT = len(FEATURES)
SEQ_LEN = 30

# 采样: 只用前 10 只股票的数据
codes_sub = sorted(df["code"].unique().to_list())[:10]
df = df.filter(pl.col("code").is_in(codes_sub))
print(f"Subsampled: {df.shape}")

class LSTMModel(nn.Module):
    def __init__(self, d_in, h=64, n_layers=2, n_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(d_in, h, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.norm = nn.LayerNorm(h * 2)
        self.fc = nn.Sequential(nn.Linear(h * 2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, n_classes))
    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(self.norm(o[:, -1, :]))

class SeqDataset(Dataset):
    def __init__(self, X, y, seq=SEQ_LEN):
        self.X, self.y, self.seq = X, y, seq
    def __len__(self): return max(0, len(self.X) - self.seq)
    def __getitem__(self, i):
        return torch.tensor(self.X[i:i+self.seq], dtype=torch.float32), torch.tensor(self.y[i+self.seq-1], dtype=torch.long)

dates = sorted(df["date"].unique().to_list())
TRAIN_WIN, TEST_WIN, STEP = 20, 5, 5
results = []
current = TRAIN_WIN
max_windows = 2

print(f"Dates: {len(dates)}, TrainWin={TRAIN_WIN}, TestWin={TEST_WIN}")
print(f"Starting rolling test...")

while current + TEST_WIN <= len(dates) and len(results) < max_windows:
    t_start = time.time()
    train_dates = set(dates[:current])
    test_dates = set(dates[current:current + TEST_WIN])
    
    train_df = df.filter(pl.col("date").is_in(train_dates))
    test_df = df.filter(pl.col("date").is_in(test_dates))
    print(f"  Window {len(results)+1}: train={len(train_dates)}d/{train_df.shape[0]}r, test={len(test_dates)}d/{test_df.shape[0]}r")
    
    # 标准化
    for c in FEATURES:
        m = train_df[c].mean(); s = train_df[c].std() or 1.0
        train_df = train_df.with_columns(((pl.col(c) - m) / s).alias(c))
        test_df = test_df.with_columns(((pl.col(c) - m) / s).alias(c))
    
    X_tr = train_df.select(FEATURES).to_numpy().astype(np.float32)
    y_tr = train_df["target"].to_numpy().astype(np.int64)
    X_te = test_df.select(FEATURES).to_numpy().astype(np.float32)
    y_te = test_df["target"].to_numpy().astype(np.int64)
    
    tr_ds = SeqDataset(X_tr, y_tr); te_ds = SeqDataset(X_te, y_te)
    tr_dl = DataLoader(tr_ds, 128, True); te_dl = DataLoader(te_ds, 256)
    
    model = LSTMModel(N_FEAT, 64, 2, 3, 0.3)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    for ep in range(5):
        model.train()
        for Xb, yb in tr_dl:
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in te_dl:
            preds.extend(torch.argmax(model(Xb), 1).numpy())
            targets.extend(yb.numpy())
    preds = np.array(preds); targets = np.array(targets)
    
    acc = (preds == targets).mean()
    precs, recs = [], []
    for cls in range(3):
        tp = ((preds == cls) & (targets == cls)).sum()
        fp = ((preds == cls) & (targets != cls)).sum()
        fn = ((preds != cls) & (targets == cls)).sum()
        precs.append(tp / (tp + fp + 1e-8))
        recs.append(tp / (tp + fn + 1e-8))
    
    f1_macro = np.mean([2*p*r/(p+r+1e-8) for p,r in zip(precs, recs)])
    
    results.append({
        "window": len(results) + 1,
        "train_end": dates[current - 1],
        "test_start": dates[current], "test_end": dates[current + TEST_WIN - 1],
        "acc": acc, "prec_up": precs[2], "rec_up": recs[2],
        "prec_down": precs[0], "rec_down": recs[0],
        "prec_flat": precs[1], "rec_flat": recs[1],
        "macro_f1": f1_macro,
    })
    
    print(f"  -> Acc={acc:.4f} | Prec_up={precs[2]:.3f} Rec_up={recs[2]:.3f} | "
          f"Prec_flat={precs[1]:.3f} Rec_flat={recs[1]:.3f} | "
          f"Down: P={precs[0]:.3f} R={recs[0]:.3f} | F1={f1_macro:.4f} | "
          f"Time={time.time()-t_start:.1f}s")
    
    current += STEP

res_df = pl.DataFrame(results)
print(f"\n=== 汇总 ===")
print(res_df)
print(f"\n平均 Acc: {res_df['acc'].mean():.4f} +/- {res_df['acc'].std():.4f}")
print(f"平均 F1:  {res_df['macro_f1'].mean():.4f} +/- {res_df['macro_f1'].std():.4f}")
print(f"平均 Prec_up: {res_df['prec_up'].mean():.4f}, Rec_up: {res_df['rec_up'].mean():.4f}")

majority = df["target"].value_counts().sort("target")
baseline = majority[1, "count"] / df.shape[0]
print(f"多数类基准(flat): {baseline:.4f}")

res_df.write_csv(OUT / "rolling_results.csv")
print(f"Saved to {OUT / 'rolling_results.csv'}")