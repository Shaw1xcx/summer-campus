from itertools import product
from pathlib import Path
import shutil
import numpy as np
import polars as pl
import zarr

# 源数据是 prepare_demo.py 生成的 24 行自编 CSV。
source = (
    pl.read_csv("data/factor-values.csv")
    .sort(["date", "symbol"])
)
dates = source["date"].unique(maintain_order=True).to_list()
symbols = source["symbol"].unique(maintain_order=True).to_list()
features = ["momentum", "volatility", "quality"]

# CSV 每一行有 3 个因子；reshape 后三个轴依次是 date、symbol、feature。
values = (
    source.select(features).to_numpy()
    .astype("float32")
    .reshape(len(dates), len(symbols), len(features))
)

store = Path("data/factors.zarr")
if store.exists():
    shutil.rmtree(store)

chunks = (2, 3, 3)   # 每块：2 天 × 3 股票 × 3 因子 = 18 个值
shards = (4, 6, 3)   # 一个 Shard 装下 2 × 2 × 1 = 4 个 Chunk
cube = zarr.create_array(
    store=str(store), shape=values.shape,
    chunks=chunks, shards=shards,
    dtype="float32", zarr_format=3,
)
cube[:] = values
reopened = zarr.open_array(str(store), mode="r")

# 根据切片的起止坐标，列出每个轴上相交的 Chunk 编号。
def touched_chunk_ids(slices, chunk_shape):
    ranges = []
    for part, chunk_size in zip(slices, chunk_shape):
        first = part.start // chunk_size
        last = (part.stop - 1) // chunk_size
        ranges.append(range(first, last + 1))
    return list(product(*ranges))

exact_slice = (slice(0, 2), slice(0, 3), slice(0, 3))
wide_slice = (slice(1, 4), slice(2, 5), slice(0, 3))
exact = reopened[exact_slice]
wide = reopened[wide_slice]
exact_ids = touched_chunk_ids(exact_slice, chunks)
wide_ids = touched_chunk_ids(wide_slice, chunks)

print("source rows:", source.height)
print("shape / chunks / shards:", reopened.shape, reopened.chunks, reopened.shards)
print("cube[0,1,0] =", reopened[0, 1, 0], "→ 20260627 / S002 / momentum")
print("exact momentum:\n", exact[:, :, 0])
print("exact chunk ids:", exact_ids, "requested:", exact.size,
      "covered:", len(exact_ids) * np.prod(chunks))
print("wide chunk ids :", wide_ids, "requested:", wide.size,
      "covered:", len(wide_ids) * np.prod(chunks))
print("wide read amplification:",
      round(len(wide_ids) * np.prod(chunks) / wide.size, 2), "x")

# 文件重新打开后必须逐值一致，证明数据确实已持久化。
assert np.allclose(reopened[:], values)
