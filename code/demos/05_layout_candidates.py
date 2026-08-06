import polars as pl

# 读取本讲义生成的虚构数据。统一把 date 转成字符串，便于和查询条件比较。
data = pl.read_parquet("data/trades.parquet").with_columns(
    pl.col("date").cast(pl.String)
)

# 先从真实数据中统计：每个日期、每只股票、每个“日期×股票”有多少行。
# 这样下面的 2,000、1,500、250 都不是手填的，而是由数据计算出来的。
rows_by_date = dict(data.group_by("date").len().iter_rows())
rows_by_symbol = dict(data.group_by("symbol").len().iter_rows())
rows_by_pair = {
    (day, symbol): rows
    for day, symbol, rows in data.group_by("date", "symbol").len().iter_rows()
}

dates = sorted(rows_by_date)
symbols = sorted(rows_by_symbol)

# 一个字典代表一个文件的“目录元数据”。
# None 表示文件名没有透露这个维度，因此不能靠目录排除它。
catalogs = {
    "A 全部放一起": [
        {"date": None, "symbol": None, "rows": data.height}
    ],
    "B 按日期": [
        {"date": day, "symbol": None, "rows": rows_by_date[day]}
        for day in dates
    ],
    "C 按股票": [
        {"date": None, "symbol": symbol, "rows": rows_by_symbol[symbol]}
        for symbol in symbols
    ],
    "D 日期×股票": [
        {"date": day, "symbol": symbol, "rows": rows_by_pair[(day, symbol)]}
        for day in dates for symbol in symbols
    ],
}

queries = {
    "查一天": {"date": "20260630"},
    "查一只股票": {"symbol": "S002"},
}

def candidate_files(files, filters):
    """只利用文件路径已经知道的值，排除肯定不匹配的文件。"""
    candidates = []
    for file_meta in files:
        can_skip = any(
            file_meta[column] is not None and file_meta[column] != wanted
            for column, wanted in filters.items()
        )
        if not can_skip:
            candidates.append(file_meta)
    return candidates

for query_name, filters in queries.items():
    condition = ", ".join(f"{key}={value}" for key, value in filters.items())
    print(f"\n{query_name}: {condition}")
    for layout_name, files in catalogs.items():
        candidates = candidate_files(files, filters)
        candidate_rows = sum(file["rows"] for file in candidates)
        print(
            f"  {layout_name:11s} "
            f"{len(candidates):2d}/{len(files):2d} 个文件，"
            f"候选行 {candidate_rows:5d}"
        )
