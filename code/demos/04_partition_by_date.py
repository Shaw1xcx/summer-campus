from pathlib import Path
import duckdb

# 源文件有 12,000 行；这里不会修改它，而是另写一套按日期摆放的文件。
# COPY 会执行括号中的查询，并把查询结果写入 data/lake。
duckdb.sql("""
COPY (
    SELECT
        *,
        -- date 的形式是 YYYYMMDD，例如 20260630。
        -- 取出前 4 位和第 5–6 位，用作上层的 year、month 目录。
        substr(date, 1, 4) AS year,
        substr(date, 5, 2) AS month
    FROM read_parquet('data/trades.parquet')
)
TO 'data/lake'
-- PARTITION_BY 告诉 DuckDB：相同 year、month、date 的行放在一起。
-- OVERWRITE 让这段课堂代码能够重复运行。
(FORMAT parquet, PARTITION_BY (year, month, date), OVERWRITE);
""")

# 列出真正的 Parquet 数据文件。
created = sorted(Path("data/lake").rglob("data_*.parquet"))
print("生成文件数:", len(created))
for path in created:
    print(" -", path)
