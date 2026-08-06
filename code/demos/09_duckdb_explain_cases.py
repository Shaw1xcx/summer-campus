import duckdb

source = """read_parquet(
    'data/lake/**/data_*.parquet', hive_partitioning = true
)"""

# 五条查询逐步增加复杂度：全列、列剪枝、聚合、Top-N、分类汇总。
# 课堂上对照前两条计划，最容易观察 SELECT * 与只选三列的差别。
queries = {
    "01_SELECT_STAR": f"""
        SELECT * FROM {source}
        WHERE date = '20260630' LIMIT 1000
    """,
    "02_THREE_COLUMNS": f"""
        SELECT symbol, Price, Volume FROM {source}
        WHERE date = '20260630' LIMIT 1000
    """,
    "03_DAILY_SUMMARY": f"""
        SELECT symbol, count(*) AS trades, sum(Volume) AS volume
        FROM {source} WHERE date = '20260630'
        GROUP BY symbol
    """,
    "04_TOP_AMOUNT": f"""
        SELECT symbol, sum(Price * Volume) / 100 AS amount_yuan
        FROM {source} WHERE date = '20260630'
        GROUP BY symbol ORDER BY amount_yuan DESC LIMIT 10
    """,
    "05_BUY_SELL_FLAGS": f"""
        SELECT BSFlag, count(*) AS trades, sum(Volume) AS volume
        FROM {source} WHERE date = '20260630'
        GROUP BY BSFlag ORDER BY BSFlag
    """,
}

# 每条 SQL 先显示少量结果，再显示真实执行计划，避免只看耗时数字。
for name, sql in queries.items():
    print("\n" + "=" * 20, name, "=" * 20)
    result = duckdb.sql(sql).fetchall()
    plan = duckdb.sql("EXPLAIN ANALYZE " + sql).fetchall()[0][1]
    print("结果前 5 行:", result[:5])
    # 在 PARQUET_SCAN 节点中关注读取列、过滤条件和处理行数。
    print(plan)
