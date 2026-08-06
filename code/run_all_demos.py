from __future__ import annotations

import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "数据处理技术补充.html"
DEMO_DIR = ROOT / "code" / "demos"
RESULT_DIR = ROOT / "code" / "results"

DEMO_NAMES = [
    "prepare_data",
    "csv_vs_parquet",
    "compression_benchmark",
    "partition_by_date",
    "layout_candidates",
    "polars_query_plan",
    "polars_eager_vs_lazy",
    "duckdb_top3",
    "duckdb_explain_cases",
    "cache_basic",
    "cache_with_version",
    "zarr_chunks",
    "threads_cpu_vs_io",
    "thread_lock",
    "end_to_end",
]


def main() -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    text = HTML_PATH.read_text(encoding="utf-8")
    blocks = [html.unescape(code) for code in re.findall(
        r"<pre><code>(.*?)</code></pre>", text, flags=re.DOTALL
    )]
    python_blocks = blocks[3:]
    if len(python_blocks) != 15:
        raise RuntimeError(f"预期 15 个 Python 代码块，实际找到 {len(python_blocks)} 个")

    records = []
    for number, (name, code) in enumerate(zip(DEMO_NAMES, python_blocks), start=1):
        script_path = DEMO_DIR / f"{number:02d}_{name}.py"
        log_path = RESULT_DIR / f"{number:02d}_{name}.txt"
        script_path.write_text(code.rstrip() + "\n", encoding="utf-8")

        start = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        elapsed = time.perf_counter() - start
        output = completed.stdout.rstrip()
        log_path.write_text(output + ("\n" if output else ""), encoding="utf-8")
        records.append({
            "number": number,
            "returncode": completed.returncode,
            "seconds": round(elapsed, 6),
            "script": script_path.name,
            "log": log_path.name,
        })
        status = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"{number:02d} {status} {elapsed:.3f}s {log_path.name}")
        if completed.returncode != 0:
            print(output)
            break

    (RESULT_DIR / "run_summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if len(records) == 15 and all(r["returncode"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
