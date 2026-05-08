"""写入 / 更新训记训练记录。

用法:
  # 从 JSON 文件读 rows 写入 (文件格式: ["row1", "row2", ...])
  python -m xunji.upsert --file rows.json

  # 直接传 rows
  python -m xunji.upsert --row "2026-04-02,休息日"

行格式参考 README + 训记导入说明：
  YYYY-MM-DD[,id:LOCALID],标题[,train_time:start-end][,备注],
  1.动作名,1组,Wkg,R次[,time:Ts],2组,...
  有氧: 2.跑步,5km,300kcal,time:1800s,140bpm
  休息日: 2026-04-02,休息日
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .client import XunjiClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="JSON file containing a list of row strings")
    ap.add_argument("--row", action="append", default=[],
                    help="single row string (can repeat)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate locally, don't POST")
    args = ap.parse_args()

    rows: list[str] = list(args.row)
    if args.file:
        loaded = json.loads(Path(args.file).read_text())
        if not isinstance(loaded, list):
            raise SystemExit("--file must contain a JSON list of strings")
        rows.extend(loaded)

    if not rows:
        ap.error("provide --file or --row")

    # 同一日期校验
    dates = {r.split(",", 1)[0].strip() for r in rows}
    if len(dates) != 1:
        raise SystemExit(f"all rows must share one date, got: {dates}")

    print(f"Upserting {len(rows)} row(s) for date {next(iter(dates))}")
    for i, r in enumerate(rows):
        print(f"  [{i}] {r[:100]}{'...' if len(r) > 100 else ''}")

    if args.dry_run:
        print("(dry-run, not sending)")
        return

    client = XunjiClient()
    resp = client.upsert(rows)
    print("--- response ---")
    print(json.dumps(resp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
