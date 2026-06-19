"""写入 / 更新训记训练记录。

默认只做本地校验，不真实 POST。真实写入必须显式传 --send。

用法:
  # dry-run（默认）
  python -m xunji.upsert --row "2026-04-02,休息日"

  # 真实写入（危险）
  python -m xunji.upsert --send --allow-create --row "2026-04-02,休息日"

  # 从 JSON 文件读 rows（文件格式: ["row1", "row2", ...]）
  python -m xunji.upsert --file rows.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import XunjiClient, XunjiError, validate_rows
from .fetch import RAW_DIR


def _date_has_cached_record(datestr: str) -> bool:
    path = RAW_DIR / f"{datestr}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except Exception:
        return True  # corrupted cache: be conservative
    res = data.get("res")
    return isinstance(res, list) and len(res) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="JSON file containing a list of row strings")
    ap.add_argument("--row", action="append", default=[],
                    help="single row string (can repeat)")
    ap.add_argument("--send", action="store_true",
                    help="actually POST to Xunji. Omit for dry-run")
    ap.add_argument("--allow-create", action="store_true",
                    help="allow rows without id: to create new entries")
    ap.add_argument("--allow-existing-date-create", action="store_true",
                    help="allow creating no-id rows when local raw cache already has records for that date")
    args = ap.parse_args()

    rows: list[str] = list(args.row)
    if args.file:
        loaded = json.loads(Path(args.file).read_text())
        if not isinstance(loaded, list) or not all(isinstance(x, str) for x in loaded):
            raise SystemExit("--file must contain a JSON list of strings")
        rows.extend(loaded)

    if not rows:
        ap.error("provide --file or --row")

    try:
        infos = validate_rows(rows)
    except XunjiError as e:
        raise SystemExit(str(e))

    datestr = infos[0].date
    creates = [i for i, info in enumerate(infos) if not info.has_id]
    existing_cached = _date_has_cached_record(datestr)
    if args.send:
        if creates and not args.allow_create:
            raise SystemExit(
                "refusing rows without id: because they create new Xunji entries. "
                "Pass --allow-create if this is intentional."
            )
        if creates and existing_cached and not args.allow_existing_date_create:
            raise SystemExit(
                f"refusing to create no-id rows for {datestr}: local raw cache already has records. "
                "Use id:LOCALID to update, or pass --allow-existing-date-create if you verified App calendar."
            )

    mode = "SEND" if args.send else "DRY-RUN"
    print(f"[{mode}] {len(rows)} row(s) for date {datestr}")
    if creates:
        print(f"  creates new entries (no id:) rows: {creates}")
        if not args.send:
            print("  [dry-run warning] real send would require --send --allow-create")
            if existing_cached:
                print("  [dry-run warning] local raw cache already has records for this date; real send would also require --allow-existing-date-create")
    for i, r in enumerate(rows):
        print(f"  [{i}] {r[:140]}{'...' if len(r) > 140 else ''}")

    if not args.send:
        print("(dry-run, not sending. Add --send to POST.)")
        return

    client = XunjiClient()
    resp = client.upsert(rows)
    print("--- response ---")
    print(json.dumps(resp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
