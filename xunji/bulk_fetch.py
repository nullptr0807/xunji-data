"""Bulk fetch all dates in a range, fast (per-date rate limit + retry).

The API primarily locks repeated calls for the same (date, endpoint). Different
uncached dates can usually be fetched with a short gap; use --gap 95 if the
server behaves like a global limiter.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

from .client import XunjiClient
from .fetch import RAW_DIR, PARSED_DIR, write_parsed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--gap", type=float, default=1.5,
                    help="seconds between uncached requests (default 1.5; use 95 for conservative global throttling)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reparse", action="store_true", help="rebuild parsed JSON from cached raw")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    client = XunjiClient()
    total = len(days)
    n_fetched = n_cached = n_reparsed = n_with_data = 0
    t0 = time.time()
    for i, day in enumerate(days):
        ds = day.isoformat()
        raw_path = RAW_DIR / f"{ds}.json"
        parsed_path = PARSED_DIR / f"{ds}.json"
        if raw_path.exists() and not args.force:
            n_cached += 1
            data = json.loads(raw_path.read_text())
            if args.reparse or not parsed_path.exists() or parsed_path.stat().st_mtime < raw_path.stat().st_mtime:
                write_parsed(ds, data)
                n_reparsed += 1
        else:
            try:
                data = client.fetch_with_retry(ds, max_retries=8)
            except Exception as e:
                print(f"[{i+1}/{total}] {ds} ERR: {e}", flush=True)
                time.sleep(args.gap)
                continue
            raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            write_parsed(ds, data)
            n_fetched += 1
            time.sleep(args.gap)

        n = len(data.get("res", []) or [])
        if n:
            n_with_data += 1
        if (i + 1) % 30 == 0 or i == total - 1:
            elapsed = time.time() - t0
            print(f"[{i+1}/{total}] {ds}  fetched={n_fetched} cached={n_cached} "
                  f"reparsed={n_reparsed} non-empty={n_with_data}  elapsed={elapsed:.0f}s", flush=True)

    print(f"DONE: total={total} fetched={n_fetched} cached={n_cached} "
          f"reparsed={n_reparsed} non-empty={n_with_data}")


if __name__ == "__main__":
    main()
