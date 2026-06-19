"""训记 API 客户端。"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://trains.xunjiapp.cn"
ENDPOINT = "/api_trains_for_llm"


class XunjiError(Exception):
    pass


class RateLimited(XunjiError):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"too frequent, retry after {retry_after}s")


@dataclass(frozen=True)
class RowInfo:
    raw: str
    date: str
    has_id: bool
    signature: str


_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{6})$")


def _retry_after_from(msg: str, default: int = 90) -> int:
    m = re.search(r"retry after\s*(\d+)", msg, re.I) or re.search(r"(\d+)s", msg)
    return int(m.group(1)) if m else default


def _row_signature(row: str) -> str:
    """Stable enough duplicate signature for one upsert batch.

    Keep id/train_time out because the dangerous duplicate pattern is usually the
    same planned content sent twice without an id.
    """
    parts = []
    for tok in (t.strip() for t in row.split(",") if t.strip()):
        if tok.startswith("id:") or tok.startswith("train_time:"):
            continue
        parts.append(tok)
    return ",".join(parts)


def parse_row_info(row: str) -> RowInfo:
    if not isinstance(row, str) or not row.strip():
        raise XunjiError("row must be a non-empty string")
    first = row.split(",", 1)[0].strip()
    if not _DATE_RE.fullmatch(first):
        raise XunjiError(f"row date invalid: {first!r}")
    has_id = any(tok.strip().startswith("id:") for tok in row.split(",")[:6])
    return RowInfo(raw=row, date=first, has_id=has_id, signature=_row_signature(row))


def validate_rows(rows: list[str]) -> list[RowInfo]:
    """Local safety checks shared by CLI and direct client callers."""
    if not rows:
        raise XunjiError("rows must be non-empty")
    if len(rows) > 12:
        raise XunjiError(f"max 12 rows per call, got {len(rows)}")

    infos: list[RowInfo] = []
    for i, row in enumerate(rows):
        if len(row) > 1500:
            raise XunjiError(f"row {i} exceeds 1500 chars ({len(row)})")
        infos.append(parse_row_info(row))

    dates = {info.date for info in infos}
    if len(dates) != 1:
        raise XunjiError(f"all rows must share one date, got: {sorted(dates)}")

    seen: dict[str, int] = {}
    for i, info in enumerate(infos):
        if info.signature in seen:
            raise XunjiError(
                f"duplicate row in one upsert batch: row {seen[info.signature]} and row {i}"
            )
        seen[info.signature] = i
    return infos


class XunjiClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("XUNJI_API_KEY")
        if not self.api_key:
            raise XunjiError("XUNJI_API_KEY not set (env or .env)")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip",
        })

    def _classify_fetch_response(self, data: dict) -> dict:
        err = data.get("error")
        if err:
            msg = str(err)
            if "too frequent" in msg.lower() or "rate" in msg.lower():
                raise RateLimited(_retry_after_from(msg))
            raise XunjiError(msg)

        res = data.get("res")
        # Xunji export quirk: success may be false while res is the valid list.
        if isinstance(res, list):
            return data

        msg = str(res or "unknown error")
        if "too frequent" in msg.lower() or "rate" in msg.lower():
            raise RateLimited(_retry_after_from(msg))
        raise XunjiError(msg)

    def _classify_upsert_response(self, data: dict) -> dict:
        err = data.get("error")
        if err:
            msg = str(err)
            if "too frequent" in msg.lower() or "rate" in msg.lower():
                raise RateLimited(_retry_after_from(msg))
            raise XunjiError(msg)

        res = data.get("res")
        # Xunji upsert success commonly returns {"res": []}. Some successful
        # updates echo row strings with ids; accept list[str] too.
        if isinstance(res, list) and all(isinstance(x, str) for x in res):
            return data

        msg = str(res or "unknown error")
        if "too frequent" in msg.lower() or "rate" in msg.lower():
            raise RateLimited(_retry_after_from(msg))
        raise XunjiError(msg)

    def fetch(self, datestr: str) -> dict:
        """抓取某天的训练数据。datestr: YYYY-MM-DD。"""
        body = {"datestr": datestr, "apikey": self.api_key}
        r = self.session.post(
            BASE_URL + ENDPOINT,
            json=body,
            params={"apikey": self.api_key},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return self._classify_fetch_response(r.json())

    def upsert(self, rows: list[str]) -> dict:
        """写入 / 更新训练记录。

        rows: 训练行字符串列表，全部必须同一日期，单次最多 12 条，每行 ≤1500 字符。
        含 ``id:...`` 的行会按 localId 更新已有记录；不含 id 的会新建。
        """
        validate_rows(rows)

        body = {"res": rows, "apikey": self.api_key}
        r = self.session.post(
            BASE_URL + "/api_upsert_trains_for_llm",
            json=body,
            params={"apikey": self.api_key},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return self._classify_upsert_response(r.json())

    def fetch_with_retry(self, datestr: str, max_retries: int = 3) -> dict:
        """碰到限流自动等待重试。"""
        for attempt in range(max_retries):
            try:
                return self.fetch(datestr)
            except RateLimited as e:
                wait = e.retry_after + 2
                print(f"  [rate-limited] sleeping {wait}s...")
                time.sleep(wait)
        raise XunjiError(f"still rate-limited after {max_retries} retries")
