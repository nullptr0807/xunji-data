"""解析 res 数组里的训练记录文本。

支持 strength / bodyweight / cardio 的基础语义层。原始 raw 永远保留；
新增字段尽量 additive，兼容旧分析脚本。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_EXERCISE_HEAD = re.compile(r"^\d+\.\D")  # "1.引体向上" not "17.5kg"
_SET = re.compile(r"^\d+组$")
_KG = re.compile(r"^([\d.]+)kg$")
_REPS = re.compile(r"^(\d+)次$")
_TIME = re.compile(r"^time:(\d+)s$")
_DATE_SHORT = re.compile(r"^\d{6}$")
_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DISTANCE = re.compile(r"^([\d.]+)km$", re.I)
_KCAL = re.compile(r"^([\d.]+)kcal$", re.I)
_BPM = re.compile(r"^([\d.]+)bpm$", re.I)
_FLOORS = re.compile(r"^floors:([\d.]+)$", re.I)


def _iso_from_short(code: str) -> str:
    yy, mm, dd = int(code[:2]), int(code[2:4]), int(code[4:6])
    year = 2000 + yy if yy < 80 else 1900 + yy
    return f"{year:04d}-{mm:02d}-{dd:02d}"


def _is_meta(tok: str) -> bool:
    return (
        _DATE_SHORT.fullmatch(tok) is not None
        or _DATE_ISO.fullmatch(tok) is not None
        or tok.startswith("id:")
        or tok.startswith("train_time:")
        or tok.startswith("calorie:")
    )


def _load_kind_for(name: str, modality: str | None = None) -> str:
    if modality == "cardio":
        return "cardio"
    if "辅助" in name and ("引体" in name or "双杠" in name or "臂屈伸" in name):
        return "assist_kg"
    if "引体" in name or "双杠臂屈伸" in name or "抬腿" in name or "举腿" in name:
        return "bodyweight"
    if "哑铃" in name or "锤式弯举" in name or "Y字" in name or "侧平举" in name:
        return "per_side_external_kg"
    return "external_kg"


def _modality_for(name: str) -> str:
    low = name.lower()
    if any(k in name for k in ["跑步", "划船机", "爬楼梯", "椭圆", "单车", "跳绳"]):
        return "cardio"
    if any(k in low for k in ["run", "rowing", "elliptical", "stepper", "stair", "bike"]):
        return "cardio"
    if any(k in name for k in ["引体", "双杠臂屈伸", "抬腿", "举腿"]):
        return "strength_bodyweight"
    return "strength"


def _parse_title(tokens: list[str]) -> str | None:
    title_parts: list[str] = []
    for tok in tokens:
        if _EXERCISE_HEAD.match(tok):
            break
        if _is_meta(tok):
            continue
        # Stop before obvious set/cardio data accidentally outside exercise.
        if _SET.match(tok) or _KG.match(tok) or _REPS.match(tok) or _TIME.match(tok):
            continue
        title_parts.append(tok)
    return ",".join(title_parts) if title_parts else None


def _finalize_cardio(ex: dict[str, Any]) -> None:
    if ex.get("modality") != "cardio":
        return
    metrics = ex.setdefault("cardio", {})
    if "duration_s" in metrics and metrics["duration_s"]:
        ex["duration_s"] = metrics["duration_s"]
    if "kcal" in metrics:
        ex["calorie"] = metrics["kcal"]


def parse_record(text: str) -> dict[str, Any]:
    """把单条训练记录文本解析成结构化 dict。"""
    rec: dict[str, Any] = {"raw": text}
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    title = _parse_title(tokens)
    if title:
        rec["title"] = title

    exercises: list[dict[str, Any]] = []
    current_ex: dict[str, Any] | None = None
    current_set: dict[str, Any] | None = None

    for tok in tokens:
        # 元数据
        if _DATE_SHORT.fullmatch(tok):
            rec["date_code"] = tok
            rec["date"] = _iso_from_short(tok)
            continue
        if _DATE_ISO.fullmatch(tok):
            rec["date"] = tok
            rec["date_code"] = tok[2:4] + tok[5:7] + tok[8:10]
            continue
        if tok.startswith("id:"):
            rec["local_id"] = tok[3:]
            continue
        if tok.startswith("train_time:"):
            m = re.match(r"train_time:(\d+)-(\d+)", tok)
            if m:
                start_ms, end_ms = int(m.group(1)), int(m.group(2))
                rec["start_ms"] = start_ms
                rec["end_ms"] = end_ms
                rec["duration_ms"] = end_ms - start_ms
                rec["start_iso"] = datetime.fromtimestamp(start_ms / 1000).isoformat()
                rec["end_iso"] = datetime.fromtimestamp(end_ms / 1000).isoformat()
            continue
        if tok.startswith("calorie:"):
            v = tok.split(":", 1)[1].strip()
            if v:
                try:
                    rec["calorie"] = int(float(v))
                except ValueError:
                    rec["calorie_raw"] = v
            continue

        # 新动作 "1.引体向上"
        if _EXERCISE_HEAD.match(tok):
            if current_ex is not None:
                _finalize_cardio(current_ex)
            name = re.sub(r"^\d+\.", "", tok)
            modality = _modality_for(name)
            current_ex = {
                "name": name,
                "sets": [],
                "modality": modality,
                "load_kind": _load_kind_for(name, modality),
            }
            if modality == "cardio":
                current_ex["cardio"] = {}
            exercises.append(current_ex)
            current_set = None
            continue

        if current_ex is None:
            continue

        # 有氧 token（无 1组/2组结构）
        if current_ex.get("modality") == "cardio":
            metrics = current_ex.setdefault("cardio", {})
            if m := _DISTANCE.match(tok):
                metrics["distance_km"] = float(m.group(1))
                continue
            if m := _KCAL.match(tok):
                metrics["kcal"] = int(float(m.group(1)))
                continue
            if m := _BPM.match(tok):
                metrics["avg_hr_bpm"] = int(float(m.group(1)))
                continue
            if m := _FLOORS.match(tok):
                metrics["floors"] = int(float(m.group(1)))
                continue
            if m := _TIME.match(tok):
                metrics["duration_s"] = int(m.group(1))
                continue

        # 组开始
        if _SET.match(tok):
            current_set = {"set": int(tok[:-1])}
            current_ex["sets"].append(current_set)
            continue

        if current_set is None:
            continue

        if m := _KG.match(tok):
            val = float(m.group(1))
            kind = current_ex.get("load_kind")
            if kind == "assist_kg":
                current_set["assist_kg"] = val
                current_set["weight_kg"] = val  # backward-compatible raw field
            elif kind == "per_side_external_kg":
                current_set["weight_kg_per_side"] = val
                current_set["weight_kg"] = val
            else:
                current_set["weight_kg"] = val
        elif m := _REPS.match(tok):
            current_set["reps"] = int(m.group(1))
        elif m := _TIME.match(tok):
            current_set["rest_s"] = int(m.group(1))

    if current_ex is not None:
        _finalize_cardio(current_ex)

    if exercises:
        rec["exercises"] = exercises
        rec["total_sets"] = sum(len(e["sets"]) for e in exercises)
        rec["total_volume_external_kg"] = sum(
            (s.get("weight_kg") or 0) * (s.get("reps") or 0)
            for e in exercises
            if e.get("load_kind") in {"external_kg", "per_side_external_kg"}
            for s in e["sets"]
        )
        # Backward-compatible parsed volume: raw external numeric tonnage. This
        # intentionally still mirrors old behavior for existing reports.
        rec["total_volume_kg"] = sum(
            (s.get("weight_kg") or 0) * (s.get("reps") or 0)
            for e in exercises for s in e["sets"]
        )
        rec["cardio_summary"] = [e for e in exercises if e.get("modality") == "cardio"]

    return rec


def parse_response(data: dict) -> list[dict]:
    """解析整个 API 响应的 res 数组。"""
    res = data.get("res", [])
    if not isinstance(res, list):
        return []
    return [parse_record(item) if isinstance(item, str) else item for item in res]
