"""Flatten parsed JSON into long-form DataFrames for analysis.

Outputs both CSV and a binary DataFrame format:
- analysis/out/sets.csv + sessions.csv
- analysis/out/sets.parquet + sessions.parquet when pyarrow/fastparquet exists
- analysis/out/sets.pkl + sessions.pkl as dependency-free fallback
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PARSED = os.path.join(ROOT, "data", "parsed")
OUT = os.path.join(ROOT, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

GROUP_ZH = {
    "chest": "胸", "back": "背", "shoulders": "肩", "biceps": "臂", "triceps": "臂",
    "quads": "腿", "hams": "腿", "glutes": "腿", "adductors": "腿", "abductors": "腿",
    "calves": "腿", "core": "核心", "cardio": "有氧", "rehab": "康复", "unknown": "其他",
}


def classify(name: str) -> str:
    from xunji.muscle_groups import lookup
    primary, _ = lookup(name)
    return GROUP_ZH.get(primary[0] if primary else "unknown", "其他")


def parse_title(raw: str) -> str:
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    title_parts = []
    for tok in toks:
        if re.match(r"^\d+\.\D", tok):
            break
        if re.fullmatch(r"\d{6}|\d{4}-\d{2}-\d{2}", tok):
            continue
        if tok.startswith(("id:", "train_time:", "calorie:")):
            continue
        title_parts.append(tok)
    return ",".join(title_parts)


def save_frame(df: pd.DataFrame, stem: str) -> None:
    csv_path = os.path.join(OUT, f"{stem}.csv")
    pkl_path = os.path.join(OUT, f"{stem}.pkl")
    parquet_path = os.path.join(OUT, f"{stem}.parquet")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    try:
        df.to_parquet(parquet_path, index=False)
    except ImportError as e:
        print(f"[warn] parquet skipped for {stem}: {e.__class__.__name__}. Using CSV/PKL fallback.")


def build():
    rows = []          # set-level
    sess_rows = []     # session-level
    files = sorted(glob.glob(f"{PARSED}/*.json"))
    for fp in files:
        with open(fp) as f:
            day_sessions = json.load(f)
        if not day_sessions:
            continue
        date = os.path.basename(fp).replace(".json", "")
        for s in day_sessions:
            title = s.get("title") or parse_title(s.get("raw", ""))
            dur_min = (s.get("duration_ms") or 0) / 60000
            n_sets = sum(len(e.get("sets", [])) for e in s.get("exercises", []))
            n_ex = len(s.get("exercises", []))
            volume = 0.0
            external_volume = 0.0
            groups_in_sess = set()
            cardio = False

            for e in s.get("exercises", []):
                name = e["name"]
                g = classify(name)
                groups_in_sess.add(g)
                if e.get("modality") == "cardio" or g == "有氧":
                    cardio = True
                for st in e.get("sets", []):
                    w = st.get("weight_kg") or 0
                    r = st.get("reps") or 0
                    rest = st.get("rest_s") or 0
                    tonnage = w * r
                    volume += tonnage
                    if e.get("load_kind", "external_kg") in {"external_kg", "per_side_external_kg"}:
                        external_volume += tonnage
                    rows.append({
                        "date": date,
                        "title": title,
                        "exercise": name,
                        "group": g,
                        "modality": e.get("modality", "strength"),
                        "load_kind": e.get("load_kind", "external_kg"),
                        "set_idx": st.get("set"),
                        "weight_kg": w,
                        "weight_kg_per_side": st.get("weight_kg_per_side"),
                        "assist_kg": st.get("assist_kg"),
                        "reps": r,
                        "rest_s": rest,
                        "tonnage": tonnage,
                        "duration_min": dur_min,
                        "session_id": s.get("local_id"),
                    })
            sess_rows.append({
                "date": date,
                "session_id": s.get("local_id"),
                "title": title,
                "duration_min": dur_min,
                "n_exercises": n_ex,
                "n_sets": n_sets,
                "volume_kg": volume,
                "external_volume_kg": external_volume,
                "groups": "/".join(sorted(groups_in_sess)),
                "is_cardio": cardio,
            })

    df_sets = pd.DataFrame(rows)
    df_sess = pd.DataFrame(sess_rows)
    if df_sets.empty or df_sess.empty:
        raise SystemExit("no parsed training data found")
    df_sets["date"] = pd.to_datetime(df_sets["date"])
    df_sess["date"] = pd.to_datetime(df_sess["date"])
    save_frame(df_sets, "sets")
    save_frame(df_sess, "sessions")

    # Backward-compatible copies for article_charts.py.
    df_sets.to_csv(os.path.join(ROOT, "analysis", "sets.csv"), index=False)
    df_sess.to_csv(os.path.join(ROOT, "analysis", "sessions.csv"), index=False)

    print(f"sets: {len(df_sets)} rows, sessions: {len(df_sess)} rows")
    print(f"date range: {df_sess.date.min().date()} → {df_sess.date.max().date()}")
    print("\ntop 20 exercises:")
    print(df_sets.exercise.value_counts().head(20))
    print("\ngroup distribution (set-count):")
    print(df_sets.group.value_counts())
    return df_sets, df_sess


if __name__ == "__main__":
    build()
