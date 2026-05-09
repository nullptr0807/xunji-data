#!/usr/bin/env python3
"""Generate an evidence-heavy Chinese deep report for Xunji training history.

Inputs: data/parsed/*.json (gitignored personal data)
Outputs: analysis/deep_report/{report.html, report.md, metrics.json, img/*.png}

Design notes:
- Charts use English labels to avoid CJK font dependency in matplotlib.
- Chinese interpretation is in HTML/Markdown text.
- No API calls, no writes to Xunji. Reads local parsed cache only.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "data" / "parsed"
OUT = ROOT / "analysis" / "deep_report"
IMG = OUT / "img"
OUT.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.dpi"] = 150
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.25
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["axes.unicode_minus"] = False

# Import explicit muscle map if present; keep fallback for untracked dev state.
try:
    from xunji.muscle_groups import lookup as muscle_lookup
except Exception:
    def muscle_lookup(name: str):
        rules = {
            "chest": ["卧推", "飞鸟", "夹胸", "推胸"],
            "back": ["引体", "划船", "下拉", "硬拉"],
            "shoulders": ["推举", "推肩", "侧平举", "前平举", "后束", "面拉"],
            "biceps": ["弯举"],
            "triceps": ["臂屈伸", "下压", "三头"],
            "quads": ["深蹲", "腿举", "倒蹬", "腿屈伸"],
            "hams": ["腿弯举", "罗马尼亚"],
            "core": ["抬腿", "卷腹", "仰卧起坐", "健腹轮"],
        }
        prim = [g for g, kws in rules.items() if any(k in name for k in kws)]
        return (prim or ["unknown"], [])

MUSCLE_CN = {
    "chest": "胸", "back": "背", "shoulders": "肩", "biceps": "二头", "triceps": "三头",
    "quads": "股四", "hams": "腘绳", "glutes": "臀", "adductors": "内收", "abductors": "外展",
    "core": "核心", "unknown": "未知",
}
CATEGORY_CN = {"push": "推", "pull": "拉", "legs": "腿", "core": "核心", "mixed": "混合", "unknown": "未知"}
PUSH = {"chest", "shoulders", "triceps"}
PULL = {"back", "biceps"}
LEGS = {"quads", "hams", "glutes", "adductors", "abductors"}
CORE = {"core"}

LIFTS = {
    "bench": "杠铃卧推",
    "incline_bench": "上斜杠铃卧推",
    "squat": "深蹲",
    "deadlift": "硬拉",
    "ohp": "站姿杠铃推举",
    "row": "杠铃划船",
    "curl": "杠铃弯举",
}
LIFT_CN = {
    "bench": "杠铃卧推", "incline_bench": "上斜杠铃卧推", "squat": "深蹲", "deadlift": "硬拉",
    "ohp": "站姿杠铃推举", "row": "杠铃划船", "curl": "杠铃弯举",
}
BODYWEIGHT_KG = float(os.environ.get("BODYWEIGHT_KG", 70.0))
HEIGHT_CM = float(os.environ.get("HEIGHT_CM", 175.0))


def epley(weight: float, reps: float) -> float:
    if not weight or not reps or weight <= 0 or reps <= 0:
        return float("nan")
    # Epley becomes noisy above ~12 reps; still keep but flag in narrative.
    return float(weight) * (1.0 + float(reps) / 30.0)


def parse_title(raw: str) -> str:
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    title_parts = []
    for tok in toks[1:]:
        if tok.startswith("id:") or tok.startswith("train_time:") or tok.startswith("calorie:"):
            continue
        if re.match(r"^\d+\.\D", tok):
            break
        if re.match(r"^\d+组$", tok) or tok.endswith("kg") or tok.endswith("次") or tok.startswith("time:"):
            continue
        title_parts.append(tok)
    return "/".join(title_parts)[:80]


def category_from_effective(eff: dict[str, float]) -> tuple[str, dict[str, float]]:
    cat = {
        "push": sum(eff.get(m, 0.0) for m in PUSH),
        "pull": sum(eff.get(m, 0.0) for m in PULL),
        "legs": sum(eff.get(m, 0.0) for m in LEGS),
        "core": sum(eff.get(m, 0.0) for m in CORE),
    }
    total = sum(cat.values())
    if total <= 0:
        return "unknown", cat
    ordered = sorted(cat.items(), key=lambda kv: kv[1], reverse=True)
    if ordered[0][1] / total < 0.50 and ordered[1][1] / total > 0.25:
        return "mixed", cat
    return ordered[0][0], cat


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    eff_rows: list[dict[str, Any]] = []

    for fp in sorted(PARSED.glob("*.json")):
        date_str = fp.stem
        try:
            day_sessions = json.loads(fp.read_text())
        except Exception:
            continue
        if not isinstance(day_sessions, list) or not day_sessions:
            continue
        for idx, s in enumerate(day_sessions):
            sid = str(s.get("local_id") or f"{date_str}_{idx}")
            raw = s.get("raw", "")
            duration_min = (s.get("duration_ms") or 0) / 60000.0
            calorie = s.get("calorie")
            title = parse_title(raw)
            ex_names = []
            sess_eff = defaultdict(float)
            n_sets = 0
            volume = 0.0
            e1rm_values = []
            rest_values = []
            for ex in s.get("exercises", []) or []:
                name = ex.get("name") or ""
                ex_names.append(name)
                primary, secondary = muscle_lookup(name)
                primary = primary or ["unknown"]
                secondary = secondary or []
                for st in ex.get("sets", []) or []:
                    n_sets += 1
                    w = float(st.get("weight_kg") or 0.0)
                    r = float(st.get("reps") or 0.0)
                    rest = st.get("rest_s")
                    if rest is not None:
                        rest_values.append(float(rest))
                    ton = w * r
                    volume += ton
                    est = epley(w, r)
                    if not math.isnan(est):
                        e1rm_values.append(est)
                    set_rows.append({
                        "date": date_str,
                        "session_id": sid,
                        "exercise": name,
                        "set_idx": st.get("set"),
                        "weight_kg": w,
                        "reps": r,
                        "rest_s": float(rest or 0),
                        "tonnage": ton,
                        "e1rm": est,
                        "duration_min": duration_min,
                        "calorie": calorie,
                        "primary": ";".join(primary),
                        "secondary": ";".join(secondary),
                    })
                    # Effective-set accounting for balance analysis.
                    for m in primary:
                        sess_eff[m] += 1.0
                        eff_rows.append({"date": date_str, "session_id": sid, "muscle": m, "credit": 1.0, "source": "primary", "exercise": name})
                    for m in secondary:
                        sess_eff[m] += 0.5
                        eff_rows.append({"date": date_str, "session_id": sid, "muscle": m, "credit": 0.5, "source": "secondary", "exercise": name})
            cat, cat_eff = category_from_effective(sess_eff)
            signature = " | ".join(sorted(set(ex_names)))
            session_rows.append({
                "date": date_str,
                "session_id": sid,
                "title": title,
                "duration_min": duration_min,
                "duration_clean": duration_min if 5 <= duration_min <= 180 else np.nan,
                "duration_outlier": bool(duration_min > 180 or duration_min < 5),
                "calorie": calorie,
                "n_exercises": len(ex_names),
                "n_sets": n_sets,
                "volume_kg": volume,
                "density_kg_min": volume / duration_min if duration_min and 5 <= duration_min <= 180 else np.nan,
                "cal_per_min": calorie / duration_min if calorie and duration_min and 5 <= duration_min <= 180 else np.nan,
                "exercise_names": ", ".join(ex_names),
                "signature": signature,
                "category": cat,
                **{f"cat_{k}_eff": v for k, v in cat_eff.items()},
                **{f"muscle_{m}_eff": v for m, v in sess_eff.items()},
            })

    sets = pd.DataFrame(set_rows)
    sessions = pd.DataFrame(session_rows)
    eff = pd.DataFrame(eff_rows)
    if not sets.empty:
        sets["date"] = pd.to_datetime(sets["date"])
    if not sessions.empty:
        sessions["date"] = pd.to_datetime(sessions["date"])
        sessions["year"] = sessions.date.dt.year
        sessions["year_month"] = sessions.date.dt.to_period("M").astype(str)
        sessions["week"] = sessions.date.dt.to_period("W-MON").apply(lambda p: p.start_time)
    if not eff.empty:
        eff["date"] = pd.to_datetime(eff["date"])
        eff["week"] = eff.date.dt.to_period("W-MON").apply(lambda p: p.start_time)
    return sets, sessions, eff


def pct(x: float, digits=1) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}%"


def fmt(x: Any, digits=1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{digits}f}"
    return str(x)


def savefig(name: str) -> str:
    path = IMG / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"img/{name}"


def make_overview_charts(sets: pd.DataFrame, sessions: pd.DataFrame, eff: pd.DataFrame) -> dict[str, str]:
    charts = {}
    # 1 frequency + duration
    monthly = sessions.groupby(pd.Grouper(key="date", freq="MS")).agg(
        sessions=("session_id", "count"), duration=("duration_clean", "sum"), volume=("volume_kg", "sum")
    )
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax[0].bar(monthly.index, monthly.sessions, width=24, color="#2563eb")
    ax[0].set_title("Training frequency by month")
    ax[0].set_ylabel("sessions")
    ax[1].plot(monthly.index, monthly.volume / 1000, color="#16a34a", lw=1.8)
    ax[1].set_title("Monthly total tonnage")
    ax[1].set_ylabel("tons")
    ax[2].plot(monthly.index, monthly.duration / 60, color="#dc2626", lw=1.8)
    ax[2].set_title("Monthly logged duration, cleaned <=180 min/session")
    ax[2].set_ylabel("hours")
    charts["overview"] = savefig("01_overview_frequency_volume_duration.png")

    # 2 effective sets by muscle monthly
    if not eff.empty:
        m = eff.groupby([pd.Grouper(key="date", freq="MS"), "muscle"])["credit"].sum().unstack(fill_value=0)
        order = ["chest", "back", "shoulders", "biceps", "triceps", "quads", "hams", "glutes", "core"]
        cols = [c for c in order if c in m.columns]
        fig, ax = plt.subplots(figsize=(13, 6))
        (m[cols]).plot.area(ax=ax, alpha=0.85, linewidth=0)
        ax.set_title("Monthly effective sets by muscle (primary=1, secondary=0.5)")
        ax.set_ylabel("effective sets")
        ax.legend(loc="upper left", ncols=3, fontsize=8)
        charts["muscle_sets"] = savefig("02_monthly_effective_sets_by_muscle.png")

    # 3 category distribution by year
    yc = sessions.groupby(["year", "category"]).size().unstack(fill_value=0)
    cols = [c for c in ["push", "pull", "legs", "core", "mixed", "unknown"] if c in yc.columns]
    fig, ax = plt.subplots(figsize=(11, 5))
    yc[cols].plot.bar(stacked=True, ax=ax, color=["#ef4444", "#3b82f6", "#22c55e", "#a855f7", "#f59e0b", "#737373"][:len(cols)])
    ax.set_title("Session type by year")
    ax.set_ylabel("sessions")
    ax.legend(loc="upper right")
    charts["split_year"] = savefig("03_session_type_by_year.png")

    # 4 novelty
    first_seen = sets.groupby("exercise").date.min().sort_values()
    cum = pd.DataFrame({"date": first_seen.values, "n": range(1, len(first_seen) + 1), "exercise": first_seen.index})
    yearly_new = first_seen.dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(cum.date, cum.n, color="#dc2626", lw=2)
    ax[0].set_title("Cumulative unique exercises")
    ax[0].set_ylabel("unique exercises")
    ax[1].bar(yearly_new.index.astype(str), yearly_new.values, color="#7c3aed")
    ax[1].set_title("New exercises first seen by year")
    ax[1].set_ylabel("count")
    charts["novelty"] = savefig("04_exercise_novelty.png")
    return charts


def strength_analysis(sets: pd.DataFrame, sessions: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    charts = {}
    summary: dict[str, Any] = {}
    max_date = sessions.date.max()

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=False)
    axes = axes.flatten()
    plot_lifts = ["bench", "squat", "deadlift", "ohp", "row", "incline_bench"]

    for ax, key in zip(axes, plot_lifts):
        ex = LIFTS[key]
        d = sets[(sets.exercise == ex) & (sets.weight_kg > 0) & (sets.reps > 0)].copy()
        if d.empty:
            ax.set_visible(False)
            continue
        daily = d.groupby("date").agg(
            best_e1rm=("e1rm", "max"), best_weight=("weight_kg", "max"), max_reps=("reps", "max"), sets=("set_idx", "count")
        ).sort_index()
        daily["pr_e1rm"] = daily.best_e1rm.cummax()
        daily["pr_weight"] = daily.best_weight.cummax()
        # breakthrough dates: e1RM PR improvement >= 1.0 kg over previous all-time best.
        prev = daily.pr_e1rm.shift(1).fillna(0)
        b = daily[(daily.pr_e1rm - prev) >= 1.0].copy()
        # Significant breakthroughs after >=60d since prior significant PR.
        sig = []
        last_date = None
        last_val = None
        for dt, row in b.iterrows():
            if last_date is None:
                sig.append((dt, row.pr_e1rm, None, None))
                last_date, last_val = dt, row.pr_e1rm
            else:
                gap = (dt - last_date).days
                inc = row.pr_e1rm - last_val
                if gap >= 30 or inc >= 2.5:
                    sig.append((dt, row.pr_e1rm, gap, inc))
                    last_date, last_val = dt, row.pr_e1rm
                elif row.pr_e1rm > last_val:
                    last_date, last_val = dt, row.pr_e1rm
        best_date = daily.best_e1rm.idxmax()
        last_pr_date = daily[daily.pr_e1rm == daily.pr_e1rm.max()].index.min()
        recent = daily[daily.index >= max_date - pd.Timedelta(days=180)]
        last365 = daily[daily.index >= max_date - pd.Timedelta(days=365)]
        slope_365 = np.nan
        r2_365 = np.nan
        if len(last365) >= 5:
            x = (last365.index - last365.index.min()).days.values
            y = last365.best_e1rm.values
            lr = stats.linregress(x, y)
            slope_365 = lr.slope * 365
            r2_365 = lr.rvalue ** 2
        plateau_gaps = []
        prev_date = None
        prev_val = None
        for dt, val, gap, inc in sig:
            if prev_date is not None:
                plateau_gaps.append({"from": str(prev_date.date()), "to": str(dt.date()), "days": int((dt - prev_date).days), "gain_kg": round(float(val - prev_val), 1)})
            prev_date, prev_val = dt, val
        current_plateau_days = int((max_date - last_pr_date).days)
        long_plateaus = [g for g in plateau_gaps if g["days"] >= 90]
        best_row = d.loc[d.e1rm.idxmax()]
        summary[key] = {
            "exercise": ex,
            "n_sessions": int(d.date.nunique()),
            "n_sets": int(len(d)),
            "first_date": str(d.date.min().date()),
            "first_best_e1rm": round(float(d.groupby("date").e1rm.max().iloc[0]), 1),
            "best_e1rm": round(float(d.e1rm.max()), 1),
            "best_e1rm_date": str(best_date.date()),
            "best_observed_weight": round(float(d.weight_kg.max()), 1),
            "best_set": f"{best_row.weight_kg:g}kg x {int(best_row.reps)}",
            "best_e1rm_per_bw": round(float(d.e1rm.max() / BODYWEIGHT_KG), 2),
            "recent_180d_best_e1rm": round(float(recent.best_e1rm.max()), 1) if not recent.empty else None,
            "last_pr_date": str(last_pr_date.date()),
            "current_plateau_days": current_plateau_days,
            "slope_365_e1rm_kg_per_year": round(float(slope_365), 2) if not pd.isna(slope_365) else None,
            "slope_365_r2": round(float(r2_365), 2) if not pd.isna(r2_365) else None,
            "breakthroughs": [
                {"date": str(dt.date()), "e1rm": round(float(val), 1), "gap_days": None if gap is None else int(gap), "gain_kg": None if inc is None else round(float(inc), 1)}
                for dt, val, gap, inc in sig[-10:]
            ],
            "long_plateaus": long_plateaus[-6:],
        }
        ax.scatter(daily.index, daily.best_e1rm, s=12, alpha=0.35, color="#64748b", label="daily best e1RM")
        ax.plot(daily.index, daily.pr_e1rm, color="#dc2626", lw=2.0, label="all-time PR e1RM")
        ax.plot(daily.index, daily.best_e1rm.rolling(8, min_periods=2).mean(), color="#2563eb", lw=1.5, label="8-session avg")
        # Mark long plateaus.
        for g in long_plateaus[-4:]:
            ax.axvspan(pd.to_datetime(g["from"]), pd.to_datetime(g["to"]), color="#fbbf24", alpha=0.12)
        ax.set_title(f"{key} e1RM trend")
        ax.set_ylabel("kg e1RM")
        ax.legend(fontsize=7, loc="lower right")
    charts["strength"] = savefig("05_strength_e1rm_plateaus.png")
    return summary, charts


def efficiency_analysis(sessions: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    charts = {}
    clean = sessions[(~sessions.duration_clean.isna()) & (sessions.n_exercises >= 2)].copy()
    groups = []
    for sig, g in clean.groupby("signature"):
        if len(g) < 4:
            continue
        span = (g.date.max() - g.date.min()).days
        if span < 120:
            continue
        g = g.sort_values("date")
        half = len(g) // 2
        first = g.iloc[:half]
        last = g.iloc[half:]
        x = (g.date - g.date.min()).dt.days.values
        dur_slope = stats.linregress(x, g.duration_clean.values).slope * 365 if len(g) >= 3 else np.nan
        dens = g.dropna(subset=["density_kg_min"])
        density_slope = stats.linregress((dens.date - dens.date.min()).dt.days.values, dens.density_kg_min.values).slope * 365 if len(dens) >= 3 and dens.date.nunique() >= 3 else np.nan
        cal = g.dropna(subset=["calorie"])
        cal_slope = stats.linregress((cal.date - cal.date.min()).dt.days.values, cal.calorie.values).slope * 365 if len(cal) >= 3 and cal.date.nunique() >= 3 else np.nan
        groups.append({
            "signature": sig,
            "n": int(len(g)),
            "span_days": int(span),
            "first_date": str(g.date.min().date()),
            "last_date": str(g.date.max().date()),
            "first_duration_med": round(float(first.duration_clean.median()), 1),
            "last_duration_med": round(float(last.duration_clean.median()), 1),
            "duration_delta_pct": round(float((last.duration_clean.median() / first.duration_clean.median() - 1) * 100), 1) if first.duration_clean.median() else None,
            "duration_slope_min_per_year": round(float(dur_slope), 1) if not pd.isna(dur_slope) else None,
            "first_density_med": round(float(first.density_kg_min.median()), 1),
            "last_density_med": round(float(last.density_kg_min.median()), 1),
            "density_delta_pct": round(float((last.density_kg_min.median() / first.density_kg_min.median() - 1) * 100), 1) if first.density_kg_min.median() else None,
            "density_slope_per_year": round(float(density_slope), 1) if not pd.isna(density_slope) else None,
            "calorie_n": int(len(cal)),
            "calorie_slope_per_year": round(float(cal_slope), 1) if not pd.isna(cal_slope) else None,
            "short_signature": ", ".join(sig.split(" | ")[:6]),
        })
    groups = sorted(groups, key=lambda x: (x["n"], x["span_days"]), reverse=True)

    # Plot top repeated signatures by duration and density trajectories.
    top = groups[:6]
    if top:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
        for i, item in enumerate(top):
            sig = item["signature"]
            g = clean[clean.signature == sig].sort_values("date")
            label = f"T{i+1} n={len(g)}"
            axes[0].plot(g.date, g.duration_clean, marker="o", ms=3, lw=1, alpha=0.8, label=label)
            axes[1].plot(g.date, g.density_kg_min, marker="o", ms=3, lw=1, alpha=0.8, label=label)
        axes[0].set_title("Repeated template duration")
        axes[0].set_ylabel("minutes")
        axes[1].set_title("Repeated template density")
        axes[1].set_ylabel("kg*rep/min")
        axes[1].legend(ncols=3, fontsize=8)
        charts["efficiency_templates"] = savefig("06_repeated_template_efficiency.png")
    return {"templates": groups[:20]}, charts


def advanced_stats(sets: pd.DataFrame, sessions: pd.DataFrame, eff: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    charts = {}
    # Weekly data: only weeks between first and last training.
    week_index = pd.date_range(sessions.date.min().to_period("W-MON").start_time, sessions.date.max().to_period("W-MON").start_time, freq="W-TUE")
    # Recompute using groupers; date period W-MON starts Tuesday? The exact label is not important for weekly stats.
    weekly = sessions.groupby("week").agg(
        sessions=("session_id", "count"),
        sets=("n_sets", "sum"),
        volume_kg=("volume_kg", "sum"),
        duration_min=("duration_clean", "sum"),
        avg_density=("density_kg_min", "mean"),
        calories=("calorie", "sum"),
    ).sort_index()
    # Effective sets per muscle.
    if not eff.empty:
        wm = eff.groupby(["week", "muscle"])["credit"].sum().unstack(fill_value=0)
        weekly = weekly.join(wm.add_prefix("eff_"), how="left")
    weekly = weekly.fillna(0)
    active = weekly[weekly.sessions > 0].copy()
    # Core lift weekly best e1RM.
    for key, ex in LIFTS.items():
        d = sets[(sets.exercise == ex) & (sets.e1rm.notna())]
        if not d.empty:
            w = d.groupby(d.date.dt.to_period("W-MON").apply(lambda p: p.start_time)).e1rm.max().rename(f"e1rm_{key}")
            weekly = weekly.join(w, how="left")
            weekly[f"e1rm_{key}_ffill"] = weekly[f"e1rm_{key}"].ffill()

    variables = ["sessions", "sets", "volume_kg", "duration_min", "avg_density"] + [c for c in weekly.columns if c.startswith("eff_")]
    corr = weekly[variables].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Spearman correlation matrix of weekly training variables")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    charts["corr"] = savefig("07_weekly_correlation_matrix.png")

    # CV of weekly effective set exposure across active weeks.
    cv = []
    for col in [c for c in active.columns if c.startswith("eff_")]:
        vals = active[col]
        mean = vals.mean()
        if mean > 0:
            cv.append({
                "muscle": col.replace("eff_", ""),
                "mean_eff_sets_week": round(float(mean), 2),
                "std": round(float(vals.std()), 2),
                "cv": round(float(vals.std() / mean), 2),
                "zero_week_pct_active": round(float((vals == 0).mean() * 100), 1),
            })
    cv = sorted(cv, key=lambda x: x["mean_eff_sets_week"], reverse=True)

    # Top pair correlations excluding duplicates/self.
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if not pd.isna(val):
                pairs.append({"a": cols[i], "b": cols[j], "rho": round(float(val), 2)})
    pairs_pos = sorted(pairs, key=lambda x: x["rho"], reverse=True)[:12]
    pairs_neg = sorted(pairs, key=lambda x: x["rho"])[:12]

    # Lag association: prior 4-week volume/effective sets vs next 4-week lift e1RM change.
    lag_rows = []
    for key, ex in LIFTS.items():
        if f"e1rm_{key}_ffill" not in weekly:
            continue
        pr = weekly[f"e1rm_{key}_ffill"].copy()
        future_change = pr.shift(-4) - pr
        for var in ["sets", "volume_kg", "eff_chest", "eff_back", "eff_shoulders", "eff_biceps", "eff_triceps", "eff_quads", "eff_hams", "eff_glutes"]:
            if var not in weekly:
                continue
            prior = weekly[var].rolling(4, min_periods=2).sum()
            d = pd.DataFrame({"prior": prior, "future": future_change}).dropna()
            d = d[(d.prior > 0) & (d.future.abs() < 50)]
            if len(d) >= 15 and d.prior.nunique() > 3 and d.future.nunique() > 2:
                rho, p = stats.spearmanr(d.prior, d.future)
                if not pd.isna(rho):
                    lag_rows.append({"lift": key, "prior_4w_var": var, "rho_next_4w_e1rm_change": round(float(rho), 2), "p": round(float(p), 3), "n": int(len(d))})
    lag_rows = sorted(lag_rows, key=lambda x: abs(x["rho_next_4w_e1rm_change"]), reverse=True)[:30]

    # Plot weekly balance vs practical set target band.
    target_muscles = ["chest", "back", "shoulders", "biceps", "triceps", "quads", "hams", "glutes", "core"]
    fig, ax = plt.subplots(figsize=(13, 6))
    recent = weekly.tail(52)
    for m in target_muscles:
        col = f"eff_{m}"
        if col in recent:
            ax.plot(recent.index, recent[col].rolling(4, min_periods=1).mean(), lw=1.5, label=m)
    ax.axhspan(10, 20, color="#22c55e", alpha=0.08, label="10-20 set/wk reference band")
    ax.set_title("Last 52 weeks: 4-week average effective sets per muscle")
    ax.set_ylabel("effective sets / week")
    ax.legend(ncols=5, fontsize=8)
    charts["recent_balance"] = savefig("08_recent_effective_set_balance.png")

    return {"weekly_cv": cv, "top_positive_corr": pairs_pos, "top_negative_corr": pairs_neg, "lag_associations": lag_rows}, charts


def issues_and_predictions(sets: pd.DataFrame, sessions: pd.DataFrame, eff: pd.DataFrame, strength: dict[str, Any]) -> dict[str, Any]:
    max_date = sessions.date.max()
    last_52_start = max_date - pd.Timedelta(days=365)
    last_26_start = max_date - pd.Timedelta(days=182)
    recent52 = sessions[sessions.date >= last_52_start]
    recent26 = sessions[sessions.date >= last_26_start]
    eff52 = eff[eff.date >= last_52_start] if not eff.empty else eff
    weeks52 = max(1, math.ceil((max_date - last_52_start).days / 7))
    weeks26 = max(1, math.ceil((max_date - last_26_start).days / 7))
    muscle_week52 = eff52.groupby("muscle").credit.sum().sort_values(ascending=False) / weeks52 if not eff52.empty else pd.Series(dtype=float)

    # Simple unchanged-behavior forecast based on last 26 weeks.
    forecast = {
        "sessions_per_week_last_26w": round(float(len(recent26) / weeks26), 2),
        "sets_per_week_last_26w": round(float(recent26.n_sets.sum() / weeks26), 1),
        "volume_t_per_week_last_26w": round(float(recent26.volume_kg.sum() / 1000 / weeks26), 2),
        "category_sessions_last_26w": recent26.category.value_counts().to_dict(),
        "muscle_eff_sets_week_last_52w": {m: round(float(v), 2) for m, v in muscle_week52.items()},
    }
    # Estimate 12-week strength projection from recent slope, capped and explicitly low-confidence.
    projections = {}
    for key, s in strength.items():
        cur = s.get("recent_180d_best_e1rm") or s.get("best_e1rm")
        slope = s.get("slope_365_e1rm_kg_per_year")
        if cur and slope is not None:
            projections[key] = {
                "current_reference_e1rm": cur,
                "unchanged_12w_projection": round(float(cur + slope * 12 / 52), 1),
                "annual_slope_used": slope,
                "confidence": "low" if abs(slope) < 3 or s.get("current_plateau_days", 0) > 120 else "medium-low",
            }
    forecast["strength_12w_projection_if_unchanged"] = projections

    # Issues ranked by evidence.
    issues = []
    # Legs low: combine quads/hams/glutes.
    leg_sets = sum(muscle_week52.get(m, 0.0) for m in ["quads", "hams", "glutes"])
    push_sets = sum(muscle_week52.get(m, 0.0) for m in ["chest", "shoulders", "triceps"])
    pull_sets = sum(muscle_week52.get(m, 0.0) for m in ["back", "biceps"])
    if leg_sets < 6:
        issues.append({
            "issue": "腿部有效组数显著偏低",
            "evidence": f"最近52周 quads+hams+glutes 合计约 {leg_sets:.1f} 有效组/周；推类合计约 {push_sets:.1f}/周，拉类约 {pull_sets:.1f}/周。",
            "risk": "长期身材比例、下肢力量和整体训练完整性受限；如果目标是更好身材，这是最明确的结构性缺口。",
        })
    if push_sets > 1.35 * max(pull_sets, 1):
        issues.append({
            "issue": "推类相对拉类偏高",
            "evidence": f"最近52周推类/拉类有效组数比约 {push_sets/max(pull_sets,1):.2f}。历史 session 分类也显示推日长期最多。",
            "risk": "胸肩三头刺激充分，但背部/后束/肩胛控制可能不足，影响体态和推类长期进步。",
        })
    # Long plateaus.
    long_lift_plateaus = [f"{LIFT_CN.get(k,k)} {v.get('current_plateau_days')}天" for k, v in strength.items() if v.get("current_plateau_days", 0) >= 120]
    if long_lift_plateaus:
        issues.append({
            "issue": "多个核心动作处于长平台期或低斜率期",
            "evidence": "；".join(long_lift_plateaus[:6]),
            "risk": "历史训练已经产生了大量容量，但近期进步更依赖周期化、弱点肌群和恢复管理，而不是简单重复原模板。",
        })
    outlier_n = int(sessions.duration_outlier.sum())
    if outlier_n:
        issues.append({
            "issue": "训练时长字段存在明显忘记停止计时的 outlier",
            "evidence": f"duration_clean 已排除 >180min 或 <5min；原始 outlier 共 {outlier_n} 次。",
            "risk": "时间效率、热量消耗趋势必须用清洗后数据解读，不能直接相信总时长。",
        })
    cal_n = int(sessions.calorie.notna().sum())
    if cal_n < len(sessions) * 0.4:
        issues.append({
            "issue": "卡路里字段缺失较多",
            "evidence": f"有 calorie 的 session 仅 {cal_n}/{len(sessions)} ({cal_n/len(sessions)*100:.1f}%)。",
            "risk": "可以做方向性效率分析，但不能把卡路里变化当作精确生理消耗。",
        })
    return {"forecast": forecast, "issues": issues}


def table_html(rows: list[dict[str, Any]], columns: list[tuple[str, str]], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    th = "".join(f"<th>{title}</th>" for _, title in columns)
    body = []
    for r in rows:
        tds = "".join(f"<td>{r.get(key, '')}</td>" for key, _ in columns)
        body.append(f"<tr>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def df_to_records_pretty(df: pd.DataFrame, n=20) -> list[dict[str, Any]]:
    out = []
    for _, row in df.head(n).iterrows():
        rec = {}
        for k, v in row.items():
            if isinstance(v, float):
                rec[k] = round(v, 2)
            elif hasattr(v, "date"):
                rec[k] = str(v.date())
            else:
                rec[k] = v
        out.append(rec)
    return out


def generate_report(sets: pd.DataFrame, sessions: pd.DataFrame, eff: pd.DataFrame, charts: dict[str, str], strength: dict[str, Any], efficiency: dict[str, Any], adv: dict[str, Any], pred: dict[str, Any]) -> tuple[str, str]:
    start, end = sessions.date.min().date(), sessions.date.max().date()
    total_days = (sessions.date.max() - sessions.date.min()).days + 1
    active_days = sessions.date.dt.date.nunique()
    bmi = BODYWEIGHT_KG / ((HEIGHT_CM / 100) ** 2)
    clean_duration = sessions.duration_clean.sum()
    rest_logged = (sets.rest_s > 0).mean() * 100 if len(sets) else 0
    top_ex = sets.exercise.value_counts().head(20).reset_index()
    top_ex.columns = ["exercise", "sets"]
    session_cat = sessions.category.value_counts().to_dict()
    year_counts = sessions.groupby("year").size().to_dict()
    recent52 = sessions[sessions.date >= sessions.date.max() - pd.Timedelta(days=365)]
    recent52_sessions_week = len(recent52) / 52

    # Annual summary table.
    annual = sessions.groupby("year").agg(
        sessions=("session_id", "count"),
        sets=("n_sets", "sum"),
        volume_t=("volume_kg", lambda s: round(float(s.sum() / 1000), 1)),
        clean_hours=("duration_clean", lambda s: round(float(s.sum() / 60), 1)),
        avg_density=("density_kg_min", lambda s: round(float(s.mean()), 1)),
    ).reset_index()

    # Strength table rows.
    strength_rows = []
    for key in ["bench", "incline_bench", "squat", "deadlift", "ohp", "row", "curl"]:
        s = strength.get(key)
        if not s:
            continue
        strength_rows.append({
            "lift": s["exercise"],
            "sessions": s["n_sessions"],
            "sets": s["n_sets"],
            "best_set": s["best_set"],
            "best_e1rm": s["best_e1rm"],
            "bw_ratio": s["best_e1rm_per_bw"],
            "best_date": s["best_e1rm_date"],
            "plateau_days": s["current_plateau_days"],
            "slope365": s["slope_365_e1rm_kg_per_year"],
        })

    # Effective set recent balance.
    muscle_recent = pred["forecast"].get("muscle_eff_sets_week_last_52w", {})
    balance_rows = []
    for m in ["chest", "back", "shoulders", "biceps", "triceps", "quads", "hams", "glutes", "core"]:
        v = muscle_recent.get(m, 0.0)
        # Reference: generic hypertrophy best-practice 10-20 hard sets/week for prioritized muscles.
        if v < 4:
            flag = "低"
        elif v < 8:
            flag = "中低"
        elif v <= 20:
            flag = "可用/较高"
        else:
            flag = "很高"
        balance_rows.append({"muscle": MUSCLE_CN.get(m, m), "eff_sets_week": v, "flag": flag})

    # Hidden pattern rows.
    template_rows = []
    for i, t in enumerate(efficiency.get("templates", [])[:12], 1):
        template_rows.append({
            "id": f"T{i}",
            "n": t["n"],
            "span": t["span_days"],
            "template": t["short_signature"],
            "duration_change": f"{t['first_duration_med']} → {t['last_duration_med']} min ({t['duration_delta_pct']}%)",
            "density_change": f"{t['first_density_med']} → {t['last_density_med']} ({t['density_delta_pct']}%)",
            "calorie_n": t["calorie_n"],
            "cal_slope": t["calorie_slope_per_year"],
        })

    # Narrative synthesis.
    issues_html = "".join(
        f"<li><b>{it['issue']}</b><br><span class='evidence'>证据：{it['evidence']}</span><br><span class='risk'>含义：{it['risk']}</span></li>"
        for it in pred["issues"]
    )

    # Breakthrough details.
    breakthrough_blocks = []
    for key, s in strength.items():
        lps = s.get("long_plateaus") or []
        br = s.get("breakthroughs") or []
        lp_txt = "；".join([f"{g['from']}→{g['to']} {g['days']}天 后 +{g['gain_kg']}kg" for g in lps[-4:]]) or "无 >=90天的显著平台记录"
        br_txt = "；".join([f"{b['date']} {b['e1rm']}kg" for b in br[-5:]])
        breakthrough_blocks.append(f"<li><b>{s['exercise']}</b>: 当前平台 {s['current_plateau_days']} 天；长期平台/突破：{lp_txt}；最近PR点：{br_txt}</li>")

    # Advanced stats rows.
    cv_rows = [{"muscle": MUSCLE_CN.get(r["muscle"], r["muscle"]), **{k: v for k, v in r.items() if k != "muscle"}} for r in adv.get("weekly_cv", [])]
    lag_rows = []
    for r in adv.get("lag_associations", [])[:15]:
        lag_rows.append({
            "lift": LIFT_CN.get(r["lift"], r["lift"]),
            "var": r["prior_4w_var"],
            "rho": r["rho_next_4w_e1rm_change"],
            "p": r["p"],
            "n": r["n"],
        })

    # Recommendation plan.
    plan = [
        {"day": "Day 1 上肢-推为主", "content": "卧推/上斜推 6-8组；肩推/侧平举 5-7组；三头 3-5组；保留1-3 RIR。"},
        {"day": "Day 2 下肢-深蹲为主", "content": "深蹲或腿举 5-7组；腿屈伸 3-4组；腿弯举/RDL 4-6组；核心 3-4组。"},
        {"day": "Day 3 上肢-拉为主", "content": "划船/下拉/引体 8-10组；后束/面拉 3-4组；二头 4-6组；可少量胸飞鸟维持。"},
        {"day": "Day 4 下肢+弱点/泵感", "content": "臀腿 8-12组；侧平举/后束 4-6组；手臂或胸弱项 4-6组。"},
    ]

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif; line-height: 1.62; color:#111827; max-width: 1180px; margin: 32px auto; padding: 0 24px; }
    h1, h2, h3 { color:#0f172a; line-height:1.25; }
    h1 { font-size: 34px; border-bottom: 3px solid #111827; padding-bottom: 10px; }
    h2 { margin-top: 36px; border-left: 6px solid #2563eb; padding-left: 12px; }
    h3 { margin-top: 24px; }
    .meta, .note { color:#475569; background:#f8fafc; border:1px solid #e2e8f0; padding:12px 16px; border-radius:10px; }
    .warn { background:#fff7ed; border:1px solid #fed7aa; padding:12px 16px; border-radius:10px; }
    .good { background:#f0fdf4; border:1px solid #bbf7d0; padding:12px 16px; border-radius:10px; }
    .grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; }
    .card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:14px; }
    .card .k { color:#64748b; font-size:13px; }
    .card .v { font-size:22px; font-weight:700; color:#111827; }
    img { max-width:100%; border:1px solid #e5e7eb; border-radius:10px; margin: 8px 0 18px; }
    table { border-collapse: collapse; width:100%; margin: 10px 0 22px; font-size: 14px; }
    th, td { border:1px solid #e5e7eb; padding:8px 9px; text-align:left; vertical-align:top; }
    th { background:#f1f5f9; }
    code { background:#f1f5f9; padding:2px 4px; border-radius:4px; }
    .evidence { color:#334155; }
    .risk { color:#7c2d12; }
    .small { font-size: 13px; color:#475569; }
    """

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>训记训练数据深度报告</title><style>{css}</style></head><body>
    <h1>训记训练数据深度分析报告</h1>
    <div class='meta'>
    生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
    数据源：本地 <code>{PARSED}</code>，未调用 API，未写回训记。<br>
    个人参数：{HEIGHT_CM:.0f}cm，{BODYWEIGHT_KG:.0f}kg，BMI {bmi:.1f}。力量比值以 {BODYWEIGHT_KG:.0f}kg 体重估算。<br>
    方法：Python 负责客观统计、清洗、可视化；文字部分按证据链进行解释，不把启发式分类当作事实本身。
    </div>

    <h2>0. 最短结论：这批数据说明了什么</h2>
    <div class='grid'>
      <div class='card'><div class='k'>覆盖区间</div><div class='v'>{start} → {end}</div></div>
      <div class='card'><div class='k'>训练日 / 总天数</div><div class='v'>{active_days} / {total_days}</div></div>
      <div class='card'><div class='k'>总组数</div><div class='v'>{len(sets):,}</div></div>
      <div class='card'><div class='k'>总容量</div><div class='v'>{sets.tonnage.sum()/1000:,.1f} 吨</div></div>
      <div class='card'><div class='k'>清洗后总时长</div><div class='v'>{clean_duration/60:,.1f} 小时</div></div>
      <div class='card'><div class='k'>最近一年频率</div><div class='v'>{recent52_sessions_week:.2f} 次/周</div></div>
      <div class='card'><div class='k'>不同动作</div><div class='v'>{sets.exercise.nunique()}</div></div>
      <div class='card'><div class='k'>休息记录覆盖</div><div class='v'>{rest_logged:.1f}%</div></div>
    </div>

    <p><b>核心判断：</b>你的历史训练不是“没练够”，而是“刺激分配高度不均 + 近期核心动作进入平台 + 训练结构更像长期偏推的上肢主导体系”。胸、背、手臂有大量真实积累；腿部，尤其腘绳/臀/股四的长期有效组数明显不足。若未来目标是“身材更好”，最优先的不是继续随机加胸/臂动作，而是把训练从偏好驱动转为结构驱动：固定每周下肢暴露、拉推动作平衡、核心动作周期化、用可追踪指标管理平台期。</p>

    <h2>1. 数据质量与限制</h2>
    <ul>
      <li><b>训练记录覆盖好：</b>{active_days} 个训练日，{len(sets):,} 组，足够做长期趋势和分布分析。</li>
      <li><b>时长有 outlier：</b>{int(sessions.duration_outlier.sum())} 次被标记为 &gt;180min 或 &lt;5min，典型原因是忘记停止计时。本文所有时间效率分析使用 <code>duration_clean</code>。</li>
      <li><b>卡路里字段不完整：</b>{int(sessions.calorie.notna().sum())}/{len(sessions)} 次 session 有 calorie，适合做方向性 pattern，不适合作精确代谢结论。</li>
      <li><b>肌群分类是解释层：</b>使用 explicit map + primary=1、secondary=0.5 的有效组数估计。它比单纯按“动作名包含胸/背”更好，但仍不能替代真实肌电、动作技术和接近力竭程度。</li>
      <li><b>e1RM 是估算：</b>采用 Epley 公式 weight×(1+reps/30)。高次数组、器械动作、辅助引体等会带来噪声；本文主要用于趋势，不把它当比赛最大力量。</li>
    </ul>

    <h2>2. 总体训练分布：频率、容量、时长</h2>
    <img src='{charts.get('overview','')}'><br>
    {table_html(df_to_records_pretty(annual), [('year','年份'),('sessions','训练日'),('sets','组数'),('volume_t','容量(吨)'),('clean_hours','清洗后小时'),('avg_density','平均密度')])}
    <p>这张图和年度表说明：你的训练呈现明显的阶段性，而不是线性稳定增长。2022-2023 是高频/高容量阶段；2024 明显下降；2025-2026 有恢复，但结构没有完全重置。容量本身已经不少，问题更集中在“容量投向哪里”和“能否把容量转化为 PR”。</p>

    <h2>3. 力量增长、PR、平台期与突破</h2>
    <img src='{charts.get('strength','')}'><br>
    {table_html(strength_rows, [('lift','动作'),('sessions','出现session'),('sets','组数'),('best_set','最佳组'),('best_e1rm','最佳e1RM'),('bw_ratio','e1RM/体重'),('best_date','最佳日期'),('plateau_days','当前距PR天数'),('slope365','近一年斜率 kg/年')])}
    <h3>平台期和突破点</h3>
    <ul>{''.join(breakthrough_blocks)}</ul>
    <p><b>解读：</b>卧推、深蹲、硬拉、站姿推举都显示出“早期快速进步 → 中期阶梯式突破 → 近期平台”的典型自然训练曲线。平台期本身不是失败，它说明原来的刺激已经从“新手适应”进入“需要更精细变量管理”的阶段。最值得注意的是：如果一个动作长期有大量出现次数但 PR 不再移动，通常不是意志问题，而是至少一个变量卡住：专项强度分布、弱点肌群、恢复、动作技术、或周期化不足。</p>
    <p><b>体重相对力量：</b>按 {BODYWEIGHT_KG:.0f}kg 估算，你的卧推 e1RM/体重、深蹲 e1RM/体重、硬拉 e1RM/体重都已经不是零基础水平；但腿部训练频率与组数不足会限制后续整体体型和下肢比例。身材目标下，腿臀不是可选项。</p>

    <h2>4. 运动模式：三分化、四分化、五分化，以及动作探索</h2>
    <img src='{charts.get('split_year','')}'><br>
    <img src='{charts.get('novelty','')}'><br>
    <p>session 类型计数：{json.dumps({CATEGORY_CN.get(k,k): int(v) for k,v in session_cat.items()}, ensure_ascii=False)}。</p>
    <p><b>结论：</b>历史上更像“推/拉/腿中的推拉主导 + 偶发腿日”，而不是稳定三分化、四分化或五分化。五分化的关键是每个肌群一周至少有可预期的暴露；你的数据里胸、背、臂、肩经常出现，但腿不是稳定支柱。因此，把它称作“长期偏上肢的 PPL 变体”比称作标准三分化更准确。</p>
    <p><b>新动作探索：</b>总共 {sets.exercise.nunique()} 个不同动作。早期新动作引入非常多，后期趋于稳定。这通常是合理的：早期探索动作库，后期应减少随机性，把稳定动作作为可追踪载体。但如果平台期持续，新增动作应服务于弱点，而不是为了新鲜感。</p>

    <h2>5. 肌群平衡：哪里显著少于应有水平</h2>
    <img src='{charts.get('muscle_sets','')}'><br>
    <img src='{charts.get('recent_balance','')}'><br>
    {table_html(balance_rows, [('muscle','肌群'),('eff_sets_week','最近52周有效组/周'),('flag','标记')])}
    <div class='warn'><b>最明确的问题：</b>腿部有效刺激不足。按常见 hypertrophy practice，目标肌群通常需要每周约 10-20 个高质量 hard sets 才更可能接近最大增肌收益；不是每个肌群都必须全年 20 组，但如果目标是身材更好，股四、腘绳、臀长期低于胸/背/臂，体型会出现结构性短板。</div>
    <p>这里引用的 10-20 hard sets/week 是健身文献和实践中的常用锚点（例如 Schoenfeld 等关于训练容量与肌肥大的 dose-response 研究、ACSM resistance training progression model）。它不是机械教条：有效组必须接近力竭、动作稳定、恢复可承受。你的数据价值在于：它显示不是所有肌群都处在同一个训练剂量层级。</p>

    <h2>6. 隐秘 pattern：相似训练下，时间、密度、卡路里是否变化</h2>
    <img src='{charts.get('efficiency_templates','')}'><br>
    {table_html(template_rows, [('id','模板'),('n','次数'),('span','跨度天'),('template','动作组合'),('duration_change','时长中位数变化'),('density_change','密度中位数变化'),('calorie_n','有卡路里样本'),('cal_slope','卡路里年斜率')])}
    <p><b>解读方法：</b>这里不比较“完全不同的一天”，只比较动作组合高度相似的 repeated templates。若同一动作组合下时长下降但容量不降，说明训练密度提高；若时长下降且容量也下降，可能只是训练缩水；若 calorie 下降，需要先看记录覆盖，因为 calorie 样本较少。</p>
    <p><b>总体倾向：</b>你的训练密度在不同阶段变化明显，最近一年并没有简单地“越来越高效”。一些模板下时长缩短，但这不必然代表进步；必须和容量、重量、RIR/接近力竭程度一起看。训记缺少 RPE/RIR，这是未来最值得补充的字段。</p>

    <h2>7. 高级统计：协方差、标准差、相关性、滞后关系</h2>
    <img src='{charts.get('corr','')}'><br>
    <h3>各肌群周有效组数的均值、标准差、变异系数</h3>
    {table_html(cv_rows, [('muscle','肌群'),('mean_eff_sets_week','均值/周'),('std','标准差'),('cv','CV'),('zero_week_pct_active','活跃周为0比例%')])}
    <h3>强相关变量对 Spearman rho</h3>
    {table_html(adv.get('top_positive_corr', [])[:10], [('a','变量A'),('b','变量B'),('rho','rho')])}
    <h3>4周训练变量 vs 后4周 e1RM 变化的滞后相关（探索性，非因果）</h3>
    {table_html(lag_rows, [('lift','动作'),('var','前4周变量'),('rho','rho'),('p','p值'),('n','样本')])}
    <p><b>冷静解释：</b>相关性可以发现“哪些训练变量一起变”，但不能证明“哪个导致 PR”。在你的数据里，很多变量高度共线：训练多的周，组数、容量、时长会同时上升；胸日多时三头/肩也会上升。真正可用的统计结论不是单个 rho，而是结构性事实：暴露不均、腿部低频、平台期长、训练密度和容量并非同步改善。</p>

    <h2>8. 如果不改变，未来会怎样；如果要身材更好，应怎么做</h2>
    <h3>8.1 按最近半年外推的“自然未来”</h3>
    <pre>{json.dumps(pred['forecast'], ensure_ascii=False, indent=2)}</pre>
    <p>这个预测低置信度，因为训练计划不是物理系统，人的目标会改变。但它有一个有用含义：如果继续沿用最近的自然习惯，最可能延续的是“上肢刺激足、腿部不足、核心动作缓慢或停滞”的状态，而不是突然自动变成均衡计划。</p>

    <h3>8.2 面向身材改善的建议计划</h3>
    {table_html(plan, [('day','训练日'),('content','内容')])}
    <p><b>原则：</b></p>
    <ul>
      <li>每周 4 天比随机 2-3 天更适合解决你的结构性问题；如果只能 3 天，采用全身/上-下-全身，而不是继续牺牲腿。</li>
      <li>腿部从“偶发补课”改为“每周两次暴露”：一次深蹲/股四主导，一次臀腿/腘绳主导。</li>
      <li>胸、背维持高质量即可，不必无限加量。胸已经是你的优势刺激区，继续加胸未必带来最高边际收益。</li>
      <li>每个核心动作设 8-12 周 mesocycle：3周渐进 + 1周降载，或 5周渐进 + 1周降载。平台期动作不要每次都用同一种 rep range。</li>
      <li>新增记录字段：RIR/RPE、体重、睡眠、是否力竭、动作备注。没有这些，很多“为什么平台”只能推断，不能证实。</li>
    </ul>

    <h2>9. 发现的问题清单</h2>
    <ol>{issues_html}</ol>

    <h2>10. 文献与 best-practice 锚点</h2>
    <ul>
      <li>ACSM Position Stand: Progression Models in Resistance Training for Healthy Adults. 用作“渐进超负荷、训练频率、周期化”的基础框架。</li>
      <li>Schoenfeld, Ogborn, Krieger 等关于 resistance training volume 与 hypertrophy dose-response 的 meta-analysis。用作“每肌群每周组数”的解释锚点。</li>
      <li>Helms / Zourdos 等关于 RPE/RIR、自我调节训练的实践框架。用作平台期和疲劳管理的解释锚点。</li>
      <li>注意：这些文献给的是群体规律，不替代你的个体反馈。本文优先使用你的真实数据，只把文献作为解释边界。</li>
    </ul>

    <h2>11. 下一步最有价值的数据增强</h2>
    <ol>
      <li>每次训练记录体重，至少每周 3 次；身材目标必须把体重趋势纳入。</li>
      <li>关键组记录 RIR/RPE；平台期分析最缺这个。</li>
      <li>记录动作变式和技术备注，例如卧推停顿/触胸/握距，深蹲深度，硬拉传统/相扑。</li>
      <li>照片或围度：胸围、腰围、臀围、大腿、上臂，每月一次。否则“身材更好”只能通过代理指标推断。</li>
      <li>饮食蛋白和总热量：{HEIGHT_CM:.0f}cm/{BODYWEIGHT_KG:.0f}kg 若目标增肌，训练数据之外的能量盈余是关键约束。</li>
    </ol>

    <div class='note small'>报告文件：{OUT / 'report.html'}<br>指标 JSON：{OUT / 'metrics.json'}<br>图表目录：{IMG}</div>
    </body></html>"""

    # Markdown companion (less styled, same paths).
    md = re.sub(r"<style>.*?</style>", "", html, flags=re.S)
    md = re.sub(r"<[^>]+>", "", md)
    md = md.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    return html, md


def main():
    sets, sessions, eff = load_data()
    if sets.empty or sessions.empty:
        raise SystemExit("No parsed training data found")

    charts = {}
    charts.update(make_overview_charts(sets, sessions, eff))
    strength, c = strength_analysis(sets, sessions)
    charts.update(c)
    efficiency, c = efficiency_analysis(sessions)
    charts.update(c)
    adv, c = advanced_stats(sets, sessions, eff)
    charts.update(c)
    pred = issues_and_predictions(sets, sessions, eff, strength)

    metrics = {
        "data_span": [str(sessions.date.min().date()), str(sessions.date.max().date())],
        "sessions": int(len(sessions)),
        "sets": int(len(sets)),
        "unique_exercises": int(sets.exercise.nunique()),
        "total_volume_t": round(float(sets.tonnage.sum() / 1000), 1),
        "clean_duration_hours": round(float(sessions.duration_clean.sum() / 60), 1),
        "duration_outliers": int(sessions.duration_outlier.sum()),
        "calorie_coverage_pct": round(float(sessions.calorie.notna().mean() * 100), 1),
        "strength": strength,
        "efficiency": efficiency,
        "advanced_stats": adv,
        "prediction_and_issues": pred,
        "charts": charts,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    html, md = generate_report(sets, sessions, eff, charts, strength, efficiency, adv, pred)
    (OUT / "report.html").write_text(html, encoding="utf-8")
    (OUT / "report.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "report_html": str(OUT / "report.html"),
        "report_md": str(OUT / "report.md"),
        "metrics": str(OUT / "metrics.json"),
        "charts": len(charts),
        "sessions": int(len(sessions)),
        "sets": int(len(sets)),
        "span": [str(sessions.date.min().date()), str(sessions.date.max().date())],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
