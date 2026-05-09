"""Flatten parsed JSON into long-form DataFrames for analysis."""
import json, glob, os, re
import pandas as pd
from datetime import datetime

PARSED = os.path.expanduser("~/xunji-data/data/parsed")
OUT = os.path.expanduser("~/xunji-data/analysis/out")
os.makedirs(OUT, exist_ok=True)

# --- muscle-group classifier (heuristic, based on Chinese exercise names) ---
GROUPS = {
    "胸": ["卧推","飞鸟","夹胸","推胸","仰卧","哑铃推","龙门夹胸","臂屈伸"],  # 注意臂屈伸更靠胸/肱三
    "背": ["引体","划船","下拉","硬拉","高位下拉","坐姿","直臂下压","山羊"],
    "腿": ["深蹲","腿举","腿屈","腿弯举","腿伸","箭步","保加利亚","臀推","踢","小腿","提踵","臀冲","硬拉","罗马尼亚"],
    "肩": ["推举","侧平举","前平举","俯身","面拉","耸肩","Y字","哑铃推举","史密斯推举"],
    "臂": ["弯举","臂弯举","锤式","集中","三头","臂屈伸","下压","窄距","绳索下压","过顶"],
    "核心": ["卷腹","平板","仰卧起坐","转体","举腿","侧弯","支撑","悬垂","龙旗"],
    "有氧": ["跑步","椭圆","划船机","动感单车","跳绳","快走","骑行","游泳","HIIT","开合跳"],
}
# priority order — first match wins, but 硬拉 should be 背/腿 disambig later
# we'll do best-effort

def classify(name: str) -> str:
    n = name
    # --- 高优先级显式规则（在通配关键字之前）---
    if "推肩" in n or "肩膀" in n or "后束" in n:
        return "肩"
    if "仰卧起坐" in n or "卷腹" in n or "健腹轮" in n or "龙旗" in n or "悬垂举腿" in n:
        return "核心"
    if "倒蹬" in n or "蹬腿" in n:
        return "腿"
    if "biceps" in n.lower() or "curl" in n.lower():
        return "臂"
    # 优先级规则（更精准）
    if "硬拉" in n and "罗马尼亚" not in n:
        return "背"
    if "罗马尼亚" in n:
        return "腿"
    # 臂屈伸：只有"双杠臂屈伸"算胸/三头，其它（绳索/仰卧/单臂等）归三头(臂)
    if "臂屈伸" in n:
        if "双杠" in n:
            return "胸"
        return "臂"
    # 引体向上 → 背
    if "引体" in n:
        return "背"
    # 抬腿/举腿 → 核心
    if "抬腿" in n or "举腿" in n:
        return "核心"
    # 弯举 → 臂（必须先于胸的"哑铃推"等检查）
    if "弯举" in n:
        return "臂"
    # 三头/下压 → 臂
    if "三头" in n or "下压" in n:
        return "臂"
    for g, kws in GROUPS.items():
        for kw in kws:
            if kw in n:
                return g
    return "其他"

def build():
    rows = []          # set-level
    sess_rows = []     # session-level
    files = sorted(glob.glob(f"{PARSED}/*.json"))
    for fp in files:
        with open(fp) as f:
            day_sessions = json.load(f)
        if not day_sessions:
            continue
        date = os.path.basename(fp).replace(".json","")
        for s in day_sessions:
            title = s.get("raw","").split(",")[2] if "," in s.get("raw","") else ""
            # session header
            dur_min = (s.get("duration_ms") or 0) / 60000
            n_sets = sum(len(e["sets"]) for e in s.get("exercises",[]))
            n_ex = len(s.get("exercises",[]))
            volume = 0.0
            groups_in_sess = set()
            cardio = False
            for e in s.get("exercises",[]):
                g = classify(e["name"])
                groups_in_sess.add(g)
                if g == "有氧":
                    cardio = True
                for st in e.get("sets",[]):
                    w = st.get("weight_kg") or 0
                    r = st.get("reps") or 0
                    rest = st.get("rest_s") or 0
                    volume += w * r
                    rows.append({
                        "date": date,
                        "title": s.get("raw","").split(",")[2] if len(s.get("raw","").split(","))>2 else "",
                        "exercise": e["name"],
                        "group": g,
                        "set_idx": st.get("set"),
                        "weight_kg": w,
                        "reps": r,
                        "rest_s": rest,
                        "tonnage": w*r,
                        "duration_min": dur_min,
                        "session_id": s.get("local_id"),
                    })
            sess_rows.append({
                "date": date,
                "session_id": s.get("local_id"),
                "title": s.get("raw","").split(",")[2] if len(s.get("raw","").split(","))>2 else "",
                "duration_min": dur_min,
                "n_exercises": n_ex,
                "n_sets": n_sets,
                "volume_kg": volume,
                "groups": "/".join(sorted(groups_in_sess)),
                "is_cardio": cardio,
            })
    df_sets = pd.DataFrame(rows)
    df_sess = pd.DataFrame(sess_rows)
    df_sets["date"] = pd.to_datetime(df_sets["date"])
    df_sess["date"] = pd.to_datetime(df_sess["date"])
    df_sets.to_parquet(f"{OUT}/sets.parquet", index=False)
    df_sess.to_parquet(f"{OUT}/sessions.parquet", index=False)
    print(f"sets: {len(df_sets)} rows, sessions: {len(df_sess)} rows")
    print(f"date range: {df_sess.date.min().date()} → {df_sess.date.max().date()}")
    print("\ntop 20 exercises:")
    print(df_sets.exercise.value_counts().head(20))
    print("\ngroup distribution (set-count):")
    print(df_sets.group.value_counts())
    return df_sets, df_sess

if __name__ == "__main__":
    build()
