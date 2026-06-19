"""跑全套深度统计。输出 JSON + 多张图，给 LLM 写报告用。"""
import os, json, glob, re, math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from xunji.muscle_groups import lookup, MAP

# Chinese font
for f in ['Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei','SimHei']:
    try:
        mpl.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.family'] = f
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'analysis'/'deep'
OUT.mkdir(parents=True, exist_ok=True)

# ============= load data =============
recs=[]
for f in sorted(glob.glob(str(ROOT/'data/parsed/*.json'))):
    for r in json.load(open(f)):
        if r.get('exercises'):
            r['date']=Path(f).stem
            recs.append(r)

set_rows=[]
for r in recs:
    for e in r['exercises']:
        primary, secondary = lookup(e['name'])
        for s in e['sets']:
            set_rows.append({
                'date': r['date'],
                'exercise': e['name'],
                'primary': primary[0] if primary else 'unknown',
                'all_primary': primary,  # for multi-group lifts like 硬拉
                'set': s.get('set'),
                'weight': s.get('weight_kg') or 0,
                'reps': s.get('reps') or 0,
                'rest_s': s.get('rest_s'),
            })
sdf = pd.DataFrame(set_rows)
sdf['date'] = pd.to_datetime(sdf['date'])
sdf['volume'] = sdf['weight'] * sdf['reps']
# Epley 1RM estimator: 1RM = w * (1 + reps/30)
sdf['est_1rm'] = sdf['weight'] * (1 + sdf['reps']/30)

# session-level
sess_rows=[]
for r in recs:
    sets=[s for e in r['exercises'] for s in e['sets']]
    vol=sum((s.get('weight_kg') or 0)*(s.get('reps') or 0) for s in sets)
    rests=[s.get('rest_s') for s in sets if s.get('rest_s')]
    primary_in_session = []
    for e in r['exercises']:
        p,_ = lookup(e['name'])
        primary_in_session.extend(p)
    primary_set = list(set(primary_in_session))
    sess_rows.append({
        'date': r['date'],
        'duration_min': (r.get('duration_ms') or 0)/60000,
        'calorie': r.get('calorie'),
        'n_exercises': len(r['exercises']),
        'n_sets': len(sets),
        'volume': vol,
        'avg_rest_s': np.mean(rests) if rests else np.nan,
        'primary_groups': primary_set,
        'exercises_set': set(e['name'] for e in r['exercises']),
    })
df = pd.DataFrame(sess_rows)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['weekday'] = df.date.dt.dayofweek
df['month'] = df.date.dt.to_period('M').astype(str)
df['year'] = df.date.dt.year
# clean obvious time outliers (forgot stop watch): cap at 180min
df['duration_clean'] = df['duration_min'].clip(upper=180)

# ============= deep stats =============

deep = {}

# 1. overall
deep['overall'] = {
    'date_range': [str(df.date.min().date()), str(df.date.max().date())],
    'span_days': (df.date.max()-df.date.min()).days+1,
    'sessions': len(df),
    'avg_per_week': len(df) / ((df.date.max()-df.date.min()).days+1) * 7,
    'total_duration_h': df.duration_clean.sum()/60,
    'total_volume_kg': df.volume.sum(),
    'total_sets': int(df.n_sets.sum()),
    'unique_exercises': sdf['exercise'].nunique(),
}

# 2. yearly trend
yearly = df.groupby('year').agg(
    sessions=('date','count'),
    avg_volume=('volume','mean'),
    avg_duration=('duration_clean','mean'),
    total_volume=('volume','sum'),
    avg_sets=('n_sets','mean'),
).round(1)
deep['yearly'] = yearly.to_dict('index')

# 3. monthly
monthly = df.groupby('month').agg(sessions=('date','count'), volume=('volume','sum')).round(0)
deep['monthly_sessions'] = monthly['sessions'].to_dict()

# 4. PR estimation per exercise
pr_data = {}
for ex in sdf['exercise'].unique():
    sub = sdf[(sdf['exercise']==ex) & (sdf['weight']>0)]
    if len(sub) < 3: continue
    # daily best estimated 1RM
    daily = sub.groupby('date')['est_1rm'].max()
    if len(daily) < 3: continue
    pr_data[ex] = {
        'n_sessions': int(daily.shape[0]),
        'first_date': str(daily.index.min().date()),
        'last_date': str(daily.index.max().date()),
        'first_1rm': round(daily.iloc[0], 1),
        'last_1rm': round(daily.iloc[-1], 1),
        'peak_1rm': round(daily.max(), 1),
        'peak_date': str(daily.idxmax().date()),
        'cur_vs_peak': round(daily.iloc[-1]/daily.max()*100, 1),
        # plateau: longest stretch without breaking running max
        'days_since_peak': (daily.index[-1] - daily.idxmax()).days,
        'series': [(str(d.date()), round(v,1)) for d,v in daily.items()],
    }

deep['pr_data'] = pr_data

# 5. muscle group volume distribution
mg_vol = defaultdict(lambda: defaultdict(float))  # year -> group -> volume
for _, row in sdf.iterrows():
    y = row['date'].year
    if row['volume'] == 0: continue
    # share volume across all primary groups (e.g. 硬拉 -> back+hams+glutes)
    primary = row['all_primary']
    if not primary or primary == ['unknown']: continue
    share = row['volume'] / len(primary)
    for g in primary:
        mg_vol[y][g] += share

# also overall
mg_vol_overall = defaultdict(float)
for y, d in mg_vol.items():
    for g,v in d.items():
        mg_vol_overall[g] += v
deep['muscle_volume_overall'] = dict(mg_vol_overall)
deep['muscle_volume_yearly'] = {y: dict(d) for y,d in mg_vol.items()}

# 6. body-part split frequency (push/pull/legs etc)
# classify each session by which muscle groups are primary
def classify_session(groups):
    s = set(groups)
    upper_push = {'chest','shoulders','triceps'}
    upper_pull = {'back','biceps'}
    legs = {'quads','hams','glutes','adductors','abductors'}
    has_push = bool(s & upper_push)
    has_pull = bool(s & upper_pull)
    has_legs = bool(s & legs)
    has_core = 'core' in s
    nontrivial = [k for k,v in [('push',has_push),('pull',has_pull),('legs',has_legs)] if v]
    if len(nontrivial) >= 2: return 'mixed'
    if has_push and not has_pull and not has_legs: return 'push'
    if has_pull and not has_push and not has_legs: return 'pull'
    if has_legs and not has_push and not has_pull: return 'legs'
    if has_core and not has_push and not has_pull and not has_legs: return 'core'
    return 'other'

df['split_type'] = df['primary_groups'].apply(classify_session)
deep['split_distribution'] = df['split_type'].value_counts().to_dict()
deep['split_by_year'] = df.groupby(['year','split_type']).size().unstack(fill_value=0).to_dict('index')

# 7. workout frequency / consistency
dates = sorted(df.date.dt.date.unique())
gaps=[(dates[i]-dates[i-1]).days for i in range(1,len(dates))]
streaks=[]
cur=1
for g in gaps:
    if g==1: cur+=1
    else:
        streaks.append(cur); cur=1
streaks.append(cur)
deep['streak'] = {
    'max': max(streaks),
    'mean_gap': float(np.mean(gaps)),
    'median_gap': float(np.median(gaps)),
    'max_gap': int(max(gaps)),
    'n_gaps_over_14d': sum(1 for g in gaps if g>14),
}

# 8. novelty over time: when each exercise first appeared
ex_first = sdf.groupby('exercise')['date'].min().sort_values()
novelty_per_year = ex_first.dt.year.value_counts().sort_index()
deep['novelty_per_year'] = novelty_per_year.to_dict()

# 9. movement variety (Shannon entropy of exercise mix per year)
def entropy(counts):
    total = sum(counts.values())
    if total==0: return 0
    return -sum((c/total)*math.log2(c/total) for c in counts.values() if c>0)

variety={}
for y in sorted(df.year.unique()):
    yr_sets = sdf[sdf.date.dt.year==y]
    cnt = yr_sets['exercise'].value_counts().to_dict()
    variety[int(y)] = {'n_unique':len(cnt), 'entropy':round(entropy(cnt),2), 'top5_share':sum(sorted(cnt.values(),reverse=True)[:5])/sum(cnt.values()) if cnt else 0}
deep['variety'] = variety

# 10. efficiency: volume per minute over time
df['vol_per_min'] = df['volume'] / df['duration_clean'].replace(0,np.nan)
deep['efficiency_yearly'] = df.groupby('year')['vol_per_min'].agg(['mean','median']).round(1).to_dict('index')

# 11. rest time distribution
rest_yearly = sdf[sdf['rest_s'].notna()].groupby(sdf['date'].dt.year)['rest_s'].agg(['mean','median','count']).round(1)
deep['rest_yearly'] = rest_yearly.to_dict('index')

# 12. compound vs isolation ratio
compound_lifts = {'杠铃卧推','上斜杠铃卧推','哑铃卧推','上斜哑铃卧推','下斜杠铃卧推','上斜史密斯机卧推',
                  '硬拉','杠铃划船','哑铃划船','坐姿划船','器械划船','V-bar划船',
                  '深蹲','史密斯机深蹲','哈克机深蹲','哑铃酒杯深蹲','腿举','器械倒蹬',
                  '站姿杠铃推举','哑铃推肩','悍马机坐姿推举','史密斯机推举',
                  '引体向上','引体向上（辅助）','宽距下拉','窄距下拉','悍马机正手下拉','悍马机下拉',
                  '双杠臂屈伸','双杠臂屈伸（辅助）','悍马机推胸','器械推胸'}
sdf['is_compound'] = sdf['exercise'].isin(compound_lifts)
comp_yearly = sdf.groupby([sdf.date.dt.year,'is_compound'])['volume'].sum().unstack(fill_value=0)
comp_yearly['compound_share'] = comp_yearly.get(True,0) / (comp_yearly.get(True,0)+comp_yearly.get(False,0))
deep['compound_share_yearly'] = comp_yearly['compound_share'].round(3).to_dict()

# save
import json as J
(OUT/'deep_stats.json').write_text(J.dumps(deep, ensure_ascii=False, indent=2, default=str))
print(f"saved deep_stats.json ({(OUT/'deep_stats.json').stat().st_size/1024:.1f} KB)")
print(f"sessions: {len(df)}, sets: {len(sdf)}, exercises: {sdf['exercise'].nunique()}")
print(f"unknown exercises: {sdf[sdf['primary']=='unknown']['exercise'].unique()}")

# also save dataframes for the LLM phase. Parquet is optional; pickle is the
# dependency-free fallback used by deep_charts.py.
df.to_pickle(OUT/'sessions.pkl')
sdf.to_pickle(OUT/'sets.pkl')
try:
    df.to_parquet(OUT/'sessions.parquet')
    sdf.to_parquet(OUT/'sets.parquet')
except ImportError as e:
    print(f"[warn] parquet skipped: {e.__class__.__name__}. Using PKL fallback.")
