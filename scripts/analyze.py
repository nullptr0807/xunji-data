"""完整分析 + 出图。"""
import os, json, glob, re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Chinese font
for f in ['Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei','SimHei','Arial Unicode MS','DejaVu Sans']:
    try:
        mpl.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.family'] = f
        break
    except Exception:
        pass
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(os.path.expanduser('~/xunji-data'))
OUT = ROOT/'analysis'
OUT.mkdir(exist_ok=True)

recs=[]
for f in sorted(glob.glob(str(ROOT/'data/parsed/*.json'))):
    for r in json.load(open(f)):
        r['date'] = Path(f).stem
        recs.append(r)

# session-level
sess=[]
for r in recs:
    if not r.get('exercises'): continue
    sets_list = [s for e in r['exercises'] for s in e['sets']]
    vol = sum((s.get('weight_kg') or 0)*(s.get('reps') or 0) for s in sets_list)
    reps_total = sum((s.get('reps') or 0) for s in sets_list)
    rests = [s.get('rest_s') for s in sets_list if s.get('rest_s')]
    sess.append({
        'date': r['date'],
        'duration_min': (r.get('duration_ms') or 0)/60000,
        'calorie': r.get('calorie'),
        'n_exercises': len(r['exercises']),
        'n_sets': len(sets_list),
        'n_reps': reps_total,
        'volume_kg': vol,
        'avg_rest_s': np.mean(rests) if rests else np.nan,
        'exercises': [e['name'] for e in r['exercises']],
    })
df = pd.DataFrame(sess)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
df['weekday'] = df.date.dt.day_name()
df['month'] = df.date.dt.to_period('M').astype(str)

# exercise-level long table
ex_rows=[]
set_rows=[]
for r in recs:
    if not r.get('exercises'): continue
    for e in r['exercises']:
        ex_rows.append({'date':r['date'],'name':e['name'],'n_sets':len(e['sets'])})
        for s in e['sets']:
            set_rows.append({
                'date':r['date'],'exercise':e['name'],'set':s.get('set'),
                'weight_kg':s.get('weight_kg'),'reps':s.get('reps'),
                'rest_s':s.get('rest_s'),
                'volume':(s.get('weight_kg') or 0)*(s.get('reps') or 0),
            })
exdf = pd.DataFrame(ex_rows); exdf['date']=pd.to_datetime(exdf['date'])
sdf = pd.DataFrame(set_rows); sdf['date']=pd.to_datetime(sdf['date'])

# ============= summary text =============
total_days=(df.date.max()-df.date.min()).days+1
lines=[]
lines.append(f"# 训记数据分析报告\n")
lines.append(f"**数据范围**: {df.date.min().date()} → {df.date.max().date()} (共 {total_days} 天)")
lines.append(f"**训练日数**: {len(df)} 天 (训练频率 {len(df)/total_days*100:.1f}%, 平均每周 {len(df)/total_days*7:.2f} 次)")
lines.append(f"**总训练时长**: {df.duration_min.sum():.0f} 分钟 ≈ {df.duration_min.sum()/60:.1f} 小时")
if df.calorie.notna().any():
    lines.append(f"**总卡路里**: {df.calorie.sum():.0f} kcal (有记录的 {df.calorie.notna().sum()} 次)")
lines.append(f"**总训练容量**: {df.volume_kg.sum():,.0f} kg·次")
lines.append(f"**总组数**: {df.n_sets.sum()}  **总次数**: {df.n_reps.sum()}")
lines.append("")

lines.append("## 单次训练分布")
desc=df[['duration_min','calorie','n_exercises','n_sets','volume_kg']].describe().round(1)
lines.append("```")
lines.append(desc.to_string())
lines.append("```\n")

# weekday
lines.append("## 周内分布 (周几更常练)")
wd_order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
wd_zh={'Monday':'周一','Tuesday':'周二','Wednesday':'周三','Thursday':'周四','Friday':'周五','Saturday':'周六','Sunday':'周日'}
wc=df.weekday.value_counts().reindex(wd_order).fillna(0).astype(int)
for k,v in wc.items():
    lines.append(f"- {wd_zh[k]}: {v} 次")
lines.append("")

# monthly trend
lines.append("## 每月训练频次")
mc=df.groupby('month').size()
for m,v in mc.items():
    lines.append(f"- {m}: {v} 次")
lines.append("")

# top exercises
lines.append("## 最常做的动作 (Top 20)")
ex_count=exdf['name'].value_counts().head(20)
for n,c in ex_count.items():
    lines.append(f"- {n}: {c} 次出现")
lines.append("")

# PRs (max weight per exercise) — common compound lifts
def pr_for(name_kw):
    sub=sdf[sdf['exercise'].str.contains(name_kw, na=False)]
    if sub.empty: return None
    sub=sub[sub['weight_kg']>0]
    if sub.empty: return None
    idx=sub['weight_kg'].idxmax()
    row=sub.loc[idx]
    return (row['exercise'],row['weight_kg'],int(row['reps'] or 0),row['date'].date())

lines.append("## 关键动作历史最高重量 (PR)")
keywords=['卧推','深蹲','硬拉','划船','推举','弯举','下拉','引体','臀推','腿举','飞鸟','箭步','登山','蹬']
for kw in keywords:
    pr=pr_for(kw)
    if pr:
        lines.append(f"- **{kw}** → {pr[0]}: {pr[1]:g}kg × {pr[2]}次 ({pr[3]})")
lines.append("")

# longest streak / gap
dates=sorted(df.date.dt.date.unique())
streak=cur=1; max_streak=1
for i in range(1,len(dates)):
    if (dates[i]-dates[i-1]).days==1:
        cur+=1; max_streak=max(max_streak,cur)
    else: cur=1
gaps=[(dates[i]-dates[i-1]).days for i in range(1,len(dates))]
lines.append("## 连续性")
lines.append(f"- 最长连续训练: **{max_streak} 天**")
lines.append(f"- 平均训练间隔: {np.mean(gaps):.1f} 天 (中位 {np.median(gaps):.0f})")
lines.append(f"- 最长断练: **{max(gaps)} 天**")
lines.append("")

# top single sessions
lines.append("## 几个有意思的单次训练")
top_vol=df.nlargest(3,'volume_kg')[['date','duration_min','volume_kg','n_sets','exercises']]
lines.append("**容量最大的 3 次:**")
for _,r in top_vol.iterrows():
    lines.append(f"- {r['date'].date()}  {r['volume_kg']:,.0f} kg·次  {r['n_sets']}组 {r['duration_min']:.0f}分钟  ({', '.join(r['exercises'][:5])})")
top_dur=df.nlargest(3,'duration_min')[['date','duration_min','volume_kg','exercises']]
lines.append("\n**时间最长的 3 次:**")
for _,r in top_dur.iterrows():
    lines.append(f"- {r['date'].date()}  {r['duration_min']:.0f}分钟  {r['volume_kg']:,.0f}kg·次  ({', '.join(r['exercises'][:5])})")

(OUT/'report.md').write_text('\n'.join(lines))
print('\n'.join(lines))

# ============= plots =============
fig, axes = plt.subplots(2,2, figsize=(14,9))
ax=axes[0,0]
df.set_index('date')['volume_kg'].plot(ax=ax,marker='o',ms=3,lw=0.5,color='#d35400')
df.set_index('date')['volume_kg'].rolling(7,min_periods=1).mean().plot(ax=ax,color='#1f3a93',lw=2,label='7次滑动均值')
ax.set_title('训练容量 (kg·次) 随时间')
ax.set_ylabel('volume_kg'); ax.legend(); ax.grid(alpha=.3)

ax=axes[0,1]
df.set_index('date')['duration_min'].plot(ax=ax,marker='o',ms=3,lw=0.5,color='#27ae60')
df.set_index('date')['duration_min'].rolling(7,min_periods=1).mean().plot(ax=ax,color='#1f3a93',lw=2,label='7次滑动均值')
ax.set_title('单次训练时长 (分钟)')
ax.set_ylabel('minutes'); ax.legend(); ax.grid(alpha=.3)

ax=axes[1,0]
mc.plot(kind='bar',ax=ax,color='#8e44ad')
ax.set_title('每月训练次数'); ax.set_xlabel(''); ax.tick_params(axis='x',rotation=45,labelsize=8); ax.grid(alpha=.3,axis='y')

ax=axes[1,1]
wc.index=[wd_zh[x] for x in wc.index]
wc.plot(kind='bar',ax=ax,color='#16a085')
ax.set_title('周内分布'); ax.set_xlabel(''); ax.tick_params(axis='x',rotation=0); ax.grid(alpha=.3,axis='y')

plt.tight_layout()
plt.savefig(OUT/'overview.png',dpi=130,bbox_inches='tight')
plt.close()

# top exercises bar
fig,ax=plt.subplots(figsize=(10,7))
ex_count.iloc[::-1].plot(kind='barh',ax=ax,color='#c0392b')
ax.set_title('最常做的动作 Top 20'); ax.grid(alpha=.3,axis='x')
plt.tight_layout(); plt.savefig(OUT/'top_exercises.png',dpi=130,bbox_inches='tight'); plt.close()

# PR progression for top 3 compound lifts
top3 = ['卧推','深蹲','硬拉','划船','推举']
fig,ax=plt.subplots(figsize=(11,6))
for kw in top3:
    sub=sdf[sdf['exercise'].str.contains(kw,na=False)&(sdf['weight_kg']>0)]
    if sub.empty: continue
    daily_max=sub.groupby('date')['weight_kg'].max()
    if len(daily_max)<2: continue
    daily_max.cummax().plot(ax=ax,marker='.',ms=4,lw=1.2,label=f'{kw} (最高 {daily_max.max():g}kg)')
ax.set_title('主要动作历史最大重量曲线'); ax.set_ylabel('kg'); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT/'pr_progression.png',dpi=130,bbox_inches='tight'); plt.close()

# heatmap calendar
import matplotlib.colors as mcolors
fig,ax=plt.subplots(figsize=(14,3))
all_dates=pd.date_range(df.date.min(),df.date.max())
ts=pd.Series(0,index=all_dates,dtype=float)
ts.loc[df.date]=df.set_index('date')['volume_kg'].values
weeks=((all_dates-all_dates[0]).days)//7
dow=all_dates.weekday
mat=np.full((7,weeks.max()+1),np.nan)
for d,w,wd,v in zip(all_dates,weeks,dow,ts.values):
    mat[wd,w]=v if v>0 else np.nan
im=ax.imshow(mat,aspect='auto',cmap='Oranges')
ax.set_yticks(range(7)); ax.set_yticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
ax.set_xticks([]); ax.set_title('训练日历热力图 (深=容量大)')
plt.colorbar(im,ax=ax,label='volume kg·次')
plt.tight_layout(); plt.savefig(OUT/'calendar.png',dpi=130,bbox_inches='tight'); plt.close()

# save tables
df.to_csv(OUT/'sessions.csv',index=False)
sdf.to_csv(OUT/'sets.csv',index=False)
print(f"\nSaved: {list(p.name for p in OUT.glob('*'))}")
