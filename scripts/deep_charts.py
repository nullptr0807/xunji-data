"""生成深度分析图表。"""
import os, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

for f in ['Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Zen Hei','SimHei']:
    try:
        mpl.font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams['font.family']=f; break
    except: pass
plt.rcParams['axes.unicode_minus']=False

ROOT = Path(os.path.expanduser('~/xunji-data'))
DEEP = ROOT/'analysis'/'deep'
df = pd.read_parquet(DEEP/'sessions.parquet')
sdf = pd.read_parquet(DEEP/'sets.parquet')
deep = json.load(open(DEEP/'deep_stats.json'))

# ============= Chart 1: 月度训练频率 + 容量 =============
fig, ax1 = plt.subplots(figsize=(15,5))
monthly = df.groupby('month').agg(sessions=('date','count'), volume=('volume','sum')).reset_index()
monthly['month_dt'] = pd.to_datetime(monthly['month'])
ax1.bar(monthly['month_dt'], monthly['sessions'], width=20, color='#3498db', alpha=.7, label='次数')
ax1.set_ylabel('训练次数', color='#2980b9')
ax1.tick_params(axis='y', labelcolor='#2980b9')
ax2 = ax1.twinx()
ax2.plot(monthly['month_dt'], monthly['volume']/1000, color='#e74c3c', marker='o', ms=3, label='总容量(吨·次)')
ax2.set_ylabel('总容量 (千 kg·次)', color='#c0392b')
ax2.tick_params(axis='y', labelcolor='#c0392b')
ax1.set_title('每月训练频次 + 总容量趋势 (2021-08 → 2026-05)', fontsize=13)
ax1.grid(alpha=.3)
plt.tight_layout(); plt.savefig(DEEP/'01_monthly_trend.png',dpi=120); plt.close()

# ============= Chart 2: 主要动作 1RM 进步 =============
key_lifts = ['杠铃卧推','深蹲','硬拉','站姿杠铃推举','杠铃划船','杠铃弯举']
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, ex in zip(axes.flat, key_lifts):
    if ex not in deep['pr_data']:
        ax.set_visible(False); continue
    series = deep['pr_data'][ex]['series']
    dates = pd.to_datetime([s[0] for s in series])
    vals = [s[1] for s in series]
    ax.scatter(dates, vals, s=15, alpha=.6, color='#7f8c8d', label='单次估算1RM')
    rolling = pd.Series(vals, index=dates).rolling('60D', min_periods=1).max()
    ax.plot(rolling.index, rolling.values, color='#c0392b', lw=2, label='60日峰值')
    ax.set_title(f'{ex}  峰值 {deep["pr_data"][ex]["peak_1rm"]}kg @ {deep["pr_data"][ex]["peak_date"]}', fontsize=10)
    ax.grid(alpha=.3)
    ax.set_ylabel('估算1RM (kg)')
    ax.tick_params(axis='x', rotation=30, labelsize=8)
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(DEEP/'02_main_lifts_1rm.png',dpi=120); plt.close()

# ============= Chart 3: 肌群容量分布 (年份) =============
mg = pd.DataFrame(deep['muscle_volume_yearly']).T.fillna(0)
group_zh = {'chest':'胸','back':'背','quads':'股四','hams':'腘绳','glutes':'臀',
            'shoulders':'肩','biceps':'二头','triceps':'三头','core':'核心',
            'adductors':'内收','abductors':'外展'}
mg.columns = [group_zh.get(c,c) for c in mg.columns]
# normalize to %
mg_pct = mg.div(mg.sum(axis=1), axis=0)*100
fig, ax = plt.subplots(figsize=(12,6))
mg_pct.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', width=0.85)
ax.set_title('每年肌群训练容量占比 (%)', fontsize=13)
ax.set_ylabel('容量占比 (%)'); ax.set_xlabel('')
ax.legend(loc='center left', bbox_to_anchor=(1, .5), ncol=1, fontsize=9)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.savefig(DEEP/'03_muscle_distribution.png',dpi=120); plt.close()

# ============= Chart 4: 推/拉/腿/混合 频次 =============
split = pd.DataFrame(deep['split_by_year']).T.fillna(0)
order = [c for c in ['push','pull','legs','mixed','core','other'] if c in split.columns]
split = split[order]
zh={'push':'推','pull':'拉','legs':'腿','mixed':'混合','core':'核心','other':'其他'}
split.columns=[zh[c] for c in split.columns]
fig,ax=plt.subplots(figsize=(11,5))
split.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
ax.set_title('每年训练分化类型分布', fontsize=13)
ax.set_ylabel('训练次数'); ax.set_xlabel('')
ax.legend(loc='center left', bbox_to_anchor=(1, .5))
ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.savefig(DEEP/'04_split_distribution.png',dpi=120); plt.close()

# ============= Chart 5: 训练间隔分布 =============
df_sorted = df.sort_values('date')
gaps = df_sorted['date'].diff().dt.days.dropna()
fig,ax=plt.subplots(figsize=(11,4))
ax.hist(gaps[gaps<=20], bins=20, color='#16a085', edgecolor='white')
ax.axvline(gaps.median(), color='#c0392b', ls='--', lw=2, label=f'中位 {gaps.median():.0f}d')
ax.axvline(gaps.mean(), color='#e67e22', ls='--', lw=2, label=f'平均 {gaps.mean():.1f}d')
ax.set_title(f'训练间隔分布 (最长 {int(gaps.max())} 天断练)', fontsize=12)
ax.set_xlabel('距上次训练间隔 (天)'); ax.set_ylabel('频次')
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(DEEP/'05_gap_distribution.png',dpi=120); plt.close()

# ============= Chart 6: 复合 vs 孤立动作占比 =============
comp = pd.Series(deep['compound_share_yearly'])
fig,ax=plt.subplots(figsize=(10,4))
comp.plot(kind='bar', ax=ax, color=['#2c3e50' if v>=0.5 else '#e74c3c' for v in comp.values])
ax.set_title('每年复合动作容量占比 (复合 = 多关节大重量)', fontsize=12)
ax.set_ylabel('占比'); ax.set_xlabel('')
ax.axhline(0.5, color='gray', ls='--', alpha=.5)
ax.set_ylim(0,1)
for i,v in enumerate(comp.values):
    ax.text(i, v+0.02, f'{v*100:.0f}%', ha='center', fontsize=9)
ax.tick_params(axis='x', rotation=0)
plt.tight_layout(); plt.savefig(DEEP/'06_compound_share.png',dpi=120); plt.close()

# ============= Chart 7: 动作丰富度 (新动作 / Shannon 熵) =============
fig, ax1 = plt.subplots(figsize=(10,4))
years = sorted(deep['variety'].keys())
nuniq = [deep['variety'][y]['n_unique'] for y in years]
ent = [deep['variety'][y]['entropy'] for y in years]
ax1.bar(years, nuniq, color='#9b59b6', alpha=.7, label='年度动作数量')
ax1.set_ylabel('独特动作数', color='#7d3c98')
ax1.tick_params(axis='y', labelcolor='#7d3c98')
ax2 = ax1.twinx()
ax2.plot(years, ent, color='#27ae60', marker='o', lw=2, label='Shannon熵 (越高越分散)')
ax2.set_ylabel('Shannon 熵', color='#1e8449')
ax2.tick_params(axis='y', labelcolor='#1e8449')
ax1.set_title('每年训练动作丰富度', fontsize=12)
ax1.set_xticks(years)
plt.tight_layout(); plt.savefig(DEEP/'07_variety.png',dpi=120); plt.close()

# ============= Chart 8: 训练效率 vol/min =============
eff = pd.DataFrame(deep['efficiency_yearly']).T
fig,ax=plt.subplots(figsize=(10,4))
ax.plot(eff.index, eff['mean'], marker='o', lw=2, color='#e74c3c', label='平均')
ax.plot(eff.index, eff['median'], marker='s', lw=2, color='#3498db', label='中位')
ax.set_title('训练效率 (kg·次 / 分钟)', fontsize=12)
ax.set_ylabel('volume per minute'); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(DEEP/'08_efficiency.png',dpi=120); plt.close()

# ============= Chart 9: 卧推/深蹲/硬拉 联合走势 (relative to peak) =============
fig,ax=plt.subplots(figsize=(12,5))
big3 = ['杠铃卧推','深蹲','硬拉']
for ex,color in zip(big3,['#e74c3c','#27ae60','#3498db']):
    s = deep['pr_data'].get(ex)
    if not s: continue
    dates = pd.to_datetime([x[0] for x in s['series']])
    vals = pd.Series([x[1] for x in s['series']], index=dates).rolling('30D', min_periods=1).max()
    ax.plot(vals.index, vals.values, label=f'{ex} (peak {s["peak_1rm"]}kg)', color=color, lw=2)
ax.set_title('三大项 30 日峰值 1RM 走势', fontsize=12)
ax.set_ylabel('估算 1RM (kg)'); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(DEEP/'09_big3.png',dpi=120); plt.close()

# ============= Chart 10: 周内活跃度 vs 年 =============
hm = df.pivot_table(index='weekday', columns='year', values='date', aggfunc='count', fill_value=0)
weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
hm.index = weekdays
fig,ax=plt.subplots(figsize=(8,4))
import matplotlib.colors as mcolors
im=ax.imshow(hm.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns)
ax.set_yticks(range(7)); ax.set_yticklabels(weekdays)
for i in range(7):
    for j in range(len(hm.columns)):
        v=hm.values[i,j]
        ax.text(j,i,int(v),ha='center',va='center',color='white' if v>hm.values.max()*.5 else 'black',fontsize=9)
ax.set_title('训练时间偏好 (热力图: 年份 × 星期几)', fontsize=12)
plt.colorbar(im,ax=ax)
plt.tight_layout(); plt.savefig(DEEP/'10_weekday_heatmap.png',dpi=120); plt.close()

print("done. saved 10 charts to", DEEP)
