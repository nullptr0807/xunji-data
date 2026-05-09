"""Generate 3 modern, magazine-style charts for the Zhihu article."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from xunji.muscle_groups import lookup

# === Modern aesthetic ===
plt.rcParams.update({
    'font.sans-serif': ['WenQuanYi Zen Hei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.facecolor': '#FAFAF7',
    'axes.facecolor': '#FAFAF7',
    'savefig.facecolor': '#FAFAF7',
    'axes.edgecolor': '#2C2C2C',
    'axes.linewidth': 0.8,
    'axes.labelcolor': '#2C2C2C',
    'xtick.color': '#5C5C5C',
    'ytick.color': '#5C5C5C',
    'text.color': '#2C2C2C',
    'axes.titleweight': 'normal',
    'axes.titlepad': 18,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#E5E2D9',
    'grid.linewidth': 0.6,
    'font.size': 11,
})

# Muted modern palette (warm neutral background, deep accents)
INK     = '#1A1A1A'
MUTED   = '#9B9B93'
ACCENT  = '#C8553D'   # warm terracotta (instead of harsh red)
GOOD    = '#6B8E5A'   # muted sage
WARN    = '#D4A574'   # warm sand
COOL    = '#4A6FA5'   # muted slate blue
SOFT    = '#E5E2D9'

OUT = Path('analysis/article_img')
OUT.mkdir(exist_ok=True)

df = pd.read_csv('analysis/sets.csv', parse_dates=['date'])
df['effective'] = ((df['weight_kg'] > 0) & (df['reps'] >= 5)).astype(int)

def primary(name):
    r = lookup(name)
    return r[0][0] if r and r[0] else 'other'

df['mg'] = df['exercise'].apply(primary)
mp = {'chest':'胸','back':'背','shoulders':'肩','triceps':'肱三','biceps':'肱二',
      'quads':'股四','hams':'腘绳','glutes':'臀','core':'核心','adductors':'内收','other':'其他'}

# ============ Chart 1: Muscle group balance ============
cutoff = df['date'].max() - pd.Timedelta(weeks=52)
recent = df[df['date'] >= cutoff]
g = recent.groupby('mg')['effective'].sum() / 52
g = g.sort_values(ascending=True)
g = g[g.index != 'other']
g.index = [mp.get(x, x) for x in g.index]

fig, ax = plt.subplots(figsize=(10, 6.2))

# color: sage if >=10, sand if 6-10, terracotta if <6
def bar_color(v):
    if v >= 10: return GOOD
    if v >= 6:  return WARN
    return ACCENT

colors = [bar_color(v) for v in g.values]
bars = ax.barh(g.index, g.values, color=colors, height=0.62, edgecolor='none', zorder=3)

# Reference bands
ax.axvline(10, color=INK, linestyle=(0, (1, 3)), alpha=0.35, linewidth=1, zorder=2)

# Value labels at end of bars
for bar, v in zip(bars, g.values):
    ax.text(v + 0.35, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}', va='center', fontsize=10.5, color=INK, weight='medium')

# Annotate threshold line
ax.text(10, len(g) - 0.25, '充足下限 10', fontsize=9, color=MUTED, ha='center')

ax.set_xlabel('')
ax.set_xlim(0, 13)
ax.tick_params(axis='y', length=0, labelsize=11.5)
ax.tick_params(axis='x', length=0, labelsize=9.5)
ax.grid(axis='x', alpha=0.5, zorder=1)

# Title block
fig.text(0.04, 0.97, '近 52 周肌群训练量', fontsize=16, weight='semibold', color=INK)
fig.text(0.04, 0.93, '单位：有效组 / 周（绿=达标，黄=临界，红=不足）',
         fontsize=10.5, color=MUTED)

plt.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.10)
plt.savefig(OUT/'01_muscle_balance.png', dpi=200, bbox_inches='tight', pad_inches=0.3)
plt.close()
print('Chart 1 done')

# ============ Chart 2: 卧推 e1RM 进步曲线 ============
bench = df[df['exercise'] == '杠铃卧推'].copy()
bench = bench[(bench['weight_kg'] > 0) & (bench['reps'] > 0)]
bench['e1rm'] = bench['weight_kg'] * (1 + bench['reps']/30)
sess = bench.groupby('date')['e1rm'].max().reset_index()
sess['rolling_pr'] = sess['e1rm'].cummax()

fig, ax = plt.subplots(figsize=(11, 5.6))

last_pr_idx = sess['rolling_pr'].idxmax()
last_pr_date = sess.loc[last_pr_idx, 'date']
last_pr_value = sess.loc[last_pr_idx, 'e1rm']
plateau_days = (sess['date'].max() - last_pr_date).days

# Plateau region — soft warm background band
ax.axvspan(last_pr_date, sess['date'].max(), color=ACCENT, alpha=0.07, zorder=1)

# Scatter — small, low alpha
ax.scatter(sess['date'], sess['e1rm'], s=11, alpha=0.28,
           color=COOL, edgecolor='none', zorder=2, label='单次 best e1RM')

# PR cummax line
ax.plot(sess['date'], sess['rolling_pr'], color=ACCENT, linewidth=2,
        zorder=3, label='历史 PR')

# Mark the plateau start point
ax.scatter([last_pr_date], [last_pr_value], s=50, color=ACCENT,
           edgecolor='#FAFAF7', linewidth=2, zorder=4)

# Annotation for plateau
ax.annotate(f'{last_pr_value:.1f} kg',
            xy=(last_pr_date, last_pr_value),
            xytext=(15, 12), textcoords='offset points',
            fontsize=10.5, color=ACCENT, weight='medium',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))

ax.set_ylabel('e1RM (kg)', fontsize=10.5, color=MUTED)
ax.set_ylim(40, 115)
ax.tick_params(axis='both', length=0)
ax.grid(axis='y', alpha=0.5)
ax.set_axisbelow(True)

# Custom legend — top-left where there's empty space
leg = ax.legend(loc='upper left', frameon=False, fontsize=10)
for text in leg.get_texts():
    text.set_color(INK)

# Title
fig.text(0.04, 0.97, '杠铃卧推 e1RM · 5 年进步曲线', fontsize=16, weight='semibold', color=INK)
fig.text(0.04, 0.93, f'前期稳步上行，最近一年高位震荡（e1RM 上限 ≈ {last_pr_value:.0f} kg）', fontsize=11, color=MUTED)

plt.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.10)
plt.savefig(OUT/'02_bench_e1rm.png', dpi=200, bbox_inches='tight', pad_inches=0.3)
plt.close()
print('Chart 2 done')

# ============ Chart 3: HR 分段诊断 ============
segments = [
    ('卧推',     130, 100),
    ('上斜哑铃', 139, 112),
    ('站姿推举', 142, 117),
    ('双杠臂屈伸', 147, 122),
    ('直杆下压', 154, 130),
]
labels  = [s[0] for s in segments]
peaks   = [s[1] for s in segments]
valleys = [s[2] for s in segments]
x = np.arange(len(segments))

fig, ax = plt.subplots(figsize=(11, 5.8))

# Range fill — soft band
ax.fill_between(x, valleys, peaks, alpha=0.18, color=COOL, zorder=2,
                label='单组峰-谷 HR 范围')

# Peak line + dots
ax.plot(x, peaks, '-', color=ACCENT, linewidth=1.6, zorder=3)
ax.scatter(x, peaks, s=70, color=ACCENT, edgecolor='#FAFAF7',
           linewidth=2, zorder=4, label='峰值 BPM')

# Valley line + dots
ax.plot(x, valleys, '-', color=GOOD, linewidth=1.6, zorder=3)
ax.scatter(x, valleys, s=70, color=GOOD, edgecolor='#FAFAF7',
           linewidth=2, zorder=4, label='组间谷值 BPM')

# Reference line — ideal valley
ax.axhline(110, color=GOOD, linestyle=(0, (2, 4)), alpha=0.5, linewidth=1, zorder=1)
ax.text(len(x) - 0.5, 110.5, '理想谷值 ≤ 110', fontsize=9, color=GOOD,
        ha='right', va='bottom', alpha=0.85)

# Value labels
for xi, v in zip(x, valleys):
    ax.text(xi, v - 5, str(v), ha='center', color=GOOD,
            fontsize=10, weight='medium')
for xi, p in zip(x, peaks):
    ax.text(xi, p + 3, str(p), ha='center', color=ACCENT,
            fontsize=10, weight='medium')

# HRR anomaly annotation — moved to upper-left empty area
ax.annotate('训练后 HR 反升\nHRR-1 = -3 bpm',
            xy=(4, 154), xytext=(0.15, 165),
            fontsize=10.5, color=ACCENT, weight='medium',
            ha='left',
            arrowprops=dict(arrowstyle='-', color=ACCENT, lw=1, alpha=0.6,
                            connectionstyle='arc3,rad=-0.15'))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11.5, color=INK)
ax.set_ylabel('心率 (BPM)', fontsize=10.5, color=MUTED)
ax.set_ylim(92, 168)
ax.tick_params(axis='both', length=0)
ax.grid(axis='y', alpha=0.5)
ax.set_axisbelow(True)

# Legend moved to upper-right (empty area, away from data)
leg = ax.legend(loc='lower right', frameon=False, fontsize=9.5,
                bbox_to_anchor=(1.0, -0.18), ncol=3)
for text in leg.get_texts():
    text.set_color(MUTED)

fig.text(0.04, 0.97, '5/8 推日 · 心率分段诊断', fontsize=16, weight='semibold', color=INK)
fig.text(0.04, 0.93, '组间谷值持续抬升 — 累积疲劳信号', fontsize=11, color=MUTED)

plt.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.10)
plt.savefig(OUT/'03_hr_segments.png', dpi=200, bbox_inches='tight', pad_inches=0.3)
plt.close()
print('Chart 3 done')

print('Saved to', OUT.resolve())
