"""Generate publication-quality charts for the report."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import os, json

# ---- Font ----
for cand in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]:
    if os.path.exists(cand):
        fm.fontManager.addfont(cand)
        prop = FontProperties(fname=cand)
        mpl.rcParams["font.sans-serif"] = [prop.get_name(), "DejaVu Sans"]
        mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False

# ---- Pro style ----
DPI = 200
PALETTE = {
    "胸": "#E63946",   # warm red
    "背": "#2A9D8F",   # teal
    "腿": "#F4A261",   # warm orange
    "肩": "#264653",   # dark slate
    "臂": "#9D4EDD",   # purple
    "核心": "#A8DADC", # light cyan
    "其他": "#CBD5E1",
}
ACCENT = "#1D3557"     # deep navy — main accent
ACCENT_2 = "#E63946"   # red — emphasis
NEUTRAL = "#475569"    # slate gray for text/axes
GRID = "#E2E8F0"
BG = "#FFFFFF"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": "#94A3B8",
    "axes.labelcolor": NEUTRAL,
    "axes.titlecolor": "#0F172A",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlepad": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
    "xtick.color": NEUTRAL,
    "ytick.color": NEUTRAL,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.facecolor": BG,
    "savefig.bbox": "tight",
    "savefig.dpi": DPI,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
IMG = os.path.join(ROOT, "analysis", "img")
os.makedirs(IMG, exist_ok=True)
# clean previous
for f in os.listdir(IMG):
    os.remove(os.path.join(IMG, f))

sets = pd.read_parquet(f"{OUT}/sets.parquet")
sess = pd.read_parquet(f"{OUT}/sessions.parquet")
sets["year_month"] = sets["date"].dt.to_period("M")
sess["year_month"] = sess["date"].dt.to_period("M")
sets["year"] = sets["date"].dt.year
sess["year"] = sess["date"].dt.year

stats_dict = {}
stats_dict["span"] = f"{sess.date.min().date()} → {sess.date.max().date()}"
stats_dict["total_sessions"] = int(len(sess))
stats_dict["total_sets"] = int(len(sets))
stats_dict["total_volume_t"] = round(float(sets.tonnage.sum())/1000, 1)
stats_dict["unique_exercises"] = int(sets.exercise.nunique())
stats_dict["total_train_minutes"] = int(sess.duration_min.sum())

def annotate_value(ax, x, y, text, **kw):
    ax.annotate(text, (x,y), fontsize=9, color=NEUTRAL, **kw)

def add_subtitle(fig, sub):
    fig.text(0.01, 0.97, sub, fontsize=9, color=NEUTRAL, style="italic", transform=fig.transFigure)

# ============ 1. 训练频率 ============
fig, ax = plt.subplots(2,1, figsize=(13,7), sharex=True)
monthly = sess.groupby("year_month").size()
monthly.index = monthly.index.to_timestamp()
bars = ax[0].bar(monthly.index, monthly.values, width=22, color=ACCENT, alpha=0.85, edgecolor="white", linewidth=0.5)
# highlight peak/trough
peak_idx = monthly.values.argmax()
ax[0].bar(monthly.index[peak_idx], monthly.values[peak_idx], width=22, color=ACCENT_2)
ax[0].set_title("每月训练次数 · Sessions per Month")
ax[0].set_ylabel("次数")
ax[0].annotate(f"峰值 {monthly.values[peak_idx]} 次",
               (monthly.index[peak_idx], monthly.values[peak_idx]),
               textcoords="offset points", xytext=(0,8), ha="center",
               fontsize=9, fontweight="bold", color=ACCENT_2)

daily = sess.groupby(sess.date.dt.date).size()
daily.index = pd.to_datetime(daily.index)
full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0)
roll = full.rolling(28).sum()
ax[1].fill_between(roll.index, 0, roll.values, color=ACCENT, alpha=0.18)
ax[1].plot(roll.index, roll.values, color=ACCENT, lw=2)
ax[1].axhline(12, color="#F4A261", ls="--", lw=1.2, label="3 次/周")
ax[1].axhline(16, color=ACCENT_2, ls="--", lw=1.2, label="4 次/周")
ax[1].set_title("28 天滚动训练频率 · Rolling 28-Day Training Days")
ax[1].set_ylabel("天数 / 28 天")
ax[1].legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{IMG}/01_frequency.png"); plt.close()

stats_dict["sessions_per_year"] = sess.groupby("year").size().to_dict()

# ============ 2. 容量按肌群堆叠 ============
vol = sets.groupby([sets["date"].dt.to_period("M"),"group"])["tonnage"].sum().unstack(fill_value=0)
vol.index = vol.index.to_timestamp()
keep = ["腿","肩","核心","臂","背","胸"]   # bottom→top order: emphasize 胸 on top
vol = vol[[g for g in keep if g in vol.columns]]
fig, ax = plt.subplots(figsize=(13,5.8))
vol_t = vol/1000
colors = [PALETTE[c] for c in vol_t.columns]
ax.stackplot(vol_t.index, vol_t.values.T, labels=vol_t.columns, colors=colors, alpha=0.92, edgecolor="white", linewidth=0.5)
ax.set_title("月度训练容量按肌群堆叠 · Monthly Volume by Muscle Group")
ax.set_ylabel("容量 (吨)")
ax.legend(title="肌群", loc="upper left", ncol=6, frameon=False)
plt.tight_layout(); plt.savefig(f"{IMG}/02_volume_stack.png"); plt.close()

stats_dict["volume_by_group_t"] = (sets.groupby("group").tonnage.sum()/1000).round(1).to_dict()
stats_dict["sets_by_group"] = sets.group.value_counts().to_dict()

# ============ 3. PR 曲线 (大图) ============
core_lifts = ["杠铃卧推","深蹲","硬拉","站姿杠铃推举","杠铃划船","上斜杠铃卧推","杠铃弯举"]
fig, axes = plt.subplots(3,3, figsize=(15,12))
axes = axes.flatten()
pr_summary = {}
for i, ex in enumerate(core_lifts):
    ax = axes[i]
    d = sets[sets.exercise==ex].copy()
    if d.empty:
        ax.set_visible(False); continue
    daily_max = d.groupby("date")["weight_kg"].max()
    rolling_pr = daily_max.cummax()
    d["e1rm"] = d.weight_kg * (1 + d.reps/30)
    e1rm_daily = d.groupby("date")["e1rm"].max()
    e1rm_pr = e1rm_daily.cummax()
    ax.scatter(daily_max.index, daily_max.values, s=10, alpha=0.30, color=ACCENT, label="单日最大")
    ax.plot(rolling_pr.index, rolling_pr.values, color=ACCENT, lw=2.2, label="实测 PR")
    ax.plot(e1rm_pr.index, e1rm_pr.values, color=ACCENT_2, lw=1.8, ls="--", alpha=0.85, label="估算 1RM")
    # endpoint label
    ax.annotate(f"{rolling_pr.iloc[-1]:.0f} kg",
                (rolling_pr.index[-1], rolling_pr.iloc[-1]),
                textcoords="offset points", xytext=(-4,8), ha="right",
                fontsize=10, fontweight="bold", color=ACCENT)
    ax.set_title(f"{ex}", fontsize=11)
    ax.set_ylabel("kg", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    pr_summary[ex] = {
        "first_date": str(d.date.min().date()),
        "first_max_kg": float(daily_max.iloc[0]),
        "current_pr_kg": float(rolling_pr.iloc[-1]),
        "current_e1rm_kg": float(e1rm_pr.iloc[-1]),
        "n_sets": int(len(d)),
        "n_sessions": int(d.date.nunique()),
    }
for j in range(len(core_lifts), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("核心复合动作 PR 进展", fontsize=15, y=0.995, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{IMG}/03_pr_curves.png"); plt.close()
stats_dict["pr"] = pr_summary

# ============ 5. 肌群比例 ============
fig, ax = plt.subplots(1,2, figsize=(14,5.8), gridspec_kw={"width_ratios":[1,1.3]})
g_total = sets.groupby("group").size().reindex(["胸","背","臂","肩","核心","腿","其他"]).fillna(0)
g_total = g_total[g_total>0]
colors = [PALETTE.get(g, "#CBD5E1") for g in g_total.index]
# donut
wedges, texts, autotexts = ax[0].pie(g_total, labels=g_total.index, autopct="%1.1f%%",
                                      colors=colors, startangle=90, pctdistance=0.78,
                                      wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2))
for t in autotexts:
    t.set_fontsize(10); t.set_color("white"); t.set_fontweight("bold")
for t in texts:
    t.set_fontsize(11); t.set_fontweight("bold")
ax[0].set_title("整体训练组数分布")
# add center text
ax[0].text(0,0,f"{int(g_total.sum())}\n组", ha="center", va="center",
           fontsize=14, fontweight="bold", color="#0F172A")

# yearly stacked
yg = sets.groupby([sets.year, "group"]).size().unstack(fill_value=0)
order = ["胸","背","臂","肩","核心","腿","其他"]
yg = yg[[c for c in order if c in yg.columns]]
yg_pct = yg.div(yg.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(yg_pct))
for col in yg_pct.columns:
    ax[1].bar(yg_pct.index.astype(str), yg_pct[col], bottom=bottom,
              color=PALETTE.get(col,"#CBD5E1"), label=col, edgecolor="white", linewidth=1.2, width=0.7)
    # write percentages > 5%
    for i, val in enumerate(yg_pct[col]):
        if val >= 5:
            ax[1].text(i, bottom[i] + val/2, f"{val:.0f}%", ha="center", va="center",
                       color="white", fontsize=9, fontweight="bold")
    bottom += yg_pct[col].values
ax[1].set_title("年度肌群占比")
ax[1].set_ylabel("%")
ax[1].set_ylim(0, 100)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,-0.06), ncol=7, frameon=False)
ax[1].grid(False)
plt.tight_layout(); plt.savefig(f"{IMG}/05_group_balance.png"); plt.close()
stats_dict["group_pct_overall"] = (g_total/g_total.sum()*100).round(1).to_dict()

# ============ 6. 分化模式 ============
def primary_group(sess_id):
    d = sets[sets.session_id==sess_id]
    if d.empty: return "未知"
    return d.group.value_counts().idxmax()
sess["primary_group"] = sess.session_id.map({sid:primary_group(sid) for sid in sess.session_id.unique()})
def bro_or_ppl(g):
    if g in ["胸","肩","臂"]: return "推日"
    if g == "背": return "拉日"
    if g == "腿": return "腿日"
    if g == "核心": return "核心日"
    return "其他"
sess["split_type"] = sess.primary_group.apply(bro_or_ppl)
fig, ax = plt.subplots(1,2, figsize=(13,5))
pg = sess.primary_group.value_counts()
bars = ax[0].barh(pg.index[::-1], pg.values[::-1],
                   color=[PALETTE.get(g,"#94A3B8") for g in pg.index[::-1]],
                   edgecolor="white", linewidth=1)
for i, (v, name) in enumerate(zip(pg.values[::-1], pg.index[::-1])):
    ax[0].text(v+5, i, str(int(v)), va="center", fontsize=10, fontweight="bold", color=NEUTRAL)
ax[0].set_title("Session 主导肌群分布")
ax[0].set_xlabel("session 数")
ax[0].grid(axis="y", visible=False)

st = sess.split_type.value_counts()
split_colors = {"推日": ACCENT_2, "拉日": "#2A9D8F", "腿日": "#F4A261", "核心日": "#A8DADC", "其他":"#CBD5E1"}
bars = ax[1].barh(st.index[::-1], st.values[::-1],
                   color=[split_colors.get(s,"#94A3B8") for s in st.index[::-1]],
                   edgecolor="white", linewidth=1)
total = st.sum()
for i, (v, name) in enumerate(zip(st.values[::-1], st.index[::-1])):
    ax[1].text(v+5, i, f"{int(v)}  ({v/total*100:.1f}%)", va="center", fontsize=10, fontweight="bold", color=NEUTRAL)
ax[1].set_title("PPL 分化匹配 (推/拉/腿)")
ax[1].set_xlabel("session 数")
ax[1].grid(axis="y", visible=False)
# annotation: 腿日仅 N
leg_n = st.get("腿日", 0)
ax[1].text(0.5, -0.2, f"⚠ 腿日仅占 {leg_n/total*100:.1f}% — 健康水平应 ≥ 25%",
           transform=ax[1].transAxes, ha="center", fontsize=10, color=ACCENT_2, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{IMG}/06_split.png"); plt.close()
stats_dict["split_counts"] = sess.split_type.value_counts().to_dict()
stats_dict["primary_group_counts"] = sess.primary_group.value_counts().to_dict()

# ============ 7. 多样性 ============
sets_sorted = sets.sort_values("date")
first_seen = sets_sorted.groupby("exercise").date.min().sort_values()
cum = first_seen.reset_index()
cum["n"] = range(1, len(cum)+1)
fig, ax = plt.subplots(figsize=(13,4.5))
ax.fill_between(cum.date, 0, cum.n, color=ACCENT, alpha=0.15)
ax.plot(cum.date, cum.n, lw=2.2, color=ACCENT)
ax.scatter(cum.date.iloc[-1], cum.n.iloc[-1], color=ACCENT_2, s=80, zorder=5)
ax.annotate(f"累计 {cum.n.iloc[-1]} 个动作",
            (cum.date.iloc[-1], cum.n.iloc[-1]),
            textcoords="offset points", xytext=(-12,12), ha="right",
            fontsize=11, fontweight="bold", color=ACCENT_2)
ax.set_title("累计尝试过的不同动作数")
ax.set_ylabel("动作数")
plt.tight_layout(); plt.savefig(f"{IMG}/07_diversity.png"); plt.close()
stats_dict["new_exercises_per_year"] = first_seen.dt.year.value_counts().sort_index().to_dict()

# ============ 9. 平台期检测 ============
ex = "杠铃卧推"
d = sets[sets.exercise==ex].copy()
e1 = d.assign(e1rm=d.weight_kg*(1+d.reps/30)).groupby("date").e1rm.max()
e1_pr = e1.cummax()
diffs = e1_pr.diff().fillna(0)
gaps = []
cur_start = None
for date, dval in diffs.items():
    if dval == 0:
        if cur_start is None:
            cur_start = date
    else:
        if cur_start is not None:
            gap_days = (date - cur_start).days
            if gap_days >= 60:
                gaps.append((cur_start, date, gap_days))
            cur_start = None
if cur_start is not None:
    gap_days = (e1_pr.index[-1] - cur_start).days
    if gap_days >= 60:
        gaps.append((cur_start, e1_pr.index[-1], gap_days))
fig, ax = plt.subplots(figsize=(13,5))
for s,e,g in gaps:
    ax.axvspan(s, e, color="#FCA5A5", alpha=0.25)
ax.plot(e1_pr.index, e1_pr.values, color=ACCENT, lw=2.5)
# label longest plateau
if gaps:
    longest = max(gaps, key=lambda x: x[2])
    mid = longest[0] + (longest[1] - longest[0])/2
    ax.annotate(f"最长平台期\n{longest[2]} 天",
                (mid, e1_pr.values.max()*0.55), ha="center",
                fontsize=11, fontweight="bold", color=ACCENT_2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=ACCENT_2, lw=1))
ax.set_title(f"{ex} · 估算 1RM 平台期检测  (红色 = ≥60 天无 PR)")
ax.set_ylabel("e1RM (kg)")
plt.tight_layout(); plt.savefig(f"{IMG}/09_plateau_bench.png"); plt.close()
stats_dict["bench_plateaus"] = [{"from":str(s.date()),"to":str(e.date()),"days":g} for s,e,g in gaps]

# ============ 11. PR drivers ============
ex = "杠铃卧推"
d = sets[sets.exercise==ex].copy()
d["ym"] = d.date.dt.to_period("M")
mvol = d.groupby("ym").tonnage.sum()
msessions = d.groupby("ym").session_id.nunique()
mpr = d.assign(e1=d.weight_kg*(1+d.reps/30)).groupby("ym").e1.max()
ms = pd.DataFrame({"vol":mvol,"freq":msessions,"e1rm_max":mpr}).dropna()
ms["pr_delta"] = ms.e1rm_max.diff()
fig, ax = plt.subplots(1,2, figsize=(13,5))
ax[0].axhline(0, color="#94A3B8", lw=1)
ax[0].scatter(ms.freq, ms.pr_delta, s=60, alpha=0.7, color=ACCENT, edgecolor="white", linewidth=1)
_ms = ms[["freq","pr_delta"]].dropna()
if len(_ms)>3:
    r = _ms.corr().iloc[0,1]
    z = np.polyfit(_ms.freq, _ms.pr_delta, 1)
    xx = np.linspace(_ms.freq.min(), _ms.freq.max(), 50)
    ax[0].plot(xx, np.polyval(z, xx), color=ACCENT_2, ls="--", lw=1.5, label=f"r = {r:.2f}")
    ax[0].legend(loc="upper right")
ax[0].set_title("月频次 vs e1RM 增量")
ax[0].set_xlabel("当月卧推 sessions 数"); ax[0].set_ylabel("e1RM 月增量 (kg)")
ax[1].axhline(0, color="#94A3B8", lw=1)
ax[1].scatter(ms.vol/1000, ms.pr_delta, s=60, alpha=0.7, color="#9D4EDD", edgecolor="white", linewidth=1)
_ms2 = ms[["vol","pr_delta"]].dropna()
if len(_ms2)>3:
    r2 = _ms2.corr().iloc[0,1]
    ax[1].set_title(f"月容量 vs e1RM 增量  (r = {r2:.2f})")
ax[1].set_xlabel("当月卧推容量 (吨)"); ax[1].set_ylabel("e1RM 月增量 (kg)")
plt.suptitle("卧推 PR 增长的可能驱动因素", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig(f"{IMG}/11_pr_drivers.png"); plt.close()

# ============ 12. PR 归一化 ============
fig, ax = plt.subplots(figsize=(13,5.5))
colors_pr = ["#E63946","#2A9D8F","#9D4EDD","#1D3557","#F4A261"]
for ex, color in zip(["杠铃卧推","深蹲","硬拉","站姿杠铃推举","杠铃划船"], colors_pr):
    d = sets[sets.exercise==ex]
    if d.empty: continue
    e1 = d.assign(e1=d.weight_kg*(1+d.reps/30)).groupby("date").e1.max().cummax()
    e1n = e1 / e1.iloc[0] * 100
    ax.plot(e1n.index, e1n.values, lw=2.5, label=f"{ex}  {e1.iloc[0]:.0f}→{e1.iloc[-1]:.0f} kg",
            color=color, alpha=0.9)
ax.axhline(100, color="#94A3B8", lw=1, ls="--")
ax.set_title("核心动作 e1RM 归一化增长 (起点 = 100)")
ax.set_ylabel("相对值 (%)")
ax.legend(loc="upper left", fontsize=10)
plt.tight_layout(); plt.savefig(f"{IMG}/12_pr_normalized.png"); plt.close()

# ============ 13. 恢复间隔 ============
fig, ax = plt.subplots(figsize=(12,5.5))
groups_ord = ["胸","背","腿","肩","臂","核心"]
data_box = []
labels = []
for g in groups_ord:
    g_dates = sets[sets.group==g].date.dt.normalize().sort_values().unique()
    if len(g_dates) < 5: continue
    diffs = pd.Series(g_dates).diff().dt.days.dropna()
    diffs = diffs[diffs<=21]
    data_box.append(diffs.values)
    labels.append(f"{g}\n中位 {int(diffs.median())} 天")
bp = ax.boxplot(data_box, labels=labels, patch_artist=True, widths=0.55,
                boxprops=dict(linewidth=1.2, edgecolor=NEUTRAL),
                medianprops=dict(color=ACCENT_2, linewidth=2.5),
                whiskerprops=dict(color=NEUTRAL, linewidth=1),
                capprops=dict(color=NEUTRAL, linewidth=1),
                flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch, g in zip(bp["boxes"], groups_ord):
    patch.set_facecolor(PALETTE.get(g, "#CBD5E1"))
    patch.set_alpha(0.85)
ax.axhline(7, color="#F4A261", lw=1, ls="--", alpha=0.7, label="一周一次参考线")
ax.set_title("肌群训练间隔天数分布  (距上次训练此肌群)")
ax.set_ylabel("天数")
ax.set_ylim(0, 22)
ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(f"{IMG}/13_recovery.png"); plt.close()
recovery = {}
for g in groups_ord:
    g_dates = sets[sets.group==g].date.dt.normalize().sort_values().unique()
    if len(g_dates)<5: continue
    diffs = pd.Series(g_dates).diff().dt.days.dropna()
    diffs = diffs[diffs<=30]
    recovery[g] = {"median_days": int(diffs.median()), "p25": int(diffs.quantile(0.25)), "p75": int(diffs.quantile(0.75))}
stats_dict["recovery_intervals"] = recovery

# ============ 14. 工作重量 ============
fig, ax = plt.subplots(2,2, figsize=(14,8.5))
core_lifts4 = ["杠铃卧推","深蹲","硬拉","站姿杠铃推举"]
colors_4 = ["#E63946","#F4A261","#9D4EDD","#1D3557"]
for i, (ex, color) in enumerate(zip(core_lifts4, colors_4)):
    a = ax.flatten()[i]
    d = sets[sets.exercise==ex].copy()
    if d.empty: continue
    top3 = d.groupby(["date","session_id"]).apply(lambda g: g.nlargest(3,"weight_kg").weight_kg.mean(), include_groups=False)
    monthly = top3.groupby(top3.index.get_level_values(0).to_period("M")).mean()
    monthly.index = monthly.index.to_timestamp()
    a.fill_between(monthly.index, 0, monthly.values, color=color, alpha=0.18)
    a.plot(monthly.index, monthly.values, lw=2.5, color=color, marker="o", ms=4, mec="white", mew=1)
    a.set_title(f"{ex}: 月均工作重量")
    a.set_ylabel("kg")
    a.set_ylim(bottom=max(0, monthly.min()*0.7))
plt.suptitle("Top-3 set/session 月均工作重量", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(); plt.savefig(f"{IMG}/14_working_weight.png"); plt.close()

with open(f"{OUT}/stats.json","w") as f:
    json.dump(stats_dict, f, ensure_ascii=False, indent=2, default=str)

print(f"Charts saved (DPI={DPI}) to {IMG}/")
print("Files:", sorted(os.listdir(IMG)))
