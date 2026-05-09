"""Generate magazine-style training feedback loop overview diagram."""
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Magazine palette
BG = "#FAFAF7"
TERRACOTTA = "#C66B4A"
SAGE = "#7A8C6F"
SLATE = "#4A5568"
SAND = "#D9C7A0"
INK = "#2B2B2B"
MUTE = "#8A8A85"

plt.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "PingFang SC", "Hiragino Sans GB",
                    "Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.edgecolor": "none",
})

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=180)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")

# Title (left-aligned)
fig.text(0.06, 0.93, "训练反馈闭环 · 流程总览",
         fontsize=17, fontweight="bold", color=INK, ha="left")
fig.text(0.06, 0.875, "Training Feedback Loop — 5 步、3 个数据源、1 份持续累积的复盘档案",
         fontsize=10.5, color=MUTE, ha="left")

# Box helper
def box(x, y, w, h, label, sub, color, fc_alpha=0.16):
    fbox = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.4,rounding_size=1.2",
                          linewidth=1.4, edgecolor=color,
                          facecolor=color, alpha=fc_alpha)
    ax.add_patch(fbox)
    ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK)
    ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
            fontsize=8.6, color=MUTE)

# Five stages, top row
stages = [
    (3,  28, 16, 12, "1 · 历史数据",     "训记 5 年记录\n10,998 组",    SLATE),
    (22, 28, 16, 12, "2 · 诊断报告",     "找平台期真因\n弱链 / 失衡",   TERRACOTTA),
    (41, 28, 16, 12, "3 · 新计划",       "PPL 周期\n写回训记 App",       SAGE),
    (60, 28, 16, 12, "4 · 执行训练",     "训记 + Apple Watch\n同时记录", SAND),
    (79, 28, 16, 12, "5 · 训后复盘",     "三源交叉 · 量化追踪",         TERRACOTTA),
]
for s in stages:
    box(*s)

# Arrows between stages
for i in range(4):
    x0 = stages[i][0] + stages[i][2]
    x1 = stages[i+1][0]
    arr = FancyArrowPatch((x0, 34), (x1, 34),
                          arrowstyle="-|>", mutation_scale=14,
                          linewidth=1.4, color=MUTE)
    ax.add_patch(arr)

# Bottom feedback row — 3 outputs from debrief
fb = [
    (10, 6,  20, 11, "风险信号",       "→ 下次必做规则",       SLATE),
    (40, 6,  20, 11, "主观↔客观",      "→ 校准 RPE 信号",      SAGE),
    (70, 6,  20, 11, "下次需验证",     "→ 自动塞进训前 briefing", TERRACOTTA),
]
for s in fb:
    box(*s)

# Arrows from stage 5 (训后复盘) down to all three feedback boxes
for fx in [20, 50, 80]:
    arr = FancyArrowPatch((87, 28), (fx, 17),
                          arrowstyle="-|>", mutation_scale=12,
                          linewidth=1.0, color=MUTE, alpha=0.55)
    ax.add_patch(arr)

# Loop back curve from feedback row to stage 4 (执行训练)
loop = FancyArrowPatch((50, 6), (68, 28),
                       connectionstyle="arc3,rad=-0.35",
                       arrowstyle="-|>", mutation_scale=14,
                       linewidth=1.5, color=TERRACOTTA, linestyle="--")
ax.add_patch(loop)
ax.text(50, 1.5, "↑ 三类产出 → 下一次训前 briefing 自动带上",
        fontsize=8.8, color=TERRACOTTA, style="italic", ha="center")

# Footer caption
fig.text(0.06, 0.04, "数据源：训记（容量）· Apple Watch / 任意分钟级 HR 设备（生理）· 主观 RPE（感受）",
         fontsize=8.6, color=MUTE, ha="left")

import os
out = os.path.join(os.path.dirname(__file__), "..", "analysis", "article_img", "00_flow_overview.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print("OK", out)
