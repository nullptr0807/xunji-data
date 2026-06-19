# 训记 (Xunji) 数据分析

抓取并分析 [训记 App](https://trains.xunjiapp.cn) 的训练数据，把长期训练记录变成可量化诊断、计划生成、训后复盘的反馈闭环。

完整故事：[`zhihu_article.md`](./zhihu_article.md) — 用训记数据 + Apple Watch + AI 搭训前 → 训中 → 训后的反馈闭环。

> ⚠️ **关于数据隐私**：本仓库**不包含任何个人训练数据**。`data/raw/`、`data/parsed/`、`feedback/`、`.env` 都在 `.gitignore` 里。文章和 `analysis/article_img/` 的截图/图表发布前仍应人工复核脱敏程度；你跑这套脚本得到的是**你自己**的报告。

---

## 快速开始

```bash
# 1. clone + 装依赖
git clone https://github.com/nullptr0807/xunji-data
cd xunji-data
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 XUNJI_API_KEY=xjllm_xxxx
#（开通训记 VIP 后，App「我的 → LLM 接口」可以查到）

# 3. 验证：抓今天
python -m xunji.fetch --date $(date +%F)

# 4. 抓全部历史（按你训练首日开始）
python -m xunji.bulk_fetch --start 2021-08-01 --end $(date +%F)
# bulk_fetch 默认按不同日期短间隔抓取；如果服务器表现为全局限流，可加 --gap 95。
# 修改 parser 后可用 --reparse 只从本地 raw 重建 parsed，不打 API。

# 5. 生成诊断报告
python analysis/build_dataset.py        # 生成 analysis/out/{sets,sessions}.{csv,pkl,parquet*}
python scripts/deep_stats.py             # 全量统计 → JSON + {pkl,parquet*}
python scripts/deep_charts.py            # 10 张深度图
python analysis/generate_deep_training_report.py  # 完整 HTML 报告

# 6. （可选）写训练计划回训记 App：默认 dry-run，不真实发送
python -m xunji.upsert --row "2026-06-01,推日,1.卧推,1组,60kg,10次"
# 真实新建必须显式确认：
python -m xunji.upsert --send --allow-create --row "2026-06-01,推日,1.卧推,1组,60kg,10次"
```

---

## API 参考

### 抓取 `POST /api_trains_for_llm`

| 字段 | 值 |
|---|---|
| 鉴权 | 客户端会在两个 headers、body、query 四处都带同一个 API key |
| Body | `{"datestr": "YYYY-MM-DD"}` |
| 返回 | gzip 压缩；以 `res` 类型和 `error` 字段判定，不能只看 `success` |
| 限流 | 同一日期 90 秒内只算一次，过快返回 `too frequent, retry after Ns` |
| 权限 | 仅训记 VIP |

**已知限制**（决定了分析能做到的精度）：

| 限制 | 影响 |
|---|---|
| 不返回 RPE / RIR | 主观强度得自己另存 |
| 不支持身体数据（体重、围度、体脂） | 长期容量 vs 体重相关性做不了 |
| 不支持饮食数据 | 蛋白/热量进不来 |
| 不支持动作备注 | App 里写的「肩膀紧」「换装备了」读不到 |
| 心率仅 avg HR | 分钟级 HR 拿不到（虽然 App 已从 Watch 同步）|
| 写入 `time:Ts` 不是真实组间 | 是「计划组间」，App 不会用它 override 计时器 |

### 写入 `POST /api_upsert_trains_for_llm`

```bash
# 单条
python -m xunji.upsert --row "2026-04-02,休息日"
python -m xunji.upsert --row "2026-04-02,胸部训练,1.卧推,1组,60kg,10次,2组,60kg,8次"

# 带 id = 更新，不带 = 新建
python -m xunji.upsert --row "2026-04-02,id:1778154285558,胸背,1.引体向上,1组,0kg,10次"

# 从 JSON 文件批量
python -m xunji.upsert --file rows.json   # 文件内容: ["row1", "row2", ...]
```

**行格式**：

```
YYYY-MM-DD[,id:LOCALID],标题[,train_time:start-end][,备注],动作组...
```

- 力量: `1.动作名,1组,Wkg,R次[,time:Ts]`
- 有氧: `2.跑步,5km,300kcal,time:1800s,140bpm`
- 休息: `YYYY-MM-DD,休息日`

**写入限制**：

| 项 | 限制 |
|---|---|
| 标题字符 | 半角 `[GPT5.5]` 已实测可用；若服务器报 `title contains unsafe characters`，再改全角 `【】` |
| 单日条数 | ≤ 12 条 |
| 单条长度 | ≤ 1500 字符 |
| 限流 | 同一 (日期, 端点) 90 秒锁定；不同日期 bulk 默认短间隔，保守模式用 `--gap 95` |
| 无 id 写入 | 无 `id:LOCALID` 会新建记录；CLI 默认 dry-run，真实新建需 `--send --allow-create` |

---

## 目录结构

```
xunji-data/
├── xunji/                          # API 客户端
│   ├── client.py                   #   底层 HTTP + 限流处理
│   ├── fetch.py                    #   单天抓取   (python -m xunji.fetch --date YYYY-MM-DD)
│   ├── bulk_fetch.py               #   区间抓取   (python -m xunji.bulk_fetch --start ... --end ...)
│   ├── parse.py                    #   res 文本 → 结构化 (strength/bodyweight/cardio + 口径字段)
│   ├── upsert.py                   #   写入计划   (python -m xunji.upsert --row "...")
│   └── muscle_groups.py            #   92 个动作 → 肌群分类器（chest/back/quads/...）
│
├── analysis/                       # 数据分析 + 图表
│   ├── build_dataset.py            #   parsed JSON → analysis/out/{sets,sessions}.{csv,pkl,parquet*}
│   ├── analyze.py                  #   基础统计 + 几张概览图
│   ├── article_charts.py           #   杂志风图（米白底 4 色哑光，文章用）
│   ├── generate_deep_training_report.py   # 一键出 HTML 深度报告
│   ├── embed_report_images.py      #   报告 HTML 里的 <img> 转 base64 内嵌
│   ├── build_pdf.py                #   HTML → PDF + 长图（需 weasyprint）
│   └── article_img/                #   文章里引用的 8 张图（脱敏过）
│
├── scripts/                        # 独立的统计 + 图表脚本
│   ├── deep_stats.py               #   全量统计 → analysis/deep/deep_stats.json
│   ├── deep_charts.py              #   10 张深度图（月度趋势、PR、肌群分布、...）
│   ├── gen_flow_overview.py        #   生成文章导航图
│   └── analyze.py                  #   等价于 analysis/analyze.py
│
├── data/                           # 你的训练数据（gitignore）
│   ├── raw/YYYY-MM-DD.json         #   API 原始响应
│   └── parsed/YYYY-MM-DD.json      #   parse.py 输出的结构化版
│
├── .env.example                    # 环境变量模板（XUNJI_API_KEY / BODYWEIGHT_KG / HEIGHT_CM）
├── requirements.txt                # Python 依赖
├── zhihu_article.md                # 知乎文章原稿
└── README.md
```

---

## 跑通分析的最小路径

假设你已经填了 `.env` 并抓完数据：

```bash
# 必须先跑：把 parsed JSON flatten 成 CSV
python analysis/build_dataset.py

# 然后任选其一
python analysis/analyze.py                          # 概览图（频率/容量/PR）
python scripts/deep_stats.py                        # → analysis/deep/deep_stats.json
python scripts/deep_charts.py                       # → analysis/deep/*.png（10 张）
python analysis/generate_deep_training_report.py    # → analysis/deep_report/report.html
```

`analysis/article_charts.py` 是作者文章里那几张杂志风的图，会读 `data/parsed/`。

---

## 配置项

`.env`：

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `XUNJI_API_KEY` | ✅ | — | 训记 LLM 接口 key（VIP 才有） |
| `BODYWEIGHT_KG` | 否 | 70 | 用于相对力量估算（e1RM/体重）|
| `HEIGHT_CM` | 否 | 175 | 用于 BMI 计算 |

---

## 文章里提到的反馈闭环

```
训记 5 年数据
      │
      ▼
  诊断报告 ──── 找进步点
      │
      ▼
  新训练计划 ── 写回训记 App
      │
      ▼
  执行训练（训记 + Watch 同时记录）
      │
      ▼
  训后三源复盘
      │
      ├─→ 风险信号 → 下次必做规则
      ├─→ 主观-客观一致性 → 校准 RPE 信号
      ├─→ e1RM 对比 → 周期推进决策
      └─→ "下次需验证项" → 进入下一次训前 briefing
              │
              └────────────► 回到执行训练
```

本仓库覆盖：① 数据抓取（`xunji/`）② 诊断报告（`analysis/`、`scripts/`）③ 计划写回（`xunji/upsert.py`）。

文章里训中/训后那部分用的是作者自己搭的 Hermes Bot + Apple Watch OCR，不在本仓库范围（因为强依赖个人 Telegram bot 和 Watch 数据），但**任何能读 set-level CSV + Watch 截图的 LLM 客户端都能复现**——文章第九节给了 30 分钟入门路径（不写代码版）。

---

## 已知坑（踩过的）

- API 返回 `success` 字段不可靠，要按 endpoint 分类：fetch 可接受 `res` 为训练列表；upsert `{res:[]}` 是成功；`error` 字段优先判失败
- 同一 (date, endpoint) 90 秒锁定；bulk 抓不同日期默认短间隔，保守模式用 `--gap 95`
- 标题半角 `[GPT5.5]` 已实测可用；若服务端报 `title contains unsafe characters` 再兜底全角 `【】`
- `time:Ts` 写入是「计划组间」，不是真实组间——真实组间只能从分钟级 HR 时间戳反推
- 导出训练日和 App streak 对不上（作者实测：App 显示 639 天，导出 557 天，差 82 天，未排查清）

---

## License

MIT。本仓库不含训记官方代码，不含个人数据。文章 `zhihu_article.md` 版权归作者。
