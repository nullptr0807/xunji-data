# 训记 (Xunji) 数据分析

抓取并分析 [训记 App](https://trains.xunjiapp.cn) 的训练 / 身体 / 饮食数据。

## 接口

- `POST https://trains.xunjiapp.cn/api_trains_for_llm`
- 鉴权: `Authorization: Bearer <APIKEY>` (也可 body/query 传 `apikey`)
- Body: `{"datestr": "YYYY-MM-DD"}`
- 返回: `{"success": true, "res": [...]}`，gzip 压缩
- 限流: **同一日期 90 秒内只算一次**，过快返回 `too frequent, retry after Ns`

## 用法

```bash
cp .env.example .env  # 填入 XUNJI_API_KEY
pip install -r requirements.txt

# 抓单天
python -m xunji.fetch --date 2026-05-07

# 抓区间（自动 sleep 应对限流）
python -m xunji.fetch --start 2026-01-01 --end 2026-05-08
```

数据落到 `data/raw/YYYY-MM-DD.json`（原始）+ `data/parsed/YYYY-MM-DD.json`（解析后）。

## 写入 (upsert)

`POST /api_upsert_trains_for_llm` — 把训练行写回训记。同一日期最多 12 条，每行 ≤1500 字符。

```bash
# 从 JSON 文件 (内容: ["row1", "row2", ...])
python -m xunji.upsert --file rows.json

# 直接传一条 (休息日)
python -m xunji.upsert --row "2026-04-02,休息日"

# 力量动作
python -m xunji.upsert --row \
  "2026-04-02,胸部训练,1.卧推,1组,60kg,10次,2组,60kg,8次"

# 带 id 是更新, 不带 id 是新建
python -m xunji.upsert --row \
  "2026-04-02,id:1778154285558,胸背,1.引体向上,1组,0kg,10次"
```

行格式：
- `YYYY-MM-DD[,id:LOCALID],标题[,train_time:start-end][,备注],动作组...`
- 力量: `1.动作名,1组,Wkg,R次[,time:Ts]`
- 有氧: `2.跑步,5km,300kcal,time:1800s,140bpm`
- 休息日: `YYYY-MM-DD,休息日`
- **upsert 不会删除当天未出现的旧记录**

## 目录

```
xunji-data/
├── xunji/
│   ├── fetch.py     # 抓取 + 限流处理
│   ├── parse.py     # 解析 res 文本 (id/train_time/locals)
│   └── client.py    # HTTP 客户端
├── data/
│   ├── raw/         # 原始 API 响应
│   └── parsed/      # 解析后 JSON
└── notebooks/       # 分析
```
