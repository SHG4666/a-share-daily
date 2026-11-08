# A股每日推荐 → 飞书

每个交易日收盘后（北京时间 16:07）自动运行，向飞书群推送一张复盘卡片：

- **市场概况**：涨跌家数、两市成交额、领涨行业 Top5
- **技术面筛选**：全市场初筛（市值、涨幅、换手、量比、市盈率）→ 按成交额取前 40 只 → 技术面打分（均线多头、MACD 金叉、放量、20 日新高、过热扣分）→ 取得分 ≥3 的前 5 只，附信号和风险提示
- **自选主题池**：机器人 / 半导体自选股的当日涨跌和技术信号
- **AI 点评**（可选）：配置 `ANTHROPIC_API_KEY` 后生成一段客观盘面点评

非交易日（周末、节假日）自动跳过。

## 部署

1. 新建 GitHub 仓库（如 `a-share-daily`），把本目录所有文件推上去（包括 `.github/workflows/daily.yml`）
2. 仓库 Settings → Secrets and variables → Actions，添加：
   - `FEISHU_WEBHOOK`（必填）：飞书群 → 设置 → 群机器人 → 自定义机器人 的 webhook 地址，可以直接复用 github-daily 那个
   - `FEISHU_SECRET`（选填）：机器人开启了“签名校验”才需要
   - `ANTHROPIC_API_KEY`（选填）：要 AI 点评才需要
3. Actions 页面 → “A股每日推荐” → Run workflow，勾选“强制运行”测试一次

## 本地调试

```bash
pip install -r requirements.txt
DRY_RUN=1 FORCE_RUN=1 python main.py   # 只打印卡片 JSON，不推送
```

## 调参

选股条件、自选股都在 `config.py`，打分规则在 `main.py` 的 `signals()`。

## 注意

- 数据来自 akshare（东方财富接口），GitHub 的海外服务器偶尔会被限流或超时，脚本已带重试；失败时会往群里发一条失败提醒
- 这是规则化的技术面筛选，不包含基本面和消息面，仅供研究参考，不构成投资建议
