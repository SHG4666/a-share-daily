"""
A股每日量化复盘 -> 飞书群机器人（收盘后运行）

环境变量：
  FEISHU_WEBHOOK    必填，飞书自定义机器人 webhook
  FEISHU_SECRET     选填，机器人开启“签名校验”时填写
  ANTHROPIC_API_KEY 选填，填了会额外生成一段 AI 盘面点评
  FORCE_RUN=1       非交易日也运行（复盘最近一个交易日）
  DRY_RUN=1         只打印卡片，不推送、不写追踪数据
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import time

import requests

import card
import config
import data
import model
import tracker


def fetch_stats(codes: list[str]) -> dict[str, dict]:
    out = {}
    for code in codes:
        try:
            out[code] = model.ts_stats(data.get_hist(code))
        except Exception as e:  # noqa: BLE001
            print("历史数据获取失败", code, e)
        time.sleep(0.2)
    return out


def ai_comment(regime: str, inds: str, picks) -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    brief = "; ".join(f"{r['名称']}({r['industry']}, 趋势年化{r['trend_ann']:.0%})" for _, r in picks.iterrows())
    prompt = (f"今日A股：{regime}。领涨行业：{inds}。量化模型入选：{brief or '无'}。"
              "请用不超过120字给出客观的盘面点评和需关注的风险，不要给买卖建议。")
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"), "max_tokens": 400,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    except Exception as e:  # noqa: BLE001
        print("AI 点评失败:", e)
        return None


def send_feishu(payload: dict):
    url = os.environ["FEISHU_WEBHOOK"]
    secret = os.getenv("FEISHU_SECRET")
    if secret:
        ts = str(int(time.time()))
        sign = base64.b64encode(hmac.new(f"{ts}\n{secret}".encode(), b"", hashlib.sha256).digest()).decode()
        payload = {"timestamp": ts, "sign": sign, **payload}
    r = requests.post(url, json=payload, timeout=20)
    print("飞书返回:", r.status_code, r.text)
    r.raise_for_status()
    if r.json().get("code", 0) != 0:
        raise RuntimeError(r.text)


def main():
    dry = os.getenv("DRY_RUN") == "1"
    today = data.today_cn()
    tdays = data.trade_days()
    if os.getenv("FORCE_RUN") != "1" and today not in tdays:
        print(f"{today} 非交易日，跳过"); return
    day = max(t for t in tdays if t <= today)
    print("复盘日期", day)

    spot = data.get_spot()
    print("全市场", len(spot), "只")
    try:
        index_df = data.get_index("sh000300")
    except Exception as e:  # noqa: BLE001
        print("指数获取失败:", e); index_df = None
    regime = model.market_regime(index_df)
    print("市场状态", regime)

    # 两阶段选股
    ind_map = data.get_industry_map(config.INDUSTRY_CACHE_DAYS)
    univ = model.universe(spot)
    print("股票池", len(univ), "只")
    cands = model.stage1(univ, ind_map)
    stats = fetch_stats(cands["代码"].tolist())
    top_n = config.TOP_N if regime["strong"] else config.TOP_N_WEAK
    picks = model.stage2(cands, stats, top_n)
    print("入选", picks[["代码", "名称", "score"]].to_string() if len(picks) else "无")

    # 自选池
    spot_idx = spot.set_index("代码")
    wstats = fetch_stats(list(config.WATCHLIST.values()))
    watches = [dict(name=n, code=c, tag=model.watch_tag(wstats.get(c)),
                    pct=spot_idx["涨跌幅"].get(c)) for n, c in config.WATCHLIST.items()]

    # 策略追踪
    hist = tracker.load()
    hist = tracker.evaluate(hist, tdays, day, config.EVAL_HORIZON, data.get_hist, index_df)
    if len(picks):
        hist = tracker.record(hist, day, picks)
    track = tracker.summary(hist, config.EVAL_HORIZON)
    if not dry:
        tracker.save(hist)

    inds = data.get_top_industries(config.TOP_INDUSTRY_N)
    ind_txt = "、".join(inds["板块名称"]) if inds is not None else "未知"
    comment = ai_comment(regime["text"], ind_txt, picks)

    payload = card.build(day, spot, regime, inds, picks, watches, track, comment)
    if dry:
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return
    send_feishu(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if os.getenv("FEISHU_WEBHOOK") and os.getenv("DRY_RUN") != "1":
            try:
                send_feishu({"msg_type": "text", "content": {"text": f"A股每日推荐运行失败：{e}"}})
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)
