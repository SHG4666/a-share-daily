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
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import alloc
import card
import config
import data
import model
import review
import strategies
import tracker


_HIST: dict = {}


def _fetch_one(code: str):
    try:
        time.sleep(0.2)
        return data.get_hist(code)
    except Exception as e:  # noqa: BLE001
        print("历史数据获取失败", code, str(e)[:80])
        return None


def fetch_hists(codes: list[str]):
    """并发拉取历史日线并缓存，同一只股票本次运行只拉一次"""
    todo = [c for c in dict.fromkeys(codes) if c not in _HIST]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for c, h in zip(todo, ex.map(_fetch_one, todo)):
            if h is not None:
                _HIST[c] = h
    print(f"历史数据：新拉取 {len(todo)} 只，用时 {time.time() - t0:.0f}s")


def hist(code: str):
    if code not in _HIST:
        _HIST[code] = data.get_hist(code)
    return _HIST[code]


def merge(cand: dict, res: dict, names: list[str]) -> pd.DataFrame:
    """按分配的名额从各策略取股票：跨策略去重、同行业不超过 MAX_PER_INDUSTRY 只；
    某策略候选不够时，剩余名额顺延给胜出概率更高的策略"""
    slots = dict(res["slots"])
    order = sorted(names, key=lambda n: -res["p_best"].get(n, 0))
    chosen, used, ind_cnt = [], set(), {}
    ptr = {n: 0 for n in names}

    def take(n) -> bool:
        df = cand[n]
        while ptr[n] < len(df):
            r = df.iloc[ptr[n]]; ptr[n] += 1
            ind = r.get("industry") if isinstance(r.get("industry"), str) else None
            if r["代码"] in used or (ind and ind_cnt.get(ind, 0) >= config.MAX_PER_INDUSTRY):
                continue
            used.add(r["代码"]); chosen.append(r)
            if ind:
                ind_cnt[ind] = ind_cnt.get(ind, 0) + 1
            return True
        return False

    left = 0
    for n in order:
        for _ in range(slots.get(n, 0)):
            if not take(n):
                left += 1
    for n in order:
        while left and take(n):
            left -= 1
    return pd.DataFrame(chosen).reset_index(drop=True) if chosen else pd.DataFrame()


def ai_comment(regime: str, inds: str, picks) -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    brief = "; ".join(f"{r['名称']}({r['strategy']})" for _, r in picks.iterrows())
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
    sent_flag = data.DATA_DIR / "last_sent.txt"
    if os.getenv("FORCE_RUN") != "1" and sent_flag.exists() and sent_flag.read_text().strip() == day.isoformat():
        print(f"{day} 已推送过，跳过（多个定时触发的兜底去重）"); return

    spot = data.get_spot()
    spot_idx = spot.set_index("代码")
    print("全市场", len(spot), "只")
    try:
        index_df = data.get_index("sh000300")
    except Exception as e:  # noqa: BLE001
        print("指数获取失败:", e); index_df = None
    regime = model.market_regime(index_df)
    print("市场状态", regime)

    ind_map = data.get_industry_map(config.INDUSTRY_CACHE_DAYS)
    univ = model.universe(spot)
    univ["industry"] = model.fill_industry(univ["代码"].map(ind_map), univ["名称"])
    print("股票池", len(univ), "只")

    # 1) 复盘：补算历史推荐的 1/3/5 日表现
    rec = tracker.load()
    past = [d.isoformat() for d in tdays if d < day]
    recent_codes = set(rec.loc[rec["date"].isin(past[-config.RECENT_EXCLUDE_DAYS:]), "code"])
    track_codes = rec.loc[rec["date"].isin(past[-15:]), "code"].tolist()

    # 2) 各策略粗筛 → 合并后一次性并发拉取历史数据
    ctx = {"ind_map": ind_map}
    pre = {s.name: s.prefilter(univ, ctx) for s in strategies.ALL}
    for n, df in pre.items():
        print(f"{n} 粗筛 {len(df)} 只")
    fetch_hists([c for df in pre.values() for c in df["代码"]] + track_codes)

    rec = tracker.evaluate(rec, tdays, day, hist, index_df)

    # 3) 各策略精选（排除最近几天已推荐过的）
    cand = {}
    for s in strategies.ALL:
        try:
            df = s.select(pre[s.name], _HIST)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"策略 {s.name} 失败：{e}")
            df = pd.DataFrame()
        if len(df):
            df = df[~df["代码"].isin(recent_codes)].copy()
            df["strategy"] = s.name
        cand[s.name] = df
        print(f"{s.name} 可用候选 {len(df)} 只")

    # 4) 根据近期表现分配名额
    names = [s.name for s in strategies.ALL]
    post = alloc.posteriors(rec, names, tdays, day)
    total = config.TOTAL_SLOTS if regime["strong"] else config.TOTAL_SLOTS_WEAK
    avail = {n: len(cand[n]) for n in names}
    res = alloc.allocate(post, avail, total, seed=int(day.strftime("%Y%m%d")))
    print("名额分配", res)
    picks = merge(cand, res, names)
    print("今日推荐", picks[["代码", "名称", "strategy", "score"]].to_string() if len(picks) else "无")

    # 5) 复盘文本、跟踪池、战绩
    prev_day = max((d for d in tdays if d < day), default=None)
    review_txt = review.yesterday(rec, prev_day, spot_idx, review.index_ret_on(index_df, day))
    alloc_hist = alloc.load()
    adjust_txt = review.adjustment(post, res, alloc.previous_slots(alloc_hist, day))
    pool_lines = review.tracking_pool(rec, tdays, day, spot_idx, _HIST)
    board_txt = review.scoreboard(tracker.stats_by_strategy(rec))
    details = {r["代码"]: strategies.safe_detail(r["strategy"], r) for _, r in picks.iterrows()}

    rec = tracker.record(rec, day, picks)
    if not dry:
        tracker.save(rec)
        alloc.save(alloc_hist, day, post, res)

    inds = data.get_top_industries(config.TOP_INDUSTRY_N)
    ind_txt = "、".join(inds["板块名称"]) if inds is not None else "未知"
    comment = ai_comment(regime["text"], ind_txt, picks)

    payload = card.build(day, spot, regime, inds, review_txt, adjust_txt, picks, pool_lines, board_txt,
                         comment, details)
    if dry:
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return
    send_feishu(payload)
    data.DATA_DIR.mkdir(exist_ok=True)
    sent_flag.write_text(day.isoformat())


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
