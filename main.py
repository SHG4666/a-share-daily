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

import accum
import alloc
import histstore
import card
import config
import data
import model
import review
import roster as roster_mod
import variants
import tracker


STORE = histstore.Store()
_HIST = STORE.h  # 代码 -> 历史日线（与缓存共用同一个字典）
_CTX = {"day": None, "traded": None}


def fetch_hists(codes: list[str], budget_sec: float | None = None):
    """确保这些股票有截至复盘日的历史数据（优先用缓存，缺的在时间预算内补拉）"""
    STORE.ensure(codes, _CTX["day"] or data.today_cn(), budget_sec or config.DAILY_FETCH_BUDGET,
                 traded=_CTX["traded"])


def hist(code: str):
    h = STORE.get(code)
    if h is None:
        h = data.get_hist(code)
        STORE.h[code] = h
    return h


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
    tdays = data.trade_days()
    day, why = data.target_day(tdays)
    if day is None:
        print(f"跳过：{why}"); return
    print("复盘日期", day, f"（{why}）")
    sent_flag = data.DATA_DIR / "last_sent.txt"
    if os.getenv("FORCE_RUN") != "1" and sent_flag.exists() and sent_flag.read_text().strip() == day.isoformat():
        print(f"{day} 已推送过，跳过（多个定时触发的兜底去重）"); return

    spot = data.get_spot()
    spot_idx = spot.set_index("代码")
    print("全市场", len(spot), "只")
    STORE.load()
    STORE.append_snapshot(spot, day)
    _CTX["day"] = day
    _CTX["traded"] = set(spot.loc[spot["成交量"] > 0, "代码"]) if "成交量" in spot else None
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

    # 2) 全市场历史数据（缓存 + 当天增量），计算当天截面指标
    fetch_hists(univ["代码"].tolist() + track_codes)
    try:
        STORE.save()
    except Exception as e:  # noqa: BLE001
        print("历史缓存保存失败:", e)
    rec = tracker.evaluate(rec, tdays, day, hist, index_df)
    U = variants.make_panel(univ, {c: STORE.get(c) for c in univ["代码"]}, {day})
    U = variants.tradable(U[U["date"] == day]) if len(U) else U
    print(f"当天截面：{len(U)} 只股票有完整指标")

    # 3) 上场策略：先做实盘熔断（近期表现太差的立即下线、补位），再各自选股
    ros = roster_mod.load()
    kills = roster_mod.apply_kill_switch(ros, rec, tdays, day)
    for c in kills:
        print("策略替换:", c)
    names = [n for n in ros["active"] if n in variants.BY_NAME] or list(variants.DEFAULT_ROSTER)
    print("今日上场策略:", names)
    cand = {}
    for n in names:
        try:
            df = variants.run(variants.BY_NAME[n], U, config.CAND_PER_STRATEGY + len(recent_codes), with_text=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"策略 {n} 失败：{e}")
            df = pd.DataFrame()
        if len(df):
            df = df[~df["代码"].isin(recent_codes)].head(config.CAND_PER_STRATEGY).copy()
            print(f"{n}：{int(df['pool'].iloc[0])} 只满足条件，可用候选 {len(df)} 只")
        else:
            print(f"{n}：0 只满足条件")
        cand[n] = df

    # 4) 根据近期 5 日涨幅表现分配名额
    try:
        import backtest
        bt = backtest.load_prior()
    except Exception as e:  # noqa: BLE001
        print("读取回测结果失败:", e); bt = {}
    post = alloc.posteriors(rec, names, tdays, day, bt)
    total = config.TOTAL_SLOTS if regime["strong"] else config.TOTAL_SLOTS_WEAK
    avail = {n: len(cand[n]) for n in names}
    res = alloc.allocate(post, avail, total, seed=int(day.strftime("%Y%m%d")))
    print("名额分配", res)
    picks = merge(cand, res, names)
    print("今日推荐", picks[["代码", "名称", "strategy", "score"]].to_string() if len(picks) else "无")

    # 4.5) 全市场扫描：疑似主力吸筹建仓
    acc_df, acc_hist, acc_track = pd.DataFrame(), None, None
    try:
        acc_df = accum.scan(U)
        acc_hist = accum.evaluate(accum.load(), tdays, day, hist, index_df)
        acc_track = "\n".join(x for x in [accum.bt_line(), accum.track_line(acc_hist)] if x) or None
        print(f"疑似吸筹：{int(acc_df['pool'].iloc[0]) if len(acc_df) else 0} 只满足条件")
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print("吸筹扫描失败:", e)

    # 5) 复盘文本、跟踪池、战绩
    prev_day = max((d for d in tdays if d < day), default=None)
    review_txt = review.yesterday(rec, prev_day, spot_idx, review.index_ret_on(index_df, day))
    alloc_hist = alloc.load()
    recent_night = {d.isoformat() for d in (prev_day, day) if d}
    ros_view = dict(ros) if ros.get("date") in recent_night else dict(ros, changes=[], warn=None)
    adjust_txt = review.adjustment(post, res, alloc.previous_slots(alloc_hist, day), ros_view, kills)
    pool_lines = review.tracking_pool(rec, tdays, day, spot_idx, _HIST)
    board_txt = review.scoreboard(tracker.stats_by_strategy(rec))
    details = {r["代码"]: r.get("detail", "") for _, r in picks.iterrows()}

    rec = tracker.record(rec, day, picks)
    if not dry:
        tracker.save(rec)
        alloc.save(alloc_hist, day, post, res)
        if kills:
            roster_mod.save(ros)
            roster_mod.log_changes(day, kills)
        if acc_hist is not None:
            accum.save(accum.record(acc_hist, day, acc_df))

    inds = data.get_top_industries(config.TOP_INDUSTRY_N)
    ind_txt = "、".join(inds["板块名称"]) if inds is not None else "未知"
    comment = ai_comment(regime["text"], ind_txt, picks)

    payload = card.build(day, spot, regime, inds, review_txt, adjust_txt, picks, pool_lines, board_txt,
                         comment, details, accum=(acc_df, acc_track))
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
