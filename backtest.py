"""
夜间回测：用真实历史行情回放过去 BT_DAYS 个交易日，对比各策略的实际效果

做法（逐日回放，只用当时能看到的数据）：
  对每个历史交易日 t：
    1. 用截至 t 的行情构造“当天快照”（涨跌幅、成交额排名）
    2. 每个策略按原规则粗筛、精选，取前 BT_TOP_K 只
    3. 记录这些股票 t→t+1、t+3、t+5 的真实收益，以及同期沪深300 收益
  另设两个对照组：
    🎲 随机基准：每天从股票池随机挑 K 只（判断策略是否真的比瞎选强）
    📈 追涨基准：每天买涨幅 5%~9% 的成交额前 K 只（判断是否只是在追涨）

评价指标：
  日均超额：每天组合次日收益 − 沪深300 次日收益，取平均
  t 值：日均超额 / (标准差 / √天数)，|t| > 2 大致表示不是运气
  3 日超额、5 日内上涨天数、连涨率（推荐后 5 天里至少 3 天上涨的比例）

近似与偏差（会在报告里说明）：
  - 股票池用今天的成交额前 BT_UNIVERSE 只（存在幸存者偏差）
  - 回放中没有历史换手率，换手率条件按满足处理
  - 稳健多因子使用今天的 PE/PB（有前视偏差），结果仅作参考
"""
import contextlib
import datetime as dt
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import card
import config
import data
import model
import strategies
from main import _HIST, fetch_hists, send_feishu

PATH = data.DATA_DIR / "backtest.csv"
BASELINES = ["随机基准", "追涨基准"]


def build_panel(codes: list[str]) -> dict[str, pd.DataFrame]:
    fetch_hists(codes)
    return {c: _HIST[c].reset_index(drop=True) for c in codes if c in _HIST and len(_HIST[c]) > 130}


def pseudo_univ(base: pd.DataFrame, hists: dict, t: dt.date) -> tuple[pd.DataFrame, dict]:
    """用截至 t 的数据构造当天的“快照”和历史切片"""
    rows, sliced = [], {}
    for _, r in base.iterrows():
        h = hists.get(r["代码"])
        if h is None:
            continue
        pos = h.index[h["date"] == t]
        if len(pos) == 0 or pos[0] < 1:
            continue
        p = int(pos[0])
        hs = h.iloc[:p + 1]
        c0, c1 = hs["close"].iloc[-2], hs["close"].iloc[-1]
        pct = (c1 / c0 - 1) * 100
        limit = 20.0 if r["代码"].startswith(("688", "300", "301")) else 10.0
        if abs(pct) >= limit - 0.3:
            continue
        rows.append({**r, "最新价": c1, "涨跌幅": pct, "成交额": c1 * hs["vol"].iloc[-1], "换手率": 3.0})
        sliced[r["代码"]] = hs
    return pd.DataFrame(rows), sliced


def fwd(h: pd.DataFrame, t: dt.date) -> dict | None:
    pos = h.index[h["date"] == t]
    if len(pos) == 0:
        return None
    p = int(pos[0])
    c = h["close"].to_numpy(float)
    out = {}
    for k in (1, 3, 5):
        out[f"r{k}"] = c[p + k] / c[p] - 1 if p + k < len(c) else np.nan
    if p + 5 < len(c):
        out["up5"] = int((np.diff(c[p:p + 6]) > 0).sum())
    else:
        out["up5"] = np.nan
    return out


def run_day(t, base, hists, ind_map, rng) -> list[dict]:
    univ, sliced = pseudo_univ(base, hists, t)
    if len(univ) < 100:
        return []
    picks = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for s in strategies.ALL:
            try:
                df = s.select(s.prefilter(univ, {"ind_map": ind_map}), sliced)
                picks[s.name] = df["代码"].head(config.BT_TOP_K).tolist() if len(df) else []
            except Exception:  # noqa: BLE001
                picks[s.name] = []
    picks["随机基准"] = list(rng.choice(univ["代码"].to_numpy(), size=config.BT_TOP_K, replace=False))
    chase = univ[univ["涨跌幅"].between(5, 9)].sort_values("成交额", ascending=False)
    picks["追涨基准"] = chase["代码"].head(config.BT_TOP_K).tolist()

    out = []
    for name, codes in picks.items():
        for c in codes:
            f = fwd(hists[c], t)
            if f:
                out.append({"date": t.isoformat(), "strategy": name, "code": c, **f})
    return out


def bench_fwd(index_df: pd.DataFrame, t: dt.date) -> dict:
    return fwd(index_df.assign(vol=0), t) or {}


def summarize(trades: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bench = {d: bench_fwd(index_df, dt.date.fromisoformat(d)) for d in trades["date"].unique()}
    for name, g in trades.groupby("strategy"):
        daily = g.groupby("date").agg(r1=("r1", "mean"), r3=("r3", "mean")).dropna(subset=["r1"])
        if daily.empty:
            continue
        ex1 = daily["r1"] - [bench[d].get("r1", 0) for d in daily.index]
        d3 = daily.dropna(subset=["r3"])
        ex3 = (d3["r3"] - [bench[d].get("r3", 0) for d in d3.index]) / 3
        sd = ex1.std(ddof=1)
        g5 = g.dropna(subset=["up5"])
        rows.append(dict(
            strategy=name, days=len(daily), n=len(g), avg1=g["r1"].mean(), win1=(g["r1"] > 0).mean(),
            ex1=ex1.mean(), t=ex1.mean() / (sd / np.sqrt(len(ex1))) if sd and len(ex1) > 2 else np.nan,
            ex3d=ex3.mean() if len(ex3) else np.nan, up5=g5["up5"].mean() if len(g5) else np.nan,
            cont=(g5["up5"] >= 3).mean() if len(g5) else np.nan,
            cum=float(np.prod(1 + daily["r1"]) - 1),
        ))
    return pd.DataFrame(rows).sort_values("ex1", ascending=False)


def report_card(day: dt.date, summ: pd.DataFrame, n_univ: int, n_days: int, secs: float) -> dict:
    tag = {**{s.name: strategies.tag(s.name) for s in strategies.ALL}, "随机基准": "🎲", "追涨基准": "📈"}
    lines = []
    for i, (_, r) in enumerate(summ.iterrows(), 1):
        sig = "显著" if abs(r["t"]) >= 2 else ("偏弱" if abs(r["t"]) >= 1 else "不显著")
        lines.append(
            f"**{i}. {tag.get(r['strategy'], '')}{r['strategy']}**　日均超额 **{card.pct_s(r['ex1'], 2)}**"
            f"（t={r['t']:.1f}，{sig}）\n"
            f"次日胜率 {r['win1']:.0%}｜次日均值 {card.pct_s(r['avg1'], 2)}｜3日超额/天 {card.pct_s(r['ex3d'], 2)}\n"
            f"5日内平均上涨 {r['up5']:.1f} 天｜连涨率 {r['cont']:.0%}｜每日调仓累计 {card.pct_s(r['cum'])}")
    rand = summ.loc[summ["strategy"] == "随机基准", "ex1"]
    rb = float(rand.iloc[0]) if len(rand) else 0.0
    real = summ[~summ["strategy"].isin(BASELINES)]
    beat = real[real["ex1"] > rb]["strategy"].tolist()
    concl = (f"跑赢随机基准的策略：{'、'.join(beat)}" if beat else "所有策略都没跑赢随机基准，近期市场不适合这些打法") + \
        "。明天的名额分配会以这份结果作为起点，再叠加实盘推荐的表现。"
    els = [
        card.md(f"回放过去 **{n_days}** 个交易日，股票池 {n_univ} 只，每个策略每天取前 {config.BT_TOP_K} 只，"
                f"按收盘价买入、持有 1/3/5 天计算真实收益。"),
        card.HR, card.md("\n\n".join(lines)), card.HR, card.md(f"**结论**　{concl}"),
        {"tag": "note", "elements": [{"tag": "plain_text", "content":
            "t 值 = 日均超额 ÷ 标准误，|t|≥2 大致说明不是运气。连涨率 = 推荐后 5 天里至少 3 天上涨的比例。"
            "近似：股票池用当前成交额前列（幸存者偏差），无历史换手率；稳健多因子用当前 PE/PB，仅供参考。"
            f"耗时 {secs / 60:.0f} 分钟。"}]},
    ]
    return {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": "indigo", "title": {"tag": "plain_text", "content": f"🌙 策略夜间回测 · {day:%m月%d日}"}},
        "elements": els}}


def load_prior() -> dict:
    """供 alloc 使用：最近一次回测的各策略日均超额"""
    if not PATH.exists():
        return {}
    df = pd.read_csv(PATH, dtype={"run_date": str})
    if df.empty:
        return {}
    last = df[df["run_date"] == df["run_date"].max()]
    return {r["strategy"]: {"ex": r["ex1"], "t": r["t"], "days": r["days"]} for _, r in last.iterrows()}


def main():
    t0 = time.time()
    dry = os.getenv("DRY_RUN") == "1"
    tdays = data.trade_days()
    day, why = data.target_day(tdays)
    if day is None:
        print(f"跳过：{why}"); return
    flag = data.DATA_DIR / "last_backtest.txt"
    if os.getenv("FORCE_RUN") != "1" and flag.exists() and flag.read_text().strip() == day.isoformat():
        print(f"{day} 已回测过，跳过"); return
    print("回测基准日", day, f"（{why}）")

    data.HIST_DAYS = 560  # 回放最早一天也要有约 250 个交易日的历史（底部启动需要一年数据）
    spot = data.get_spot()
    index_df = data.get_index("sh000300")
    ind_map = data.get_industry_map(config.INDUSTRY_CACHE_DAYS)
    base = model.universe(spot)
    base["industry"] = model.fill_industry(base["代码"].map(ind_map), base["名称"])
    base = base.sort_values("成交额", ascending=False).head(config.BT_UNIVERSE)
    hists = build_panel(base["代码"].tolist())
    base = base[base["代码"].isin(hists)]
    print(f"回测股票池 {len(base)} 只，历史数据就绪，用时 {time.time() - t0:.0f}s")

    dates = [d for d in tdays if d < day][-(config.BT_DAYS + 1):-1]
    rng = np.random.default_rng(int(day.strftime("%Y%m%d")))
    trades = []
    for i, t in enumerate(dates, 1):
        trades += run_day(t, base, hists, ind_map, rng)
        if i % 10 == 0:
            print(f"回放进度 {i}/{len(dates)}，用时 {time.time() - t0:.0f}s")
    trades = pd.DataFrame(trades)
    summ = summarize(trades, index_df)
    print(summ.to_string())

    summ.insert(0, "run_date", day.isoformat())
    if not dry:
        old = pd.read_csv(PATH, dtype={"run_date": str}) if PATH.exists() else pd.DataFrame()
        old = old[old["run_date"] != day.isoformat()] if len(old) else old
        data.DATA_DIR.mkdir(exist_ok=True)
        pd.concat([old, summ], ignore_index=True).to_csv(PATH, index=False)

    payload = report_card(day, summ, len(base), len(dates), time.time() - t0)
    if dry:
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return
    send_feishu(payload)
    flag.write_text(day.isoformat())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if os.getenv("FEISHU_WEBHOOK") and os.getenv("DRY_RUN") != "1":
            try:
                send_feishu({"msg_type": "text", "content": {"text": f"夜间回测运行失败：{e}"}})
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)
