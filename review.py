"""复盘：上一交易日推荐的今日表现、名额调整说明、跟踪池"""
import datetime as dt

import numpy as np
import pandas as pd

import config
from card import link, pct_s
from strategies import consec_up, tag


def index_ret_on(index_df: pd.DataFrame | None, day: dt.date) -> float | None:
    if index_df is None or len(index_df) < 2 or index_df["date"].iloc[-1] != day:
        return None
    c = index_df["close"]
    return float(c.iloc[-1] / c.iloc[-2] - 1)


def yesterday(rec: pd.DataFrame, prev_day: dt.date | None, spot_idx: pd.DataFrame,
              bench: float | None) -> str | None:
    if prev_day is None:
        return None
    x = rec[rec["date"] == prev_day.isoformat()].copy()
    if x.empty:
        return None
    x["today"] = x["code"].map(spot_idx["涨跌幅"]) / 100
    x = x.dropna(subset=["today"])
    if x.empty:
        return None
    up, down = int((x["today"] > 0).sum()), int((x["today"] < 0).sum())
    b = f"（沪深300 {pct_s(bench)}）" if bench is not None else ""
    best, worst = x.loc[x["today"].idxmax()], x.loc[x["today"].idxmin()]
    lines = [f"{prev_day:%m月%d日} 推荐 {len(x)} 只，今日平均 **{pct_s(x['today'].mean())}**{b}，{up} 涨 {down} 跌",
             f"最好 {link(best['name'], best['code'])} {pct_s(best['today'])}　"
             f"最差 {link(worst['name'], worst['code'])} {pct_s(worst['today'])}"]
    parts = [f"{tag(s)}{s} {pct_s(g['today'].mean())}（{len(g)}只）"
             for s, g in sorted(x.groupby("strategy"), key=lambda kv: -kv[1]["today"].mean())]
    lines.append("　".join(parts))
    return "\n".join(lines)


def adjustment(post: dict, res: dict, prev: dict) -> str:
    slots, p = res["slots"], res["p_best"]
    order = sorted(post, key=lambda n: -p.get(n, 0))
    seg = []
    for n in order:
        s, before = slots.get(n, 0), prev.get(n)
        change = f"{before}→{s}" if before is not None and before != s else f"{s}"
        parts = []
        if post[n].get("bt"):
            b = post[n]["bt"]
            parts.append(f"回测日均超额 {pct_s(b['ex'], 2)}（t={b['t']:.1f}）")
        parts.append(f"实盘日均超额 {pct_s(post[n]['raw'], 2)}（有效样本 {post[n]['n_eff']:.1f}）"
                     if post[n]["n"] else "实盘暂无数据")
        ev = "｜".join(parts)
        seg.append(f"{tag(n)}{n} 名额 **{change}**｜胜出概率 {p.get(n, 0):.0%}｜{ev}")
    head = "以夜间回测为起点、叠加近期实盘表现做贝叶斯分配：表现好的多给名额，数据少的保留探索机会"
    if res.get("cut"):
        head += "\n⚠️ 近期所有策略都跑输沪深300，今日减少推荐数量"
    return head + "\n" + "\n".join(seg)


def tracking_pool(rec: pd.DataFrame, tdays: list[dt.date], day: dt.date, spot_idx: pd.DataFrame,
                  hists: dict) -> list[str]:
    window = [d.isoformat() for d in tdays if d < day][-config.TRACK_DAYS:]
    x = rec[rec["date"].isin(window)].sort_values("date").drop_duplicates("code", keep="last")
    if x.empty:
        return []
    rows = []
    for _, r in x.iterrows():
        price = spot_idx["最新价"].get(r["code"])
        if price is None or pd.isna(price):
            continue
        cum = price / r["close"] - 1
        h = hists.get(r["code"])
        streak = consec_up(h["close"].to_numpy(float)) if h is not None and h["date"].iloc[-1] == day else None
        held = sum(1 for d in window if d > r["date"]) + 1
        if pd.notna(r["stop"]) and price < r["stop"]:
            status = "⚠️ 跌破止损"
        elif cum >= 0.10:
            status = "🎯 涨超10%"
        elif streak and streak >= 3:
            status = "🔥 连涨中"
        else:
            status = ""
        rows.append((cum, f"{link(r['name'], r['code'])} {tag(r['strategy'])}{r['date'][5:].replace('-', '/')}"
                          f"　累计 **{pct_s(cum)}**｜第 {held} 天"
                          + (f"｜连涨 {streak} 天" if streak else "") + (f"　{status}" if status else "")))
    rows.sort(key=lambda t: -t[0])
    return [t[1] for t in rows[:15]]


def scoreboard(stats: pd.DataFrame) -> str | None:
    if stats is None or stats.empty:
        return None
    lines = []
    for _, s in stats.sort_values("avg1", ascending=False).iterrows():
        t = f"{tag(s['strategy'])}{s['strategy']}　{s['n']:.0f} 只｜次日胜率 {s['win1']:.0%}｜次日均值 {pct_s(s['avg1'])}"
        if s["n3"] > 0 and not np.isnan(s["ex3"]):
            t += f"｜3日超额 {pct_s(s['ex3'])}"
        if not np.isnan(s["up5"]):
            t += f"｜5日内平均上涨 {s['up5']:.1f} 天"
        lines.append(t)
    return "\n".join(lines)
