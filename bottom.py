"""
进攻策略：底部启动 —— 连续下跌、位于底部、刚开始放量上攻

把"底部启动"拆成可计算的统计条件：

1. 连续下跌：近一年高点到近 60 日低点跌幅 ≥ 30%，且下跌持续 ≥ 40 个交易日
2. 位于底部：当前价处于一年价格区间的低位（分位 ≤ 45%）
3. 底部已确认：最低点出现在至少 3 天前，没有继续创新低
4. 刚开始启动：
   - 距底部反弹 5%~35%（启动了，但还不晚）
   - 收盘站上 20 日均线，且 5 日均线上穿 20 日均线
   - 近 5 日均量 ≥ 60 日均量 1.3 倍（资金进场）
   - 近 10 日对数价格回归斜率为正

满足全部条件后打分：
   放量强度 log(5日均量/60日均量)、启动质量（10日回归斜率×R²）、
   前期跌幅深度、底部收敛度（筑底期波动/长期波动，越小说明抛压衰竭）、
   位置越低越好、反弹越早越好
"""
import math

import numpy as np
import pandas as pd

import config
from model import diversify, neutral_z

LABELS = {
    "vol_surge": "放量明显，资金进场",
    "launch": "启动走势干净有力",
    "depth": "前期跌幅深，反弹空间大",
    "base": "底部波动收敛，抛压衰竭",
    "low_pos": "仍处于低位",
    "early": "刚启动，位置早",
}


def prefilter(univ: pd.DataFrame) -> pd.DataFrame:
    """用当日快照粗筛：今天在涨、有换手，再按成交额取前 N 只拉历史数据"""
    df = univ[(univ["涨跌幅"] >= config.BOTTOM_MIN_PCT) & (univ["换手率"] >= config.BOTTOM_MIN_TURNOVER)]
    return df.sort_values("成交额", ascending=False).head(config.BOTTOM_CANDIDATES)


def bottom_stats(h: pd.DataFrame) -> dict | None:
    if len(h) < 250:
        return None
    c = h["close"].to_numpy(float)
    v = h["vol"].to_numpy(float)
    low = h["low"].to_numpy(float)
    c250 = c[-250:]

    peak_i = int(np.argmax(c250))
    trough_i60 = int(np.argmin(c[-60:]))
    trough_i = 250 - 60 + trough_i60
    peak, trough = c250[peak_i], c250[trough_i]
    lo250, hi250 = c250.min(), c250.max()

    y = np.log(c[-10:])
    x = np.arange(10)
    b, a = np.polyfit(x, y, 1)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ss if ss > 0 else 0.0

    rets = np.diff(np.log(c))
    long_vol = rets[-120:].std()
    return dict(
        close=c[-1],
        decline=trough / peak - 1,
        down_days=trough_i - peak_i,
        days_since_trough=59 - trough_i60,
        rebound=c[-1] / trough - 1,
        position=(c[-1] - lo250) / (hi250 - lo250) if hi250 > lo250 else 0.5,
        ma5=c[-5:].mean(),
        ma20=c[-20:].mean(),
        vol_ratio=v[-5:].mean() / v[-60:].mean() if v[-60:].mean() > 0 else 0.0,
        slope10=b,
        launch=b * 10 * max(r2, 0.0),
        tight=rets[-30:-5].std() / long_vol if long_vol > 0 else 1.0,
        stop=float(low[-20:].min()),
    )


def select(cands: pd.DataFrame, stats: dict[str, dict], top_n: int) -> pd.DataFrame:
    rows = [{**r, **stats[r["代码"]]} for _, r in cands.iterrows() if stats.get(r["代码"])]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    lo, hi = config.BOTTOM_REBOUND_RANGE
    ok = (
        (df["decline"] <= -config.BOTTOM_MIN_DECLINE)
        & (df["down_days"] >= config.BOTTOM_MIN_DOWN_DAYS)
        & (df["days_since_trough"] >= 3)
        & df["rebound"].between(lo, hi)
        & (df["position"] <= config.BOTTOM_MAX_POSITION)
        & (df["close"] > df["ma20"]) & (df["ma5"] > df["ma20"])
        & (df["vol_ratio"] >= config.BOTTOM_MIN_VOL_RATIO)
        & (df["slope10"] > 0)
    )
    df = df[ok].copy()
    print(f"底部启动：{len(rows)} 只候选，{len(df)} 只满足全部条件")
    if df.empty:
        return df
    factors = {
        "vol_surge": np.log(df["vol_ratio"]),
        "launch": df["launch"],
        "depth": -df["decline"],
        "base": -df["tight"],
        "low_pos": -df["position"],
        "early": -df["rebound"],
    }
    contrib = pd.DataFrame({f: config.BOTTOM_WEIGHTS[f] * neutral_z(v) for f, v in factors.items()})
    df["score"] = contrib.sum(axis=1)
    df["reasons"] = [[LABELS[f] for f, v in row.sort_values(ascending=False).items() if v > 0.05][:3]
                     for _, row in contrib.iterrows()]
    df["risks"] = [risk_notes(r) for _, r in df.iterrows()]
    df["pool"] = len(df)
    return diversify(df.sort_values("score", ascending=False), top_n)


def risk_notes(r) -> list[str]:
    out = []
    if not r["市盈率-动态"] > 0:
        out.append("公司亏损")
    if r["rebound"] > 0.25:
        out.append(f"已反弹{r['rebound']:.0%}")
    if r["换手率"] > 15:
        out.append(f"换手率{r['换手率']:.0f}%，筹码不稳")
    if (r["close"] - r["stop"]) / r["close"] > 0.15:
        out.append("距底部止损位较远")
    return out
