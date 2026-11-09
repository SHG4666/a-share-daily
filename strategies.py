"""
策略库：每个策略用统一接口
  prefilter(univ, ctx) -> DataFrame   用当日快照粗筛候选（决定要拉哪些历史数据）
  select(cands, hists) -> DataFrame   用历史数据精选，返回按得分排序、行业分散后的候选
  detail(row) -> str                  卡片上展示的关键数据

返回的 DataFrame 至少包含：代码 名称 最新价 涨跌幅 close score reasons risks stop industry
"""
import math

import numpy as np
import pandas as pd

import bottom
import config
import model
from model import diversify, neutral_z

N = config.CAND_PER_STRATEGY


def linreg(y: np.ndarray) -> tuple[float, float]:
    """线性回归，返回 (斜率, R²)"""
    x = np.arange(len(y))
    b, a = np.polyfit(x, y, 1)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ss if ss > 0 else 0.0
    return float(b), float(max(r2, 0.0))


def consec_up(c: np.ndarray) -> int:
    n = 0
    for i in range(len(c) - 1, 0, -1):
        if c[i] > c[i - 1]:
            n += 1
        else:
            break
    return n


def _score(df: pd.DataFrame, factors: dict, weights: dict, labels: dict, risk_fn) -> pd.DataFrame:
    contrib = pd.DataFrame({f: weights[f] * neutral_z(v) for f, v in factors.items()})
    df = df.copy()
    df["score"] = contrib.sum(axis=1)
    df["reasons"] = [[labels[f] for f, v in row.sort_values(ascending=False).items() if v > 0.05][:3]
                     for _, row in contrib.iterrows()]
    df["risks"] = [risk_fn(r) for _, r in df.iterrows()]
    df["pool"] = len(df)
    return diversify(df.sort_values("score", ascending=False), N)


def _join(cands: pd.DataFrame, hists: dict, stat_fn) -> pd.DataFrame:
    rows = []
    for _, r in cands.iterrows():
        h = hists.get(r["代码"])
        s = stat_fn(h) if h is not None else None
        if s:
            rows.append({**r, **s})
    return pd.DataFrame(rows)


# ================================================================ 稳健多因子
class Stable:
    name = "稳健多因子"
    desc = "行业内估值/ROE/低换手占优，且 60 日趋势平稳向上"

    def prefilter(self, univ, ctx):
        return model.stage1(univ, ctx["ind_map"])

    def select(self, cands, hists):
        stats = {c: model.ts_stats(h) for c, h in hists.items() if c in set(cands["代码"])}
        return model.stage2(cands, stats, N)

    def detail(self, p):
        return (f"60日趋势年化 {p['trend_ann']:.0%}（R² {p['r2']:.2f}）｜波动 {p['vol']:.0%}｜"
                f"ROE≈{p['roe']:.0%}｜PE {p['市盈率-动态']:.0f}")


# ================================================================ 底部启动
class Bottom:
    name = "底部启动"
    desc = "长期下跌后在低位放量启动"

    def prefilter(self, univ, ctx):
        return bottom.prefilter(univ)

    def select(self, cands, hists):
        stats = {c: bottom.bottom_stats(h) for c, h in hists.items() if c in set(cands["代码"])}
        return bottom.select(cands, stats, N)

    def detail(self, p):
        return (f"前期跌幅 {p['decline']:.0%}｜距底部反弹 {p['rebound']:+.0%}｜年内位置 {p['position']:.0%}｜"
                f"量能 {p['vol_ratio']:.1f} 倍")


# ================================================================ 强势延续
class Momentum:
    """已经在稳步上涨、量能温和放大、还没有过热 —— 押注上涨延续"""
    name = "强势延续"
    desc = "均线多头、20 日稳步上涨、接近新高、未过热"
    LABELS = {"trend": "上涨稳定且有力", "up_days": "上涨天数占比高", "not_extended": "偏离均线不大，未过热",
              "volume": "量能温和放大", "near_high": "贴近新高"}

    def prefilter(self, univ, ctx):
        c = config
        limit = np.where(univ["代码"].str.startswith(("688", "300", "301")), 20.0, 10.0)
        df = univ[(univ["涨跌幅"] > 0) & (univ["涨跌幅"] < limit - 2)
                  & univ["换手率"].between(*c.MOM_TURNOVER_RANGE)]
        return df.sort_values("成交额", ascending=False).head(c.MOM_CANDIDATES)

    @staticmethod
    def stats(h):
        if h is None or len(h) < 120:
            return None
        c, v = h["close"].to_numpy(float), h["vol"].to_numpy(float)
        b20, r2 = linreg(np.log(c[-20:]))
        ma = {n: c[-n:].mean() for n in (5, 10, 20, 60)}
        return dict(
            close=c[-1], r20=c[-1] / c[-21] - 1, r5=c[-1] / c[-6] - 1, r2_20=r2,
            trend20=b20 * 20 * r2, up10=int((np.diff(c[-11:]) > 0).sum()), consec=consec_up(c),
            near_high=c[-1] / c[-60:].max(), bias20=c[-1] / ma[20] - 1,
            vol_trend=v[-5:].mean() / v[-20:].mean() if v[-20:].mean() > 0 else 0.0,
            aligned=ma[5] > ma[10] > ma[20] > ma[60], stop=ma[20],
        )

    def select(self, cands, hists):
        c = config
        df = _join(cands, hists, self.stats)
        if df.empty:
            return df
        lo, hi = c.MOM_R20_RANGE
        ok = (df["aligned"] & df["r20"].between(lo, hi) & (df["r2_20"] >= c.MOM_MIN_R2)
              & (df["near_high"] >= c.MOM_NEAR_HIGH) & (df["bias20"] <= c.MOM_MAX_BIAS20)
              & (df["r5"] <= c.MOM_MAX_R5) & df["vol_trend"].between(*c.MOM_VOL_TREND_RANGE))
        df = df[ok]
        print(f"强势延续：{len(cands)} 只候选，{len(df)} 只满足全部条件")
        if df.empty:
            return df
        factors = {"trend": df["trend20"], "up_days": df["up10"], "not_extended": -df["bias20"],
                   "volume": np.log(df["vol_trend"].clip(upper=2.0)), "near_high": df["near_high"]}
        return _score(df, factors, c.MOM_WEIGHTS, self.LABELS, self.risks)

    @staticmethod
    def risks(r):
        out = []
        if r["consec"] >= 5:
            out.append(f"已连涨{int(r['consec'])}天，警惕短期回调")
        if r["r5"] > 0.12:
            out.append(f"近5日已涨{r['r5']:.0%}")
        if r["换手率"] > 15:
            out.append(f"换手率{r['换手率']:.0f}%，分歧较大")
        return out

    def detail(self, p):
        return (f"20日 {p['r20']:+.0%}（R² {p['r2_20']:.2f}）｜近10日 {int(p['up10'])} 天上涨｜连涨 {int(p['consec'])} 天｜"
                f"偏离20日线 {p['bias20']:+.0%}｜量能 {p['vol_trend']:.1f} 倍")


# ================================================================ 趋势回踩
class Pullback:
    """上升趋势中缩量回调到 20 日线附近、今天企稳 —— 等趋势恢复后的再次上涨"""
    name = "趋势回踩"
    desc = "上升趋势中缩量回踩 20 日线后企稳"
    LABELS = {"trend": "中期上升趋势稳定", "shrink": "回调缩量，抛压轻", "support": "贴近20日线支撑",
              "strength": "前期走势强"}

    def prefilter(self, univ, ctx):
        c = config
        df = univ[univ["涨跌幅"].between(*c.PB_PCT_RANGE) & (univ["换手率"] >= 0.8)]
        return df.sort_values("成交额", ascending=False).head(c.PB_CANDIDATES)

    @staticmethod
    def stats(h):
        if h is None or len(h) < 130:
            return None
        c, v, low = h["close"].to_numpy(float), h["vol"].to_numpy(float), h["low"].to_numpy(float)
        b60, r2 = linreg(np.log(c[-60:]))
        ma20, ma60 = c[-20:].mean(), c[-60:].mean()
        return dict(
            close=c[-1], r60=c[-1] / c[-61] - 1, r2_60=r2, trend60=b60 * 60 * r2,
            ma_ok=(ma20 > ma60) and (ma60 > c[-70:-10].mean()),
            pull=c[-1] / c[-20:].max() - 1, touch=low[-3:].min() / ma20 - 1, dist20=c[-1] / ma20 - 1,
            vol_c=v[-5:].mean() / v[-20:].mean() if v[-20:].mean() > 0 else 9.0,
            up_today=c[-1] > c[-2], stop=ma20 * 0.97,
        )

    def select(self, cands, hists):
        c = config
        df = _join(cands, hists, self.stats)
        if df.empty:
            return df
        lo, hi = c.PB_PULL_RANGE
        ok = (df["ma_ok"] & (df["trend60"] > 0) & (df["r2_60"] >= c.PB_MIN_R2) & df["pull"].between(lo, hi)
              & (df["touch"] <= 0.02) & (df["dist20"] >= -0.02) & (df["vol_c"] <= c.PB_MAX_VOL_RATIO)
              & df["up_today"])
        df = df[ok]
        print(f"趋势回踩：{len(cands)} 只候选，{len(df)} 只满足全部条件")
        if df.empty:
            return df
        factors = {"trend": df["trend60"], "shrink": -df["vol_c"], "support": -df["dist20"].abs(),
                   "strength": df["r60"]}
        return _score(df, factors, c.PB_WEIGHTS, self.LABELS, self.risks)

    @staticmethod
    def risks(r):
        out = []
        if r["r60"] > 0.6:
            out.append(f"60日已涨{r['r60']:.0%}，注意高位")
        if not r["市盈率-动态"] > 0:
            out.append("公司亏损")
        return out

    def detail(self, p):
        return (f"60日 {p['r60']:+.0%}（R² {p['r2_60']:.2f}）｜从高点回踩 {p['pull']:.0%}｜"
                f"缩量至 {p['vol_c']:.1f} 倍｜距20日线 {p['dist20']:+.1%}")


ALL = [Momentum(), Pullback(), Bottom(), Stable()]
BY_NAME = {s.name: s for s in ALL}


def tag(name: str) -> str:
    return {"强势延续": "🔥", "趋势回踩": "↩️", "底部启动": "🚀", "稳健多因子": "🛡️"}.get(name, "•")


def safe_detail(name: str, row) -> str:
    try:
        return BY_NAME[name].detail(row)
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["ALL", "BY_NAME", "tag", "safe_detail", "linreg", "consec_up", "math"]
