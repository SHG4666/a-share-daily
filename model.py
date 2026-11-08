"""
两阶段量化选股模型

第一阶段（全市场截面）：先选“好公司”
  因子：盈利收益率 EP、账面市值比 BP、推算 ROE（=PB/PE）、低换手
  处理：MAD 去极值 → 行业内标准化（行业中性）→ 加权合成

第二阶段（候选池时间序列）：再选“好时机”
  趋势强度：对 ln(价格) 做 60 日线性回归，年化斜率 × R²（涨得多且涨得稳）
  低波动：60 日年化波动率
  回撤：60 日最大回撤
  短期反转：近 5 日涨幅（A 股短期反转效应显著，不追急涨）
  硬条件：趋势向上、R² 达标、站上 60 日均线
"""
import math

import numpy as np
import pandas as pd

import config

LABELS = {
    "trend": "上涨趋势稳定性强",
    "fundamental": "盈利质量/估值在同行业中占优",
    "low_vol": "波动率低",
    "reversal": "近期有回调，不追高",
    "drawdown": "回撤控制好",
}


# ---------------------------------------------------------------- 统计工具
def winsorize_mad(s: pd.Series, k: float = 5.0) -> pd.Series:
    med = s.median()
    mad = 1.4826 * (s - med).abs().median()
    if not mad or math.isnan(mad):
        return s
    return s.clip(med - k * mad, med + k * mad)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def neutral_z(s: pd.Series, groups: pd.Series | None = None, min_group: int = 5) -> pd.Series:
    """去极值后标准化；有行业信息时在行业内部标准化，消除行业偏差（比如银行天然低 PE）"""
    s = winsorize_mad(s)
    out = zscore(s)
    if groups is not None:
        for _, idx in s.groupby(groups).groups.items():
            if len(idx) >= min_group:
                out.loc[idx] = zscore(s.loc[idx])
    return out.fillna(0.0)


# ---------------------------------------------------------------- 行业补全
NAME_INDUSTRY = [("银行", "银行"), ("证券", "证券"), ("保险", "保险"), ("人寿", "保险")]


def fill_industry(ind: pd.Series, names: pd.Series) -> pd.Series:
    """行业缺失时按名称识别金融股（低估值、低换手，最容易在截面比较中霸榜）"""
    ind = ind.copy()
    for kw, label in NAME_INDUSTRY:
        mask = ind.isna() & names.str.contains(kw, na=False)
        ind[mask] = label
    return ind


# ---------------------------------------------------------------- 股票池
def universe(spot: pd.DataFrame) -> pd.DataFrame:
    df = spot.dropna(subset=["最新价", "涨跌幅"]).copy()
    df = df[~df["名称"].str.contains("ST|退", na=False)]
    df = df[~df["代码"].str.startswith(config.EXCLUDE_PREFIX)]
    df = df[(df["总市值"] >= config.MIN_MARKET_CAP) & (df["成交额"] >= config.MIN_AMOUNT)]
    limit = np.where(df["代码"].str.startswith(("688", "300", "301")), 20.0, 10.0)
    df = df[df["涨跌幅"].abs() < limit - 0.3]  # 涨跌停买不进卖不出，排除
    return df


# ---------------------------------------------------------------- 第一阶段
def stage1(univ: pd.DataFrame, ind_map: dict[str, str]) -> pd.DataFrame:
    df = univ[(univ["市盈率-动态"] > 0) & (univ["市净率"] > 0) & (univ["换手率"] > 0)].copy()
    df["ep"] = 1 / df["市盈率-动态"]
    df["bp"] = 1 / df["市净率"]
    df["roe"] = df["市净率"] / df["市盈率-动态"]
    df["low_turn"] = -np.log(df["换手率"])
    df["industry"] = fill_industry(df["代码"].map(ind_map) if ind_map else pd.Series(None, index=df.index, dtype=object),
                                   df["名称"])
    groups = df["industry"]
    df["s1"] = sum(w * neutral_z(df[f], groups) for f, w in config.STAGE1_WEIGHTS.items())
    print(f"第一阶段：{len(df)} 只盈利股参与打分，行业覆盖 {df['industry'].notna().mean():.0%}")
    return df.sort_values("s1", ascending=False).head(config.STAGE1_KEEP)


# ---------------------------------------------------------------- 第二阶段
def ts_stats(h: pd.DataFrame) -> dict | None:
    if len(h) < config.MIN_HISTORY_DAYS:
        return None
    c = h["close"].to_numpy(float)
    w = config.TREND_WINDOW
    y = np.log(c[-w:])
    x = np.arange(w)
    b, a = np.polyfit(x, y, 1)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - (a + b * x)) ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    rets = np.diff(np.log(c))
    seg = c[-60:]
    hi, lo, prev = h["high"].to_numpy(float), h["low"].to_numpy(float), np.roll(c, 1)
    tr = np.maximum.reduce([hi - lo, np.abs(hi - prev), np.abs(lo - prev)])[1:]
    return dict(
        close=c[-1],
        trend_ann=math.exp(b * 250) - 1,
        r2=r2,
        vol=rets[-60:].std() * math.sqrt(250),
        mdd=(seg / np.maximum.accumulate(seg) - 1).min(),
        rev5=c[-1] / c[-6] - 1,
        ma60=seg.mean(),
        atr=float(np.mean(tr[-14:])),
    )


def stage2(cands: pd.DataFrame, stats: dict[str, dict], top_n: int) -> pd.DataFrame:
    rows = [{**r, **stats[r["代码"]]} for _, r in cands.iterrows() if stats.get(r["代码"])]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["trend_ann"] > 0) & (df["r2"] >= config.MIN_TREND_R2) & (df["close"] > df["ma60"])].copy()
    print(f"第二阶段：{len(df)} 只通过趋势硬条件")
    if df.empty:
        return df
    factors = {
        "trend": df["trend_ann"] * df["r2"],
        "fundamental": df["s1"],
        "low_vol": -df["vol"],
        "reversal": -df["rev5"],
        "drawdown": df["mdd"],
    }
    contrib = pd.DataFrame({f: config.STAGE2_WEIGHTS[f] * neutral_z(v) for f, v in factors.items()})
    df["score"] = contrib.sum(axis=1)
    df["reasons"] = [
        [LABELS[f] for f, v in row.sort_values(ascending=False).items() if v > 0.1][:3]
        for _, row in contrib.iterrows()
    ]
    df["stop"] = df["close"] - config.ATR_MULT * df["atr"]
    df["risks"] = [risk_notes(r) for _, r in df.iterrows()]
    df["pool"] = len(df)
    return diversify(df.sort_values("score", ascending=False), top_n)


def diversify(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """同一行业最多选 1 只，分散风险"""
    ind = fill_industry(df["industry"], df["名称"])
    seen, keep = set(), []
    for i, g in ind.items():
        if isinstance(g, str):
            if g in seen:
                continue
            seen.add(g)
        keep.append(i)
        if len(keep) == top_n:
            break
    return df.loc[keep]


def risk_notes(r) -> list[str]:
    out = []
    if r["rev5"] > 0.10:
        out.append(f"近5日已涨{r['rev5']:.0%}")
    if r["vol"] > 0.55:
        out.append(f"波动率偏高（{r['vol']:.0%}）")
    if r["trend_ann"] > 2.0:
        out.append("趋势斜率过陡，警惕均值回归")
    return out


# ---------------------------------------------------------------- 市场状态
def market_regime(idx: pd.DataFrame | None) -> dict:
    if idx is None or len(idx) < 60:
        return {"strong": True, "text": "未知"}
    c = idx["close"]
    ma60 = c.iloc[-60:].mean()
    r20 = c.iloc[-1] / c.iloc[-21] - 1
    strong = c.iloc[-1] > ma60
    text = f"沪深300 {'站上' if strong else '跌破'} 60 日线，近 20 日 {r20:+.1%}"
    return {"strong": strong, "text": text}


def watch_tag(s: dict | None) -> str:
    if not s:
        return "数据不足"
    if s["trend_ann"] > 0 and s["r2"] >= 0.5 and s["close"] > s["ma60"]:
        t = f"📈 稳健上行（年化{s['trend_ann']:.0%}，R² {s['r2']:.2f}）"
    elif s["close"] < s["ma60"]:
        t = "📉 60日线下方"
    else:
        t = f"〰️ 震荡（R² {s['r2']:.2f}）"
    return f"{t}｜5日 {s['rev5']:+.1%}"
