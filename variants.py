"""
策略库：7 类策略 × 若干参数版本（变体）

每个变体 = 一类策略的规则（硬条件 + 打分因子）+ 一组参数。
所有变体都基于 features.py 的同一套指标，实盘和回测完全一致。
每晚回测给所有变体打分，表现最好的几个“上场”参与每日推荐（见 roster.py）。

输入 U：某一天的全市场截面，每行一只股票，包含 features 指标 + 名称/industry/PE/PB/s1 等静态信息。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from model import diversify, neutral_z

FAMILY_TAG = {"accum": "🕵️", "momentum": "🔥", "pullback": "↩️", "bottom": "🚀", "stable": "🛡️",
              "breakout": "💥", "oversold": "🪃", "lowvol": "🐢", "limitup": "⚡"}
FAMILY_NAME = {"accum": "主力吸筹", "momentum": "强势延续", "pullback": "趋势回踩", "bottom": "底部启动", "stable": "稳健多因子",
               "breakout": "放量突破", "oversold": "超跌反弹", "lowvol": "低波稳涨",
               "limitup": "涨停回踩"}


def limit_of(codes: pd.Series) -> np.ndarray:
    return np.where(codes.str.startswith(("688", "300", "301")), 20.0, 10.0)


# ================================================================ 各类策略规则
def _momentum(U, p):
    lim = limit_of(U["code"])
    m = ((U["pct"] > 0) & (U["pct"] < lim - 2)
         & (U["ma5"] > U["ma10"]) & (U["ma10"] > U["ma20"]) & (U["ma20"] > U["ma60"])
         & U["r20"].between(*p["r20"]) & (U["r2_20"] >= p["r2"]) & (U["near_high60"] >= p["near"])
         & (U["dist20"] <= p["bias"]) & (U["r5"] <= p["r5max"]) & U["v5_20"].between(*p["vt"]))
    F = lambda d: {"trend": (.35, d["trend20"], "上涨稳定且有力"),  # noqa: E731
                   "up": (.20, d["up10"], "上涨天数占比高"),
                   "ext": (.20, -d["dist20"], "偏离均线不大，未过热"),
                   "vol": (.15, np.log(d["v5_20"].clip(upper=2.0)), "量能温和放大"),
                   "high": (.10, d["near_high60"], "贴近新高")}
    return m, F


def _pullback(U, p):
    m = (U["pct"].between(0.01, 4.0) & (U["ma20"] > U["ma60"]) & (U["ma60"] > U["ma60_lag10"])
         & (U["trend60"] > 0) & (U["r2_60"] >= p["r2"]) & U["pull20"].between(*p["pull"])
         & (U["touch20"] <= 0.02) & (U["dist20"] >= -0.02) & (U["v5_20"] <= p["vmax"]))
    F = lambda d: {"trend": (.35, d["trend60"], "中期上升趋势稳定"),  # noqa: E731
                   "shrink": (.25, -d["v5_20"], "回调缩量，抛压轻"),
                   "support": (.20, -d["dist20"].abs(), "贴近20日线支撑"),
                   "strength": (.20, d["r60"], "前期走势强")}
    return m, F


def _bottom(U, p):
    m = ((U["pct"] >= 1.0) & (U["decline"] <= -p["decline"]) & (U["down_days"] >= p["down"])
         & (U["since_trough"] >= 3) & U["rebound"].between(*p["rebound"]) & (U["pos250"] <= p["pos"])
         & (U["close"] > U["ma20"]) & (U["ma5"] > U["ma20"]) & (U["v5_60"] >= p["vr"]) & (U["slope10"] > 0))
    F = lambda d: {"vol": (.25, np.log(d["v5_60"]), "放量明显，资金进场"),  # noqa: E731
                   "launch": (.25, d["trend10"], "启动走势干净有力"),
                   "depth": (.15, -d["decline"], "前期跌幅深，反弹空间大"),
                   "base": (.15, -d["tight"], "底部波动收敛，抛压衰竭"),
                   "low": (.10, -d["pos250"], "仍处于低位"),
                   "early": (.10, -d["rebound"], "刚启动，位置早")}
    return m, F


def _stable(U, p):
    s1 = U["s1"]
    top = s1.rank(ascending=False, method="first") <= p["keep"]
    m = (top & s1.notna() & (U["slope60"] > 0) & (U["r2_60"] >= p["r2"]) & (U["close"] > U["ma60"]))
    F = lambda d: {"trend": (.35, d["trend60"], "上涨趋势稳定"),  # noqa: E731
                   "fund": (.25, d["s1"], "盈利质量/估值在同行业中占优"),
                   "lowvol": (.20, -d["vol60"], "波动率低"),
                   "rev": (.10, -d["r5"], "近期有回调，不追高"),
                   "dd": (.10, d["mdd60"], "回撤控制好")}
    return m, F


def _breakout(U, p):
    lim = limit_of(U["code"])
    m = ((U["close"] > U["prev_high60"]) & (U["box60"] <= p["box"]) & (U["v5_20"] >= p["vr"])
         & (U["pct"] >= 2.0) & (U["pct"] < lim - 1.5) & (U["r20"] <= 0.40))
    F = lambda d: {"vol": (.30, np.log(d["v5_20"]), "放量突破，资金认可"),  # noqa: E731
                   "box": (.30, -d["box60"], "突破前横盘整理充分"),
                   "trend": (.20, d["trend20"], "突破前走势已转强"),
                   "fresh": (.20, -d["r20"], "累计涨幅不大，刚突破")}
    return m, F


def _oversold(U, p):
    m = ((U["r5"] <= p["r5"]) & (U["dist20"] <= p["dist"]) & U["pct"].between(0.5, 6.0)
         & (U["r60"] >= -0.35) & (U["v5_20"] >= 0.9))
    F = lambda d: {"drop": (.30, -d["r5"], "短期跌幅大，反弹空间足"),  # noqa: E731
                   "dev": (.30, -d["dist20"], "远离20日线，超跌"),
                   "bounce": (.20, d["pct"], "今日已出现反弹"),
                   "vol": (.20, d["v5_20"], "有资金承接")}
    return m, F


def _lowvol(U, p):
    m = ((U["vol60"] <= p["vol"]) & U["r60"].between(0.05, 0.50) & (U["r2_60"] >= p["r2"])
         & (U["close"] > U["ma20"]) & (U["dist20"] <= 0.08))
    F = lambda d: {"trend": (.40, d["trend60"], "稳步上涨"),  # noqa: E731
                   "lowvol": (.30, -d["vol60"], "波动小"),
                   "dd": (.20, d["mdd60"], "回撤小"),
                   "ext": (.10, -d["dist20"], "未偏离均线")}
    return m, F


def _limitup(U, p):
    """涨停后的回踩/承接：涨停当天买不进，等之后缩量回踩、不破涨停支撑时介入。
    since：距最近一次涨停的交易日数；hold：涨停后最低点相对涨停收盘价的下限；vz：今日量/涨停当天量 上限"""
    m = (U["since_zt"].between(*p["since"]) & (U["hold_zt"] >= p["hold"]) & (U["v_zt"] <= p["vz"])
         & U["pct"].between(*p["pct"]) & (U["close"] >= U["zt_lowp"]) & (U["r20"] <= 0.60)
         & (U["ma5"] >= U["ma20"]))
    if p.get("first"):
        m &= U["zt_first"] == 1
    if p.get("mcap_max"):  # 小盘低价：涨停统计显示启动价低、市值小的首板更容易走出连板
        mc = U["mcap"] if "mcap" in U.columns else pd.Series(np.nan, index=U.index)
        m &= (mc <= p["mcap_max"]) & (U["close"] <= p["px_max"])
    F = lambda d: {"shrink": (.30, -d["v_zt"], "涨停后明显缩量，抛压轻"),  # noqa: E731
                   "hold": (.30, d["hold_zt"], "回踩不破涨停价，承接强"),
                   "trend": (.20, d["trend20"], "短期趋势向上"),
                   "fresh": (.20, -d["r20"], "累计涨幅不大，位置不高")}
    return m, F


def _accum(U, p):
    """疑似主力吸筹建仓：深跌后的低位，40 日横盘不涨，但成交量比之前明显放大、上涨日量能占优、OBV 上行，底部抬高。
    即“量增价平、资金净流入却不拉价”——还没启动，是建仓的特征，而不是拉升或出货"""
    m = ((U["pos250"] <= p["pos"]) & (U["dd250"] <= -p["dd"]) & (U["box40"] <= p["box"])
         & (U["r40"].abs() <= 0.12) & (U["r5"] <= 0.08) & (U["r20"] <= 0.15)
         & (U["updown40"] >= p["ud"]) & (U["obv_trend"] > p["obv"]) & (U["vol_up"] >= p["vu"])
         & (U["low_rise"] >= -0.02) & (U["vol_squeeze"] <= 1.1))
    F = lambda d: {"flow": (.25, np.log(d["updown40"].clip(upper=4)), "上涨日成交量明显多于下跌日，资金净流入"),  # noqa: E731
                   "obv": (.25, d["obv_trend"], "OBV 上行而股价横盘，量价背离"),
                   "vol": (.15, np.log(d["vol_up"].clip(upper=4)), "横盘期成交量比之前放大"),
                   "box": (.15, -d["box40"], "箱体窄，筹码在收集"),
                   "lowrise": (.10, d["low_rise"], "底部逐步抬高"),
                   "low": (.10, -d["pos250"], "处于一年低位")}
    return m, F


RULES = {"accum": _accum, "momentum": _momentum, "pullback": _pullback, "bottom": _bottom, "stable": _stable,
         "breakout": _breakout, "oversold": _oversold, "lowvol": _lowvol,
         "limitup": _limitup}


def stop_price(fam: str, d: pd.DataFrame) -> pd.Series:
    if fam in ("momentum",):
        return d["ma20"]
    if fam in ("pullback",):
        return d["ma20"] * 0.97
    if fam in ("bottom",):
        return d["low20"]
    if fam in ("breakout", "oversold"):
        return d["low5"]
    if fam in ("lowvol",):
        return d["ma60"]
    if fam in ("limitup",):
        return d["zt_lowp"]
    if fam in ("accum",):
        return d["low40"] * 0.98  # 跌破箱体底部说明吸筹判断失效
    return d["close"] - 2 * d["atr14"]


def detail(fam: str, r) -> str:
    try:
        if fam == "momentum":
            return (f"20日 {r['r20']:+.0%}（R² {r['r2_20']:.2f}）｜近10日 {int(r['up10'])} 天上涨｜连涨 {int(r['consec'])} 天｜"
                    f"偏离20日线 {r['dist20']:+.0%}｜量能 {r['v5_20']:.1f} 倍")
        if fam == "pullback":
            return (f"60日 {r['r60']:+.0%}（R² {r['r2_60']:.2f}）｜从高点回踩 {r['pull20']:.0%}｜"
                    f"缩量至 {r['v5_20']:.1f} 倍｜距20日线 {r['dist20']:+.1%}")
        if fam == "bottom":
            return (f"前期跌幅 {r['decline']:.0%}（跌了 {int(r['down_days'])} 天）｜距底部反弹 {r['rebound']:+.0%}｜"
                    f"年内位置 {r['pos250']:.0%}｜量能 {r['v5_60']:.1f} 倍")
        if fam == "stable":
            return (f"60日 {r['r60']:+.0%}（R² {r['r2_60']:.2f}）｜波动 {r['vol60']:.0%}｜60日回撤 {r['mdd60']:.0%}｜"
                    f"PE {r['市盈率-动态']:.0f}")
        if fam == "breakout":
            return (f"突破60日新高｜突破前箱体 {r['box60']:.0%}｜量能 {r['v5_20']:.1f} 倍｜20日 {r['r20']:+.0%}")
        if fam == "oversold":
            return (f"5日 {r['r5']:+.0%}｜距20日线 {r['dist20']:+.0%}｜60日 {r['r60']:+.0%}｜量能 {r['v5_20']:.1f} 倍")
        if fam == "accum":
            return (f"距一年高点 {r['dd250']:.0%}｜40日箱体 {r['box40']:.0%}（40日 {r['r40']:+.0%}）｜"
                    f"上涨/下跌日量比 {r['updown40']:.1f}｜横盘期量能 {r['vol_up']:.1f} 倍｜"
                    f"OBV 40日净增 {r['obv_trend']:+.0%}｜近20日低点较前20日 {r['low_rise']:+.1%}")
        if fam == "limitup":
            return (f"{int(r['since_zt'])} 天前{'首板' if r['zt_first'] == 1 else '涨停'}｜涨停后最低 {r['hold_zt']:+.1%}（相对涨停价）｜"
                    f"今日量为涨停日的 {r['v_zt']:.0%}｜近20日涨停 {int(r['zt_n20'])} 次｜20日 {r['r20']:+.0%}")
        if fam == "lowvol":
            return (f"60日 {r['r60']:+.0%}（R² {r['r2_60']:.2f}）｜波动 {r['vol60']:.0%}｜60日回撤 {r['mdd60']:.0%}")
    except Exception:  # noqa: BLE001
        pass
    return ""


def risks(fam: str, r) -> list[str]:
    out = []
    if r.get("consec", 0) >= 5:
        out.append(f"已连涨{int(r['consec'])}天，警惕短期回调")
    if r.get("r5", 0) > 0.15 and fam != "oversold":
        out.append(f"近5日已涨{r['r5']:.0%}")
    if not (r.get("市盈率-动态", 1) > 0):
        out.append("公司亏损")
    if r.get("vol60", 0) > 0.6:
        out.append(f"波动率高（{r['vol60']:.0%}）")
    if fam == "oversold":
        out.append("超跌反弹属于博短线，失败要果断止损")
    return out


# ================================================================ 变体
@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    params: dict = field(default_factory=dict, hash=False, compare=False)

    @property
    def tag(self) -> str:
        return FAMILY_TAG[self.family]


VARIANTS = [
    Variant("强势延续", "momentum", dict(r20=(0.08, 0.45), r2=0.60, near=0.95, bias=0.15, r5max=0.18, vt=(0.9, 2.5))),
    Variant("强势延续·激进", "momentum", dict(r20=(0.15, 0.60), r2=0.50, near=0.97, bias=0.20, r5max=0.25, vt=(1.0, 3.0))),
    Variant("强势延续·稳健", "momentum", dict(r20=(0.05, 0.30), r2=0.70, near=0.93, bias=0.10, r5max=0.10, vt=(0.8, 2.0))),
    Variant("趋势回踩", "pullback", dict(pull=(-0.15, -0.04), r2=0.50, vmax=0.90)),
    Variant("趋势回踩·深", "pullback", dict(pull=(-0.25, -0.10), r2=0.40, vmax=1.00)),
    Variant("趋势回踩·浅", "pullback", dict(pull=(-0.08, -0.02), r2=0.60, vmax=0.80)),
    Variant("底部启动", "bottom", dict(decline=0.30, down=40, rebound=(0.05, 0.25), pos=0.45, vr=1.3)),
    Variant("底部启动·深跌", "bottom", dict(decline=0.45, down=40, rebound=(0.05, 0.20), pos=0.30, vr=1.5)),
    Variant("底部启动·宽松", "bottom", dict(decline=0.20, down=20, rebound=(0.03, 0.30), pos=0.55, vr=1.2)),
    Variant("稳健多因子", "stable", dict(keep=300, r2=0.40)),
    Variant("稳健多因子·严格", "stable", dict(keep=150, r2=0.60)),
    Variant("放量突破", "breakout", dict(vr=1.8, box=0.35)),
    Variant("放量突破·强", "breakout", dict(vr=2.5, box=0.25)),
    Variant("超跌反弹", "oversold", dict(r5=-0.10, dist=-0.08)),
    Variant("超跌反弹·深", "oversold", dict(r5=-0.15, dist=-0.12)),
    Variant("低波稳涨", "lowvol", dict(vol=0.30, r2=0.60)),
    Variant("低波稳涨·宽", "lowvol", dict(vol=0.40, r2=0.50)),
    # 涨停系（来自每日涨停复盘：首板后缩量回踩不破涨停价，是常见的“主力洗盘后二波”形态）
    Variant("主力吸筹", "accum", dict(pos=0.35, dd=0.30, box=0.25, ud=1.2, obv=0.0, vu=1.1)),
    Variant("主力吸筹·严格", "accum", dict(pos=0.30, dd=0.35, box=0.18, ud=1.4, obv=0.05, vu=1.25)),
    Variant("涨停回踩", "limitup", dict(since=(2, 8), hold=-0.05, vz=0.60, pct=(-3.0, 3.0), first=True)),
    Variant("涨停回踩·紧", "limitup", dict(since=(2, 6), hold=-0.02, vz=0.50, pct=(-2.0, 2.0), first=True)),
    Variant("涨停次日承接", "limitup", dict(since=(1, 1), hold=-0.04, vz=1.80, pct=(-3.0, 4.0), first=True)),
    # 小盘低价版（股票池本身要求市值≥50亿，这里在池内再取市值≤100亿、股价≤15元）
    Variant("涨停回踩·小盘低价", "limitup", dict(since=(2, 8), hold=-0.05, vz=0.60, pct=(-3.0, 3.0), first=True,
                                           mcap_max=100e8, px_max=15.0)),
    Variant("涨停次日承接·小盘低价", "limitup", dict(since=(1, 1), hold=-0.04, vz=1.80, pct=(-3.0, 4.0), first=True,
                                             mcap_max=100e8, px_max=15.0)),
]
BY_NAME = {v.name: v for v in VARIANTS}
DEFAULT_ROSTER = ["强势延续", "趋势回踩", "底部启动", "稳健多因子"]


def family_of(name: str) -> str | None:
    v = BY_NAME.get(name)
    if v:
        return v.family
    for fam, cn in FAMILY_NAME.items():
        if name.startswith(cn):
            return fam
    return None


def tag(name: str) -> str:
    return FAMILY_TAG.get(family_of(name) or "", "•")


# ================================================================ 引擎
def run(v: Variant, U: pd.DataFrame, k: int, with_text: bool = False) -> pd.DataFrame:
    """在截面 U 上执行变体 v，返回得分最高、行业分散后的前 k 只"""
    if U.empty:
        return U
    mask, F = RULES[v.family](U, v.params)
    d = U[mask.fillna(False)]
    if d.empty:
        return d
    facs = F(d)
    contrib = pd.DataFrame({k_: w * neutral_z(s.astype(float)) for k_, (w, s, _) in facs.items()}, index=d.index)
    d = d.assign(score=contrib.sum(axis=1).astype(float), strategy=v.name)
    d = d.assign(stop=np.maximum(stop_price(v.family, d), d["close"] * (1 - config.MAX_STOP_LOSS)))
    d = diversify(d.sort_values("score", ascending=False), k)
    if with_text:
        labels = {k_: lab for k_, (_, _, lab) in facs.items()}
        c = contrib.loc[d.index]
        d = d.assign(
            reasons=[[labels[f] for f, x in row.sort_values(ascending=False).items() if x > 0.05][:3]
                     for _, row in c.iterrows()],
            risks=[risks(v.family, r) for _, r in d.iterrows()],
            detail=[detail(v.family, r) for _, r in d.iterrows()],
            pool=int(mask.sum()))
    return d


def s1_scores(static: pd.DataFrame) -> pd.Series:
    """基本面截面分（行业中性化 EP/BP/ROE/低换手），按代码索引；用于稳健多因子"""
    df = static[(static["市盈率-动态"] > 0) & (static["市净率"] > 0) & (static["换手率"] > 0)].copy()
    if df.empty:
        return pd.Series(dtype=float)
    fac = {"ep": 1 / df["市盈率-动态"], "bp": 1 / df["市净率"], "roe": df["市净率"] / df["市盈率-动态"],
           "low_turn": -np.log(df["换手率"])}
    s1 = sum(w * neutral_z(fac[f], df["industry"]) for f, w in config.STAGE1_WEIGHTS.items())
    return pd.Series(s1.values, index=df["代码"].values)


# ================================================================ 截面构造（实盘与回测共用）
STATIC_COLS = ["代码", "名称", "industry", "市盈率-动态", "市净率", "换手率", "最新价", "涨跌幅", "总市值"]


def make_panel(static: pd.DataFrame, hists: dict, dates: set | None) -> pd.DataFrame:
    """指标长表 + 静态信息（名称、行业、PE/PB、基本面分 s1）"""
    import features
    F = features.panel(hists, dates)
    if F.empty:
        return F
    st = static[[c for c in STATIC_COLS if c in static.columns]].drop_duplicates("代码").copy()
    st["s1"] = st["代码"].map(s1_scores(st))
    df = F.merge(st, left_on="code", right_on="代码", how="inner")
    if "总市值" in df.columns:  # 历史市值 ≈ 当前总股本 × 当时前复权价
        df["mcap"] = (df["总市值"] / df["最新价"] * df["close"]).astype("float32")
    return df


def tradable(U: pd.DataFrame) -> pd.DataFrame:
    """当天可交易：成交额达标、不在涨跌停（涨跌停买不进卖不出）"""
    lim = limit_of(U["code"])
    return U[(U["amount"] >= config.MIN_AMOUNT) & (U["pct"].abs() < lim - 0.3) & U["close"].notna()]
