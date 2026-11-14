"""
每日盘面数据摘要（给“每日股市观点报告”用）

收盘后运行，输出 data/market/YYYY-MM-DD.txt（给人和 AI 读的文字摘要）和同名 .json。
报告本身由 Claude 定时任务根据这份摘要 + 当天新闻撰写，写好后提交到 data/reports/，由 report.yml 推送飞书。

摘要内容：
  1. 主要指数：收盘、涨跌、5/20 日涨跌、相对 20/60 日均线位置、近 20 日高低点（作为支撑/压力参考）
  2. 量能：全市场成交额，及（缓存覆盖的约 3000 只）成交额相对 20 日均值
  3. 涨跌结构：上涨/下跌家数、涨停/跌停、中位数涨幅，以及近 10 个交易日的变化
  4. 风格：大盘 / 中盘 / 小盘的平均涨跌
  5. 行业：领涨、领跌行业
  6. 资金焦点：成交额前 15 的个股
  7. 主力行为线索（量价异动，属推测）：
     - 高位放量冲高回落：20 日涨 ≥20%，当天量 ≥ 20 日均量 2.5 倍，收盘在当日振幅下沿（上影长）→ 疑似派发
     - 高位放量下跌：20 日涨 ≥20%，当天放量 2 倍以上且跌 ≥5% → 疑似出货/踩踏
     - 趋势中缩量回调：20 日线在 60 日线上、60 日涨 ≥15%，从 20 日高点回撤 5–12%，近 5 日量 ≤ 20 日均量 0.7 倍 → 疑似洗盘
     - 底部放量异动：一年内从高点跌 ≥40%，当天量 ≥ 20 日均量 3 倍且涨 ≥5% → 疑似建仓/试盘
     - 放量突破新高：收盘创 60 日新高，量 ≥ 20 日均量 2 倍 → 疑似拉升
  8. 今日明星股档案（最多 8 只）：成交额第一、成交额前20中涨/跌幅最大、最高连板、涨停股中成交额最大、
     底部放量异动和高位放量下跌中成交额最大的。每只给出 1/5/20/60 日走势、一年区间位置、量比、
     上涨日/下跌日成交量比（资金进出的代理）、20/60 日成交量加权均价（近期持股成本区）、K 线形态、
     连板数、龙虎榜席位，以及按规则推测的阶段（吸筹/洗盘/拉升/派发/出逃）
  9. 情绪（来自涨停复盘 limitup.py 的摘要，如有）
  10. 量化系统视角：上场策略、今日推荐、最新回测结论
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

import config
import data
import histstore
import model

OUT = data.DATA_DIR / "market"
INDEXES = [("sh000001", "上证指数"), ("sz399001", "深证成指"), ("sz399006", "创业板指"),
           ("sh000300", "沪深300"), ("sh000852", "中证1000"), ("sh000688", "科创50")]


def pct(x, nd=2):
    return "-" if x is None or pd.isna(x) else f"{x * 100:+.{nd}f}%"


def yi(x):
    return "-" if x is None or pd.isna(x) else f"{x / 1e8:.1f}亿"


def index_lines(day: dt.date) -> tuple[list[str], dict]:
    lines, js = [], {}
    for sym, name in INDEXES:
        try:
            df = data.get_index(sym)
            df = df[df["date"] <= day].reset_index(drop=True)
            c = df["close"].astype(float)
            if len(c) < 61 or df["date"].iloc[-1] != day:
                continue
            ma20, ma60 = c.tail(20).mean(), c.tail(60).mean()
            hi20, lo20 = c.tail(20).max(), c.tail(20).min()
            r = dict(close=float(c.iloc[-1]), d1=float(c.iloc[-1] / c.iloc[-2] - 1), d5=float(c.iloc[-1] / c.iloc[-6] - 1),
                     d20=float(c.iloc[-1] / c.iloc[-21] - 1), ma20=float(ma20), ma60=float(ma60),
                     hi20=float(hi20), lo20=float(lo20))
            js[name] = r
            lines.append(f"{name} {r['close']:.2f}（{pct(r['d1'])}，5日{pct(r['d5'])}，20日{pct(r['d20'])}）"
                         f"｜20日线 {ma20:.0f}（{'上方' if r['close'] > ma20 else '下方'}）"
                         f"｜60日线 {ma60:.0f}（{'上方' if r['close'] > ma60 else '下方'}）"
                         f"｜近20日区间 {lo20:.0f}–{hi20:.0f}")
        except Exception as e:  # noqa: BLE001
            print("指数失败", sym, e)
    return lines, js


def breadth_history(store: histstore.Store, days: list[dt.date]) -> pd.DataFrame:
    """用历史缓存算近 N 个交易日的涨跌家数、涨停家数、成交额（覆盖缓存中的股票）"""
    rows = {d: [0, 0, 0, 0, 0.0] for d in days}
    want = set(days)
    for code, h in store.h.items():
        if h is None or len(h) < 2:
            continue
        t = h.tail(len(days) + 1)
        c, v, ds = t["close"].to_numpy(float), t["vol"].to_numpy(float), list(t["date"])
        lim = 0.195 if code.startswith(("688", "300", "301")) else 0.097
        for i in range(1, len(ds)):
            d = ds[i]
            if d not in want:
                continue
            r = c[i] / c[i - 1] - 1
            row = rows[d]
            row[0] += r > 0
            row[1] += r < 0
            row[2] += r >= lim
            row[3] += r <= -lim
            row[4] += c[i] * v[i] * 100
    df = pd.DataFrame([[d, *v] for d, v in rows.items()], columns=["date", "up", "down", "zt", "dt", "amount"])
    return df.sort_values("date")


def anomalies(store: histstore.Store, spot: pd.DataFrame, day: dt.date) -> dict[str, list[dict]]:
    """量价异动：主力行为线索（推测）"""
    info = spot.set_index("代码")
    liquid = set(spot.loc[(spot["成交额"] >= 3e8) & ~spot["名称"].str.contains("ST|退", na=False), "代码"])
    out = {k: [] for k in ["高位放量冲高回落", "高位放量下跌", "趋势中缩量回调", "底部放量异动", "放量突破新高"]}
    for code in liquid:
        h = store.get(code)
        if h is None or len(h) < 250 or h["date"].iloc[-1] != day:
            continue
        c, v = h["close"].to_numpy(float), h["vol"].to_numpy(float)
        hi, lo = h["high"].to_numpy(float), h["low"].to_numpy(float)
        vr = v[-1] / v[-21:-1].mean() if v[-21:-1].mean() > 0 else np.nan
        r1, r20, r60 = c[-1] / c[-2] - 1, c[-2] / c[-22] - 1, c[-1] / c[-61] - 1
        rng_ = hi[-1] - lo[-1]
        upper = (hi[-1] - c[-1]) / rng_ if rng_ > 0 else 0
        ma20, ma60 = c[-20:].mean(), c[-60:].mean()
        pull = c[-1] / c[-20:].max() - 1
        v5_20 = v[-5:].mean() / v[-20:].mean()
        dec = c[-250:].min() / c[-250:].max() - 1 if c[-250:].argmax() < c[-250:].argmin() else 0
        name = info["名称"].get(code, code)
        base = dict(code=code, name=name, pct=r1, vr=vr, amount=float(info["成交额"].get(code, np.nan)))
        if r20 >= 0.20 and vr >= 2.5 and upper >= 0.6:
            out["高位放量冲高回落"].append({**base, "r20": r20, "upper": upper})
        if r20 >= 0.20 and vr >= 2.0 and r1 <= -0.05:
            out["高位放量下跌"].append({**base, "r20": r20})
        if ma20 > ma60 and r60 >= 0.15 and -0.12 <= pull <= -0.05 and v5_20 <= 0.7:
            out["趋势中缩量回调"].append({**base, "r60": r60, "pull": pull, "v5_20": v5_20})
        if dec <= -0.40 and vr >= 3 and r1 >= 0.05:
            out["底部放量异动"].append({**base, "dec": dec})
        if c[-1] > c[-61:-1].max() and vr >= 2:
            out["放量突破新高"].append({**base, "r20": r20})
    for k in out:
        out[k].sort(key=lambda x: -(x["amount"] if pd.notna(x["amount"]) else 0))
    return out


# ================================================================ 当日明星股档案
def limit_pct(code: str, name: str = "") -> float:
    if "ST" in str(name):
        return 0.05
    return 0.20 if code.startswith(("688", "689", "300", "301")) else 0.10


def consec_limit(c: np.ndarray, lim: float) -> int:
    n = 0
    for i in range(len(c) - 1, 0, -1):
        if c[i] / c[i - 1] - 1 >= lim - 0.003:
            n += 1
        else:
            break
    return n


def pick_stars(spot: pd.DataFrame, store: histstore.Store, day: dt.date, an: dict) -> list[tuple[str, str]]:
    """挑当日明星股：返回 [(代码, 入选理由)]，最多 8 只"""
    s = spot[~spot["名称"].str.contains("ST|退", na=False) & spot["成交额"].notna()].copy()
    top20 = s.sort_values("成交额", ascending=False).head(20)
    picks: list[tuple[str, str]] = []

    def add(code, why):
        if code and code not in [c for c, _ in picks]:
            picks.append((code, why))

    if len(top20):
        add(top20.iloc[0]["代码"], "全市场成交额第一，资金焦点")
        up = top20.sort_values("涨跌幅", ascending=False).iloc[0]
        if up["涨跌幅"] > 0:
            add(up["代码"], f"成交额前20中涨幅最大（{up['涨跌幅']:+.1f}%），资金抢筹最强")
        dn = top20.sort_values("涨跌幅").iloc[0]
        if dn["涨跌幅"] < 0:
            add(dn["代码"], f"成交额前20中跌幅最大（{dn['涨跌幅']:+.1f}%），资金出逃最凶")
    # 连板最高（情绪龙头）
    best, best_n = None, 1
    for r in s[s["涨跌幅"] >= 9.5].itertuples():
        h = store.get(r.代码)
        if h is None or h["date"].iloc[-1] != day:
            continue
        n = consec_limit(h["close"].to_numpy(float)[-15:], limit_pct(r.代码, r.名称))
        if n > best_n or (n == best_n and best is not None and r.成交额 > best[1]):
            best, best_n = (r.代码, r.成交额), n
    if best and best_n >= 2:
        add(best[0], f"{best_n}连板，市场最高标，情绪龙头")
    # 涨停股里成交额最大（题材核心）
    zt = s[[r.涨跌幅 / 100 >= limit_pct(r.代码, r.名称) - 0.003 for r in s.itertuples()]]
    if len(zt):
        z = zt.sort_values("成交额", ascending=False).iloc[0]
        add(z["代码"], f"涨停股中成交额最大（{z['成交额'] / 1e8:.0f}亿），题材核心")
    # 底部放量异动、高位放量下跌里成交额最大的
    for k, why in [("底部放量异动", "底部放量异动中成交额最大，疑似资金建仓/试盘"),
                   ("高位放量下跌", "高位放量下跌中成交额最大，疑似出货")]:
        if an.get(k):
            add(an[k][0]["code"], why)
    return picks[:8]


def stage_of(f: dict) -> tuple[str, list[str]]:
    """基于量价和成本的阶段推测（规则），返回 (阶段, 依据)"""
    ev = []
    if f["r20"] >= 0.25 and ((f["upper"] >= 0.5 and f["vr20"] >= 2) or (f["updown"] < 1 and f["close"] < f["ma5"])):
        ev.append("20日涨幅已大" + ("，今日放量收长上影" if f["upper"] >= 0.5 else "，近20日下跌日成交量反超上涨日且跌破5日线"))
        return "高位派发嫌疑", ev
    if f["close"] < f["ma20"] < f["ma60"] and f["updown"] < 0.8 and f["r1"] <= -0.03:
        ev.append("均线空头排列，下跌日放量，资金持续流出")
        return "资金出逃/下跌中继", ev
    if f["r20"] >= 0.15 and f["close"] > f["ma5"] > f["ma20"] and f["updown"] >= 1.5:
        ev.append(f"沿5日线上行，上涨日成交量是下跌日的 {f['updown']:.1f} 倍")
        return "主升/拉升阶段", ev
    if f["ma20"] > f["ma60"] and -0.15 <= f["pull20"] <= -0.04 and f["v5_20"] < 0.8:
        ev.append(f"中期趋势向上，从20日高点回撤 {f['pull20']:.0%}，量能缩至 {f['v5_20']:.1f} 倍")
        return "洗盘嫌疑", ev
    if f["pos250"] <= 0.3 and f["updown"] >= 1.3 and -0.05 <= f["r20"] <= 0.20:
        ev.append(f"处于一年区间低位（{f['pos250']:.0%}），上涨日量能占优（{f['updown']:.1f} 倍）")
        return "低位吸筹嫌疑", ev
    if f["r1"] >= 0.05 and f["vr20"] >= 2.5 and f["pos250"] <= 0.4:
        ev.append("低位突然放巨量大涨")
        return "底部异动/试盘", ev
    return "分歧震荡", ["多空信号交织，没有明确阶段特征"]


def star_profile(code: str, name: str, why: str, h: pd.DataFrame, row, lb: dict | None) -> tuple[str, dict]:
    c, v = h["close"].to_numpy(float), h["vol"].to_numpy(float)
    hi, lo = h["high"].to_numpy(float), h["low"].to_numpy(float)
    ret = np.diff(c[-21:]) / c[-21:-1]
    vv = v[-20:]
    f = dict(
        close=c[-1], r1=c[-1] / c[-2] - 1, r5=c[-1] / c[-6] - 1, r20=c[-1] / c[-21] - 1, r60=c[-1] / c[-61] - 1,
        ma5=c[-5:].mean(), ma20=c[-20:].mean(), ma60=c[-60:].mean(),
        vr5=v[-1] / v[-6:-1].mean(), vr20=v[-1] / v[-21:-1].mean(), v5_20=v[-5:].mean() / v[-20:].mean(),
        updown=(vv[ret > 0].sum() / vv[ret < 0].sum()) if vv[ret < 0].sum() > 0 else 9.9,
        vwap20=(c[-20:] * v[-20:]).sum() / v[-20:].sum(), vwap60=(c[-60:] * v[-60:]).sum() / v[-60:].sum(),
        pos250=(c[-1] - c[-250:].min()) / (c[-250:].max() - c[-250:].min()) if c[-250:].max() > c[-250:].min() else 0.5,
        pull20=c[-1] / c[-20:].max() - 1, hi20=c[-20:].max(), lo20=c[-20:].min(),
        upper=(hi[-1] - c[-1]) / (hi[-1] - lo[-1]) if hi[-1] > lo[-1] else 0.0,
        amp=(hi[-1] - lo[-1]) / c[-2], boards=consec_limit(c[-15:], limit_pct(code, name)),
        turnover=float(getattr(row, "换手率", np.nan)), amount=float(getattr(row, "成交额", np.nan)),
    )
    stage, ev = stage_of(f)
    lines = [f"■ {name}（{code}）— {why}",
             f"  走势：收 {f['close']:.2f}，今日 {pct(f['r1'])}，5日 {pct(f['r5'])}，20日 {pct(f['r20'])}，60日 {pct(f['r60'])}；"
             f"一年区间位置 {f['pos250']:.0%}" + (f"；{f['boards']}连板" if f["boards"] >= 2 else ""),
             f"  量能：成交 {yi(f['amount'])}，换手 {f['turnover']:.1f}%，量比（对5日）{f['vr5']:.1f}、（对20日）{f['vr20']:.1f}；"
             f"近20日上涨日/下跌日成交量之比 {f['updown']:.1f}",
             f"  成本：20日成交量加权均价 {f['vwap20']:.2f}（现价{pct(f['close'] / f['vwap20'] - 1, 1)}），"
             f"60日 {f['vwap60']:.2f}（{pct(f['close'] / f['vwap60'] - 1, 1)}）；"
             f"20日线 {f['ma20']:.2f}，60日线 {f['ma60']:.2f}，近20日区间 {f['lo20']:.2f}–{f['hi20']:.2f}",
             f"  当日K线：振幅 {f['amp']:.1%}，收盘位于当日振幅的 {(1 - f['upper']):.0%} 位置" +
             ("（长上影，冲高回落）" if f["upper"] >= 0.6 else "（收在高位）" if f["upper"] <= 0.15 else "")]
    if lb:
        seats = "、".join(lb.get("买方", [])[:5]) if lb.get("买方") else ""
        lines.append(f"  龙虎榜：净买 {yi(lb.get('净买额'))}" + (f"，买方前五 {seats}" if seats else "")
                     + (f"，买一 {lb['买一']}（占前五买入 {lb['买一占比']:.0%}）" if lb.get("买一") and pd.notna(lb.get("买一占比")) else "")
                     + (f"，机构净买 {yi(lb['机构净买'])}" if pd.notna(lb.get("机构净买", np.nan)) else "")
                     + (f"；上榜原因 {lb['原因']}" if lb.get("原因") else ""))
    lines.append(f"  规则推测阶段：{stage}（{'；'.join(ev)}）")
    return "\n".join(lines), {"code": code, "name": name, "why": why, "stage": stage, **{k: float(x) for k, x in f.items()}}


def system_view(day: dt.date) -> list[str]:
    lines = []
    try:
        r = json.loads((data.DATA_DIR / "roster.json").read_text())
        lines.append("上场策略：" + "、".join(r.get("active", [])) + (f"（{r['warn']}）" if r.get("warn") else ""))
    except Exception:  # noqa: BLE001
        pass
    try:
        p = pd.read_csv(data.DATA_DIR / "picks.csv", dtype={"code": str, "date": str})
        t = p[p["date"] == day.isoformat()]
        if len(t):
            lines.append("今日量化推荐：" + "、".join(f"{a}({s})" for a, s in zip(t["name"], t["strategy"])))
        ev = p.dropna(subset=["r5"])
        if len(ev):
            lines.append(f"历史推荐已满5日的 {len(ev)} 只：5日平均 {pct(ev['r5'].mean())}，上涨比例 {(ev['r5'] > 0).mean():.0%}")
    except Exception:  # noqa: BLE001
        pass
    try:
        b = pd.read_csv(data.DATA_DIR / "backtest.csv", dtype={"run_date": str})
        b = b[b["run_date"] == b["run_date"].max()].dropna(subset=["mean5"]).sort_values("mean5", ascending=False)
        rand = b.loc[b["strategy"] == "随机基准", "mean5"]
        lines.append(f"最近一次回测（{b['run_date'].iloc[0]}，过去120个交易日，持有5日）：随机选股平均 {pct(rand.iloc[0]) if len(rand) else '-'}；"
                     "前三 " + "、".join(f"{s} {pct(m)}" for s, m in zip(b["strategy"].head(3), b["mean5"].head(3))) +
                     "；后三 " + "、".join(f"{s} {pct(m)}" for s, m in zip(b["strategy"].tail(3), b["mean5"].tail(3))))
    except Exception:  # noqa: BLE001
        pass
    return lines


def main():
    tdays = data.trade_days()
    day, why = data.target_day(tdays)
    if day is None:
        print(f"盘面摘要跳过：{why}"); return
    print("盘面摘要日期", day)
    spot = data.get_spot()
    spot = spot[~spot["代码"].str.startswith(config.EXCLUDE_PREFIX)]
    store = histstore.Store().load()
    store.append_snapshot(spot, day)
    days = [d for d in tdays if d <= day][-21:]

    L = [f"【盘面数据摘要】{day}（收盘后自动生成）", ""]
    idx_lines, idx_js = index_lines(day)
    L += ["一、主要指数"] + idx_lines + [""]

    bh = breadth_history(store, days)
    bh = bh[bh["date"].isin(days)]
    amt_today = spot["成交额"].sum()
    if len(bh) and bh["amount"].iloc[-1] > 0:  # 缓存口径只用于看相对变化，绝对值按今天快照的全市场成交额缩放
        bh["amount"] = bh["amount"] * amt_today / bh["amount"].iloc[-1]
    ratio = bh["amount"].iloc[-1] / bh["amount"].iloc[-21:-1].mean() if len(bh) >= 6 else np.nan
    L += ["二、量能与涨跌结构",
          f"全市场成交额 {amt_today / 1e12:.2f} 万亿；为近20日均值的 {ratio:.2f} 倍",
          f"上涨 {int((spot['涨跌幅'] > 0).sum())} 家，下跌 {int((spot['涨跌幅'] < 0).sum())} 家，"
          f"中位数涨跌 {spot['涨跌幅'].median():+.2f}%",
          "近10个交易日（成交额按今天口径折算）：" + "；".join(
              f"{r.date:%m-%d} 涨{int(r.up)}/跌{int(r.down)} 涨停{int(r.zt)} 跌停{int(r.dt)} 额{r.amount / 1e12:.2f}万亿"
              for r in bh.tail(10).itertuples()), ""]

    cap = spot.dropna(subset=["总市值", "涨跌幅"])
    cap = cap[cap["成交额"] > 0]
    q = cap["总市值"].quantile([0.2, 0.8])
    big, small = cap[cap["总市值"] >= q[0.8]], cap[cap["总市值"] <= q[0.2]]
    mid = cap[(cap["总市值"] > q[0.2]) & (cap["总市值"] < q[0.8])]
    L += ["三、风格",
          f"大盘股（市值前20%）平均 {big['涨跌幅'].mean():+.2f}%｜中盘 {mid['涨跌幅'].mean():+.2f}%｜"
          f"小盘股（后20%）平均 {small['涨跌幅'].mean():+.2f}%", ""]

    try:
        import akshare as ak
        sec = ak.stock_sector_spot(indicator="新浪行业").rename(columns={"板块": "板块名称"})
        sec["涨跌幅"] = pd.to_numeric(sec["涨跌幅"], errors="coerce")
        sec = sec.sort_values("涨跌幅", ascending=False)
        L += ["四、行业（新浪行业）",
              "领涨：" + "、".join(f"{a}{b:+.2f}%" for a, b in zip(sec["板块名称"].head(6), sec["涨跌幅"].head(6))),
              "领跌：" + "、".join(f"{a}{b:+.2f}%" for a, b in zip(sec["板块名称"].tail(6), sec["涨跌幅"].tail(6))), ""]
    except Exception as e:  # noqa: BLE001
        print("行业失败", e)

    top = spot.sort_values("成交额", ascending=False).head(15)
    L += ["五、资金焦点（成交额前15）",
          "；".join(f"{r.名称} {r.涨跌幅:+.2f}% {yi(r.成交额)} 换手{r.换手率:.1f}%" for r in top.itertuples()), ""]

    an = anomalies(store, spot, day)
    L.append("六、主力行为线索（量价异动，属推测；按成交额排序，每类最多列8只）")
    desc = {"高位放量冲高回落": "20日涨≥20%、放量≥2.5倍、长上影，疑似派发",
            "高位放量下跌": "20日涨≥20%、放量≥2倍、跌≥5%，疑似出货或踩踏",
            "趋势中缩量回调": "上升趋势中回撤5–12%且明显缩量，疑似洗盘",
            "底部放量异动": "一年内跌≥40%后放量≥3倍、涨≥5%，疑似建仓或试盘",
            "放量突破新高": "创60日新高且放量≥2倍，疑似主动拉升"}
    for k, items in an.items():
        ex = "、".join(f"{x['name']}({x['pct'] * 100:+.1f}%,量{x['vr']:.1f}倍)" for x in items[:8])
        L.append(f"{k}（{desc[k]}）：{len(items)} 只" + (f"，如 {ex}" if items else ""))
    L.append("")

    stars_js = []
    try:
        stars = pick_stars(spot, store, day, an)
        info = spot.set_index("代码")
        try:
            import limitup
            data.EM_STATE["dead"] = False
            LB = limitup.lhb(day, [c for c, _ in stars])
        except Exception as e:  # noqa: BLE001
            print("明星股龙虎榜失败:", e); LB = {}
        L.append("七、今日明星股档案（量价、成本与阶段为规则推测，供撰写主力操盘分析用）")
        for code, why in stars:
            h = store.get(code)
            if h is None or len(h) < 250 or h["date"].iloc[-1] != day:
                continue
            row = info.loc[code]
            txt, js = star_profile(code, row["名称"], why, h, row, LB.get(code))
            L.append(txt); stars_js.append(js)
        L.append("")
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print("明星股档案失败:", e)

    zt_file = OUT / f"{day.isoformat()}-limitup.txt"
    if zt_file.exists():
        L += ["八、涨停与情绪（来自涨停复盘）", zt_file.read_text().strip(), ""]
    L += ["九、量化系统视角"] + system_view(day)

    text = "\n".join(L)
    print(text)
    if os.getenv("DRY_RUN") == "1":
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{day.isoformat()}.txt").write_text(text)
    (OUT / f"{day.isoformat()}.json").write_text(json.dumps(
        {"date": day.isoformat(), "indexes": idx_js, "amount": amt_today, "amount_ratio20": ratio,
         "anomalies": {k: v[:20] for k, v in an.items()}, "stars": stars_js}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
