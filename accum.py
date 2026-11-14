"""
全市场“疑似主力吸筹建仓”扫描 + 跟踪

规则见 variants._accum（威科夫吸筹的量化版）：
  深跌后的低位（距一年高点 ≥30%、一年区间下 35%）
  40 日横盘（箱体 ≤25%、40 日涨跌 ≤12%），还没启动（5 日 ≤8%、20 日 ≤15%）
  量增价平：近 60 日均量 ≥ 之前 120 日的 1.1 倍
  资金净流入却不拉价：40 日上涨日成交量 ≥ 下跌日 1.2 倍，OBV 上行
  底部不再创新低（近 20 日低点不低于前 20 日 2% 以上），波动收敛

吸筹往往要几周到几个月才见效，所以除了夜间回测（5 日口径）外，这里单独记录每天的扫描结果，
跟踪之后 5 / 10 / 20 个交易日的涨幅，检验“疑似吸筹”到底有没有用。
"""
import datetime as dt

import numpy as np
import pandas as pd

import variants
from data import DATA_DIR

PATH = DATA_DIR / "accum.csv"
COLS = ["date", "code", "name", "close", "stop", "score", "detail", "r5", "r10", "r20", "b5", "b10", "b20"]
SCAN_K = 10


def scan(U: pd.DataFrame, k: int = SCAN_K) -> pd.DataFrame:
    """全市场扫描，返回得分最高的 k 只（行业分散）；pool 列是满足条件的总数"""
    if U is None or U.empty:
        return pd.DataFrame()
    d = variants.run(variants.BY_NAME["主力吸筹"], U, k, with_text=True)
    if len(d):
        strict = variants.run(variants.BY_NAME["主力吸筹·严格"], U, 50)
        d = d.assign(strict=d["代码"].isin(set(strict["代码"])) if len(strict) else False)
    return d


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_csv(PATH, dtype={"code": str, "date": str}).reindex(columns=COLS)
    return pd.DataFrame(columns=COLS)


def record(df: pd.DataFrame, day: dt.date, picks: pd.DataFrame) -> pd.DataFrame:
    df = df[df["date"] != day.isoformat()]
    if picks is None or picks.empty:
        return df
    new = pd.DataFrame({"date": day.isoformat(), "code": picks["代码"].values, "name": picks["名称"].values,
                        "close": picks["close"].round(3).values, "stop": picks["stop"].round(3).values,
                        "score": picks["score"].round(3).values, "detail": picks["detail"].values}).reindex(columns=COLS)
    return pd.concat([df, new], ignore_index=True) if len(df) else new


def evaluate(df: pd.DataFrame, tdays: list[dt.date], day: dt.date, hist_fn, index_df) -> pd.DataFrame:
    df = df.copy()
    recent = {d.isoformat() for d in tdays if d <= day}
    recent = set(sorted(recent)[-40:])
    for i, r in df[df["r20"].isna() & df["date"].isin(recent)].iterrows():
        d = dt.date.fromisoformat(r["date"])
        after = [t for t in tdays if d < t <= day][:20]
        if not after:
            continue
        try:
            h = hist_fn(r["code"])
        except Exception:  # noqa: BLE001
            continue
        m = dict(zip(h["date"], h["close"])) if h is not None else {}
        b = dict(zip(index_df["date"], index_df["close"])) if index_df is not None else {}
        for k in (5, 10, 20):
            if len(after) >= k:
                p0, p1 = m.get(d), m.get(after[k - 1])
                if p0 and p1:
                    df.at[i, f"r{k}"] = round(p1 / p0 - 1, 5)
                b0, b1 = b.get(d), b.get(after[k - 1])
                if b0 and b1:
                    df.at[i, f"b{k}"] = round(b1 / b0 - 1, 5)
    return df


def save(df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True)
    df[COLS].to_csv(PATH, index=False)


def track_line(df: pd.DataFrame) -> str | None:
    parts = []
    for k in (5, 10, 20):
        x = df.dropna(subset=[f"r{k}"])
        if len(x):
            ex = (x[f"r{k}"] - x[f"b{k}"].fillna(0)).mean()
            parts.append(f"{k}日：{len(x)} 只，平均 {x[f'r{k}'].mean():+.1%}，上涨 {(x[f'r{k}'] > 0).mean():.0%}，"
                         f"相对沪深300 {ex:+.1%}")
    return "历史扫描结果跟踪｜" + "；".join(parts) if parts else None


def bt_line() -> str | None:
    """最近一次夜间回测里“主力吸筹”相对随机选股的表现，让读者知道这份名单历史上有没有用"""
    try:
        b = pd.read_csv(DATA_DIR / "backtest.csv", dtype={"run_date": str})
        b = b[b["run_date"] == b["run_date"].max()].set_index("strategy")
        a, r = b.loc["主力吸筹"], b.loc["随机基准"]
        verdict = ("跑赢随机选股" if a["mean5"] > r["mean5"] and a["mean10"] > r["mean10"] else
                   "和随机选股差不多" if abs(a["mean5"] - r["mean5"]) < 0.003 else "跑输随机选股")
        return (f"回测（近120个交易日，{int(a['n'])} 次信号）：5日平均 {a['mean5']:+.1%}、10日平均 {a['mean10']:+.1%}；"
                f"随机选股 5日 {r['mean5']:+.1%}、10日 {r['mean10']:+.1%}——{verdict}，仅作观察名单")
    except Exception:  # noqa: BLE001
        return None
