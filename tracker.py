"""推荐记录与回测：每只推荐记录次日 / 3 日 / 5 日收益、同期沪深300 收益、5 日内上涨天数"""
import datetime as dt

import numpy as np
import pandas as pd

from data import DATA_DIR

PATH = DATA_DIR / "picks.csv"
COLS = ["date", "strategy", "code", "name", "close", "stop", "score",
        "r1", "r3", "r5", "b1", "b3", "b5", "up5"]
LEGACY = {"稳健": "稳健多因子"}


def load() -> pd.DataFrame:
    if not PATH.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(PATH, dtype={"code": str, "date": str})
    if "strategy" not in df.columns:
        df["strategy"] = "稳健多因子"
    df["strategy"] = df["strategy"].replace(LEGACY)
    df["code"] = df["code"].str.zfill(6)
    return df.reindex(columns=COLS)


def save(df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True)
    df[COLS].to_csv(PATH, index=False)


def record(df: pd.DataFrame, day: dt.date, picks: pd.DataFrame) -> pd.DataFrame:
    df = df[df["date"] != day.isoformat()]
    if picks is None or picks.empty:
        return df
    new = pd.DataFrame({"date": day.isoformat(), "strategy": picks["strategy"].values,
                        "code": picks["代码"].values, "name": picks["名称"].values,
                        "close": picks["close"].round(3).values, "stop": picks["stop"].round(3).values,
                        "score": picks["score"].round(3).values}).reindex(columns=COLS)
    return pd.concat([df, new], ignore_index=True) if len(df) else new


def _closes(series: pd.DataFrame | None, days: list[dt.date]) -> list[float | None]:
    if series is None:
        return [None] * len(days)
    m = dict(zip(series["date"], series["close"]))
    return [m.get(d) for d in days]


def evaluate(df: pd.DataFrame, tdays: list[dt.date], day: dt.date, hist_fn,
             index_df: pd.DataFrame | None, lookback: int = 15) -> pd.DataFrame:
    """补算还没完成 5 日评估、且在最近 lookback 个交易日内的推荐"""
    df = df.copy()
    recent = {d.isoformat() for d in tdays if d <= day}
    recent = set(sorted(recent)[-lookback:])
    todo = df[df["r5"].isna() & df["date"].isin(recent)]
    for i, r in todo.iterrows():
        d = dt.date.fromisoformat(r["date"])
        after = [t for t in tdays if d < t <= day][:5]
        if not after:
            continue
        try:
            h = hist_fn(r["code"])
        except Exception as e:  # noqa: BLE001
            print("回测取数失败", r["code"], e)
            continue
        p0, *ps = _closes(h, [d] + after)
        b0, *bs = _closes(index_df, [d] + after)
        for k in (1, 3, 5):
            if len(after) >= k and ps[k - 1] and p0:
                df.at[i, f"r{k}"] = round(ps[k - 1] / p0 - 1, 5)
                if bs[k - 1] and b0:
                    df.at[i, f"b{k}"] = round(bs[k - 1] / b0 - 1, 5)
        if len(after) >= 5 and p0 and all(ps):
            seq = np.array([p0] + ps, float)
            df.at[i, "up5"] = int((np.diff(seq) > 0).sum())
    return df


def stats_by_strategy(df: pd.DataFrame) -> pd.DataFrame:
    ev = df.dropna(subset=["r1"])
    if ev.empty:
        return pd.DataFrame()
    rows = []
    for s, x in ev.groupby("strategy"):
        x3 = x.dropna(subset=["r3"])
        x5 = x.dropna(subset=["up5"])
        rows.append(dict(
            strategy=s, n=len(x), win1=(x["r1"] > 0).mean(), avg1=x["r1"].mean(),
            n3=len(x3), ex3=(x3["r3"] - x3["b3"].fillna(0)).mean() if len(x3) else np.nan,
            up5=x5["up5"].mean() if len(x5) else np.nan,
        ))
    return pd.DataFrame(rows)
