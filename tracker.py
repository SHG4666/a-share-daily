"""策略追踪：记录每日推荐，N 个交易日后用真实价格回测，统计胜率和超额收益"""
import datetime as dt

import pandas as pd

from data import DATA_DIR

PATH = DATA_DIR / "picks.csv"
COLS = ["date", "code", "name", "close", "score", "ret", "bench"]


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_csv(PATH, dtype={"code": str, "date": str})
    return pd.DataFrame(columns=COLS)


def save(df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True)
    df[COLS].to_csv(PATH, index=False)


def record(df: pd.DataFrame, day: dt.date, picks: pd.DataFrame) -> pd.DataFrame:
    df = df[df["date"] != day.isoformat()]
    new = pd.DataFrame({"date": day.isoformat(), "code": picks["代码"], "name": picks["名称"],
                        "close": picks["close"].round(3), "score": picks["score"].round(3)})
    return pd.concat([df, new], ignore_index=True) if len(df) else new.reindex(columns=COLS)


def _close_on(series: pd.DataFrame, d: dt.date) -> float | None:
    hit = series.loc[series["date"] == d, "close"]
    return float(hit.iloc[0]) if len(hit) else None


def evaluate(df: pd.DataFrame, tdays: list[dt.date], day: dt.date, horizon: int,
             hist_fn, index_df: pd.DataFrame | None) -> pd.DataFrame:
    df = df.copy()
    for i, r in df[df["ret"].isna()].iterrows():
        d = dt.date.fromisoformat(r["date"])
        after = [t for t in tdays if d < t <= day]
        if len(after) < horizon:
            continue
        target = after[horizon - 1]
        try:
            h = hist_fn(r["code"])
            p0, p1 = _close_on(h, d), _close_on(h, target)
            if p0 and p1:
                df.at[i, "ret"] = round(p1 / p0 - 1, 5)
            if index_df is not None:
                b0, b1 = _close_on(index_df, d), _close_on(index_df, target)
                if b0 and b1:
                    df.at[i, "bench"] = round(b1 / b0 - 1, 5)
        except Exception as e:  # noqa: BLE001
            print("回测失败", r["code"], e)
    return df


def summary(df: pd.DataFrame, horizon: int) -> str | None:
    ev = df.dropna(subset=["ret"])
    if ev.empty:
        return None

    def line(x, label):
        ex = (x["ret"] - x["bench"]).mean()
        return (f"{label} {len(x)} 只：{horizon}日胜率 **{(x['ret'] > 0).mean():.0%}**，"
                f"平均收益 **{x['ret'].mean():+.2%}**，相对沪深300 **{ex:+.2%}**")

    out = [line(ev, "累计")]
    if len(ev) > 20:
        out.append(line(ev.tail(20), "最近"))
    return "\n".join(out)
