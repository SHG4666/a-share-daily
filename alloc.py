"""
名额分配：根据各策略近期的真实表现，用贝叶斯方法分配每天的推荐名额

1. 每只历史推荐的“奖励” = 推荐后每天的平均超额收益
     有 3 日数据用 (3日收益 − 沪深300 3日收益) / 3，只有 1 日数据用次日超额收益
   先验：如果有夜间回测结果，以回测的日均超额（打折后）作为起点
2. 时间衰减：越近的推荐权重越大，半衰期 HALF_LIFE 个交易日（市场风格会变）
3. 每个策略的期望收益看作正态分布：
     均值 = 加权平均奖励向 0 收缩（先验：没有证据时认为策略不赚不亏）
     标准差 = σ / √(有效样本数 + 先验样本数)（样本越少越不确定）
4. 蒙特卡洛模拟 5000 次，统计每个策略“是当前最好策略”的概率 P(best)
5. 名额按 P(best) 比例分配：表现好的多给；数据少、不确定的策略仍有机会（自动探索）
"""
import datetime as dt

import numpy as np
import pandas as pd

import config
from data import DATA_DIR

PATH = DATA_DIR / "alloc.csv"


def rewards(rec: pd.DataFrame) -> pd.DataFrame:
    ev = rec.dropna(subset=["r1"]).copy()
    if ev.empty:
        return ev
    for k in (1, 3):
        ev[f"x{k}"] = (ev[f"r{k}"] - ev[f"b{k}"].fillna(0)) / k
    ev["reward"] = ev["x3"].where(ev["r3"].notna(), ev["x1"])
    return ev


def posteriors(rec: pd.DataFrame, names: list[str], tdays: list[dt.date], day: dt.date,
               bt: dict | None = None) -> dict:
    """bt：夜间回测结果 {策略: {"ex": 日均超额, ...}}，作为先验均值；没有时先验为 0"""
    bt = bt or {}
    ev = rewards(rec)
    idx = {d.isoformat(): i for i, d in enumerate(tdays)}
    today_i = idx.get(day.isoformat(), len(tdays))
    out = {}
    for n in names:
        x = ev[ev["strategy"] == n] if len(ev) else ev
        if len(x):
            age = np.array([today_i - idx.get(d, today_i) for d in x["date"]], float)
            w = 0.5 ** (age / config.HALF_LIFE)
            r = x["reward"].to_numpy(float)
            n_eff = float(w.sum())
            mean_raw = float((w * r).sum() / n_eff)
            sigma = float(np.sqrt((w * (r - mean_raw) ** 2).sum() / n_eff)) if len(x) >= 3 else config.PRIOR_SIGMA
            sigma = max(sigma, 0.008)
        else:
            n_eff, mean_raw, sigma = 0.0, 0.0, config.PRIOR_SIGMA
        if n in bt and pd.notna(bt[n]["ex"]):
            mu0, n0 = config.BT_PRIOR_SHRINK * float(bt[n]["ex"]), config.BT_PRIOR_N
        else:
            mu0, n0 = 0.0, config.PRIOR_N
        k = n_eff + n0
        out[n] = dict(n=len(x), n_eff=n_eff, raw=mean_raw, mean=(mean_raw * n_eff + mu0 * n0) / k,
                      sd=sigma / np.sqrt(k), bt=bt.get(n))
    return out


def allocate(post: dict, avail: dict[str, int], total: int, seed: int) -> dict:
    names = [n for n in post if avail.get(n, 0) > 0]
    slots = {n: 0 for n in post}
    if not names:
        return {"slots": slots, "p_best": {n: 0.0 for n in post}, "total": 0, "cut": False}

    rng = np.random.default_rng(seed)
    draws = np.column_stack([rng.normal(post[n]["mean"], post[n]["sd"], 5000) for n in names])
    p = np.bincount(draws.argmax(axis=1), minlength=len(names)) / 5000
    p_best = {n: 0.0 for n in post} | dict(zip(names, p.round(3)))

    # 风控：证据充分且所有策略近期都跑输，减少推荐数
    total_eff = sum(post[n]["n_eff"] for n in post)
    cut = total_eff >= 10 and max(post[n]["mean"] for n in names) < 0
    if cut:
        total = max(2, total - 2)

    # 按 P(best) 比例分配（最大余数法），不超过各策略可用候选数
    remaining = total
    quota = {n: p_best[n] * total for n in names}
    for n in names:
        slots[n] = min(int(quota[n]), avail[n])
        remaining -= slots[n]
    order = sorted(names, key=lambda n: (quota[n] - int(quota[n]), p_best[n]), reverse=True)
    while remaining > 0:
        progressed = False
        for n in order:
            if remaining and slots[n] < avail[n]:
                slots[n] += 1; remaining -= 1; progressed = True
        if not progressed:
            break
    return {"slots": slots, "p_best": p_best, "total": total, "cut": cut}


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_csv(PATH, dtype={"date": str})
    return pd.DataFrame(columns=["date", "strategy", "slots", "mean", "n_eff", "p_best"])


def save(hist: pd.DataFrame, day: dt.date, post: dict, res: dict):
    rows = pd.DataFrame([{"date": day.isoformat(), "strategy": n, "slots": res["slots"].get(n, 0),
                          "mean": round(post[n]["mean"], 5), "n_eff": round(post[n]["n_eff"], 2),
                          "p_best": res["p_best"].get(n, 0.0)} for n in post])
    hist = hist[hist["date"] != day.isoformat()]
    DATA_DIR.mkdir(exist_ok=True)
    pd.concat([hist, rows], ignore_index=True).to_csv(PATH, index=False)


def previous_slots(hist: pd.DataFrame, day: dt.date) -> dict:
    prev = hist[hist["date"] < day.isoformat()]
    if prev.empty:
        return {}
    last = prev[prev["date"] == prev["date"].max()]
    return dict(zip(last["strategy"], last["slots"].astype(int)))
