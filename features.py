"""
全量向量化指标：对一只股票的整段历史，一次算出每一天的全部指标。

实盘（取最后一天）和夜间回测（取历史上的每一天）用的是同一份计算，保证回测里验证过的策略就是实盘在跑的策略。
所有指标只用“当天及以前”的数据；以 f 开头的前瞻收益（f1/f3/f5/f10/up5）只用于回测评分，选股时绝不使用。
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

MIN_ROWS = 130  # 少于这个行数的不计算


def _roll_linreg(y: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """滚动线性回归（对 log 价格），闭式解：返回 (斜率, R²)"""
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    r = lambda s: s.rolling(n, min_periods=n).sum()  # noqa: E731
    sy, st, sty, stt, syy = r(y), r(t), r(t * y), r(t * t), r(y * y)
    sxy = n * sty - st * sy
    sxx = n * stt - st * st
    syy_ = n * syy - sy * sy
    slope = sxy / sxx
    r2 = (sxy * sxy) / (sxx * syy_)
    return slope, r2.clip(0, 1).fillna(0)


def _window_stat(arr: np.ndarray, n: int, fn) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) >= n:
        out[n - 1:] = fn(swv(arr, n))
    return out


def compute(h: pd.DataFrame, lim: float = 10.0) -> pd.DataFrame | None:
    """lim：涨跌停幅度（%），主板 10，创业板/科创板 20"""
    if h is None or len(h) < MIN_ROWS:
        return None
    h = h.reset_index(drop=True)
    c = h["close"].astype(float)
    hi, lo, v = h["high"].astype(float), h["low"].astype(float), h["vol"].astype(float)
    logc = np.log(c)
    ret = logc.diff()
    f = pd.DataFrame({"date": h["date"], "close": c})

    f["pct"] = (c / c.shift(1) - 1) * 100
    f["amount"] = c * v * 100  # 成交量单位：手
    for n in (5, 10, 20, 60):
        f[f"ma{n}"] = c.rolling(n).mean()
    f["ma60_lag10"] = f["ma60"].shift(10)
    for n in (5, 20, 60):
        f[f"r{n}"] = c / c.shift(n) - 1
    for n in (10, 20, 60):
        s, r2 = _roll_linreg(logc, n)
        f[f"slope{n}"], f[f"r2_{n}"] = s, r2
        f[f"trend{n}"] = s * n * r2  # 窗口内对数涨幅 × 拟合优度：涨得多且涨得稳
    vm5, vm20, vm60 = v.rolling(5).mean(), v.rolling(20).mean(), v.rolling(60).mean()
    f["v5_20"], f["v5_60"] = vm5 / vm20, vm5 / vm60
    f["near_high60"] = c / c.rolling(60).max()
    f["pull20"] = c / c.rolling(20).max() - 1
    f["touch20"] = lo.rolling(3).min() / f["ma20"] - 1
    f["dist20"] = c / f["ma20"] - 1
    up = (c.diff() > 0).astype(float)
    f["up10"] = up.rolling(10).sum()
    grp = (up == 0).cumsum()
    f["consec"] = up.groupby(grp).cumsum()
    f["vol60"] = ret.rolling(60).std() * np.sqrt(250)
    f["prev_high60"] = c.shift(1).rolling(60).max()
    f["box60"] = c.shift(5).rolling(55).max() / c.shift(5).rolling(55).min() - 1  # 突破前的箱体宽度
    prev = c.shift(1)
    tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    f["atr14"] = tr.rolling(14).mean()
    f["low20"] = lo.rolling(20).min()
    f["low5"] = lo.rolling(5).min()

    ca = c.to_numpy()
    f["mdd60"] = _window_stat(ca, 60, lambda w: (w / np.maximum.accumulate(w, axis=1) - 1).min(axis=1))

    # 底部特征（需要一年数据）
    if len(ca) >= 250:
        w = swv(ca, 250)
        peak_i = w.argmax(axis=1)
        tr_i = w[:, 190:].argmin(axis=1) + 190
        rows = np.arange(len(w))
        pad = np.full(249, np.nan)
        f["decline"] = np.r_[pad, w[rows, tr_i] / w[rows, peak_i] - 1]
        f["down_days"] = np.r_[pad, (tr_i - peak_i).astype(float)]
        f["since_trough"] = np.r_[pad, (249 - tr_i).astype(float)]
        wmin, wmax = w.min(axis=1), w.max(axis=1)
        f["pos250"] = np.r_[pad, np.where(wmax > wmin, (w[:, -1] - wmin) / (wmax - wmin), 0.5)]
        f["rebound"] = np.r_[pad, w[:, -1] / w[rows, tr_i] - 1]
    else:
        for k in ("decline", "down_days", "since_trough", "pos250", "rebound"):
            f[k] = np.nan
    f["tight"] = ret.shift(5).rolling(25).std() / ret.rolling(120).std()

    # 涨停特征（日线近似：涨幅达到涨停幅度且收在最高价 = 封板；最高价触及涨停但没封住 = 炸板）
    pct = f["pct"]
    zt = (pct >= lim - 0.2) & (c >= hi * 0.9995)
    f["zt"] = zt.astype(float)
    f["zb"] = ((hi / prev - 1) * 100 >= lim - 0.2).astype(float) * (~zt)
    f["board"] = zt.astype(float).groupby((~zt).cumsum()).cumsum()       # 当前连板数（今天没涨停为 0）
    f["zt_n20"] = zt.astype(float).rolling(20).sum()                      # 近 20 日涨停次数
    pos = pd.Series(np.arange(len(c), dtype=float))
    f["since_zt"] = pos - pos.where(zt).ffill()                           # 距最近一次涨停的交易日数
    first = zt & (zt.astype(float).rolling(21, min_periods=1).sum() == 1)  # 前 20 日没涨停过的“首板”
    f["zt_first"] = first.astype(float).where(zt).ffill()                 # 最近一次涨停是否首板
    f["zt_close"] = c.where(zt).ffill()
    f["zt_lowp"] = lo.where(zt).ffill()                                   # 涨停当天最低价（常用支撑位）
    f["v_zt"] = v / v.where(zt).ffill()                                   # 今日量 / 涨停当天量
    seg = zt.cumsum()
    low_since = lo.mask(zt, np.inf).groupby(seg).cummin().replace(np.inf, np.nan)
    f["hold_zt"] = low_since / f["zt_close"] - 1                          # 涨停后最低点相对涨停收盘价

    # 前瞻收益（只用于回测评分）
    for k in (1, 3, 5, 10):
        f[f"f{k}"] = c.shift(-k) / c - 1
    upf = (c.shift(-1) > c).astype(float)
    f["up5"] = sum(upf.shift(-i) for i in range(5))
    f.loc[f["f5"].isna(), "up5"] = np.nan
    return f


def panel(hists: dict, dates: set | None = None) -> pd.DataFrame:
    """多只股票的指标拼成长表（code, date, 指标...）；dates 给定时只保留这些日期，节省内存"""
    parts = []
    for code, h in hists.items():
        f = compute(h, 20.0 if str(code).startswith(("688", "300", "301")) else 10.0)
        if f is None:
            continue
        if dates is not None:
            f = f[f["date"].isin(dates)]
        if len(f):
            f.insert(0, "code", code)
            parts.append(f)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    fl = df.select_dtypes("float64").columns
    df[fl] = df[fl].astype("float32")
    return df
