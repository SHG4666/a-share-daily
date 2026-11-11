"""数据获取层：所有 akshare 调用都在这里"""
import datetime as dt
import os
import threading
import time
from functools import wraps
from pathlib import Path

import pandas as pd

TZ_CN = dt.timezone(dt.timedelta(hours=8))
DATA_DIR = Path(__file__).parent / "data"


def retry(times=3, wait=3):
    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            last = None
            for i in range(times):
                try:
                    return fn(*a, **kw)
                except Exception as e:  # noqa: BLE001
                    last = e
                    print(f"[retry {i + 1}/{times}] {fn.__name__}{a}: {e}")
                    time.sleep(wait * (i + 1))
            raise last
        return wrapper
    return deco


EM_STATE = {"dead": False}  # 东方财富一旦被判定限流，本次运行不再请求它


def em_call(fn, *a, timeout: float = 30, **kw):
    """带硬超时调用东方财富接口；失败或超时即标记限流，后续请求全部走备用源"""
    if EM_STATE["dead"]:
        raise RuntimeError("东方财富已限流，本次跳过")
    box = {}

    def run():
        try:
            box["v"] = fn(*a, **kw)
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=run, daemon=True)  # daemon：超时后不阻塞进程退出
    t.start()
    t.join(timeout)
    if t.is_alive():
        EM_STATE["dead"] = True
        raise TimeoutError(f"东方财富请求超过 {timeout:.0f}s，判定限流")
    if "e" in box:
        EM_STATE["dead"] = True
        raise box["e"]
    return box["v"]


def today_cn() -> dt.date:
    return dt.datetime.now(TZ_CN).date()


def target_day(tdays: list[dt.date], now: dt.datetime | None = None) -> tuple[dt.date | None, str]:
    """根据当前北京时间决定复盘哪一个交易日。

    GitHub 定时任务经常延迟几个小时，甚至拖到第二天凌晨才触发，所以不能简单地用“今天”：
      - 交易日 15:30 之后：复盘今天
      - 交易日 9:15–15:30（盘中）：行情不完整，不运行
      - 其他时间（凌晨、早盘前、周末）：复盘上一个交易日（此时行情快照正是它的收盘数据）
    """
    now = now or dt.datetime.now(TZ_CN)
    today, hm = now.date(), now.hour * 100 + now.minute
    if today in tdays:
        if hm >= 1530:
            return today, "收盘后"
        if hm >= 915:
            return None, "盘中，行情不完整"
    prev = [t for t in tdays if t < today]
    return (prev[-1] if prev else None), "复盘上一交易日"


def num(s):
    return pd.to_numeric(s, errors="coerce")


def to_date(s) -> pd.Series:
    return pd.to_datetime(s).dt.date


@retry()
def trade_days() -> list[dt.date]:
    import akshare as ak
    return sorted(to_date(ak.tool_trade_date_hist_sina()["trade_date"]))


SPOT_COLS = ["代码", "名称", "最新价", "涨跌幅", "成交额", "换手率", "市盈率-动态", "市净率", "总市值",
             "昨收", "最高", "最低", "成交量"]


def market_symbol(code: str) -> str:
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _spot_em() -> pd.DataFrame:
    import akshare as ak
    df = em_call(ak.stock_zh_a_spot_em, timeout=150)
    DATA_DIR.mkdir(exist_ok=True)  # 缓存代码表，供备用源使用
    df[["代码"]].rename(columns={"代码": "code"}).to_csv(DATA_DIR / "codes.csv", index=False)
    return df


def _spot_tencent() -> pd.DataFrame:
    """备用源：腾讯行情批量接口（不走东方财富，不受其限流影响）"""
    import requests
    cache = DATA_DIR / "codes.csv"
    if cache.exists():
        codes = pd.read_csv(cache, dtype={"code": str})["code"].str.zfill(6).tolist()
    else:  # 没有缓存时枚举沪深全部号段，腾讯只返回真实存在的代码
        ranges = [(600000, 606000), (688000, 690000), (1, 4000), (300000, 303000)]
        codes = [f"{i:06d}" for a, b in ranges for i in range(a, b)]
    print("腾讯源：代码表", len(codes), "只")
    rows = []
    for i in range(0, len(codes), 300):
        q = ",".join(market_symbol(c) for c in codes[i:i + 300])
        txt = retry()(requests.get)("http://qt.gtimg.cn/q=" + q, timeout=15).content.decode("gbk", "ignore")
        for line in txt.split(";"):
            if '"' not in line:
                continue
            f = line.split('"')[1].split("~")
            if len(f) < 47:
                continue
            rows.append({"代码": f[2], "名称": f[1], "最新价": f[3], "涨跌幅": f[32], "成交额": f[37],
                         "换手率": f[38], "市盈率-动态": f[39], "市净率": f[46], "总市值": f[45],
                         "昨收": f[4], "最高": f[33], "最低": f[34], "成交量": f[36]})  # 成交量单位：手
            if f[2] == "600519":
                print("腾讯样本(茅台):", {k: v for k, v in rows[-1].items()})
        time.sleep(0.1)
    if len(rows) < 3000:
        raise RuntimeError(f"腾讯行情只返回 {len(rows)} 只，数据不完整")
    df = pd.DataFrame(rows)
    df["成交额"] = num(df["成交额"]) * 1e4   # 万元 -> 元
    df["总市值"] = num(df["总市值"]) * 1e8   # 亿元 -> 元
    return df


def get_spot() -> pd.DataFrame:
    sources = [("东方财富", _spot_em), ("腾讯", _spot_tencent)]
    if os.getenv("SPOT_SOURCE") == "tencent":
        sources.reverse()
    last = None
    for name, fn in sources:
        try:
            df = fn()
            for c in SPOT_COLS[2:]:
                df[c] = num(df[c]) if c in df.columns else float("nan")
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            df.loc[df["最新价"] <= 0, "最新价"] = float("nan")  # 停牌
            print(f"行情来源：{name}，{len(df)} 只")
            return df
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"行情源 {name} 失败：{str(e)[:120]}")
    raise last


def _hist_em(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    df = em_call(ak.stock_zh_a_hist, symbol=code, period="daily", start_date=start,
                 end_date=end, adjust="qfq", timeout=20)
    return df.rename(columns={"日期": "date", "收盘": "close", "最高": "high", "最低": "low", "成交量": "vol"})


def _hist_sina(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_daily(symbol=market_symbol(code), start_date=start, end_date=end, adjust="qfq")
    df = df.rename(columns={"volume": "vol"})
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce") / 100  # 新浪是股，统一成手（与东方财富、腾讯一致）
    return df


HIST_DAYS = 600  # 拉取多少个自然日的历史（约 400 个交易日，够夜间回测回放 60 天 + 底部启动所需一年）


def get_hist(code: str, days: int | None = None) -> pd.DataFrame:
    """前复权日线，列：date close high low vol"""
    days = days or HIST_DAYS
    end = today_cn()
    start, end_s = (end - dt.timedelta(days=days)).strftime("%Y%m%d"), end.strftime("%Y%m%d")
    df = None
    if not EM_STATE["dead"]:
        try:
            df = _hist_em(code, start, end_s)
        except Exception as e:  # noqa: BLE001
            print(f"东方财富历史数据失败（{code}），后续切换新浪：{str(e)[:80]}")
    if df is None:
        df = retry(times=2, wait=2)(_hist_sina)(code, start, end_s)
    df["date"] = to_date(df["date"])
    for c in ["close", "high", "low", "vol"]:
        df[c] = num(df[c])
    return df[["date", "close", "high", "low", "vol"]].dropna().reset_index(drop=True)


@retry()
def get_index(symbol: str = "sh000300") -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = to_date(df["date"])
    df["close"] = num(df["close"])
    return df[["date", "close"]].reset_index(drop=True)


def get_top_industries(n: int) -> pd.DataFrame | None:
    try:
        import akshare as ak
        df = em_call(ak.stock_board_industry_name_em, timeout=30)
        df["涨跌幅"] = num(df["涨跌幅"])
        return df.sort_values("涨跌幅", ascending=False).head(n)
    except Exception as e:  # noqa: BLE001
        print("行业涨幅（东方财富）失败，改用新浪:", str(e)[:60])
    try:
        import akshare as ak
        df = ak.stock_sector_spot(indicator="新浪行业").rename(columns={"板块": "板块名称"})
        df["涨跌幅"] = num(df["涨跌幅"])
        return df.sort_values("涨跌幅", ascending=False).head(n)[["板块名称", "涨跌幅"]]
    except Exception as e:  # noqa: BLE001
        print("行业涨幅（新浪）失败:", e)
        return None


def _industry_em() -> tuple[list, bool]:
    import akshare as ak
    boards = em_call(ak.stock_board_industry_name_em, timeout=30)["板块名称"].tolist()
    rows = []
    for b in boards:
        try:
            cons = em_call(ak.stock_board_industry_cons_em, symbol=b, timeout=20)
            rows += [(str(c).zfill(6), b) for c in cons["代码"]]
        except Exception as e:  # noqa: BLE001
            print("东方财富行业成分失败", b, str(e)[:80])
            return rows, False
        time.sleep(0.3)
    return rows, True


def _industry_sina() -> tuple[list, bool]:
    import akshare as ak
    sec = ak.stock_sector_spot(indicator="新浪行业")
    rows, ok = [], True
    for _, r in sec.iterrows():
        try:
            d = ak.stock_sector_detail(sector=r["label"])
            rows += [(str(c).zfill(6), r["板块"]) for c in d["code"]]
        except Exception as e:  # noqa: BLE001
            print("新浪行业成分失败", r.get("板块"), str(e)[:80])
            ok = False
        time.sleep(0.3)
    return rows, ok


def get_industry_map(max_age_days: int) -> dict[str, str]:
    """股票 -> 行业。东方财富优先，失败用新浪；完整结果缓存到 data/industry.csv"""
    path = DATA_DIR / "industry.csv"
    cached = None
    if path.exists():
        cached = pd.read_csv(path, dtype={"code": str})
        updated = dt.date.fromisoformat(str(cached["updated"].iloc[0]))
        if (today_cn() - updated).days < max_age_days:
            return dict(zip(cached["code"], cached["industry"]))
    best = []
    for name, fn in [("东方财富", _industry_em), ("新浪", _industry_sina)]:
        try:
            rows, complete = fn()
        except Exception as e:  # noqa: BLE001
            print(f"行业分类（{name}）失败：{str(e)[:120]}")
            continue
        print(f"行业分类（{name}）：{len(rows)} 条，{'完整' if complete else '不完整'}")
        if complete and len(rows) > 3000:
            df = pd.DataFrame(rows, columns=["code", "industry"]).drop_duplicates("code")
            df["updated"] = today_cn().isoformat()
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(path, index=False)
            return dict(zip(df["code"], df["industry"]))
        if len(rows) > len(best):
            best = rows
    if cached is not None:
        return dict(zip(cached["code"], cached["industry"]))
    return dict(best)
