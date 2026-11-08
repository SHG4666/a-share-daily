"""数据获取层：所有 akshare 调用都在这里"""
import datetime as dt
import os
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


def today_cn() -> dt.date:
    return dt.datetime.now(TZ_CN).date()


def num(s):
    return pd.to_numeric(s, errors="coerce")


def to_date(s) -> pd.Series:
    return pd.to_datetime(s).dt.date


@retry()
def trade_days() -> list[dt.date]:
    import akshare as ak
    return sorted(to_date(ak.tool_trade_date_hist_sina()["trade_date"]))


SPOT_COLS = ["代码", "名称", "最新价", "涨跌幅", "成交额", "换手率", "市盈率-动态", "市净率", "总市值"]


def market_symbol(code: str) -> str:
    if code.startswith(("8", "4", "92")):
        return "bj" + code
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def _spot_em() -> pd.DataFrame:
    import akshare as ak
    return ak.stock_zh_a_spot_em()


def _spot_tencent() -> pd.DataFrame:
    """备用源：腾讯行情批量接口（不走东方财富，不受其限流影响）"""
    import akshare as ak
    import requests
    codes = ak.stock_info_a_code_name()["code"].astype(str).str.zfill(6).tolist()
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
                         "换手率": f[38], "市盈率-动态": f[39], "市净率": f[46], "总市值": f[45]})
            if f[2] == "600519":
                print("腾讯样本(茅台):", {k: v for k, v in rows[-1].items()})
        time.sleep(0.1)
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
                df[c] = num(df[c])
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
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
    return df.rename(columns={"日期": "date", "收盘": "close", "最高": "high", "最低": "low", "成交量": "vol"})


def _hist_sina(code: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_daily(symbol=market_symbol(code), start_date=start, end_date=end, adjust="qfq")
    return df.rename(columns={"volume": "vol"})


_hist_em_dead = False  # 东方财富历史接口被限流后，本次运行剩余请求全部走新浪


def get_hist(code: str, days: int = 420) -> pd.DataFrame:
    """前复权日线，列：date close high low vol"""
    global _hist_em_dead
    end = today_cn()
    start, end_s = (end - dt.timedelta(days=days)).strftime("%Y%m%d"), end.strftime("%Y%m%d")
    df = None
    if not _hist_em_dead:
        try:
            df = _hist_em(code, start, end_s)
        except Exception as e:  # noqa: BLE001
            print(f"东方财富历史数据失败（{code}），切换新浪：{str(e)[:80]}")
            _hist_em_dead = True
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
        df = ak.stock_board_industry_name_em()
        df["涨跌幅"] = num(df["涨跌幅"])
        return df.sort_values("涨跌幅", ascending=False).head(n)
    except Exception as e:  # noqa: BLE001
        print("行业涨幅获取失败:", e)
        return None


def get_industry_map(max_age_days: int) -> dict[str, str]:
    """股票 -> 东财行业。结果缓存到 data/industry.csv，过期才重新抓取"""
    path = DATA_DIR / "industry.csv"
    cached = None
    if path.exists():
        cached = pd.read_csv(path, dtype={"code": str})
        updated = dt.date.fromisoformat(str(cached["updated"].iloc[0]))
        if (today_cn() - updated).days < max_age_days:
            return dict(zip(cached["code"], cached["industry"]))
    try:
        import akshare as ak
        boards = ak.stock_board_industry_name_em()["板块名称"].tolist()
        rows, fails = [], 0
        for b in boards:
            try:
                cons = ak.stock_board_industry_cons_em(symbol=b)
                rows += [(str(c).zfill(6), b) for c in cons["代码"]]
                fails = 0
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("行业成分获取失败", b, str(e)[:80])
                if fails >= 3:
                    print("连续失败，放弃本次行业分类更新")
                    break
            time.sleep(0.3)
        if fails >= 3 and cached is None and rows:
            return dict(rows)  # 本次部分可用，但不写缓存
        if fails < 3 and len(rows) > 3000:
            df = pd.DataFrame(rows, columns=["code", "industry"]).drop_duplicates("code")
            df["updated"] = today_cn().isoformat()
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(path, index=False)
            print("行业分类已更新", len(df), "只")
            return dict(zip(df["code"], df["industry"]))
    except Exception as e:  # noqa: BLE001
        print("行业分类更新失败:", e)
    return dict(zip(cached["code"], cached["industry"])) if cached is not None else {}
