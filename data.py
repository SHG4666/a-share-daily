"""数据获取层：所有 akshare 调用都在这里"""
import datetime as dt
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


@retry()
def get_spot() -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    for c in ["最新价", "涨跌幅", "成交额", "换手率", "市盈率-动态", "市净率", "总市值"]:
        df[c] = num(df[c])
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    return df


@retry()
def get_hist(code: str, days: int = 420) -> pd.DataFrame:
    """前复权日线，列：date close high low vol"""
    import akshare as ak
    end = today_cn()
    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=(end - dt.timedelta(days=days)).strftime("%Y%m%d"),
                            end_date=end.strftime("%Y%m%d"), adjust="qfq")
    df = df.rename(columns={"日期": "date", "收盘": "close", "最高": "high", "最低": "low", "成交量": "vol"})
    df["date"] = to_date(df["date"])
    return df[["date", "close", "high", "low", "vol"]].reset_index(drop=True)


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
        rows = []
        for b in boards:
            try:
                cons = retry(times=2, wait=2)(ak.stock_board_industry_cons_em)(symbol=b)
                rows += [(str(c).zfill(6), b) for c in cons["代码"]]
            except Exception as e:  # noqa: BLE001
                print("行业成分获取失败", b, e)
            time.sleep(0.2)
        if len(rows) > 3000:
            df = pd.DataFrame(rows, columns=["code", "industry"]).drop_duplicates("code")
            df["updated"] = today_cn().isoformat()
            DATA_DIR.mkdir(exist_ok=True)
            df.to_csv(path, index=False)
            print("行业分类已更新", len(df), "只")
            return dict(zip(df["code"], df["industry"]))
    except Exception as e:  # noqa: BLE001
        print("行业分类更新失败:", e)
    return dict(zip(cached["code"], cached["industry"])) if cached is not None else {}
