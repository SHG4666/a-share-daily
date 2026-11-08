"""
A股每日推荐 -> 飞书群机器人
数据源：akshare（东方财富），收盘后运行
环境变量：
  FEISHU_WEBHOOK   必填，飞书自定义机器人 webhook
  FEISHU_SECRET    选填，机器人开启“签名校验”时填写
  ANTHROPIC_API_KEY 选填，填了会额外生成一段 AI 盘面点评
  FORCE_RUN=1      选填，非交易日也强制运行（手动调试用）
  DRY_RUN=1        选填，只打印卡片不推送
"""
import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import sys
import time
from functools import wraps

import pandas as pd
import requests

import config

TZ_CN = dt.timezone(dt.timedelta(hours=8))


# ---------------------------------------------------------------- 工具
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
                    print(f"[retry {i + 1}/{times}] {fn.__name__}: {e}")
                    time.sleep(wait * (i + 1))
            raise last
        return wrapper
    return deco


def today_cn() -> dt.date:
    return dt.datetime.now(TZ_CN).date()


def num(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------- 数据
@retry()
def trade_days() -> set[dt.date]:
    import akshare as ak
    days = ak.tool_trade_date_hist_sina()["trade_date"]
    return set(pd.to_datetime(days).dt.date)


def last_trade_day(d: dt.date, days: set[dt.date]) -> dt.date:
    return max(x for x in days if x <= d)


@retry()
def get_spot() -> pd.DataFrame:
    import akshare as ak
    df = ak.stock_zh_a_spot_em()
    for c in ["最新价", "涨跌幅", "成交额", "量比", "换手率", "市盈率-动态", "总市值"]:
        df[c] = num(df[c])
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    return df


@retry()
def get_hist(code: str, days: int = 200) -> pd.DataFrame:
    import akshare as ak
    end = today_cn()
    start = end - dt.timedelta(days=days)
    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start.strftime("%Y%m%d"),
                            end_date=end.strftime("%Y%m%d"), adjust="qfq")
    return df.rename(columns={"收盘": "close", "成交量": "vol"})[["日期", "close", "vol"]]


def get_top_industries(n: int) -> pd.DataFrame | None:
    try:
        import akshare as ak
        df = ak.stock_board_industry_name_em()
        df["涨跌幅"] = num(df["涨跌幅"])
        return df.sort_values("涨跌幅", ascending=False).head(n)
    except Exception as e:  # noqa: BLE001
        print("行业数据获取失败:", e)
        return None


# ---------------------------------------------------------------- 指标与打分
def indicators(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    c = h["close"]
    for n in (5, 10, 20, 60):
        h[f"ma{n}"] = c.rolling(n).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    h["dif"] = ema12 - ema26
    h["dea"] = h["dif"].ewm(span=9, adjust=False).mean()
    h["vol_ma5"] = h["vol"].rolling(5).mean().shift(1)  # 不含当日
    return h


def signals(h: pd.DataFrame) -> tuple[int, list[str], list[str]]:
    """返回 (得分, 正面信号, 风险提示)"""
    if len(h) < 61:
        return 0, [], ["上市时间短，数据不足"]
    h = indicators(h)
    t = h.iloc[-1]
    score, good, risk = 0, [], []

    if t.ma5 > t.ma10 > t.ma20:
        score += 2; good.append("均线多头排列")
    if t.close > t.ma60:
        score += 1; good.append("站上60日线")
    elif t.close < t.ma20:
        risk.append("收盘在20日线下方")

    recent = h.iloc[-4:]
    cross = any(recent.dif.iloc[i - 1] <= recent.dea.iloc[i - 1] and recent.dif.iloc[i] > recent.dea.iloc[i]
                for i in range(1, len(recent)))
    if cross:
        score += 2; good.append("MACD近3日金叉" + ("（零轴上）" if t.dif > 0 else ""))
    elif t.dif > t.dea and t.dif > 0:
        score += 1; good.append("MACD零轴上运行")

    if pd.notna(t.vol_ma5) and t.vol > 1.5 * t.vol_ma5:
        score += 1; good.append(f"放量（{t.vol / t.vol_ma5:.1f}倍5日均量）")

    if t.close >= h.close.iloc[-21:-1].max():
        score += 1; good.append("突破20日新高")

    pct20 = (t.close / h.close.iloc[-21] - 1) * 100
    if pct20 > config.OVERHEAT_20D_PCT:
        score -= 2; risk.append(f"20日已涨{pct20:.0f}%，追高风险")

    return score, good, risk


# ---------------------------------------------------------------- 选股
def screen(spot: pd.DataFrame) -> pd.DataFrame:
    df = spot.copy()
    df = df[~df["名称"].str.contains("ST|退", na=False)]
    df = df[~df["代码"].str.startswith(config.EXCLUDE_PREFIX)]
    df = df[df["总市值"] >= config.MIN_MARKET_CAP]
    df = df[df["涨跌幅"].between(*config.PCT_CHG_RANGE)]
    df = df[df["换手率"].between(*config.TURNOVER_RANGE)]
    df = df[df["量比"] >= config.MIN_VOLUME_RATIO]
    if config.PE_RANGE:
        lo, hi = config.PE_RANGE
        df = df[(df["市盈率-动态"] > lo) & (df["市盈率-动态"] < hi)]
    return df.sort_values("成交额", ascending=False).head(config.CANDIDATE_LIMIT)


def pick(cands: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in cands.iterrows():
        try:
            s, good, risk = signals(get_hist(r["代码"]))
        except Exception as e:  # noqa: BLE001
            print("跳过", r["代码"], e)
            continue
        out.append(dict(code=r["代码"], name=r["名称"], price=r["最新价"], pct=r["涨跌幅"],
                        turnover=r["换手率"], vr=r["量比"], cap=r["总市值"] / 1e8,
                        amount=r["成交额"] / 1e8, score=s, good=good, risk=risk))
        time.sleep(0.3)
    out.sort(key=lambda x: (x["score"], x["amount"]), reverse=True)
    return [x for x in out if x["score"] >= 3][: config.TOP_N]


def watch(spot: pd.DataFrame) -> list[dict]:
    idx = spot.set_index("代码")
    out = []
    for name, code in config.WATCHLIST.items():
        try:
            s, good, risk = signals(get_hist(code))
            pct = idx.loc[code, "涨跌幅"] if code in idx.index else float("nan")
            price = idx.loc[code, "最新价"] if code in idx.index else float("nan")
            out.append(dict(code=code, name=name, pct=pct, price=price, score=s, good=good, risk=risk))
        except Exception as e:  # noqa: BLE001
            print("自选获取失败", code, e)
        time.sleep(0.3)
    return out


# ---------------------------------------------------------------- AI 点评（可选）
def ai_comment(breadth: str, inds: str, picks: list[dict]) -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    brief = "; ".join(f"{p['name']}({p['pct']:+.1f}%,{'/'.join(p['good'])})" for p in picks)
    prompt = (f"今日A股：{breadth}。领涨行业：{inds}。量化初筛标的：{brief or '无'}。"
              "请用不超过120字给出客观的盘面点评和明日需关注的风险，不要给买卖建议。")
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"), "max_tokens": 400,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    except Exception as e:  # noqa: BLE001
        print("AI 点评失败:", e)
        return None


# ---------------------------------------------------------------- 飞书卡片
def color_pct(p: float) -> str:
    if pd.isna(p):
        return "-"
    c = "red" if p > 0 else ("green" if p < 0 else "grey")
    return f"<font color='{c}'>{p:+.2f}%</font>"


def build_card(day: dt.date, spot: pd.DataFrame, inds, picks, watches, comment) -> dict:
    up = int((spot["涨跌幅"] > 0).sum()); down = int((spot["涨跌幅"] < 0).sum())
    amount = spot["成交额"].sum() / 1e12
    els = [{"tag": "markdown",
            "content": f"**市场概况**　上涨 <font color='red'>{up}</font> / 下跌 <font color='green'>{down}</font>"
                       f"　成交额 **{amount:.2f} 万亿**"}]
    if inds is not None and len(inds):
        els.append({"tag": "markdown", "content": "**领涨行业**　" + "　".join(
            f"{r['板块名称']} {color_pct(r['涨跌幅'])}" for _, r in inds.iterrows())})
    if comment:
        els.append({"tag": "markdown", "content": f"**AI 点评**\n{comment}"})

    els.append({"tag": "hr"})
    if picks:
        lines = ["**📈 今日技术面筛选**"]
        for i, p in enumerate(picks, 1):
            lines.append(
                f"**{i}. {p['name']}（{p['code']}）** {p['price']:.2f} {color_pct(p['pct'])}　评分 {p['score']}\n"
                f"换手 {p['turnover']:.1f}%｜量比 {p['vr']:.1f}｜市值 {p['cap']:.0f}亿｜成交 {p['amount']:.1f}亿\n"
                f"✅ {'、'.join(p['good']) or '-'}"
                + (f"\n⚠️ {'、'.join(p['risk'])}" if p["risk"] else ""))
        els.append({"tag": "markdown", "content": "\n\n".join(lines)})
    else:
        els.append({"tag": "markdown", "content": "**📈 今日技术面筛选**\n今天没有满足条件的标的，空仓也是一种选择。"})

    if watches:
        els.append({"tag": "hr"})
        lines = ["**👀 自选主题池**"]
        for w in sorted(watches, key=lambda x: x["score"], reverse=True):
            tags = "、".join(w["good"][:2] + [f"⚠️{r}" for r in w["risk"][:1]]) or "无明显信号"
            lines.append(f"{w['name']} {color_pct(w['pct'])}　{tags}")
        els.append({"tag": "markdown", "content": "\n".join(lines)})

    els.append({"tag": "note", "elements": [{"tag": "plain_text",
               "content": "规则化筛选结果，仅供研究参考，不构成投资建议。股市有风险，决策需独立判断。"}]})
    return {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": "red", "title": {"tag": "plain_text", "content": f"A股每日复盘 · {day:%m月%d日}"}},
        "elements": els}}


def send_feishu(payload: dict):
    url = os.environ["FEISHU_WEBHOOK"]
    secret = os.getenv("FEISHU_SECRET")
    if secret:
        ts = str(int(time.time()))
        sign = base64.b64encode(hmac.new(f"{ts}\n{secret}".encode(), b"", hashlib.sha256).digest()).decode()
        payload = {"timestamp": ts, "sign": sign, **payload}
    r = requests.post(url, json=payload, timeout=20)
    print("飞书返回:", r.status_code, r.text)
    r.raise_for_status()
    if r.json().get("code", 0) != 0:
        raise RuntimeError(r.text)


# ---------------------------------------------------------------- 入口
def main():
    today = today_cn()
    days = trade_days()
    if os.getenv("FORCE_RUN") != "1" and today not in days:
        print(f"{today} 非交易日，跳过"); return
    day = last_trade_day(today, days)  # 非交易日强制运行时，复盘最近一个交易日
    print("复盘日期", day)

    spot = get_spot()
    print("全市场", len(spot), "只")
    cands = screen(spot)
    print("初筛候选", len(cands), "只")
    picks = pick(cands)
    watches = watch(spot)
    inds = get_top_industries(config.TOP_INDUSTRY_N)

    breadth = f"上涨{int((spot['涨跌幅'] > 0).sum())}家，下跌{int((spot['涨跌幅'] < 0).sum())}家"
    ind_txt = "、".join(inds["板块名称"]) if inds is not None else "未知"
    comment = ai_comment(breadth, ind_txt, picks)

    card = build_card(day, spot, inds, picks, watches, comment)
    if os.getenv("DRY_RUN") == "1":
        print(json.dumps(card, ensure_ascii=False, indent=2)); return
    send_feishu(card)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print("运行失败:", e)
        if os.getenv("FEISHU_WEBHOOK") and os.getenv("DRY_RUN") != "1":
            try:
                send_feishu({"msg_type": "text", "content": {"text": f"A股每日推荐运行失败：{e}"}})
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)
