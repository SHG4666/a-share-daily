"""
涨停统计与连板股特征研究（夜间运行，在回测之后）

1. 全市场（剔除 ST/北交所，不限市值）日线：缓存里没有的小市值股，用腾讯 K 线接口分批补齐（有时间预算，几晚补完）
2. 涨停事件：每天每只涨停股和连板高度 -> data/zt_events.csv；每日/每周/每月汇总 -> data/zt_daily.csv
3. 连板股特征：把每一段连续涨停（首板到断板）当一个样本，看“启动日”（首板前一天）的特征：
     启动市值、启动价、股东户数、户均持股市值、股东户数变化、前 20/60 日涨幅、年内位置、波动率、成交额、
     首板量比、是否一字板、近 60 日涨停次数
   对比“首板即断 / 2 板 / 3-4 板 / 5 板+”各组的中位数，并按每个特征分五档统计“首板晋级到 3 板”的概率
   -> data/zt_study.csv
4. 每周最后一个交易日（或 ZT_CARD=1）推送一张「📊 涨停周报 / 月报 + 连板股特征」飞书卡片

近似：价格是前复权价；历史市值 = 当前总股本 × 当时前复权价（送转后口径一致）；股东户数取启动日之前已公告的最近一期；
ST 按当前名称剔除；涨停按日线判断（涨幅达到涨停幅度且收在最高价）。
"""
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import card
import data
import features
import histstore
from data import num
from main import send_feishu

FLAG = data.DATA_DIR / "last_ztstats.txt"
EVENTS = data.DATA_DIR / "zt_events.csv"
DAILY = data.DATA_DIR / "zt_daily.csv"
STUDY = data.DATA_DIR / "zt_study.csv"
GDHS_DIR = Path(__file__).parent / "cache" / "gdhs"
FETCH_BUDGET = float(os.getenv("ZT_FETCH_BUDGET", 20 * 60))
WINDOW = 480  # 研究最近多少个交易日（约两年）


def lim_of(code: str) -> float:
    return 20.0 if str(code).startswith(("688", "300", "301")) else 10.0


# ================================================================ 补齐全市场日线（腾讯）
def hist_tencent(code: str, n: int = histstore.KEEP_ROWS) -> pd.DataFrame:
    sym = data.market_symbol(code)
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={sym},day,,,{n},qfq"
    j = requests.get(url, timeout=15).json()
    d = j["data"][sym]
    rows = d.get("qfqday") or d.get("day") or []
    if not rows:
        raise ValueError("腾讯 K 线为空")
    df = pd.DataFrame([r[:6] for r in rows], columns=["date", "open", "close", "high", "low", "vol"])
    df["date"] = data.to_date(df["date"])
    for c in ["close", "high", "low", "vol"]:
        df[c] = num(df[c])
    return df[histstore.COLS].dropna().reset_index(drop=True)


def ensure_all(store: histstore.Store, codes: list[str], day: dt.date, traded: set | None) -> int:
    todo = [c for c in codes if c not in store.h or c in store.stale
            or (store.h[c]["date"].iloc[-1] < day and (traded is None or c in traded))]
    if not todo:
        return 0
    t0, ok, bad = time.time(), 0, 0

    def one(c):
        if time.time() - t0 > FETCH_BUDGET:
            return c, "skip"
        time.sleep(0.1)
        try:
            return c, hist_tencent(c)
        except Exception as e:  # noqa: BLE001
            try:
                return c, data.get_hist(c)
            except Exception:  # noqa: BLE001
                return c, e

    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(one, c) for c in todo]):
            c, h = f.result()
            if isinstance(h, pd.DataFrame) and len(h):
                store.h[c] = h.tail(histstore.KEEP_ROWS).reset_index(drop=True)
                store.stale.discard(c); ok += 1
            elif not (isinstance(h, str) and h == "skip"):
                bad += 1
                if bad <= 3:
                    print("补拉失败", c, str(h)[:100])
    left = len(todo) - ok - bad
    print(f"全市场日线：需补 {len(todo)} 只，成功 {ok}，失败 {bad}，超时未拉 {left}，用时 {time.time() - t0:.0f}s")
    return left


# ================================================================ 股东户数（东方财富，按季度快照缓存）
def quarter_ends(day: dt.date, n: int = 10) -> list[str]:
    out, y, q = [], day.year, (day.month - 1) // 3
    for _ in range(n):
        if q == 0:
            y, q = y - 1, 4
        m = q * 3
        last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1))
        out.append(last.strftime("%Y%m%d"))
        q -= 1
    return out


def load_gdhs(day: dt.date) -> pd.DataFrame:
    import akshare as ak
    GDHS_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, qe in enumerate(quarter_ends(day)):
        p = GDHS_DIR / f"{qe}.csv.gz"
        fresh = i < 2  # 最近两期还在陆续公告，每周刷新
        if p.exists() and not (fresh and time.time() - p.stat().st_mtime > 6 * 86400):
            parts.append(pd.read_csv(p, dtype={"代码": str})); continue
        try:
            df = data.em_call(ak.stock_zh_a_gdhs, symbol=qe, timeout=60)
            keep = [c for c in df.columns if c in ("代码", "股东户数-本次", "股东户数-增减比例", "股东户数统计截止日-本次",
                                                    "公告日期", "总股本", "户均持股市值")]
            df = df[keep].copy()
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            df.to_csv(p, index=False)
            parts.append(df)
            print(f"股东户数 {qe}：{len(df)} 条")
        except Exception as e:  # noqa: BLE001
            print(f"股东户数 {qe} 失败：{str(e)[:100]}")
            data.EM_STATE["dead"] = False
            if p.exists():
                parts.append(pd.read_csv(p, dtype={"代码": str}))
    if not parts:
        return pd.DataFrame()
    g = pd.concat(parts, ignore_index=True)
    def c(name):
        return g[name] if name in g.columns else pd.Series(np.nan, index=g.index)
    g["end"] = pd.to_datetime(c("股东户数统计截止日-本次"), errors="coerce")
    ann = pd.to_datetime(c("公告日期"), errors="coerce")
    g["avail"] = ann.fillna(g["end"] + pd.Timedelta(days=45))  # 没有公告日就按截止日后 45 天才可见
    g["hs"] = num(c("股东户数-本次"))
    g["hs_chg"] = num(c("股东户数-增减比例")) / 100
    g = g.dropna(subset=["avail", "hs"]).sort_values("avail")
    return g[["代码", "avail", "hs", "hs_chg"]].drop_duplicates(["代码", "avail"], keep="last")


# ================================================================ 事件与样本
def scan(store, codes, names, day):
    ev, runs = [], []
    for code in codes:
        h = store.get(code)
        if h is None or len(h) < 150:
            continue
        h = h[h["date"] <= day]
        f = features.compute(h, lim_of(code))
        if f is None:
            continue
        zt = f["zt"].to_numpy() == 1
        n = len(f)
        lo_i = max(0, n - WINDOW)
        board = f["board"].to_numpy()
        dates = f["date"].to_numpy()
        idx = np.where(zt[lo_i:])[0] + lo_i
        for i in idx:
            ev.append((dates[i], code, names.get(code, ""), int(board[i])))
        # 每段连续涨停
        starts = [i for i in idx if i >= 61 and not zt[i - 1]]
        vol = h["vol"].to_numpy(float)
        hi, lo = h["high"].to_numpy(float), h["low"].to_numpy(float)
        zt60 = pd.Series(zt.astype(float)).rolling(60).sum().to_numpy()
        for s in starts:
            e = s
            while e + 1 < n and zt[e + 1]:
                e += 1
            L = s - 1
            r = f.iloc[L]
            runs.append(dict(
                code=code, name=names.get(code, ""), start=dates[s], height=e - s + 1, open_end=(e == n - 1),
                close=r["close"], r20=r["r20"], r60=r["r60"], pos250=r["pos250"], vol60=r["vol60"],
                amount=r["close"] * vol[L] * 100, vr=vol[s] / np.nanmean(vol[L - 4:L + 1]),
                yizi=float(hi[s] == lo[s]), zt60=zt60[L]))
    return pd.DataFrame(ev, columns=["date", "code", "name", "board"]), pd.DataFrame(runs)


FEATS = [("mcap", "启动市值", lambda x: f"{x / 1e8:.0f}亿"), ("close", "启动价", lambda x: f"{x:.2f}元"),
         ("hs", "股东户数", lambda x: f"{x / 1e4:.1f}万"), ("per_hs", "户均持股市值", lambda x: f"{x / 1e4:.0f}万"),
         ("hs_chg", "股东户数变化", lambda x: f"{x:+.0%}"), ("r20", "前20日涨幅", lambda x: f"{x:+.0%}"),
         ("r60", "前60日涨幅", lambda x: f"{x:+.0%}"), ("pos250", "年内位置", lambda x: f"{x:.0%}"),
         ("vol60", "波动率", lambda x: f"{x:.0%}"), ("amount", "启动日成交额", lambda x: f"{x / 1e8:.1f}亿"),
         ("vr", "首板量比", lambda x: f"{x:.1f}倍"), ("zt60", "近60日涨停次数", lambda x: f"{x:.1f}")]
GROUPS = [("首板即断", 1, 1), ("2板", 2, 2), ("3-4板", 3, 4), ("5板+", 5, 99)]


def study(R: pd.DataFrame):
    R = R[~R["open_end"]].copy()
    base3, base2 = (R["height"] >= 3).mean(), (R["height"] >= 2).mean()
    med = {}
    for g, a, b in GROUPS:
        sub = R[R["height"].between(a, b)]
        med[g] = {k: sub[k].median() for k, _, _ in FEATS} | {"n": len(sub)}
    rows = []
    for k, label, fmt in FEATS:
        s = R[k].replace([np.inf, -np.inf], np.nan)
        ok = s.notna()
        if ok.sum() < 200:
            continue
        try:
            q = pd.qcut(s[ok], 5, duplicates="drop")
        except ValueError:
            continue
        grp = R[ok].groupby(q, observed=True)
        for iv, sub in grp:
            rows.append(dict(feature=k, label=label, lo=iv.left, hi=iv.right, n=len(sub),
                             p2=(sub["height"] >= 2).mean(), p3=(sub["height"] >= 3).mean()))
    B = pd.DataFrame(rows)
    return R, med, B, base2, base3


def lift_lines(B: pd.DataFrame, base3: float, top: int = 6) -> list[str]:
    fm = {k: f for k, _, f in FEATS}
    out = []
    for k, g in B.groupby("feature", sort=False):
        g = g.reset_index(drop=True)
        best, worst = g.loc[g["p3"].idxmax()], g.loc[g["p3"].idxmin()]
        spread = best["p3"] / max(worst["p3"], 1e-3)

        def rng(r):
            f = fm[k]
            if r.name == 0:
                return f"≤{f(r['hi'])}"
            if r.name == len(g) - 1:
                return f"≥{f(r['lo'])}"
            return f"{f(r['lo'])}~{f(r['hi'])}"
        out.append((spread, f"**{best['label']}**　{rng(best)}：晋级3板 **{best['p3']:.1%}**｜{rng(worst)}：{worst['p3']:.1%}"
                            f"（相差 {spread:.1f} 倍）"))
    out.sort(key=lambda x: -x[0])
    return [s for _, s in out[:top]]


def period_top(E: pd.DataFrame, start: dt.date, end: dt.date, k: int = 10) -> tuple[str, int, int]:
    s = E[(E["date"] >= start) & (E["date"] <= end)]
    if s.empty:
        return "无", 0, 0
    g = s.groupby(["code", "name"]).agg(n=("board", "size"), mx=("board", "max")).reset_index()
    g = g.sort_values(["n", "mx"], ascending=False).head(k)
    txt = "、".join(f"{card.link(r['name'], r['code'])}×{r['n']}" + (f"（{r['mx']}连板）" if r["mx"] >= 2 else "")
                   for _, r in g.iterrows())
    return txt, len(s), s["code"].nunique()


def main():
    t0 = time.time()
    dry = os.getenv("DRY_RUN") == "1"
    tdays = data.trade_days()
    day, why = data.target_day(tdays)
    if day is None:
        print("涨停统计跳过:", why); return
    if os.getenv("FORCE_RUN") != "1" and FLAG.exists() and FLAG.read_text().strip() == day.isoformat():
        print(f"{day} 涨停统计已做过，跳过"); return
    print("涨停统计日期", day)

    spot = data.get_spot()
    U = spot[~spot["名称"].str.contains("ST|退", na=False) & ~spot["代码"].str.startswith(("8", "4", "92"))]
    codes = U["代码"].tolist()
    names = dict(zip(U["代码"], U["名称"]))
    shares = (U["总市值"] / U["最新价"]).set_axis(U["代码"]).replace([np.inf, -np.inf], np.nan)
    store = histstore.Store().load()
    store.append_snapshot(spot, day)
    traded = set(spot.loc[spot["成交量"] > 0, "代码"]) if "成交量" in spot else None
    left = ensure_all(store, codes, day, traded)
    store.save()

    E, R = scan(store, [c for c in codes if store.get(c) is not None], names, day)
    print(f"涨停事件 {len(E)} 条，连续涨停样本 {len(R)} 段，用时 {time.time() - t0:.0f}s")
    E = E.sort_values(["date", "board"], ascending=[True, False])

    # 每日汇总
    D = E.groupby("date").agg(zt=("code", "size"), lianban=("board", lambda b: int((b >= 2).sum())),
                              max_board=("board", "max")).reset_index()

    # 股东户数 & 市值
    R["mcap"] = R["code"].map(shares) * R["close"]
    G = load_gdhs(day)
    if len(G) and len(R):
        R["start_ts"] = pd.to_datetime(R["start"]).astype("datetime64[ns]")
        G = G.assign(avail=G["avail"].astype("datetime64[ns]"))
        R = pd.merge_asof(R.sort_values("start_ts"), G.rename(columns={"代码": "code"}).sort_values("avail"),
                          left_on="start_ts", right_on="avail", by="code", direction="backward")
        R["per_hs"] = R["mcap"] / R["hs"]
    else:
        for c in ("hs", "hs_chg", "per_hs"):
            R[c] = np.nan
    R2, med, B, base2, base3 = study(R)
    print(f"样本（已结束的连续涨停段）{len(R2)}：晋级2板 {base2:.1%}，晋级3板 {base3:.1%}")
    for g, m in med.items():
        print(g, {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()})
    print(B.to_string())

    # ---------- 卡片
    wk0 = day - dt.timedelta(days=day.weekday())
    mo0 = day.replace(day=1)
    wk_txt, wk_n, wk_u = period_top(E, wk0, day)
    mo_txt, mo_n, mo_u = period_top(E, mo0, day)
    today = E[E["date"] == day]
    fm = {k: (lab, f) for k, lab, f in FEATS}
    med_lines = []
    for k, _, _ in FEATS:
        lab, f = fm[k]
        vals = [f"{g} {f(med[g][k])}" if pd.notna(med[g][k]) else f"{g} -" for g, _, _ in GROUPS]
        med_lines.append(f"**{lab}**　" + "｜".join(vals))
    period = f"{E['date'].min():%Y-%m-%d} 至 {day:%Y-%m-%d}" if len(E) else ""
    cnt = "｜".join(f"{g} {med[g]['n']} 段" for g, _, _ in GROUPS)
    els = [
        card.md(f"**今日**　涨停 **{len(today)}** 只｜连板 {int((today['board'] >= 2).sum())} 只｜最高 {int(today['board'].max()) if len(today) else 0} 板\n"
                f"**本周**（{wk0:%m-%d} 起）涨停 {wk_n} 次 / {wk_u} 只\n涨停次数最多：{wk_txt}\n"
                f"**本月**（{mo0:%m月}）涨停 {mo_n} 次 / {mo_u} 只\n涨停次数最多：{mo_txt}"),
        card.HR,
        card.md(f"**🔬 连板股特征**（{period}，共 {len(R2)} 段连续涨停：{cnt}）\n"
                f"首板晋级 2 板 **{base2:.1%}**，晋级 3 板 **{base3:.1%}**。以下是“启动日”（首板前一天）各组的中位数：\n"
                + "\n".join(med_lines)),
        card.HR,
        card.md("**🎯 哪些特征最能区分能走出连板的股票**（按特征分五档，比较首板晋级 3 板的概率）\n"
                + "\n".join(lift_lines(B, base3)) if len(B) else "样本不足"),
        {"tag": "note", "elements": [{"tag": "plain_text", "content":
            f"全市场剔除 ST/北交所，不限市值；有历史数据 {sum(store.get(c) is not None for c in codes)} 只"
            + (f"（还有 {left} 只下次补齐）" if left else "") +
            "。日线判断涨停；价格前复权，历史市值=当前总股本×当时价格；股东户数取启动前已公告的最近一期。"
            "统计规律不代表未来，仅供研究参考。"}]},
    ]
    payload = {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": "purple", "title": {"tag": "plain_text", "content": f"📊 涨停统计 · {day:%m月%d日}"}},
        "elements": els}}

    nxt = next((d for d in tdays if d > day), None)
    send = os.getenv("ZT_CARD") == "1" or nxt is None or nxt.isocalendar()[1] != day.isocalendar()[1]
    if dry:
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return
    data.DATA_DIR.mkdir(exist_ok=True)
    E.to_csv(EVENTS, index=False)
    D.to_csv(DAILY, index=False)
    B.assign(run_date=day.isoformat(), base2=base2, base3=base3).to_csv(STUDY, index=False)
    if send:
        send_feishu(payload)
    else:
        print("非周末最后一个交易日，不推送卡片（数据已更新）")
    FLAG.write_text(day.isoformat())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if os.getenv("FEISHU_WEBHOOK") and os.getenv("DRY_RUN") != "1":
            try:
                send_feishu({"msg_type": "text", "content": {"text": f"涨停统计运行失败：{e}"}})
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)
