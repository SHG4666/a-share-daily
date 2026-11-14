"""
每日涨停复盘 -> 飞书（收盘后运行，在每日推荐之后）

内容：
  1. 情绪温度：涨停/跌停/炸板家数、炸板率、昨日涨停今日溢价、连板晋级率、最高板（和前几天对比，判断情绪周期）
  2. 连板梯队与主线：按连板高度和所属行业聚集看资金主攻方向
  3. 主力手法推测：根据封板时间、炸板次数、换手率、龙虎榜席位（机构/北向/营业部）给每只代表股打“手法标签”
  4. 心得：有 ANTHROPIC_API_KEY 时由 AI 根据以上数据写；没有或失败时用规则生成
  5. 涨停系策略的回测验证（⚡涨停回踩等，来自 variants.py，夜间回测打分，跑赢才上场）

数据源：东方财富涨停池/炸板池/跌停池/昨日涨停池 + 龙虎榜；东方财富失败时，用全市场快照 + 历史缓存自行计算（没有封板时间和封单）。
环境变量同 main.py；DRY_RUN=1 只打印不推送；FORCE_RUN=1 忽略“今天已发过”。
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

import card
import data
import histstore
from data import em_call, num
from main import send_feishu

FLAG = data.DATA_DIR / "last_limitup.txt"
HIST = data.DATA_DIR / "limitup.csv"


# ================================================================ 工具
def lim_of(code: str, name: str = "") -> float:
    if "ST" in str(name):
        return 5.0
    return 20.0 if str(code).startswith(("688", "300", "301")) else 10.0


def hhmm(x) -> str:
    s = str(x).split(".")[0].zfill(6)
    return f"{s[:2]}:{s[2:4]}" if s.isdigit() else "-"


def col(df: pd.DataFrame, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series([np.nan] * len(df), index=df.index)


def yi(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x / 1e8:.1f}亿"


def main_board(code: str) -> bool:
    return not str(code).startswith(("8", "4", "92"))


# ================================================================ 数据：东方财富涨停池
def em_pools(day: dt.date) -> dict | None:
    import akshare as ak
    d = day.strftime("%Y%m%d")
    out = {}
    try:
        zt = em_call(ak.stock_zt_pool_em, date=d, timeout=40)
    except Exception as e:  # noqa: BLE001
        print("东方财富涨停池失败:", str(e)[:120]); return None
    zt = zt.copy()
    zt["代码"] = zt["代码"].astype(str).str.zfill(6)
    out["zt"] = pd.DataFrame({
        "代码": zt["代码"], "名称": col(zt, "名称"), "涨跌幅": num(col(zt, "涨跌幅")),
        "成交额": num(col(zt, "成交额")), "流通市值": num(col(zt, "流通市值")), "换手率": num(col(zt, "换手率")),
        "封板资金": num(col(zt, "封板资金")), "首次封板": col(zt, "首次封板时间").map(hhmm),
        "最后封板": col(zt, "最后封板时间").map(hhmm), "炸板次数": num(col(zt, "炸板次数")).fillna(0),
        "连板": num(col(zt, "连板数")).fillna(1), "涨停统计": col(zt, "涨停统计"),
        "行业": col(zt, "所属行业")})
    for key, fn in [("zb", ak.stock_zt_pool_zbgc_em), ("dt", ak.stock_zt_pool_dtgc_em),
                    ("prev", ak.stock_zt_pool_previous_em)]:
        try:
            df = em_call(fn, date=d, timeout=30).copy()
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            df["涨跌幅"] = num(df["涨跌幅"])
            out[key] = df
        except Exception as e:  # noqa: BLE001
            print(f"东方财富 {key} 池失败:", str(e)[:120])
            out[key] = None
    return out


# ================================================================ 数据：备用（快照 + 历史缓存自算）
def self_pools(day: dt.date) -> dict:
    spot = data.get_spot()
    spot = spot[spot["代码"].map(main_board) & spot["最新价"].notna() & (spot["成交量"] > 0)].copy()
    lim = np.array([lim_of(c, n) for c, n in zip(spot["代码"], spot["名称"])])
    hi_pct = (spot["最高"] / spot["昨收"] - 1) * 100
    is_zt = (spot["涨跌幅"] >= lim - 0.2) & (spot["最新价"] >= spot["最高"] * 0.9995)
    is_zb = (hi_pct >= lim - 0.2) & ~is_zt
    is_dt = (spot["涨跌幅"] <= -(lim - 0.2)) & (spot["最新价"] <= spot["最低"] * 1.0005)

    store = histstore.Store().load()
    ind = data.get_industry_map(30)

    def boards_before(code: str) -> tuple[int, bool]:
        """今天之前的连续涨停数；以及昨天是否涨停"""
        h = store.get(code)
        if h is None or len(h) < 3:
            return 0, False
        h = h[h["date"] < day].tail(30)
        c, hi = h["close"].to_numpy(float), h["high"].to_numpy(float)
        l = lim_of(code)
        n = 0
        for i in range(len(c) - 1, 0, -1):
            if (c[i] / c[i - 1] - 1) * 100 >= l - 0.2 and c[i] >= hi[i] * 0.9995:
                n += 1
            else:
                break
        return n, n > 0

    z = spot[is_zt].copy()
    z["连板"] = [boards_before(c)[0] + 1 for c in z["代码"]]
    zt = pd.DataFrame({"代码": z["代码"], "名称": z["名称"], "涨跌幅": z["涨跌幅"], "成交额": z["成交额"],
                       "流通市值": np.nan, "换手率": z["换手率"], "封板资金": np.nan, "首次封板": "-",
                       "最后封板": "-", "炸板次数": 0, "连板": z["连板"], "涨停统计": "",
                       "行业": z["代码"].map(ind)})
    # 昨日涨停今日表现：从缓存找昨天涨停的股票
    prev_rows = []
    for code in spot["代码"]:
        n, yzt = boards_before(code)
        if yzt:
            prev_rows.append((code, n))
    pidx = spot.set_index("代码")
    prev = pd.DataFrame([{"代码": c, "名称": pidx.at[c, "名称"], "涨跌幅": pidx.at[c, "涨跌幅"], "昨日连板数": n}
                         for c, n in prev_rows]) if prev_rows else None
    print(f"自算涨停池：涨停 {len(zt)}，炸板 {int(is_zb.sum())}，跌停 {int(is_dt.sum())}，昨日涨停 {len(prev_rows)}")
    return {"zt": zt.reset_index(drop=True), "zb": spot[is_zb], "dt": spot[is_dt], "prev": prev, "src": "self"}


# ================================================================ 龙虎榜
SEAT_RETAIL = ("拉萨", "东方财富证券股份有限公司拉萨")


def seat_type(name: str) -> str:
    n = str(name)
    if "机构专用" in n:
        return "机构"
    if "沪股通" in n or "深股通" in n:
        return "北向"
    if "拉萨" in n:
        return "散户席位"
    if "量化" in n:
        return "量化"
    return "营业部"


def lhb(day: dt.date, codes: list[str], max_detail: int = 12) -> dict[str, dict]:
    """返回 代码 -> {净买额, 上榜原因, 买方结构...}；只对传入的涨停股取明细"""
    import akshare as ak
    d = day.strftime("%Y%m%d")
    try:
        df = em_call(ak.stock_lhb_detail_em, start_date=d, end_date=d, timeout=40)
    except Exception as e:  # noqa: BLE001
        print("龙虎榜失败:", str(e)[:120]); return {}
    if df is None or df.empty:
        print("龙虎榜：今天暂无数据（通常 17:00 后陆续公布）"); return {}
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df = df.drop_duplicates("代码")
    out = {}
    for _, r in df.iterrows():
        out[r["代码"]] = {"净买额": num(pd.Series([r.get("龙虎榜净买额")])).iloc[0],
                         "原因": str(r.get("上榜原因", ""))[:30]}
    hit = [c for c in codes if c in out][:max_detail]
    for c in hit:
        try:
            buy = em_call(ak.stock_lhb_stock_detail_em, symbol=c, date=d, flag="买入", timeout=20)
            time.sleep(0.3)
        except Exception as e:  # noqa: BLE001
            print("龙虎榜明细失败", c, str(e)[:80]); break
        if buy is None or buy.empty:
            continue
        name_col = next((x for x in buy.columns if "营业部" in x), buy.columns[1])
        amt_col = next((x for x in buy.columns if x.startswith("买入金额") and "占" not in x), None)
        net_col = next((x for x in buy.columns if "净额" in x), None)
        buy = buy.head(5)
        types = [seat_type(n) for n in buy[name_col]]
        amts = num(buy[amt_col]) if amt_col else pd.Series(dtype=float)
        top_share = float(amts.iloc[0] / amts.sum()) if len(amts) and amts.sum() > 0 else np.nan
        out[c].update({"买方": types, "买一占比": top_share,
                       "机构净买": float(num(buy.loc[[t == "机构" for t in types], net_col]).sum()) if net_col else np.nan,
                       "买一": str(buy[name_col].iloc[0])[:24]})
    return out


# ================================================================ 统计
def metrics(day: dt.date, P: dict) -> dict:
    zt, zb, dtp, prev = P["zt"], P.get("zb"), P.get("dt"), P.get("prev")
    n_zt, n_zb = len(zt), (len(zb) if zb is not None else np.nan)
    m = {"date": day.isoformat(), "zt": n_zt, "zb": n_zb, "dt": len(dtp) if dtp is not None else np.nan,
         "zb_rate": n_zb / (n_zt + n_zb) if n_zt + (n_zb or 0) > 0 else np.nan,
         "max_board": int(zt["连板"].max()) if n_zt else 0,
         "lianban": int((zt["连板"] >= 2).sum()) if n_zt else 0}
    if prev is not None and len(prev):
        m["prev_n"] = len(prev)
        m["prev_ret"] = float(prev["涨跌幅"].mean()) / 100
        m["prev_up"] = float((prev["涨跌幅"] > 0).mean())
        lb = num(col(prev, "昨日连板数"))
        m["prev_lb_ret"] = float(prev.loc[lb >= 2, "涨跌幅"].mean()) / 100 if (lb >= 2).any() else np.nan
        m["promote"] = m["lianban"] / len(prev)  # 晋级率：今日连板家数 / 昨日涨停家数
    return m


def phase(m: dict, hist: pd.DataFrame) -> str:
    """情绪周期粗判（规则，仅供参考）"""
    pr, zr = m.get("prev_ret", np.nan), m.get("zb_rate", np.nan)
    prev_max = hist["max_board"].tail(3).max() if len(hist) else np.nan
    if pd.notna(pr) and pr > 0.03 and pd.notna(zr) and zr < 0.25 and m["zt"] >= 60:
        return "🔥 高潮/发酵：涨停多、溢价高、封板率高，接力情绪强（高潮次日易分化）"
    if pd.notna(pr) and pr < 0 and ((pd.notna(zr) and zr > 0.35) or m.get("dt", 0) >= 20):
        return "🧊 退潮：昨日涨停今天平均收绿、炸板多，亏钱效应明显，打板接力风险大"
    if pd.notna(prev_max) and m["max_board"] < prev_max - 1 and pd.notna(pr) and pr < 0.01:
        return "📉 高度压缩：最高板下降，龙头断板，情绪走弱"
    if m["zt"] <= 30 and pd.notna(pr) and pr < 0.01:
        return "🌫️ 冰点/低迷：涨停少、溢价低，往往是新周期酝酿期"
    return "⚖️ 分歧/震荡：有赚钱效应但不强，资金在主线与轮动之间切换"


def style_tags(r, lb: dict | None) -> list[str]:
    """手法推测标签（基于公开数据的推测，不是事实）"""
    t = []
    first, zb_n, turn = r.get("首次封板", "-"), r.get("炸板次数", 0) or 0, r.get("换手率", np.nan)
    if first != "-":
        if first <= "09:30" and zb_n == 0:
            t.append("开盘即封：筹码锁定，多为消息/预期驱动或主力提前吸筹后一字拉升")
        elif first <= "10:00" and zb_n == 0:
            t.append("早盘快速封板：资金抢筹坚决")
        elif first >= "14:30":
            t.append("尾盘封板：尾盘偷袭做形态，次日溢价往往不稳")
    if zb_n >= 3:
        t.append(f"炸板{int(zb_n)}次后回封：分歧大，换手充分，可能边拉边出或借分歧洗盘")
    elif zb_n >= 1:
        t.append("炸板后回封：盘中有分歧但承接住了")
    if pd.notna(turn):
        if turn >= 20:
            t.append(f"高换手{turn:.0f}%：游资接力换手板")
        elif turn <= 3:
            t.append(f"低换手{turn:.1f}%：缩量封死，控盘度高")
    fd, amt = r.get("封板资金", np.nan), r.get("成交额", np.nan)
    if pd.notna(fd) and pd.notna(amt) and amt > 0 and fd / amt >= 0.5:
        t.append(f"封单{yi(fd)}达成交额{fd / amt:.0%}：封板意愿强")
    if lb:
        types = lb.get("买方", [])
        if types.count("机构") >= 2 or (lb.get("机构净买") or 0) > 0:
            t.append("龙虎榜机构净买：机构参与，持续性相对好")
        if types.count("营业部") >= 4:
            t.append("买方前五多为营业部：游资合力")
        if pd.notna(lb.get("买一占比", np.nan)) and lb["买一占比"] >= 0.5:
            t.append(f"买一独大（占前五 {lb['买一占比']:.0%}，{lb.get('买一', '')}）：一家主导，次日防砸盘")
        if types.count("散户席位") >= 2:
            t.append("拉萨等散户席位扎堆：散户追涨，筹码不稳")
        if pd.notna(lb.get("净买额")) and lb["净买额"] < 0:
            t.append(f"龙虎榜净卖出 {yi(-lb['净买额'])}：上榜资金在出货")
    return t


# ================================================================ 心得
def ai_journal(digest: str) -> str | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    prompt = (
        "你是一位冷静、重视证据的 A 股短线复盘研究员。下面是今天收盘后的涨停数据摘要（含最近几天的情绪指标对比、"
        "连板梯队、行业聚集、代表股的封板细节和龙虎榜席位、以及我们的涨停系策略回测结果）。\n\n"
        f"{digest}\n\n"
        "请写一份今日涨停复盘心得，400 字以内，分四段，每段用加粗小标题：\n"
        "**情绪周期**：结合最近几天数据判断处于哪个阶段（冰点/启动/发酵/高潮/分歧/退潮），说明依据。\n"
        "**主线与逻辑**：资金集中在哪些方向，涨停背后可能的驱动（只基于数据推断，不编造新闻）。\n"
        "**主力手法推测**：挑 2–3 只代表股，根据封板时间、炸板、换手、龙虎榜席位推测操盘手法；"
        "必须写明这是推测和依据，不要说成事实。\n"
        "**明日观察与策略启示**：明天该观察什么信号来验证今天的判断；对“涨停后缩量回踩”类策略有什么启示。\n"
        "要求：客观克制，不给具体买卖建议，不承诺收益，不用表情符号。")
    models = [m for m in (os.getenv("LIMITUP_MODEL"), "claude-sonnet-4-5", os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")) if m]
    for mdl in dict.fromkeys(models):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
                "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": mdl, "max_tokens": 1500, "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            print("AI 心得模型:", mdl)
            return r.json()["content"][0]["text"].strip()
        except Exception as e:  # noqa: BLE001
            print(f"AI 心得失败（{mdl}）:", str(e)[:160])
    return None


def rule_journal(m: dict, ph: str, top_ind: list, ladder: pd.DataFrame, bt_line: str | None) -> str:
    s = [f"**情绪周期**　{ph}。"]
    if pd.notna(m.get("prev_ret", np.nan)):
        s.append(f"昨日涨停股今天平均 {m['prev_ret']:+.1%}，上涨比例 {m['prev_up']:.0%}；"
                 f"炸板率 {m['zb_rate']:.0%}，晋级率 {m.get('promote', np.nan):.0%}。")
    if top_ind:
        s.append(f"\n**主线**　涨停集中在 {'、'.join(f'{a}({b}家)' for a, b in top_ind[:3])}，"
                 "同一方向多股涨停说明资金在合力攻击，单股孤立涨停持续性通常更差。")
    if len(ladder):
        r = ladder.iloc[0]
        s.append(f"\n**龙头**　最高 {int(r['连板'])} 板 {r['名称']}，它的强弱决定短线情绪高度：明天继续晋级说明情绪延续，断板放量则警惕退潮。")
    s.append("\n**启示**　涨停当天买不进，策略只在涨停后缩量回踩、不破涨停价时介入；情绪退潮期这类形态成功率会明显下降。")
    if bt_line:
        s.append("\n" + bt_line)
    return "".join(s)


def period_ranks(day: dt.date, zt: pd.DataFrame, k: int = 6) -> str:
    """本周 / 本月涨停次数榜：历史来自夜间 ztstats.py 生成的 data/zt_events.csv，今天来自涨停池"""
    path = data.DATA_DIR / "zt_events.csv"
    E = pd.read_csv(path, dtype={"code": str}) if path.exists() else pd.DataFrame(columns=["date", "code", "name"])
    E["date"] = pd.to_datetime(E["date"]).dt.date
    E = pd.concat([E[E["date"] < day][["date", "code", "name"]],
                   pd.DataFrame({"date": day, "code": zt["代码"], "name": zt["名称"]})], ignore_index=True)
    out = []
    for lab, start in (("本周", day - dt.timedelta(days=day.weekday())), ("本月", day.replace(day=1))):
        s = E[E["date"] >= start]
        g = s.groupby(["code", "name"]).size().sort_values(ascending=False)
        top = "、".join(f"{card.link(n, c)}×{v}" for (c, n), v in g.head(k).items() if v >= 2) or "暂无涨停 2 次以上的股票"
        out.append(f"**{lab}**涨停 {len(s)} 次 / {s['code'].nunique()} 只｜次数最多：{top}")
    return "**📅 涨停统计**\n" + "\n".join(out)


# ================================================================ 主流程
def main():
    dry = os.getenv("DRY_RUN") == "1"
    tdays = data.trade_days()
    day, why = data.target_day(tdays)
    if day is None:
        print(f"涨停复盘跳过：{why}"); return
    if os.getenv("FORCE_RUN") != "1" and FLAG.exists() and FLAG.read_text().strip() == day.isoformat():
        print(f"{day} 涨停复盘已发过，跳过"); return
    print("涨停复盘日期", day, f"（{why}）")

    P = em_pools(day)
    src = "东方财富涨停池"
    if P is None or P.get("zt") is None or P["zt"].empty:
        print("改用快照 + 历史缓存自算")
        data.EM_STATE["dead"] = False
        P = self_pools(day); src = "行情快照自算（无封板时间/封单）"
    zt = P["zt"]
    zt = zt[zt["代码"].map(main_board) & ~zt["名称"].astype(str).str.contains("ST|退", na=False)].copy()
    P["zt"] = zt

    hist = pd.read_csv(HIST) if HIST.exists() else pd.DataFrame()
    if len(hist):
        hist = hist[hist["date"] < day.isoformat()]
    m = metrics(day, P)
    ph = phase(m, hist)
    print("情绪指标", m, ph)

    # 连板梯队 & 行业聚集
    ladder = zt.sort_values(["连板", "封板资金"], ascending=[False, False])
    ind_cnt = zt["行业"].dropna().value_counts()
    top_ind = list(ind_cnt.head(5).items())

    # 代表股：最高的几个连板 + 主线行业里封单最大的首板
    reps = ladder[ladder["连板"] >= 2].head(6)
    main_inds = [a for a, _ in top_ind[:3]]
    firsts = zt[(zt["连板"] == 1) & zt["行业"].isin(main_inds)].sort_values("封板资金", ascending=False)
    reps = pd.concat([reps, firsts.head(4)]).drop_duplicates("代码").head(10)
    try:
        data.EM_STATE["dead"] = False  # 龙虎榜只有东方财富，涨停池失败也再试一次
        L = lhb(day, reps["代码"].tolist())
    except Exception as e:  # noqa: BLE001
        print("龙虎榜处理失败:", e); L = {}

    # 涨停系策略回测结果
    bt_line = None
    try:
        import backtest
        bt = backtest.load_prior()
        rand = bt.get("随机基准", {}).get("mean5")
        rows = [(n, v) for n, v in bt.items() if n.startswith("涨停")]
        if rows:
            bt_line = "**⚡ 涨停系策略回测**（过去 120 个交易日，5 日平均涨幅）　" + "　".join(
                f"{n} {card.pct_s(v['mean5'])}（t={v['t']:.1f}）" for n, v in rows) + \
                (f"　随机选股 {card.pct_s(rand)}" if rand is not None else "")
    except Exception as e:  # noqa: BLE001
        print("读取回测失败:", e)

    # ---------- 摘要给 AI
    def fmt_hist_row(r):
        return (f"{r['date']}: 涨停{int(r['zt'])} 炸板率{r['zb_rate']:.0%} 跌停{r['dt']:.0f} 最高{int(r['max_board'])}板 "
                f"昨涨停今均{r.get('prev_ret', np.nan):+.1%} 晋级率{r.get('promote', np.nan):.0%}")
    lines = [f"日期 {day}，数据源：{src}", "最近几天情绪指标："]
    lines += [fmt_hist_row(r) for _, r in hist.tail(5).iterrows()]
    lines.append(fmt_hist_row(m) + "（今天）")
    lines.append(f"规则判断：{ph}")
    lines.append("涨停行业分布：" + "、".join(f"{a}{b}家" for a, b in top_ind))
    lines.append("连板梯队：" + "；".join(f"{int(r['连板'])}板 {r['名称']}({r['行业']})"
                                      for _, r in ladder[ladder["连板"] >= 2].head(12).iterrows()))
    lines.append("代表股细节：")
    for _, r in reps.iterrows():
        lb = L.get(r["代码"])
        tg = style_tags(r, lb)
        lines.append(f"- {r['名称']} {int(r['连板'])}板 {r['行业']} 首封{r['首次封板']} 末封{r['最后封板']} 炸板{int(r['炸板次数'])} "
                     f"换手{r['换手率']:.1f}% 成交{yi(r['成交额'])} 封单{yi(r['封板资金'])}"
                     + (f" 龙虎榜净买{yi(lb['净买额'])} 买方{lb.get('买方', '')}" if lb else "")
                     + f" | 规则标签：{'；'.join(tg) or '无'}")
    if bt_line:
        lines.append(bt_line.replace("**", ""))
    digest = "\n".join(lines)
    print(digest)
    if not dry:  # 给每日观点报告用（market.py 会并入盘面摘要）
        (data.DATA_DIR / "market").mkdir(parents=True, exist_ok=True)
        (data.DATA_DIR / "market" / f"{day.isoformat()}-limitup.txt").write_text(digest)
    journal = ai_journal(digest)
    jsrc = "AI" if journal else "规则"
    if not journal:
        journal = rule_journal(m, ph, top_ind, ladder, None)

    # ---------- 卡片
    def pct(x):
        return "-" if x is None or pd.isna(x) else f"{x:.0%}"
    prev_txt = (f"昨日涨停今日 {card.pct_s(m['prev_ret'])}（上涨 {pct(m['prev_up'])}）"
                + (f"｜昨日连板今日 {card.pct_s(m['prev_lb_ret'])}" if pd.notna(m.get('prev_lb_ret', np.nan)) else "")
                if pd.notna(m.get("prev_ret", np.nan)) else "昨日涨停表现：无数据")
    trend = ""
    if len(hist):
        y = hist.iloc[-1]
        trend = f"\n较上一交易日：涨停 {int(m['zt']) - int(y['zt']):+d}｜最高板 {int(y['max_board'])} → {m['max_board']}"
    els = [card.md(f"**情绪温度**　涨停 <font color='red'>{m['zt']}</font>　炸板 {m['zb']:.0f}（炸板率 {pct(m['zb_rate'])}）"
                   f"　跌停 <font color='green'>{m['dt']:.0f}</font>\n"
                   f"{prev_txt}\n连板 {m['lianban']} 家｜晋级率 {pct(m.get('promote'))}｜最高 {m['max_board']} 板{trend}\n"
                   f"**周期判断**　{ph}")]
    if top_ind:
        els.append(card.md("**涨停主线**　" + "　".join(f"{a} **{b}**" for a, b in top_ind)))
    try:
        els.append(card.md(period_ranks(day, zt)))
    except Exception as e:  # noqa: BLE001
        print("本周/本月涨停榜失败:", e)
    lad = ladder[ladder["连板"] >= 2]
    if len(lad):
        rows = []
        for b, g in lad.groupby("连板", sort=False):
            rows.append(f"**{int(b)}板**　" + "、".join(card.link(r["名称"], r["代码"]) for _, r in g.head(8).iterrows()))
        els.append(card.md("**🪜 连板梯队**\n" + "\n".join(rows)))
    tag_lines = []
    for _, r in reps.iterrows():
        tg = style_tags(r, L.get(r["代码"]))
        if tg:
            info = f"首封 {r['首次封板']}｜" if r["首次封板"] != "-" else ""
            tag_lines.append(f"**{card.link(r['名称'], r['代码'])}** {int(r['连板'])}板 {r['行业'] if isinstance(r['行业'], str) else ''}\n"
                             f"{info}换手 {r['换手率']:.1f}%｜成交 {yi(r['成交额'])}\n🔍 " + "；".join(tg[:3]))
    if tag_lines:
        els += [card.HR, card.md("**🕵️ 主力手法推测**（依据封板时间、炸板、换手、龙虎榜席位，属推测）\n\n" + "\n\n".join(tag_lines[:8]))]
    els += [card.HR, card.md(f"**📓 今日心得**（{jsrc}）\n{journal}")]
    if bt_line:
        els += [card.HR, card.md(bt_line + "\n涨停系策略已加入策略库，每晚和其他策略一起回测，前后半段都跑赢随机选股才会上场。")]
    els.append({"tag": "note", "elements": [{"tag": "plain_text", "content":
               f"数据：{src}" + ("、东方财富龙虎榜" if L else "（龙虎榜暂无）") +
               "。手法标签和心得基于公开数据推测，不代表主力真实意图。仅供研究参考，不构成投资建议。"}]})
    payload = {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": "orange", "title": {"tag": "plain_text", "content": f"⚡ 涨停复盘 · {day:%m月%d日}"}},
        "elements": els}}
    if dry:
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return
    send_feishu(payload)
    data.DATA_DIR.mkdir(exist_ok=True)
    row = pd.DataFrame([m])
    out = pd.concat([pd.read_csv(HIST), row], ignore_index=True) if HIST.exists() else row
    out = out.drop_duplicates("date", keep="last").sort_values("date")
    out.to_csv(HIST, index=False)
    FLAG.write_text(day.isoformat())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if os.getenv("FEISHU_WEBHOOK") and os.getenv("DRY_RUN") != "1":
            try:
                send_feishu({"msg_type": "text", "content": {"text": f"涨停复盘运行失败：{e}"}})
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)
