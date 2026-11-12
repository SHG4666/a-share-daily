"""
夜间回测 + 策略评选：用全市场真实历史行情，回放过去 BT_DAYS 个交易日，给策略库里每个变体打分，选出明天上场的策略

做法（逐日回放，只用当时能看到的数据）：
  对每个历史交易日 t：
    1. 用截至 t 的行情计算全市场指标（与实盘同一套 features.py）
    2. 每个变体按规则选出前 BT_TOP_K 只
    3. 记录这些股票之后 1 / 5 / 10 个交易日的真实涨幅
  对照组：
    🎲 随机基准：每天从可交易股票里随机挑 K 只（判断策略是否真的比瞎选强）
    📈 追涨基准：每天买涨幅 5%~9% 的成交额前 K 只

评判（以涨幅为准）：
  5日平均涨幅、5日上涨比例、涨超 5% 的比例、10日平均涨幅
  t 值：每 5 天取一个不重叠的组合 5 日收益算 t 统计量，|t| ≥ 2 大致说明不是运气
  前半段 / 后半段：分别算 5 日平均涨幅，两段都跑赢随机选股才有资格上场（防止只在某段行情碰巧有效）

近似与偏差：
  - 股票池按当前上市状态和市值筛选（有幸存者偏差），每天再按当天成交额过滤
  - 稳健多因子用的是当前 PE/PB/换手率（有前视偏差），结果仅作参考
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import card
import config
import data
import main as daily
import model
import roster as roster_mod
import tracker
import variants
from main import send_feishu

PATH = data.DATA_DIR / "backtest.csv"
BASELINES = ["随机基准", "追涨基准"]


def replay(P: pd.DataFrame, dates: list[dt.date], seed: int) -> pd.DataFrame:
    """逐日回放所有变体，返回每笔推荐的前瞻收益"""
    rng = np.random.default_rng(seed)
    out = []
    t0 = time.time()
    for i, (t, U) in enumerate(P.groupby("date", sort=True), 1):
        U = variants.tradable(U)
        if len(U) < 200:
            continue
        picks = {}
        for v in variants.VARIANTS:
            try:
                picks[v.name] = variants.run(v, U, config.BT_TOP_K)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ {t} {v.name} 出错：{e}")
        picks["随机基准"] = U.sample(n=min(config.BT_TOP_K, len(U)), random_state=int(rng.integers(1e9)))
        picks["追涨基准"] = U[U["pct"].between(5, 9)].sort_values("amount", ascending=False).head(config.BT_TOP_K)
        for name, d in picks.items():
            if len(d):
                out.append(pd.DataFrame({"date": t, "strategy": name, "code": d["code"].values,
                                         "f1": d["f1"].values, "f5": d["f5"].values, "f10": d["f10"].values,
                                         "up5": d["up5"].values}))
        if i % 20 == 0:
            print(f"回放进度 {i}/{len(dates)}，用时 {time.time() - t0:.0f}s")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize(trades: pd.DataFrame, dates: list[dt.date]) -> pd.DataFrame:
    half = dates[len(dates) // 2] if dates else None
    rows = []
    for name, g in trades.groupby("strategy"):
        g5 = g.dropna(subset=["f5"])
        daily_ = g5.groupby("date")["f5"].mean().sort_index()
        nonover = daily_.iloc[::config.EVAL_HORIZON]  # 不重叠的 5 日窗口，避免样本自相关夸大显著性
        sd = nonover.std(ddof=1)
        t = nonover.mean() / (sd / np.sqrt(len(nonover))) if len(nonover) > 2 and sd > 0 else np.nan
        h1 = daily_[daily_.index < half].mean() if half else np.nan
        h2 = daily_[daily_.index >= half].mean() if half else np.nan
        rows.append(dict(
            strategy=name, days=g["date"].nunique(), n=len(g),
            mean5=g5["f5"].mean() if len(g5) else np.nan, win5=(g5["f5"] > 0).mean() if len(g5) else np.nan,
            big5=(g5["f5"] >= 0.05).mean() if len(g5) else np.nan, t=t, h1=h1, h2=h2,
            mean1=g["f1"].mean(), mean10=g["f10"].mean(), up5=g5["up5"].mean() if len(g5) else np.nan,
        ))
    return pd.DataFrame(rows).sort_values("mean5", ascending=False, na_position="last")


def report_card(day: dt.date, summ: pd.DataFrame, ros: dict, coverage: str, n_days: int, secs: float) -> dict:
    def tg(n):
        return {"随机基准": "🎲", "追涨基准": "📈"}.get(n, variants.tag(n))

    rand = summ[summ["strategy"] == "随机基准"]
    r5 = float(rand["mean5"].iloc[0]) if len(rand) else 0.0
    active = set(ros.get("active", []))
    lines = []
    for i, (_, r) in enumerate(summ.iterrows(), 1):
        if pd.isna(r["mean5"]):
            continue
        mark = "　✅上场" if r["strategy"] in active else ""
        sig = "显著" if abs(r["t"]) >= 2 else ("偏弱" if abs(r["t"]) >= 1 else "不显著")
        lines.append(
            f"**{i}. {tg(r['strategy'])}{r['strategy']}**{mark}　5日平均 **{card.pct_s(r['mean5'])}**"
            f"（随机 {card.pct_s(r5)}，t={r['t']:.1f} {sig}）\n"
            f"5日上涨 {r['win5']:.0%}｜涨超5% {r['big5']:.0%}｜10日平均 {card.pct_s(r['mean10'])}"
            f"｜前/后半段 {card.pct_s(r['h1'])} / {card.pct_s(r['h2'])}｜{int(r['days'])} 天有信号")
    ch = ros.get("changes", [])
    ch_txt = "\n".join(f"{'🔺' if c['action'] == '上线' else '🔻'} {c['action']} {variants.tag(c['strategy'])}"
                       f"{c['strategy']}：{c['reason']}" for c in ch) or "上场名单不变"
    els = [
        card.md(f"回放过去 **{n_days}** 个交易日，{coverage}。策略库 {len(variants.VARIANTS)} 个策略版本，"
                f"每个每天选前 {config.BT_TOP_K} 只，按收盘价买入，**以之后 5 个交易日的涨幅评判**。"),
        card.HR, card.md("**明日上场策略**　" + "、".join(f"{variants.tag(n)}{n}" for n in ros.get("active", []))
                         + "\n" + ch_txt + (f"\n⚠️ {ros['warn']}" if ros.get("warn") else "")),
        card.HR, card.md("\n\n".join(lines)),
        {"tag": "note", "elements": [{"tag": "plain_text", "content":
            "入选条件：前、后半段的 5 日平均涨幅都跑赢随机选股，按回测与实盘综合排名取前 "
            f"{config.ROSTER_SIZE} 个（同类最多 {config.ROSTER_MAX_PER_FAMILY} 个）。t 值用不重叠的 5 日窗口计算。"
            "近似：股票池按当前上市状态和市值筛选（有幸存者偏差）；稳健多因子用当前 PE/PB，仅供参考。"
            f"耗时 {secs / 60:.0f} 分钟。"}]},
    ]
    return {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": "indigo", "title": {"tag": "plain_text", "content": f"🌙 策略夜间回测 · {day:%m月%d日}"}},
        "elements": els}}


def load_prior() -> dict:
    """供 alloc 使用：最近一次回测的各策略 5 日平均涨幅"""
    if not PATH.exists():
        return {}
    df = pd.read_csv(PATH, dtype={"run_date": str})
    if df.empty or "mean5" not in df.columns:
        return {}
    last = df[df["run_date"] == df["run_date"].max()]
    return {r["strategy"]: {"mean5": r["mean5"], "t": r["t"], "days": r["days"]} for _, r in last.iterrows()}


def main():
    t0 = time.time()
    if os.getenv("BT_MODE") == "zt":
        print("只跑涨停统计，跳过回测"); return
    dry = os.getenv("DRY_RUN") == "1"
    tdays = data.trade_days()
    day, why = data.target_day(tdays)
    if day is None:
        print(f"跳过：{why}"); return
    flag = data.DATA_DIR / "last_backtest.txt"
    if os.getenv("FORCE_RUN") != "1" and os.getenv("BT_MODE") != "build" and flag.exists() \
            and flag.read_text().strip() == day.isoformat():
        print(f"{day} 已回测过，跳过"); return
    print("回测基准日", day, f"（{why}）")

    spot = data.get_spot()
    ind_map = data.get_industry_map(config.INDUSTRY_CACHE_DAYS)
    # 全市场：剔除 ST、北交所、市值过小（与实盘规则一致）；流动性按每个历史交易日的成交额单独过滤
    base = spot[~spot["名称"].str.contains("ST|退", na=False) & ~spot["代码"].str.startswith(config.EXCLUDE_PREFIX)
                & (spot["总市值"] >= config.MIN_MARKET_CAP)].copy()
    base["industry"] = model.fill_industry(base["代码"].map(ind_map), base["名称"])
    store = daily.STORE.load()
    store.append_snapshot(spot, day)
    traded = set(spot.loc[spot["成交量"] > 0, "代码"]) if "成交量" in spot else None
    left = store.ensure(base["代码"].tolist(), day, budget_sec=config.BT_FETCH_BUDGET, traded=traded)
    store.save()
    have = [c for c in base["代码"] if store.get(c) is not None]
    coverage = (f"全市场 {len(spot)} 只，剔除 ST/北交所/市值<{config.MIN_MARKET_CAP / 1e8:.0f}亿 后 {len(base)} 只，"
                f"有历史数据 {len(have)} 只" + (f"（还有 {left} 只下次补齐）" if left else "")
                + f"，每天再按当天成交额≥{config.MIN_AMOUNT / 1e8:.0f}亿过滤")
    print(coverage, f"用时 {time.time() - t0:.0f}s")
    if os.getenv("BT_MODE") == "build":
        print("仅构建历史缓存，结束"); return

    dates = [d for d in tdays if d < day][-config.BT_DAYS:]
    P = variants.make_panel(base, {c: store.get(c) for c in have}, set(dates))
    print(f"指标面板 {len(P)} 行，用时 {time.time() - t0:.0f}s")
    trades = replay(P, dates, seed=int(day.strftime("%Y%m%d")))
    if trades.empty:
        raise RuntimeError(f"回放没有产生任何推荐（指标面板 {len(P)} 行），请检查数据")
    summ = summarize(trades, dates)
    print(summ.to_string())

    ros = roster_mod.load()
    rec = tracker.load()
    changes = roster_mod.select_nightly(ros, summ, rec, tdays, day)
    print("明日上场:", ros["active"], "变动:", changes)

    payload = report_card(day, summ, ros, coverage, len(dates), time.time() - t0)
    if dry:
        print(json.dumps(payload, ensure_ascii=False, indent=2)); return
    out = summ.copy()
    out.insert(0, "run_date", day.isoformat())
    old = pd.read_csv(PATH, dtype={"run_date": str}) if PATH.exists() else pd.DataFrame()
    if len(old):
        old = old[old["run_date"] != day.isoformat()]
    data.DATA_DIR.mkdir(exist_ok=True)
    pd.concat([old, out], ignore_index=True).to_csv(PATH, index=False)
    roster_mod.save(ros)
    roster_mod.log_changes(day, changes)
    send_feishu(payload)
    flag.write_text(day.isoformat())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        if os.getenv("FEISHU_WEBHOOK") and os.getenv("DRY_RUN") != "1":
            try:
                send_feishu({"msg_type": "text", "content": {"text": f"夜间回测运行失败：{e}"}})
            except Exception:  # noqa: BLE001
                pass
        sys.exit(1)
