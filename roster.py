"""
上场策略名单：哪些变体参与每日推荐，以及什么时候替换

夜间评选（select_nightly，每晚回测后）：
  综合分 = 回测 5 日平均涨幅(按样本数收缩) × (1 − w) + 实盘 5 日平均涨幅 × w，w = 实盘样本 / (实盘样本 + 20)
  入选条件：
    - 回测里有推荐的交易日 ≥ ROSTER_MIN_DAYS，推荐总数 ≥ ROSTER_MIN_PICKS
    - t 值 ≥ ROSTER_MIN_T（用不重叠的 5 日窗口计算）
    - 前半段和后半段的 5 日平均涨幅都高于同期随机选股（防止只在某一段行情里碰巧有效）
    - 不在冷却期
  取综合分最高的 ROSTER_SIZE 个，同一类策略最多 ROSTER_MAX_PER_FAMILY 个
  没有任何变体满足条件时，保留综合分最高的 2 个，并在卡片上提示

实盘熔断（apply_kill_switch，每天推荐前）：
  某个上场变体最近 KILL_N 只已满 5 日的推荐，平均 5 日涨幅 < KILL_R5，立即下线并冷却 COOLDOWN_DAYS 个交易日，
  由回测排名里下一个合格的变体补上
"""
import datetime as dt
import json

import numpy as np
import pandas as pd

import config
import variants
from data import DATA_DIR

PATH = DATA_DIR / "roster.json"
LOG = DATA_DIR / "roster_log.csv"


def load() -> dict:
    if PATH.exists():
        try:
            return json.loads(PATH.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"date": None, "active": list(variants.DEFAULT_ROSTER), "cooldown": {}, "ranking": [], "changes": []}


def save(r: dict):
    DATA_DIR.mkdir(exist_ok=True)
    PATH.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))


def log_changes(day: dt.date, changes: list[dict]):
    if not changes:
        return
    df = pd.DataFrame([{"date": day.isoformat(), **c} for c in changes])
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    DATA_DIR.mkdir(exist_ok=True)
    pd.concat([old, df], ignore_index=True).to_csv(LOG, index=False)


def _in_cooldown(r: dict, name: str, day: dt.date) -> bool:
    until = r.get("cooldown", {}).get(name)
    return bool(until) and day.isoformat() <= until


def _cooldown_until(tdays: list[dt.date], day: dt.date) -> str:
    after = [t for t in tdays if t > day]
    n = config.COOLDOWN_DAYS
    return (after[n - 1] if len(after) >= n else day + dt.timedelta(days=int(n * 1.5))).isoformat()


def live_stats(rec: pd.DataFrame) -> dict:
    ev = rec.dropna(subset=["r5"])
    out = {}
    for s, g in ev.groupby("strategy"):
        g = g.sort_values("date")
        out[s] = {"n": len(g), "mean5": float(g["r5"].mean()),
                  "recent_n": min(len(g), config.KILL_N), "recent5": float(g["r5"].tail(config.KILL_N).mean())}
    return out


def apply_kill_switch(r: dict, rec: pd.DataFrame, tdays, day: dt.date) -> list[dict]:
    """实盘熔断：近期表现太差的上场策略立即下线，用回测排名中下一个合格的补上"""
    stats = live_stats(rec)
    changes = []
    for name in list(r["active"]):
        s = stats.get(name)
        if s and s["recent_n"] >= config.KILL_MIN_N and s["recent5"] < config.KILL_R5:
            r["active"].remove(name)
            r.setdefault("cooldown", {})[name] = _cooldown_until(tdays, day)
            changes.append({"action": "下线", "strategy": name,
                            "reason": f"实盘最近 {s['recent_n']} 只推荐 5 日平均 {s['recent5']:+.1%}，触发熔断，冷却至 {r['cooldown'][name]}"})
    fams = [variants.family_of(n) for n in r["active"]]
    for cand in r.get("ranking", []):
        if len(r["active"]) >= config.ROSTER_SIZE:
            break
        name = cand["strategy"]
        if (name in r["active"] or not cand.get("eligible") or _in_cooldown(r, name, day)
                or fams.count(variants.family_of(name)) >= config.ROSTER_MAX_PER_FAMILY):
            continue
        r["active"].append(name); fams.append(variants.family_of(name))
        changes.append({"action": "上线", "strategy": name,
                        "reason": f"替补：回测 5 日平均 {cand['mean5']:+.1%}，综合排名第 {cand['rank']}"})
    return changes


def select_nightly(r: dict, summ: pd.DataFrame, rec: pd.DataFrame, tdays, day: dt.date) -> list[dict]:
    """夜间评选：根据回测结果 + 实盘战绩，重新决定上场名单"""
    stats = live_stats(rec)
    rand = summ[summ["strategy"] == "随机基准"]
    rh1 = float(rand["h1"].iloc[0]) if len(rand) else 0.0
    rh2 = float(rand["h2"].iloc[0]) if len(rand) else 0.0
    rows = []
    for _, s in summ.iterrows():
        name = s["strategy"]
        if name not in variants.BY_NAME:
            continue
        live = stats.get(name, {"n": 0, "mean5": 0.0})
        w = live["n"] / (live["n"] + 20)
        # 样本少的向 0 收缩，避免几只大涨股把平均值撑高
        bt5 = (float(s["mean5"]) * s["n"] / (s["n"] + config.SHRINK_N)) if pd.notna(s["mean5"]) else -1.0
        combined = bt5 * (1 - w) + live["mean5"] * w
        eligible = bool(s["days"] >= config.ROSTER_MIN_DAYS and s["n"] >= config.ROSTER_MIN_PICKS
                        and pd.notna(s["t"]) and s["t"] >= config.ROSTER_MIN_T
                        and pd.notna(s["h1"]) and pd.notna(s["h2"])
                        and s["h1"] > rh1 and s["h2"] > rh2 and not _in_cooldown(r, name, day))
        why = ("冷却中" if _in_cooldown(r, name, day) else
               f"回测信号太少（{int(s['days'])} 天 / {int(s['n'])} 只）"
               if s["days"] < config.ROSTER_MIN_DAYS or s["n"] < config.ROSTER_MIN_PICKS else
               f"统计上不够可靠（t={s['t']:.1f}）" if not (pd.notna(s["t"]) and s["t"] >= config.ROSTER_MIN_T) else
               "回测前后两段没有都跑赢随机选股" if not eligible else "")
        rows.append({"strategy": name, "mean5": float(s["mean5"]) if pd.notna(s["mean5"]) else None,
                     "live_n": live["n"], "live5": live["mean5"], "combined": combined, "eligible": eligible,
                     "why": why})
    rows.sort(key=lambda x: -x["combined"])
    for i, x in enumerate(rows, 1):
        x["rank"] = i

    new, fams = [], []
    for x in rows:
        fam = variants.family_of(x["strategy"])
        if x["eligible"] and fams.count(fam) < config.ROSTER_MAX_PER_FAMILY:
            new.append(x["strategy"]); fams.append(fam)
        if len(new) >= config.ROSTER_SIZE:
            break
    warn = None
    if not new:
        new = [x["strategy"] for x in rows[:2]]
        warn = "没有任何策略在前后两段都跑赢随机选股，暂时保留综合排名前 2 的策略，建议谨慎参考"

    old = list(r.get("active", []))
    info = {x["strategy"]: x for x in rows}
    changes = []
    for n in old:
        if n not in new:
            x = info.get(n)
            why = (x["why"] if x and not x["eligible"] else
                   f"综合排名降到第 {x['rank']}" if x else "不在策略库中")
            changes.append({"action": "下线", "strategy": n, "reason": why})
    for n in new:
        if n not in old:
            x = info[n]
            changes.append({"action": "上线", "strategy": n,
                            "reason": f"回测 5 日平均 {x['mean5']:+.1%}，综合排名第 {x['rank']}"})
    r.update(date=day.isoformat(), active=new, ranking=rows, changes=changes, warn=warn)
    r["cooldown"] = {k: v for k, v in r.get("cooldown", {}).items() if v >= day.isoformat()}
    return changes
