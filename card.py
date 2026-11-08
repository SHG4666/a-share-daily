"""飞书卡片渲染"""
import datetime as dt

import pandas as pd


def xueqiu(code: str) -> str:
    if code.startswith(("8", "4", "92")):
        mkt = "BJ"
    elif code.startswith(("6", "9")):
        mkt = "SH"
    else:
        mkt = "SZ"
    return f"https://xueqiu.com/S/{mkt}{code}"


def link(name: str, code: str) -> str:
    return f"[{name}]({xueqiu(code)})"


def color_pct(p) -> str:
    if p is None or pd.isna(p):
        return "-"
    c = "red" if p > 0 else ("green" if p < 0 else "grey")
    return f"<font color='{c}'>{p:+.2f}%</font>"


def md(text: str) -> dict:
    return {"tag": "markdown", "content": text}


def build(day: dt.date, spot, regime, inds, picks: pd.DataFrame, watches, track, comment) -> dict:
    up, down = int((spot["涨跌幅"] > 0).sum()), int((spot["涨跌幅"] < 0).sum())
    els = [md(f"**市场概况**　上涨 <font color='red'>{up}</font> / 下跌 <font color='green'>{down}</font>"
              f"　成交额 **{spot['成交额'].sum() / 1e12:.2f} 万亿**\n"
              f"**市场状态**　{regime['text']}" + ("" if regime["strong"] else "　⚠️ 弱势，推荐数已下调"))]
    if inds is not None and len(inds):
        els.append(md("**领涨行业**　" + "　".join(f"{r['板块名称']} {color_pct(r['涨跌幅'])}"
                                                  for _, r in inds.iterrows())))
    if comment:
        els.append(md(f"**AI 点评**\n{comment}"))

    els.append({"tag": "hr"})
    if len(picks):
        lines = [f"**📊 量化精选**（{picks['pool'].iloc[0]} 只通过全部条件，取综合分最高的 {len(picks)} 只）"]
        for i, (_, p) in enumerate(picks.iterrows(), 1):
            s = (f"**{i}. {link(p['名称'], p['代码'])}** {p['代码']}　{p['最新价']:.2f} {color_pct(p['涨跌幅'])}"
                 f"　综合分 {p['score']:.2f}\n"
                 f"趋势年化 {p['trend_ann']:.0%}（R² {p['r2']:.2f}）｜波动 {p['vol']:.0%}｜60日回撤 {p['mdd']:.0%}\n"
                 f"ROE≈{p['roe']:.0%}｜PE {p['市盈率-动态']:.0f}｜{p['industry'] if isinstance(p['industry'], str) else ''}"
                 f"｜参考止损 {p['stop']:.2f}\n"
                 f"💡 {'、'.join(p['reasons']) or '各项均衡'}")
            if p["risks"]:
                s += f"\n⚠️ {'、'.join(p['risks'])}"
            lines.append(s)
        els.append(md("\n\n".join(lines)))
    else:
        els.append(md("**📊 量化精选**\n今天没有同时满足基本面和趋势条件的标的，空仓也是一种选择。"))

    if watches:
        els.append({"tag": "hr"})
        els.append(md("**👀 自选主题池**\n" + "\n".join(
            f"{link(w['name'], w['code'])} {color_pct(w['pct'])}　{w['tag']}" for w in watches)))

    if track:
        els.append({"tag": "hr"})
        els.append(md(f"**📈 策略追踪**\n{track}"))

    els.append({"tag": "note", "elements": [{"tag": "plain_text", "content":
               "模型：行业中性化 EP/BP/ROE/低换手 → 趋势回归(斜率×R²)、波动、回撤、短期反转综合打分。"
               "点击股票名查看雪球走势。仅供研究参考，不构成投资建议。"}]})
    return {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": "red", "title": {"tag": "plain_text", "content": f"A股每日复盘 · {day:%m月%d日}"}},
        "elements": els}}
