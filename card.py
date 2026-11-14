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
    return f"[{str(name).replace(' ', '')}]({xueqiu(code)})"


def pct_s(x, nd: int = 1) -> str:
    """小数收益率 → 带颜色百分比（A 股习惯：红涨绿跌）"""
    if x is None or pd.isna(x):
        return "-"
    c = "red" if x > 0 else ("green" if x < 0 else "grey")
    return f"<font color='{c}'>{x * 100:+.{nd}f}%</font>"


def md(text: str) -> dict:
    return {"tag": "markdown", "content": text}


HR = {"tag": "hr"}


def build(day: dt.date, spot, regime, inds, review_txt, adjust_txt, picks: pd.DataFrame,
          pool_lines: list[str], board_txt, comment, details: dict, accum=None) -> dict:
    from variants import tag, VARIANTS
    n_var = len(VARIANTS)
    up, down = int((spot["涨跌幅"] > 0).sum()), int((spot["涨跌幅"] < 0).sum())
    els = [md(f"**市场概况**　上涨 <font color='red'>{up}</font> / 下跌 <font color='green'>{down}</font>"
              f"　成交额 **{spot['成交额'].sum() / 1e12:.2f} 万亿**\n"
              f"**市场状态**　{regime['text']}" + ("" if regime["strong"] else "　⚠️ 弱势"))]
    if inds is not None and len(inds):
        els.append(md("**领涨行业**　" + "　".join(f"{r['板块名称']} {pct_s(r['涨跌幅'] / 100, 2)}"
                                                  for _, r in inds.iterrows())))
    if comment:
        els.append(md(f"**AI 点评**\n{comment}"))

    els += [HR, md("**📝 昨日复盘**\n" + (review_txt or "暂无上一交易日的推荐记录。"))]
    els.append(md("**🔧 今日策略调整**\n" + adjust_txt))

    els.append(HR)
    if len(picks):
        lines = [f"**🎯 今日推荐**（{len(picks)} 只）"]
        for i, (_, p) in enumerate(picks.iterrows(), 1):
            ind = p.get("industry")
            s = (f"**{i}. {link(p['名称'], p['代码'])}** {p['代码']}　{p['最新价']:.2f} {pct_s(p['涨跌幅'] / 100, 2)}"
                 f"　{tag(p['strategy'])}{p['strategy']}" + (f"｜{ind}" if isinstance(ind, str) else "") + "\n"
                 f"{details.get(p['代码'], '')}\n"
                 f"止损参考 {p['stop']:.2f}（距现价 {p['stop'] / p['close'] - 1:.0%}）\n"
                 f"💡 {'、'.join(p['reasons']) or '各项均衡'}")
            if p["risks"]:
                s += f"\n⚠️ {'、'.join(p['risks'])}"
            lines.append(s)
        els.append(md("\n\n".join(lines)))
    else:
        els.append(md("**🎯 今日推荐**\n今天所有策略都没有满足条件的标的，空仓也是一种选择。"))

    if accum is not None:
        acc_df, acc_track = accum
        if acc_df is not None and len(acc_df):
            lines = [f"**🕵️ 疑似主力吸筹**（全市场扫描，{int(acc_df['pool'].iloc[0])} 只符合，列得分前 {len(acc_df)} 只；"
                     "吸筹见效慢，适合放进观察池，等放量突破箱体再考虑）"]
            for i, (_, p) in enumerate(acc_df.iterrows(), 1):
                ind = p.get("industry")
                lines.append(f"{i}. {link(p['名称'], p['代码'])} {p['代码']}　{p['最新价']:.2f} {pct_s(p['涨跌幅'] / 100, 2)}"
                             + (f"｜{ind}" if isinstance(ind, str) else "") + ("　⭐严格条件也满足" if p.get("strict") else "")
                             + f"\n{p['detail']}\n失效位 {p['stop']:.2f}（跌破箱体底部说明判断失效，距现价 {p['stop'] / p['close'] - 1:.0%}）")
            if acc_track:
                lines.append(f"<font color='grey'>{acc_track}</font>")
            els += [HR, md("\n\n".join(lines))]
        elif acc_track:
            els += [HR, md("**🕵️ 疑似主力吸筹**\n今天全市场没有符合条件的股票。\n" + acc_track)]

    if pool_lines:
        els += [HR, md(f"**📌 推荐跟踪池**（最近 {len(pool_lines)} 只，按推荐后累计涨幅排序）\n" + "\n".join(pool_lines))]
    if board_txt:
        els += [HR, md("**📈 策略实盘战绩**（全部历史推荐，以推荐后 5 日涨幅评判）\n" + board_txt)]

    els.append({"tag": "note", "elements": [{"tag": "plain_text", "content":
               f"策略库：🔥强势延续 ↩️趋势回踩 🚀底部启动 🛡️稳健多因子 💥放量突破 🪃超跌反弹 🐢低波稳涨 ⚡涨停回踩（共{n_var}个参数版本）。"
               "每晚全市场回测评选上场策略，实盘表现差的立即下线替换。"
               "点击股票名查看雪球走势。仅供研究参考，不构成投资建议。"}]})
    return {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": "red", "title": {"tag": "plain_text", "content": f"A股每日复盘 · {day:%m月%d日}"}},
        "elements": els}}
