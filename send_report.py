"""
把 data/reports/ 下的观点报告以纯文字消息推送到飞书。

用法：python send_report.py data/reports/2026-11-13.md [更多文件...]
报告第一行是标题，其余是正文。飞书单条文本消息有长度上限，超长会按段落拆成多条发送。
"""
import sys
from pathlib import Path

from main import send_feishu

MAX_CHARS = 3800


def chunks(text: str) -> list[str]:
    out, cur = [], ""
    for para in text.split("\n\n"):
        if len(cur) + len(para) + 2 > MAX_CHARS and cur:
            out.append(cur.rstrip()); cur = ""
        cur += para + "\n\n"
    if cur.strip():
        out.append(cur.rstrip())
    return out


def send(path: Path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    title = lines[0].lstrip("# ").strip()
    body = "\n".join(lines[1:]).strip()
    parts = chunks(body)
    for i, part in enumerate(parts, 1):
        head = title + (f"（{i}/{len(parts)}）" if len(parts) > 1 else "")
        send_feishu({"msg_type": "text", "content": {"text": f"{head}\n\n{part}"}})
    print(f"已发送 {path}，共 {len(parts)} 条")


if __name__ == "__main__":
    files = [Path(a) for a in sys.argv[1:] if a.endswith(".md") and Path(a).exists()]
    if not files:
        print("没有要发送的报告"); sys.exit(0)
    for f in files:
        send(f)
