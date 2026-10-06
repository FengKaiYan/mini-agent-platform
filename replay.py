# 事件重放与 trace 对比（Harness 职责4/加分5：事件重放、场景回放、调试面板的 CLI 形态）
#
# 用法：
#   .venv/bin/python replay.py traces/xxx.jsonl              # 按录制时序重放，终端渲染成调试面板
#   .venv/bin/python replay.py a.jsonl --diff b.jsonl        # 两条 trace 结构对比（路由/节点序列/工具调用差异）
#
# 防：改 prompt/改图之后「感觉变好了」→ diff 给出结构级差异，好坏有据可查。

import json
import sys


def load(path):
    header, events = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "header":
                header = rec
            else:
                events.append(rec)
    return header, events


def render(header, events):
    print("══ trace: %s" % (header or {}).get("thread_id", "?"))
    print("   输入: %s" % (header or {}).get("message", "?"))
    stream_buf = []
    for rec in events:
        kind, node, t = rec["event"], rec.get("node", ""), rec["t"]
        if kind == "on_chat_model_stream":
            stream_buf.append(rec.get("chunk") or "")
        elif kind == "on_chain_start":
            if stream_buf:
                print("   💬 %s" % "".join(stream_buf)[:80])
                stream_buf = []
            print("  [%6.2fs] ▶ %s" % (t, node))
        elif kind == "on_chain_end":
            out = rec.get("output") or {}
            extra = ""
            if "route" in out:
                extra = " route=%s" % out["route"]
            if "ctx" in out:
                extra += " rag=%s" % "、".join(out["ctx"])
            print("  [%6.2fs] ✔ %s%s" % (t, node, extra))
        elif kind == "on_tool_end":
            print("  [%6.2fs] 🔧 tool → %s…" % (t, (rec.get("tool_output") or "")[:60]))
    if stream_buf:
        print("   💬 %s" % "".join(stream_buf)[:80])


def signature(events):
    """结构签名：节点进出序列 + 工具调用 + 路由，用于 diff。"""
    sig = []
    for rec in events:
        if rec["event"] == "on_chain_start":
            sig.append(("enter", rec.get("node")))
        elif rec["event"] == "on_chain_end":
            out = rec.get("output") or {}
            sig.append(("exit", rec.get("node"), out.get("route"), tuple(out.get("ctx") or [])))
        elif rec["event"] == "on_tool_end":
            sig.append(("tool", rec.get("tool_output", "")[:40]))
    return sig


def diff(path_a, path_b):
    _, ea = load(path_a)
    _, eb = load(path_b)
    sa, sb = signature(ea), signature(eb)
    if sa == sb:
        print("结构一致（节点序列/路由/RAG 命中/工具调用全同）")
        return
    print("结构差异：")
    la, lb = len(sa), len(sb)
    for i in range(max(la, lb)):
        a = sa[i] if i < la else None
        b = sb[i] if i < lb else None
        if a != b:
            print("  #%d  A=%s" % (i, a))
            print("       B=%s" % (b,))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--diff" in args:
        i = args.index("--diff")
        diff(args[i - 1], args[i + 1])
    else:
        header, events = load(args[0])
        render(header, events)
