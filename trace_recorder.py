# 会话/事件回放（Harness 职责4：场景启动器、会话回放、事件重放、调试面板）
#
# 把一次 Agent 运行的事件流录成 JSONL trace（seq + 相对时间 + 事件类型 + 节点 + 载荷摘要），
# 之后可离线重放（replay.py）或两条 trace 对比（replay.py --diff）。
# 防：线上 badcase 不可复现 → 录下来的 trace 就是复现载体，归因不再靠人肉重跑。
#
# 用法：
#   .venv/bin/python trace_recorder.py "这个候选人和 JD-001 岗位匹配吗" [thread_id] [out.jsonl]

import asyncio
import json
import os
import sys
import time

import app

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")

# 只录对回放/归因有用的事件，避免 trace 膨胀成噪音。
EVENT_FILTER = {"on_chat_model_stream", "on_chain_start", "on_chain_end", "on_tool_end"}
NODES = {"supervisor", "screener", "matcher", "interviewer", "chatter", "tools", "memory", "recall"}


def _safe_output(out):
    """节点输出只保留归因字段（route/ctx），messages 太大且含对象不可序列化。"""
    if isinstance(out, dict):
        keep = {}
        for k in ("route", "ctx"):
            if k in out:
                keep[k] = out[k]
        return keep or None
    return None


async def record(message: str, thread_id: str, out_path: str) -> str:
    cfg = {"configurable": {"thread_id": thread_id}}
    t0 = time.time()
    seq = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "header", "message": message,
                            "thread_id": thread_id, "recorded_at": time.time()},
                           ensure_ascii=False) + "\n")
        async for ev in app.graph.astream_events(
            {"messages": [{"role": "user", "content": message}]},
            config=cfg, version="v2",
        ):
            kind = ev["event"]
            if kind not in EVENT_FILTER:
                continue
            name = ev.get("name", "")
            if kind in ("on_chain_start", "on_chain_end") and name not in NODES:
                continue
            rec = {"seq": seq, "t": round(time.time() - t0, 3), "event": kind, "node": name}
            if kind == "on_chat_model_stream":
                rec["chunk"] = ev["data"]["chunk"].content
            elif kind == "on_chain_end":
                rec["output"] = _safe_output(ev["data"].get("output"))
            elif kind == "on_tool_end":
                out = ev["data"].get("output")
                rec["tool_output"] = str(getattr(out, "content", out))[:200]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seq += 1
    return out_path


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "这个候选人和 JD-001 岗位匹配吗"
    tid = sys.argv[2] if len(sys.argv) > 2 else "trace-demo"
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(TRACE_DIR, tid + ".jsonl")
    path = asyncio.run(record(msg, tid, out))
    n = sum(1 for _ in open(path, encoding="utf-8")) - 1
    print("recorded %d events → %s" % (n, path))
