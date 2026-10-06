# 统一 CLI 入口 —— Harness 加分5：CLI / 自动化调试工具 / 场景回放系统
#
# 用法：
#   .venv/bin/python cli.py serve [--port 8000]
#   .venv/bin/python cli.py mcp
#   .venv/bin/python cli.py record "问题" [thread_id] [out.jsonl]
#   .venv/bin/python cli.py replay traces/xxx.jsonl
#   .venv/bin/python cli.py diff a.jsonl b.jsonl
#   .venv/bin/python cli.py screen "简历文本" [position_id]
#
# 防：能力散在多个入口脚本（app/mcp_server/trace_recorder/replay 各有 __main__）
# → 调试与演示路径不统一、交接成本高；CLI 收敛成单入口，子命令即能力清单。

import argparse
import asyncio
import json
import os

import app
import mcp_server
import replay as replay_mod
import trace_recorder

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


def cmd_serve(args):
    import uvicorn

    print("打开 http://127.0.0.1:%d （MOCK=%s）" % (args.port, app.MOCK))
    uvicorn.run(app.api, host="127.0.0.1", port=args.port)


def cmd_mcp(args):
    mcp_server.main()


def cmd_record(args):
    tid = args.thread or "cli"
    out = args.out or os.path.join(TRACE_DIR, tid + ".jsonl")
    path = asyncio.run(trace_recorder.record(args.message, tid, out))
    n = sum(1 for _ in open(path, encoding="utf-8")) - 1
    print("recorded %d events → %s" % (n, path))


def cmd_replay(args):
    header, events = replay_mod.load(args.trace)
    replay_mod.render(header, events)


def cmd_diff(args):
    replay_mod.diff(args.a, args.b)


def cmd_screen(args):
    position = next((d for d in app.JD_CORPUS if d["id"] == args.position_id), app.JD_CORPUS[0])
    item = app.score_resume({"id": "input", "name": "输入简历", "text": args.resume_text}, position)
    print(json.dumps(item.model_dump(), ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(prog="cli.py", description="mini Agent 平台统一 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="起 FastAPI 服务（SSE 流式 + Web UI）")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("mcp", help="以 MCP server 挂 stdio")
    s.set_defaults(fn=cmd_mcp)

    s = sub.add_parser("record", help="录制会话/事件 trace（JSONL）")
    s.add_argument("message")
    s.add_argument("thread", nargs="?")
    s.add_argument("out", nargs="?")
    s.set_defaults(fn=cmd_record)

    s = sub.add_parser("replay", help="重放 trace 渲染调试面板")
    s.add_argument("trace")
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("diff", help="两条 trace 结构对比")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("screen", help="单份简历 × 岗位 → 结构化初筛结论")
    s.add_argument("resume_text")
    s.add_argument("position_id", nargs="?", default="JD-001")
    s.set_defaults(fn=cmd_screen)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
