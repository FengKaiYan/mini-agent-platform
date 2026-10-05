import asyncio, os
os.environ.pop("DASHSCOPE_API_KEY", None)  # 强制 mock 模式

import app
assert app.MOCK, "应为 MOCK 模式"

async def main():
    cfg = {"configurable": {"thread_id": "verify"}}
    result = await app.graph.ainvoke(
        {"messages": [{"role": "user", "content": "这个候选人和 JD-001 岗位匹配吗"}]},
        config=cfg,
    )
    print("=== 消息序列 ===")
    saw_tool_call = False
    saw_tool_msg = False
    for m in result["messages"]:
        tc = getattr(m, "tool_calls", None)
        typ = type(m).__name__
        if tc:
            saw_tool_call = True
        if typ == "ToolMessage":
            saw_tool_msg = True
        print(f"  {typ:14s} | tool_calls={tc} | content={(m.content or '')[:50]!r}")
    print()
    print("AIMessage 带 tool_calls :", saw_tool_call)
    print("ToolMessage 工具执行回灌 :", saw_tool_msg)
    print("route                  :", result.get("route"))
    print("ctx(RAG 命中)           :", result.get("ctx"))
    print()
    assert saw_tool_call, "FAIL: 没有 tool_calls，FC 未落地"
    assert saw_tool_msg, "FAIL: 没有 ToolMessage，工具未执行"
    print("PASS: FC 执行 + ReAct 回环 真实落地（mock 模式，无需 key）")

asyncio.run(main())
