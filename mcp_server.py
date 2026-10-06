# MCP Server（手写零依赖 JSON-RPC 2.0 over stdio）
#
# 为什么手写而不装官方 SDK：官方 mcp SDK 要求 Python 3.10+，本仓承诺 3.9 可跑、零依赖可离线；
# MCP 本质是 JSON-RPC 2.0 + stdio 帧（Content-Length 头），协议层手写反而把「协议适配层」
# 这个能力本身做成了可看、可跑的交付物。
#
# 运行：
#   .venv/bin/python mcp_server.py            # 作为 MCP server 挂在 stdio 上
#   .venv/bin/python test_mcp_client.py       # 客户端验证 initialize→tools/list→tools/call
#
# 阅读约定：每个组件下的「防：…」标注它要防的失败类型。

import json
import subprocess
import sys
import textwrap

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "mini-agent-mcp"
SERVER_VERSION = "0.1.0"

# 工具执行沙箱参数：超时/输出截断/进程隔离，防失控工具拖死 server。
TOOL_TIMEOUT_SECONDS = 5
TOOL_OUTPUT_LIMIT = 4000

TOOLS = [
    {
        "name": "query_position",
        "description": "根据职位 ID 查询该招聘职位的详细要求。JD 一律以工具查询结果为准，可核对。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "position_id": {"type": "string", "description": '职位唯一标识，例如 "JD-001"'},
            },
            "required": ["position_id"],
        },
    }
]


# ---------- stdio 帧：Content-Length 头 + JSON body（LSP/MCP 标准帧格式） ----------
# 防：按行读 stdin → JSON body 里可能含换行，帧边界错乱。用 Content-Length 精确读字节。
def read_message(stream):
    headers = {}
    while True:
        line = stream.readline()
        if not line:
            return None  # EOF
        line = line.strip()
        if not line:
            break
        if b":" in line:
            k, v = line.split(b":", 1)
            headers[k.decode().strip().lower()] = v.decode().strip()
    length = int(headers.get("content-length", 0))
    if length <= 0:
        return None
    body = stream.read(length)
    return json.loads(body.decode())


def write_message(obj):
    body = json.dumps(obj, ensure_ascii=False).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def rpc_result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def rpc_error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


# ---------- 工具执行：子进程沙箱 ----------
# 防：工具逻辑在 server 进程内直接跑 → 一次死循环/内存爆炸整个 MCP server 陪葬。
# 子进程 + timeout + 输出截断，把爆炸半径限制在单个工具调用内。
SANDBOX_RUNNER = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo_dir!r})
    from app import JD_CORPUS
    args = json.loads(sys.argv[1])
    pid = args.get("position_id", "")
    for doc in JD_CORPUS:
        if doc["id"] == pid:
            print(json.dumps({{"found": True, "id": doc["id"], "title": doc["title"], "text": doc["text"]}}, ensure_ascii=False))
            break
    else:
        print(json.dumps({{"found": False, "id": pid}}, ensure_ascii=False))
    """
)


def run_tool_sandboxed(name, arguments):
    if name != "query_position":
        return None, "unknown tool: %s" % name
    import os

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    script = SANDBOX_RUNNER.format(repo_dir=repo_dir)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, json.dumps(arguments, ensure_ascii=False)],
            capture_output=True,
            timeout=TOOL_TIMEOUT_SECONDS,
            cwd=repo_dir,
        )
    except subprocess.TimeoutExpired:
        return None, "tool execution timed out after %ss (sandbox kill)" % TOOL_TIMEOUT_SECONDS
    if proc.returncode != 0:
        return None, "tool process exited %d: %s" % (
            proc.returncode,
            proc.stderr.decode(errors="replace")[:500],
        )
    out = proc.stdout.decode(errors="replace")
    if len(out) > TOOL_OUTPUT_LIMIT:
        out = out[:TOOL_OUTPUT_LIMIT] + "…[truncated by sandbox]"
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        return None, "tool output is not valid JSON: %r" % out[:200]
    return payload, None


def handle(msg):
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # 通知（无 id）不回复：initialized / cancelled 等。
    if msg_id is None:
        return None

    if method == "initialize":
        return rpc_result(msg_id, {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    if method == "ping":
        return rpc_result(msg_id, {})
    if method == "tools/list":
        return rpc_result(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        payload, err = run_tool_sandboxed(name, arguments)
        if err:
            # MCP 约定：工具执行失败用 isError 内容返回，而不是 JSON-RPC error。
            return rpc_result(msg_id, {
                "content": [{"type": "text", "text": err}],
                "isError": True,
            })
        return rpc_result(msg_id, {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": False,
        })
    return rpc_error(msg_id, -32601, "method not found: %s" % method)


def main():
    stdin = sys.stdin.buffer
    while True:
        msg = read_message(stdin)
        if msg is None:
            break
        resp = handle(msg)
        if resp is not None:
            write_message(resp)


if __name__ == "__main__":
    main()
