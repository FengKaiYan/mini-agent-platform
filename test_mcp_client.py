# MCP 客户端验证脚本：跑通 initialize → tools/list → tools/call 全链路。
# 用法：.venv/bin/python test_mcp_client.py
import json
import subprocess
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))


class StdioMcpClient:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO
        )
        self._next_id = 1

    def _send(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body))
        self.proc.stdin.write(body)
        self.proc.stdin.flush()

    def _recv(self):
        headers = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("server closed stdout")
            line = line.strip()
            if not line:
                break
            if b":" in line:
                k, v = line.split(b":", 1)
                headers[k.decode().lower()] = v.decode().strip()
        n = int(headers["content-length"])
        return json.loads(self.proc.stdout.read(n).decode())

    def request(self, method, params=None):
        msg_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
        resp = self._recv()
        assert resp.get("id") == msg_id, "id mismatch: %r" % resp
        return resp

    def notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


def main():
    client = StdioMcpClient([sys.executable, os.path.join(REPO, "mcp_server.py")])

    # 1) initialize 握手
    r = client.request("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "verify-client", "version": "0.1.0"},
    })
    info = r["result"]["serverInfo"]
    print("initialize OK →", info["name"], info["version"], "| protocol:", r["result"]["protocolVersion"])
    client.notify("notifications/initialized")

    # 2) tools/list
    r = client.request("tools/list")
    tools = r["result"]["tools"]
    print("tools/list OK →", [t["name"] for t in tools])
    assert tools and tools[0]["name"] == "query_position"
    assert tools[0]["inputSchema"]["required"] == ["position_id"]

    # 3) tools/call 正常路径
    r = client.request("tools/call", {"name": "query_position", "arguments": {"position_id": "JD-001"}})
    content = r["result"]["content"][0]["text"]
    payload = json.loads(content)
    print("tools/call OK → found:", payload["found"], "| title:", payload.get("title"))
    assert payload["found"] is True and "智能体" in payload["text"]
    assert r["result"]["isError"] is False

    # 4) tools/call 未命中路径
    r = client.request("tools/call", {"name": "query_position", "arguments": {"position_id": "JD-999"}})
    payload = json.loads(r["result"]["content"][0]["text"])
    print("tools/call miss OK → found:", payload["found"])
    assert payload["found"] is False

    # 5) 未知工具 → isError 内容（MCP 约定，非 JSON-RPC error）
    r = client.request("tools/call", {"name": "nope", "arguments": {}})
    print("unknown tool OK → isError:", r["result"]["isError"], "|", r["result"]["content"][0]["text"][:40])
    assert r["result"]["isError"] is True

    # 6) 未知 method → JSON-RPC -32601
    r = client.request("bogus/method")
    print("unknown method OK → code:", r["error"]["code"])
    assert r["error"]["code"] == -32601

    client.close()
    print("\nPASS: MCP Server 全链路验证通过（initialize/tools/list/tools/call/错误路径）")


if __name__ == "__main__":
    main()
