# mini-agent-platform · mini Agent 平台（智能体招聘助手）

一个**单文件可跑**的智能体招聘助手：supervisor 多智能体编排 + 本地 RAG + 双层记忆 + SSE 流式 + 工具调用，
并带**结构化简历初筛**（简历 × 岗位 → 分数 / 结论 / 理由 / 证据）。
对齐「智能体平台研发」JD 的四块核心能力，全部在 `app.py` 内闭环。

> 样例简历为**合成虚构数据**，仅用于 demo；请勿替换为真实候选人简历（PII）。

---

## 快速开始

```bash
git clone https://github.com/FengKaiYan/mini-agent-platform.git
cd mini-agent-platform
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py        # 无需 API key，内置 mock 模型即可看全流程
# 打开 http://127.0.0.1:8000
```

配了 key 会自动切换真实 qwen 并启用工具调用：

```bash
export DASHSCOPE_API_KEY=sk-xxx
.venv/bin/python app.py        # MOCK=False
```

## 两个 demo 入口

| 入口 | 说明 |
|---|---|
| 页面聊天框 | 自然语言 → supervisor 路由到 screener/matcher/interviewer/chat，SSE 流式回包，实时显示「路由 / RAG 命中 / 记忆更新」 |
| 页面「批量初筛 demo」按钮 | 5 份合成简历 × JD-001 → 排序 + 每人分数/结论/理由/证据（结构化） |

## HTTP 接口

| 方法 | 路径 | 返回 |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/chat` | SSE 流式（`data: {"result":…}` / `{"meta":…}` / `[DONE]`） |
| POST | `/screen` | 单份简历文本 × 岗位 → `ScreenItem`（结构化） |
| GET | `/rank?position_id=JD-001` | 合成样例批量初筛 → `ScreenReport`（排序） |

结构化输出示例：

```json
{
  "resume_id": "R-01",
  "candidate": "示例·林知远(合成)",
  "score": 94,
  "verdict": "推荐进入下一轮",
  "reasons": ["命中 Agent 方向信号：智能体、langgraph、langchain、多智能体、编排",
              "有可验证硬信号：性能、资损、qps",
              "与岗位语义相似度 cos=0.46"],
  "evidence": ["JD-001 智能体平台高级研发工程师", "CR-02 Agent 方向加分项"]
}
```

## 架构

```mermaid
graph TD
    U[用户/前端] -->|POST /chat SSE| API[FastAPI]
    API --> G[LangGraph StateGraph]
    G --> R[recall 注入长期记忆]
    R --> S{supervisor 路由}
    S -->|screen| SC[screener 简历初筛]
    S -->|match| M[matcher 人岗匹配]
    S -->|interview| I[interviewer 面试安排]
    S -->|chat| C[chatter 通用对话]
    M -->|tools_condition 有 tool_calls| T[tools ToolNode 执行 query_position]
    T -->|ReAct 回环:观察结果再思考| M
    M -->|无 tool_calls 收敛| MEM
    SC --> RAG[RAG 检索 n-gram+余弦]
    M --> RAG
    I --> RAG
    C --> RAG
    SC --> MEM[memory 写长期记忆]
    I --> MEM
    C --> MEM
    MEM --> E[END]
    SC -->|结构化 ScreenItem| OUT[/screen /rank]
```

## 能力 ↔ JD 映射

| JD 要求 | 本实现 | 位置 |
|---|---|---|
| 多智能体编排 | supervisor 条件路由 → 4 个专家 agent | `supervisor` / `StateGraph` |
| RAG 检索 | 本地字符 n-gram 向量 + 余弦，top-k 注入 prompt | `_embed` / `retrieve` |
| 记忆 | 会话=LangGraph checkpointer(thread_id)；长期=跨会话 LT store | `checkpointer` / `LT` |
| 流式 + 工具 | `astream_events` → SSE；matcher 绑定 `query_position`，经 `ToolNode` + `tools_condition` 形成 ReAct 回环（思考→调用→观察→再思考），mock 模式也可跑通 | `/chat` / `@tool` / `matcher_node` |
| 结构化输出 | 初筛返回 Pydantic `ScreenItem/ScreenReport` | `/screen` `/rank` |

## 实现边界（诚实标注）

区分「生产复用」与「自驱组装」，以及工具调用的真实落地范围，避免过度声称：

- **生产层复用**：流式协议与 SSE `{"result"}/{"meta"}/[DONE]` 事件格式，复用生产流式 SDK（fetchSse）的协议约定；评测门控纪律与 `harness-skill` 同源。
- **自驱组装（非生产）**：`supervisor` 多智能体路由、`retrieve` n-gram RAG、双层记忆（`checkpointer` + `LT`）均为本仓自驱实现，用于验证编排层取舍，**非线上生产系统**。
- **工具调用（Function Calling）真实落地**：`matcher` 节点绑定 `query_position`，`AIMessage.tool_calls` → `ToolNode` 执行 → `ToolMessage` 回灌 → 二次思考，构成完整 ReAct 回环；`tools_condition` 负责「有调用去 tools / 无调用收敛 memory」分流。**mock 模式**用 `MockToolChatModel` 模拟首轮吐 tool_call，**无需 API key 即可跑通全链路**；配 `DASHSCOPE_API_KEY` 则走真实 qwen 的 function calling。
- **ReAct 的边界**：回环由 `tools_condition` 驱动，工具执行后必然回到 `matcher` 收敛（`MockToolChatModel` 见到 `ToolMessage` 即出终答），**不含无限步自主规划**；真实模式下多步深度取决于 qwen 的 function calling 行为。
- **RAG 的取舍**：本地字符 n-gram + 余弦，以精度换零依赖可离线；接口已抽象成 `retrieve()`，可换真实 embedding / 向量库。
- **人工确认（human-in-the-loop）**：**未实现**。当前为全自动回环，副作用动作不做执行前人工审批；如需可加 LangGraph `interrupt` 断点，本仓未落地，故不声称。

## 设计取舍（为什么这么做）

- **supervisor 而非平铺**：招聘意图差异大（筛/匹配/约面/闲聊），先路由再专家处理，prompt 更聚焦、可扩展新专家。
- **本地 n-gram RAG 而非向量库**：零外部依赖、mock 可跑、对中文友好；接口抽象成 `retrieve()`，可无缝换 embedding/向量库。
- **双层记忆**：会话记忆保证多轮连贯；长期记忆跨会话召回用户偏好/历史，体现"记得你"。
- **SSE 而非 WebSocket**：单向流式足够、实现轻、与前端 `fetchSse` 协议一致。
- **mock/真实双模**：无 key 也能完整演示路由/RAG/记忆/流式，降低评审门槛。

## 已知限制 / Roadmap

- [ ] 真实 embedding 替换 n-gram（接口已抽象）
- [ ] 长期记忆落盘（当前 in-memory）
- [ ] 初筛 eval：标注集 + precision/recall 报表
- [ ] 真实模式接入 LLM 结构化判定（`with_structured_output`）

## 运行环境

Python 3.9+ · langgraph · fastapi · uvicorn · numpy · pydantic（见 `requirements.txt`）
