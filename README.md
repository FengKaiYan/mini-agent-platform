# ai-recruiter · mini Agent 平台

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
    S -->|match| M[matcher 人岗匹配 + 工具调用]
    S -->|interview| I[interviewer 面试安排]
    S -->|chat| C[chatter 通用对话]
    SC --> RAG[RAG 检索 n-gram+余弦]
    M --> RAG
    I --> RAG
    C --> RAG
    SC --> MEM[memory 写长期记忆]
    M --> MEM
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
| 流式 + 工具 | `astream_events` → SSE；matcher 绑定 `query_position` 工具 | `/chat` / `@tool` |
| 结构化输出 | 初筛返回 Pydantic `ScreenItem/ScreenReport` | `/screen` `/rank` |

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
