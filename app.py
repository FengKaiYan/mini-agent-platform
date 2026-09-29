# day5：mini Agent 平台 —— 把前几课串成一个「可链接」的智能体应用
#
# 对齐千问 Agent 平台 JD 的四块能力，全部在本文件内闭环：
#   1) 多智能体编排：supervisor 路由 -> screener / matcher / interviewer / chat 四个专家 agent
#   2) RAG 检索：本地字符 n-gram 向量 + 余弦相似度（无外部依赖，可换 embedding/向量库）
#   3) 记忆：会话记忆 = LangGraph checkpointer(thread_id)；长期记忆 = 跨会话 LT store
#   4) 流式 + 工具调用：astream_events -> SSE（沿用 day3 / 你手写 fetchSse 的 data: {"result":…} 协议）
#
# 运行（无需 API key，内置假模型也能看流式 + 路由 + RAG）：
#   .venv/bin/python day5_platform.py
#   浏览器打开 http://127.0.0.1:8000
#   若配了 DASHSCOPE_API_KEY，则自动改用真实 qwen（并启用工具调用）。
#   阅读约定：每个组件下的「防：…」标注它要防的失败类型。

import asyncio
import json
import os
import re
from typing import Annotated, Iterator, List, Optional

import numpy as np
from typing_extensions import TypedDict

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse

from langchain_core.messages import (
    AIMessage, AIMessageChunk, AnyMessage, BaseMessage, SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ================= 1. 模型：有 key 用真实 qwen，没 key 用逐字假模型 =================
# 防：缺 key、断网时整个 demo 起不来 → 教学链路必须离线可跑，真实模型只在这一处开关。
api_key = os.environ.get("DASHSCOPE_API_KEY")
if api_key:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=os.environ.get("QWEN_MODEL", "qwen-plus"),
        base_url=os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        api_key=api_key,
        temperature=0,
        streaming=True,
    )
    MOCK = False
else:
    from langchain_core.language_models.chat_models import BaseChatModel

    # 防：假模型不会逐字产出 → astream 只拿到整块 chunk，离线环境验证不了流式渲染。
    class CharFakeChatModel(BaseChatModel):
        reply: str = "你好，我是 AI 招聘助手。"

        @property
        def _llm_type(self) -> str:
            return "char-fake"

        def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

        def _stream(self, messages, stop=None, run_manager=None, **kw) -> Iterator[ChatGenerationChunk]:
            for ch in self.reply:
                yield ChatGenerationChunk(message=AIMessageChunk(content=ch))

    llm = None  # mock 下按节点动态构造
    MOCK = True


# 防：mock/真实两种模式在节点里各写一套分支 → 总有一套长期没被验证；构造统一收口于此。
def make_llm(reply: Optional[str] = None, with_tools: bool = False):
    """mock 模式每次按节点合成回复；真实模式复用全局 llm（可选绑工具）。"""
    if MOCK:
        return CharFakeChatModel(reply=reply or "收到。")
    return llm.bind_tools([query_position]) if with_tools else llm


# 防：节点里 invoke 一次拿全量 → on_chat_model_stream 不产出，SSE 前端空白到结束。
async def run_llm(messages: List[BaseMessage], config: RunnableConfig,
                  reply: Optional[str] = None, with_tools: bool = False) -> str:
    """节点内用 astream 收流，保证 on_chat_model_stream 事件一定产出（mock/真实通用）。"""
    model = make_llm(reply, with_tools)
    parts: List[str] = []
    async for chunk in model.astream(messages, config):
        if chunk.content:
            parts.append(chunk.content)
    return "".join(parts) or (reply or "")


# ================= 2. 工具（沿用 day2 的 ReAct 工具调用） =================
# 防：模型凭记忆编造职位要求（幻觉）→ JD 一律以工具查询结果为准，可核对。
@tool
def query_position(position_id: str) -> str:
    """根据职位 ID 查询该招聘职位的详细要求。

    Args:
        position_id: 职位唯一标识，例如 "JD-001"
    """
    for doc in JD_CORPUS:
        if doc["id"] == position_id:
            return doc["text"]
    return f"未找到职位 {position_id}"


# ================= 3. RAG：本地字符 n-gram 向量 + 余弦（可换 embedding/向量库） =================
JD_CORPUS = [
    {"id": "JD-001", "title": "智能体平台高级研发工程师",
     "text": "负责智能体开发范式的设计、开发及应用；熟悉 coze/dify/langflow，了解 RAG、记忆、Agent 机制，"
             "使用过 langchain 等多智能体研发经验优先；了解检索引擎、kv 存储、在线应用框架、批流处理等中间件。"},
    {"id": "JD-002", "title": "高级前端工程师",
     "text": "5 年经验，熟悉 React 与 AI SDK，负责流式渲染、组件体系与性能优化。"},
    {"id": "JD-003", "title": "算法工程师",
     "text": "3 年经验，熟悉 LLM 与推荐系统，负责召回排序与效果优化。"},
]
CRITERIA_CORPUS = [
    {"id": "CR-01", "title": "初筛标准",
     "text": "学历本科及以上；工作年限达标；有可验证的项目硬信号（性能、资损、规模）优先。"},
    {"id": "CR-02", "title": "Agent 方向加分项",
     "text": "有智能体编排、RAG、记忆、流式输出实战经验者优先；纯消费侧经验次之。"},
]


# 防：外部 embedding API/向量库断网、限流即检索瘫痪 → 本地字符 n-gram，零依赖可离线跑。
def _embed(text: str) -> np.ndarray:
    """字符 2-gram 词频向量（对中文友好），L2 归一化后做余弦。"""
    grams = [text[i:i + 2] for i in range(max(0, len(text) - 1))]
    vocab: dict = {}
    for g in grams:
        vocab[g] = vocab.get(g, 0) + 1
    keys = sorted(vocab)
    vec = np.array([vocab[k] for k in keys], dtype=float) if keys else np.zeros(1)
    # 用哈希把变长词表映射到定长桶，保证可比较
    dim = 512
    out = np.zeros(dim)
    for k, v in vocab.items():
        out[hash(k) % dim] += v
    norm = np.linalg.norm(out)
    return out / norm if norm else out


# 防：不检索、让模型凭记忆作答 → 依据不可控、结论不可溯源；命中结果进 prompt 也进 SSE meta。
def retrieve(query: str, corpus: List[dict], top_k: int = 2) -> List[dict]:
    qv = _embed(query)
    scored = []
    for doc in corpus:
        sv = _embed(doc["title"] + doc["text"])
        scored.append((float(qv @ sv), doc))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:top_k]]


# ================= 4. 记忆：会话(checkpointer) + 长期(LT store) =================
checkpointer = MemorySaver()          # 会话记忆：同 thread_id 上下文连续（防：多轮对话丢上下文）
LT: dict = {}                          # 长期记忆：thread_id -> [notes]，跨会话保留（防：会话结束即遗忘）


# ================= 4.5 结构化初筛：合成样例 + 评分 + 结构化输出 =================
# 防：模型自由发挥整段文字 → 下游解析脆弱、字段缺失无从校验；schema 在 Pydantic 里钉死。
class ScreenItem(BaseModel):
    resume_id: str
    candidate: str
    score: int                          # 0-100
    verdict: str                        # 推荐进入下一轮 / 待定·建议补面 / 不推荐
    reasons: List[str]
    evidence: List[str]                 # 命中的 JD / 初筛标准 证据


class ScreenReport(BaseModel):
    position_id: str
    position_title: str
    items: List[ScreenItem]             # 按 score 降序


# 合成样例简历（虚构数据，仅用于 demo；切勿替换为真实候选人简历，避免 PII 泄露）
# 防：demo 混入真实简历 → PII 泄露与合规风险；样例全部合成虚构，可公开演示。
SAMPLE_RESUMES = [
    {"id": "R-01", "name": "示例·林知远(合成)",
     "text": "6 年后端研发。主导智能体编排平台设计，熟悉 langgraph/langchain 多智能体研发；"
             "落地 RAG 检索与跨会话记忆机制；负责流式输出(SSE)链路；"
             "曾优化检索引擎性能，QPS 提升 3 倍，年度资损降为 0；带 4 人小组。"},
    {"id": "R-02", "name": "示例·苏晚(合成)",
     "text": "4 年全栈。熟悉 React 与 AI SDK，负责流式渲染与组件体系；"
             "参与过内部 AI 问答助手的二次开发；了解 RAG 基本概念；"
             "做过 kv 存储与在线应用框架的性能优化，P99 下降 40%。"},
    {"id": "R-03", "name": "示例·陈屿(合成)",
     "text": "3 年算法。熟悉 LLM 与推荐系统，负责召回排序与效果优化；"
             "无智能体编排与 RAG 实战；无流式输出经验；项目以离线指标为主。"},
    {"id": "R-04", "name": "示例·周叙(合成)",
     "text": "5 年前端。熟悉 React 组件体系与性能优化；"
             "未接触 Agent/RAG/记忆机制；无后端与中间件经验；项目为消费侧页面。"},
    {"id": "R-05", "name": "示例·顾行舟(合成)",
     "text": "7 年基础平台。熟悉检索引擎、kv 存储、批流处理等中间件；"
             "近一年转向智能体方向，用 langchain 搭过内部问答助手(含 RAG)；"
             "保障高并发系统稳定性，支撑日均亿级请求；但多智能体编排经验较浅。"},
]

# 防：只靠语义余弦会把「用词相近」当成「能力匹配」→ 信号词常数提供可解释的加减分。
AGENT_KWS = ["智能体", "agent", "langgraph", "langchain", "多智能体", "编排",
             "rag", "记忆", "流式", "sse", "coze", "dify"]
HARD_KWS = ["性能", "资损", "规模", "并发", "qps", "p99", "亿级", "稳定性", "一致性"]


# 防：「未接触 RAG / 无流式经验」被读成命中信号 → 否定句整句作废（见 _clause_negated）。
NEG = ("未", "无", "没", "缺", "不", "非", "否")


def _clause_negated(clause: str) -> bool:
    """分句句首附近出现否定词 → 整句视为'缺少该能力'的表述。"""
    head = clause.strip()
    for n in NEG:
        if 0 <= head.find(n) <= 3:
            return True
    return False


def _kw_hits(text: str, kws: List[str]) -> List[str]:
    """分句级关键词命中；否定分句(未接触/无/缺少…)内的关键词不计为正向信号。"""
    hits: List[str] = []
    for clause in re.split(r"[；;。！!？?\n]", text):
        if _clause_negated(clause):
            continue
        low = clause.lower()
        for k in kws:
            if k in low and k not in hits:
                hits.append(k)
    return hits


# 防：让 LLM 直接打分 → 每次结果漂移、无法复现对账；确定性公式保证同输入同输出。
def score_resume(resume: dict, position: dict) -> ScreenItem:
    """启发式评分（mock/真实通用）：语义余弦 + Agent 信号 + 硬信号 -> 结构化结论。

    真实模式下可在此替换/叠加 LLM 结构化判定（with_structured_output），
    接口与返回结构保持不变。
    """
    qv = _embed(resume["text"])
    pv = _embed(position["title"] + position["text"])
    cos = float(qv @ pv)
    crit = retrieve(resume["text"], CRITERIA_CORPUS, top_k=1)
    agent_hits = _kw_hits(resume["text"], AGENT_KWS)
    hard_hits = _kw_hits(resume["text"], HARD_KWS)
    score = 40 + int(round(cos * 40)) + len(agent_hits) * 3 + len(hard_hits) * 3
    score = max(0, min(100, score))
    if score >= 75:
        verdict = "推荐进入下一轮"
    elif score >= 60:
        verdict = "待定·建议补面"
    else:
        verdict = "不推荐"
    reasons = []
    if agent_hits:
        reasons.append("命中 Agent 方向信号：" + "、".join(agent_hits[:5]))
    else:
        reasons.append("缺少 Agent/RAG/编排 实战信号")
    if hard_hits:
        reasons.append("有可验证硬信号：" + "、".join(hard_hits[:4]))
    else:
        reasons.append("缺少性能/资损/规模等硬信号")
    reasons.append(f"与岗位语义相似度 cos={cos:.2f}")
    evidence = [f"{position['id']} {position['title']}"] + \
               [f"{d['id']} {d['title']}" for d in crit]
    return ScreenItem(resume_id=resume["id"], candidate=resume["name"],
                      score=score, verdict=verdict, reasons=reasons,
                      evidence=evidence)


# 防：各调用方各自排序、口径不一 → 报告在服务端按 score 降序定稿。
def rank_resumes(position_id: str) -> ScreenReport:
    position = next((d for d in JD_CORPUS if d["id"] == position_id), JD_CORPUS[0])
    items = sorted((score_resume(r, position) for r in SAMPLE_RESUMES),
                   key=lambda x: -x.score)
    return ScreenReport(position_id=position["id"],
                        position_title=position["title"], items=items)


# ================= 5. 图：supervisor 路由 + 四个专家 agent =================
# 防：后写节点把 messages 覆盖成只剩本轮 → add_messages reducer 保证追加进历史。
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    lt: list
    ctx: list


# 防：supervisor 吐出图里没有的分支名 → 取值收敛到 ROUTES 并兜底 chat，条件边不会 KeyError。
ROUTES = ["screen", "match", "interview", "chat"]
ROLE_DESC = {
    "screen": "简历初筛专家",
    "match": "人岗匹配专家",
    "interview": "面试安排专家",
    "chat": "通用对话助手",
}
ROUTE_KEYWORDS = {
    "screen": ["筛", "初筛", "简历", "过不了", "资格"],
    "match": ["匹配", "适合", "岗位", "职位", "jd"],
    "interview": ["面试", "约", "安排", "时间"],
}


# 防：抓错输入（把 AI/系统消息当用户问题）→ 只认最后一条 type=="human"。
def _last_user(state: State) -> str:
    for m in reversed(state["messages"]):
        if getattr(m, "type", "") == "human":
            return m.content
    return ""


# 防：长期记忆只写不读 → 每次会话开头把历史 notes 注入 state.lt。
async def recall(state: State, config: RunnableConfig) -> dict:
    """会话开始注入长期记忆。"""
    tid = config["configurable"]["thread_id"]
    return {"lt": LT.get(tid, [])}


# 防：所有问题塞给同一个万能 prompt → 职责混杂、答非所问；先定路由再进对应专家。
async def supervisor(state: State, config: RunnableConfig) -> dict:
    q = _last_user(state).lower()
    if MOCK:
        route = "chat"
        for r, kws in ROUTE_KEYWORDS.items():
            if any(k in q for k in kws):
                route = r
                break
    else:
        prompt = [SystemMessage(
            "你是路由 supervisor。只回复一个词，从 screen/match/interview/chat 中选最合适的：\n"
            "screen=简历初筛；match=人岗/职位匹配；interview=面试安排；chat=其它。"
        ), AIMessage(content="可选: screen match interview chat"),
            SystemMessage(content="用户问题：" + q)]
        raw = (await run_llm(prompt, config)).strip().lower()
        route = next((r for r in ROUTES if r in raw), "chat")
    return {"route": route}


# 防：专家不查资料凭空答、也看不到历史 → 节点内先 RAG 检索，再把长期记忆拼进系统提示。
def _specialist(role: str, corpus: List[dict]):
    async def node(state: State, config: RunnableConfig) -> dict:
        q = _last_user(state)
        docs = retrieve(q, corpus)                       # RAG 检索
        ctx_txt = "\n".join(f"- [{d['id']}] {d['title']}: {d['text']}" for d in docs)
        lt_txt = "\n".join(state.get("lt") or []) or "（无）"
        sys = SystemMessage(
            f"你是{ROLE_DESC[role]}。优先依据下面 RAG 检索结果与长期记忆回答，简洁、给结论。\n"
            f"【RAG 检索】\n{ctx_txt}\n【长期记忆】\n{lt_txt}"
        )
        mock_reply = (f"【{ROLE_DESC[role]}】结合检索到的《{docs[0]['title']}》：{q[:20]}… "
                      f"结论：符合度较高，建议进入下一轮。")
        text = await run_llm([sys] + list(state["messages"]), config,
                             reply=mock_reply, with_tools=(role == "match"))
        return {"messages": [AIMessage(content=text)],
                "ctx": [d["id"] + " " + d["title"] for d in docs]}
    return node


# 防：会话结束即遗忘、记忆无限膨胀 → 结论写回 LT 且只留最近 5 条。
async def memory_write(state: State, config: RunnableConfig) -> dict:
    """会话结束写长期记忆（跨会话可召回）。"""
    tid = config["configurable"]["thread_id"]
    note = f"[{state.get('route') or 'chat'}] {_last_user(state)[:40]}"
    LT.setdefault(tid, []).append(note)
    LT[tid] = LT[tid][-5:]
    return {}


g = StateGraph(State)
g.add_node("recall", recall)
g.add_node("supervisor", supervisor)
g.add_node("screener", _specialist("screen", CRITERIA_CORPUS))
g.add_node("matcher", _specialist("match", JD_CORPUS))
g.add_node("interviewer", _specialist("interview", CRITERIA_CORPUS))
g.add_node("chatter", _specialist("chat", JD_CORPUS))
g.add_node("memory", memory_write)

g.add_edge(START, "recall")
g.add_edge("recall", "supervisor")
g.add_conditional_edges("supervisor", lambda s: s["route"], {
    "screen": "screener", "match": "matcher",
    "interview": "interviewer", "chat": "chatter",
})
for n in ("screener", "matcher", "interviewer", "chatter"):
    g.add_edge(n, "memory")
g.add_edge("memory", END)
# 防：编译时不挂 checkpointer → thread_id 状态无处存放，多轮对话与中断恢复全部失效。
graph = g.compile(checkpointer=checkpointer)


# ================= 6. FastAPI：/chat SSE 流式 + / 前端页 =================
api = FastAPI()


# 防：手测或前端漏传参数直接 422 → 给默认值，一条 curl 就能看全链路。
class ChatReq(BaseModel):
    message: str = "帮我看看这份简历和智能体平台岗位匹配吗"
    thread_id: str = "demo"


@api.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))


# 防：前端干等到整段生成完才有输出（首字延迟差）→ 逐 chunk 转 SSE，路由/RAG 依据用 meta 透出。
@api.post("/chat")
async def chat(req: ChatReq):
    async def gen():
        cfg = {"configurable": {"thread_id": req.thread_id}}
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": req.message}]},
            config=cfg, version="v2",
        ):
            name = event.get("name", "")
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data: {json.dumps({'result': chunk}, ensure_ascii=False)}\n\n"
                    if MOCK:
                        await asyncio.sleep(0.03)
            elif event["event"] == "on_chain_end" and name == "supervisor":
                yield f"data: {json.dumps({'meta': '路由 → ' + (event['data']['output'].get('route') or 'chat')}, ensure_ascii=False)}\n\n"
            elif event["event"] == "on_chain_end" and name in ("screener", "matcher", "interviewer", "chatter"):
                ctx = event["data"]["output"].get("ctx") or []
                if ctx:
                    yield f"data: {json.dumps({'meta': 'RAG 命中 → ' + '、'.join(ctx)}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'meta': '记忆已更新（thread=' + req.thread_id + '）'}, ensure_ascii=False)}\n\n"
        # 防：前端不知道流何时结束（一直转圈）→ [DONE] 哨兵收尾，与手写 fetchSse 协议对齐。
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


class ScreenReq(BaseModel):
    resume_text: str
    position_id: str = "JD-001"


# 防：直传简历与批量排序走两套口径 → 复用 score_resume，结论同源可比。
@api.post("/screen", response_model=ScreenItem)
async def screen(req: ScreenReq):
    """单份简历 × 岗位 -> 结构化初筛结论（分数/结论/理由/证据）。"""
    position = next((d for d in JD_CORPUS if d["id"] == req.position_id), JD_CORPUS[0])
    return score_resume({"id": "input", "name": "输入简历",
                         "text": req.resume_text}, position)


# 防：demo 必须准备真实数据才能看效果 → 内置合成样例零输入出排序报告。
@api.get("/rank", response_model=ScreenReport)
async def rank(position_id: str = "JD-001"):
    """内置合成样例简历 × 岗位 -> 排序后的结构化初筛报告（demo 用）。"""
    return rank_resumes(position_id)


if __name__ == "__main__":
    import uvicorn
    print("打开 http://127.0.0.1:8000  （MOCK=%s）" % MOCK)
    uvicorn.run(api, host="127.0.0.1", port=8000)
