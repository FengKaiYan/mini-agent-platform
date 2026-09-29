# day3：流式输出（streaming）—— 把 LLM「逐字生成」的过程实时推给浏览器
#
# 串联前几课 + 你手写的 fetchSse：
#   LangGraph astream_events 逐 token 产出
#     -> FastAPI StreamingResponse 包成 SSE（每条形如  data: {"result":"…"}\n\n）
#     -> 前端 fetch + ReadableStream 边收边拼（和我在生产里封装的 fetchSse 传输层同构）
#   ReAct 循环里「发一个 chunk、处理、再发」和这里是同构的，只是这里把 chunk 直接推给了浏览器。
#
# 运行（无需 API key，内置假模型也能看到流式）：
#   .venv/bin/python day3_stream.py
#   浏览器打开 http://127.0.0.1:8000
#   若配了 DASHSCOPE_API_KEY，则自动改用真实 qwen 做流式输出。

import asyncio
import json
import os
from typing import Annotated, Iterator, List, Optional

from typing_extensions import TypedDict

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse

from langchain_core.messages import AnyMessage, AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# ---- 1. 选模型：有 key 用真实 qwen（开 streaming），没 key 用逐字假模型 ----
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
        streaming=True,  # 关键：开启流式，astream_events 才会逐 token 产出
    )
    MOCK = False
else:
    from langchain_core.language_models.chat_models import BaseChatModel

    class CharFakeChatModel(BaseChatModel):
        """没配 API key 时的替身：把一句固定回复「逐字」吐出来，先让你看到流式效果。"""

        reply: str = "你好，我是 AI 招聘助手。我可以帮你筛选简历、匹配职位、给候选人打分。"

        @property
        def _llm_type(self) -> str:
            return "char-fake"

        def _generate(
            self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
            run_manager=None, **kwargs,
        ) -> ChatResult:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

        def _stream(
            self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
            run_manager=None, **kwargs,
        ) -> Iterator[ChatGenerationChunk]:
            for ch in self.reply:  # 一个字一个 chunk -> 前端就看到「打字机」效果
                yield ChatGenerationChunk(message=AIMessageChunk(content=ch))

    llm = CharFakeChatModel()
    MOCK = True


# ---- 2. State & 图：最小 agent 节点 ----
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


async def agent(state: State, config: RunnableConfig):
    # 关键：节点必须是 async，且把 config 透传给模型，astream_events 才能捕获到 token 流。
    #（Python 3.9 下若不透传 config，会收不到 on_chat_model_stream 事件 —— 踩过的坑）
    return {"messages": [await llm.ainvoke(state["messages"], config)]}


g = StateGraph(State)
g.add_node("agent", agent)
g.add_edge(START, "agent")
g.add_edge("agent", END)
graph = g.compile()  # 注意：这里叫 graph，别和下面的 FastAPI 实例 api 撞名（原文件的 bug）


# ---- 3. FastAPI：/chat 走 SSE 流式，/ 返回前端页面 ----
api = FastAPI()


class ChatReq(BaseModel):
    message: str = "介绍一下你能做什么"


@api.get("/")
async def index():
    # 和 /chat 同源，省掉 CORS；直接浏览器打开 http://127.0.0.1:8000 即可
    return FileResponse(os.path.join(os.path.dirname(__file__), "day3_stream.html"))


@api.post("/chat")
async def chat(req: ChatReq):
    async def gen():
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": req.message}]},
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    # 对齐你 fetchSse 里的 parseSseResultsWithCache：它只认 data: {"result": …}
                    yield f"data: {json.dumps({'result': chunk}, ensure_ascii=False)}\n\n"
                    if MOCK:
                        await asyncio.sleep(0.04)  # 假模型吐太快，稍微限速才看得到「逐字」

    # text/event-stream = SSE；不要设 Content-Length，让连接保持流式
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    print("打开 http://127.0.0.1:8000  （MOCK=%s）" % MOCK)
    uvicorn.run(api, host="127.0.0.1", port=8000)
