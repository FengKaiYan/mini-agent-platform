# 核心心智：agent → tools → agent → tools → ... 这个循环就是 ReAct 模式。你之前手写的 fetchSse 里的"发一个 chunk、处理、再发"是同构的，只不过 LangGraph 把控制权交给 LLM 决定下一步
import os
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ---- 1. 定义工具 query_position(之前它没被定义,却在第 4/11 行被引用 → NameError)----
@tool
def query_position(position_id: str) -> str:
    """根据职位 ID 查询该招聘职位的详细要求。

    Args:
        position_id: 职位唯一标识,例如 "JD-001"
    """
    # 学习示例:数据先写死,以后可换成真实数据库/接口
    fake_db = {
        "JD-001": "高级前端工程师:5 年经验,熟悉 React 与 AI SDK",
        "JD-002": "算法工程师:3 年经验,熟悉 LLM 与推荐系统",
    }
    return fake_db.get(position_id, f"未找到职位 {position_id}")


# ---- 2. 配置 LLM(api_key 从环境变量读,严禁硬编码进源码!)----
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    raise SystemExit("缺少 API key:请先执行  export DASHSCOPE_API_KEY=你的key")

llm = ChatOpenAI(
    # 你的网关若确实支持 qwen3.8-max,就 export QWEN_MODEL=qwen3.8-max
    model=os.environ.get("QWEN_MODEL", "qwen-plus"),
    # key 来自哪个服务,就把 base_url 指向哪个服务的 OpenAI 兼容地址
    base_url=os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    api_key=api_key,
    temperature=0,
).bind_tools([query_position])


# ---- 3. State:messages 用 add_messages 做 reducer,新消息「追加」而非覆盖 ----
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# ---- 4. 组图:agent ⇄ tools 的经典 tool-calling 循环 ----
g = StateGraph(State)
g.add_node("agent", agent)
g.add_node("tools", ToolNode([query_position]))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)  # LLM 要调工具→tools,否则→END
g.add_edge("tools", "agent")                        # 工具跑完把结果带回 agent
app = g.compile()


# ---- 5. 运行:问一个必须调用工具才能回答的问题 ----
if __name__ == "__main__":
    result = app.invoke(
        {"messages": [{"role": "user", "content": "帮我查一下职位 JD-001 的招聘要求"}]}
    )
    for m in result["messages"]:
        m.pretty_print()
