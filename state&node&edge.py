# Node 返回的 dict 是怎么合并进 State 的？（默认浅合并，可用 Annotated[list, add_messages] 改成追加）
# 如果两个 Node 并行执行都改同一个字段会怎样？（会报冲突，除非声明 reducer）
# Edge 除了固定顺序还能怎么决定下一步？（add_conditional_edges + 一个返回字符串的函数）

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    resume_text: str
    jd_text: str
    score: int
    reason: str

def extract(state: State) -> dict:
    # 假装调 LLM 抽关键信息，先写死
    return {"reason": "5年前端 + AI SDK 经验"}

def score(state: State) -> dict:
    return {"score": 85}

g = StateGraph(State)
g.add_node("extract", extract)
g.add_node("score", score)
g.add_edge(START, "extract")
g.add_edge("extract", "score")
g.add_edge("score", END)

app = g.compile()
result = app.invoke({"resume_text": "...", "jd_text": "..."})
print(result)