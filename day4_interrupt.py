# 中断（interrupt）到底是什么？—— 用"发邮件前等人工批准"来演示
#
# 串联前几课的概念：
#   should_interrupt 判断该停 -> 抛 GraphInterrupt 暂停 -> put 存档 -> 人工确认 -> get 读档 -> 继续跑
#
# 场景：AI 给简历打完分后，要"发邮件通知候选人"。
#      但发邮件是有副作用的动作，我们让它【执行前暂停】，等人类点头再继续。
#      这就是 human-in-the-loop（人工介入）。

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # 最简单的存档器：存内存里


class State(TypedDict):
    resume_text: str
    score: int
    email_sent: bool


def score(state: State) -> dict:
    # 假装调 LLM 打分，先写死
    print("  [节点 score] 正在给简历打分...")
    return {"score": 85}


def send_email(state: State) -> dict:
    # 这个动作有副作用（真的发邮件），所以我们想在它执行【前】暂停
    print(f"  [节点 send_email] 邮件已发送！候选人得分 {state['score']}")
    return {"email_sent": True}


g = StateGraph(State)
g.add_node("score", score)
g.add_node("send_email", send_email)
g.add_edge(START, "score")
g.add_edge("score", "send_email")
g.add_edge("send_email", END)

# 关键 1：编译时挂一个"存档器"（checkpointer），暂停时才能把状态存下来
# 关键 2：interrupt_before=["send_email"] -> 在 send_email 执行【前】暂停
app = g.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send_email"],
)

# thread_id 相当于"存档槽位"：同一个 thread_id 才能读回同一份存档
config = {"configurable": {"thread_id": "user-001"}}

print("\n=== 第 1 次运行：会自动跑到 send_email 前暂停 ===")
result = app.invoke({"resume_text": "5年前端经验", "email_sent": False}, config)
print("暂停时的状态:", result)

# 看一下当前存档：下一步本来要跑哪个节点？（should_interrupt 让它停在了 send_email 前）
snapshot = app.get_state(config)
print("下一步将执行的节点:", snapshot.next)  # 预期是 ('send_email',)

# 这里模拟"人工审核"：老板看了打分觉得 OK，批准发邮件
print("\n=== 人工批准，第 2 次运行：从暂停处继续（传 None 表示不喂新数据，接着跑） ===")
result = app.invoke(None, config)  # 传 None = 从存档恢复，继续跑剩下的节点
print("最终状态:", result)
