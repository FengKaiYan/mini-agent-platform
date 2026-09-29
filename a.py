from langchain.tools import tool
from langchain.chat_models import init_chat_model

from langchain.message import AnyMessage
from typing_extensions import TypedDict,Annotated

model = init_chat_model(
    'qwen3.8-max',
    temperature=0,
)

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`. 
    
    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`. 
    
    Args:
        a: First int
        b: Second int
    """
    return a + b

@tool
def divide(a: int, b: int) -> int:
    """Divide `a` and `b`. 
    
    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [multiply, add, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind(tools=tools)
