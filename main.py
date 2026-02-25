from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from chain import revisor, first_responder
from tool_executor import tool_executor

MAX_ITERATIONS = 2

def draft_node(state:MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state:MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state:MessagesState) -> Literal["tool_executor", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iter = count_tool_visits
    if num_iter > MAX_ITERATIONS:
        return END
    return "tool_executor"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("tool_executor", tool_executor)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "tool_executor")
builder.add_edge("tool_executor", "revise")
builder.add_conditional_edges("revise", event_loop, ["tool_executor", END])
graph = builder.compile()
print(graph.get_graph().draw_mermaid())

result = graph.invoke({"messages": [{
    "role": "user",
    "content": "Write about AI-Powered SOC / autonomous SOC problem domain, list startup that do that and raised capital"
}]})
last_msg = result["messages"][-1]
if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
    print(last_msg.tool_calls[0]["args"]["answer"])
print (result)