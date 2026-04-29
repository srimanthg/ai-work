from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, List, Dict, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")


class AgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)


def call_llm(state: AgentState):
    conversation_history = state.messages
    response = llm.invoke(conversation_history)
    return {"messages": response}


graph = StateGraph(AgentState)
graph.add_node("llm_node", call_llm)
graph.add_edge(START, "llm_node")
graph.add_edge("llm_node", END)
agent = graph.compile()

human_msg = HumanMessage(
    content="I like bananas very much. What kind of flavor does it have?"
)
response = agent.invoke({"messages": human_msg})

human_msg = HumanMessage(
    content="Can you suggest a dish that has this as the main ingredient?"
)
response = agent.invoke({"messages": response["messages"] + [human_msg]})

print(response)
