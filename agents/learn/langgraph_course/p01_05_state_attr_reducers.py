"""
This script shows the importance of 'Reducers' which are
functions that update state's attributes
"""

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages, BaseMessage
from typing import Annotated, List, TypedDict
from pydantic import BaseModel, Field
from operator import add


def add_values(a: int, b: int) -> int:
    """
    Reducer which adds step-counts
    """
    return a + b


# State
class TypingCounterState(TypedDict):
    messages: List[str] = []
    step_count: int = 0
    animals: List[str] = []


class PydanticCounterState(BaseModel):
    """
    Contains reducers using `Annotated` which accumulate or add when needed
    """

    messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
    step_count: Annotated[int, add_values] = Field(default=0)
    animals: Annotated[List[str], add] = Field(default_factory=list)


def build_graph_and_run(state_cls):

    def node_a(state: state_cls):
        print("Running node A...")
        return {"messages": ["Node A"], "step_count": 1, "animals": ["cat"]}

    def node_b(state: state_cls):
        print("Running node B...")
        return {"messages": ["Node B"], "step_count": 1, "animals": ["dog"]}

    graph = StateGraph(state_cls)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    agent = graph.compile()
    result = agent.invoke(state_cls())
    print("\n--FINAL--")
    print(result)


build_graph_and_run(TypingCounterState)
build_graph_and_run(PydanticCounterState)
