from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict, Annotated
from langgraph.graph.message import add_messages
from PIL.Image import Image
from operator import add


# 1. State
class SimpleState(TypedDict):
    messages: Annotated[list, add_messages]


graph = StateGraph(SimpleState)


# 2. Nodes
def say_hello(state: SimpleState):
    print("Executing say_hello node...")
    # Return the parts of the state we want to update
    # Here due to `add_messages` function, it will be appended to `messages`
    return {"messages": ["Hello"]}


def say_world(state: SimpleState):
    print("Executing say_world node...")
    # Return the parts of the state we want to update
    # Here due to `add_messages` function, it will be appended to `messages`
    return {"messages": ["World"]}


graph.add_node("hello_node", say_hello)
graph.add_node("world_node", say_world)


# 3. Edges
# START -> hello_node -> world_node -> END
graph.add_edge(START, "hello_node")
graph.add_edge("hello_node", "world_node")
graph.add_edge("world_node", END)


# 4. Compile Graph
agent = graph.compile()
agent.get_graph().draw_mermaid_png(output_file_path="p01_04_graph_api.png")


# 5. Run Graph
initial_state = {"messages": []}
final_state = agent.invoke(initial_state)

print("\n--FINAL STATE--")
print(final_state)
