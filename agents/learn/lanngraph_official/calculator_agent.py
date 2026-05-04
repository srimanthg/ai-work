"""
A calculator agent that does all calculator operations

Flow:
    - [LLM] Node: Asks LLM the prompt
    - [Decision] Evaluate: Parses LLM output and

Example:
    Input: "Add 3 and 4"
    Output: "7"
"""

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from pathlib import Path
from typing import Literal

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


#
# Tools
#
@tool
def add(a: int, b: int) -> int:
    """
    Adds 2 numbers and returns their sum
    """
    logging.info(f"TOOL: {a} + {b} = {a+b}")
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """
    Subtracts the second number from the first number and returns the result
    """
    logging.info(f"TOOL: {a} - {b} = {a-b}")
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    "Multiplies both numbers and returns their product"
    logging.info(f"TOOL: {a} * {b} = {a*b}")
    return a * b


@tool
def divide(a: int, b: int) -> int:
    """
    Divides the first number with the second number and returns the result
    """
    logging.info(f"TOOL: {a} / {b} = {a/b}")
    return a / b


#
# LLM definition with tools hookup
#
tools = [add, subtract, multiply, divide]
tool_names = {x.name: x for x in tools}
llm = ChatOpenAI(model="gpt-5")
llm = llm.bind_tools(tools=tools)


def ask_solve_problem(state: MessagesState) -> Command:
    user_input = interrupt(
        {
            "interrupt": "ask",
            "question": "Do you want to solve a Math problem (no/[question]?",
        }
    )
    if "no" == user_input.lower():
        return Command(goto=END)
    else:
        return Command(
            goto="node_llm_call", update={"messages": [HumanMessage(user_input)]}
        )


def node_llm_call(state: MessagesState):
    logging.info(f"LLM")
    return {
        "messages": [
            llm.invoke(
                [
                    SystemMessage(
                        content="You are a helpful calculator who will use the provided tools to solve complex math equations."
                    )
                ]
                + state["messages"]
            )
        ]
    }


def node_tool_invoke(state: dict):
    results = []
    last_message = state["messages"][-1]
    logging.info(f"TOOLS: {len(last_message.tool_calls)}")
    for tool_call in last_message.tool_calls:
        tool = tool_names[tool_call["name"]]
        tool_args = tool_call["args"]
        result = tool.invoke(tool_args)
        results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
    return {"messages": results}


def has_tool_calls(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "node_tool_invoke"
    return "ask_solve_problem"


#
# Agent state
#
calculator_graph = StateGraph(state_schema=MessagesState)
calculator_graph.add_node("ask_solve_problem", ask_solve_problem)
calculator_graph.add_node("node_llm_call", node_llm_call)
calculator_graph.add_node("node_tool_invoke", node_tool_invoke)
calculator_graph.add_edge(START, "ask_solve_problem")
calculator_graph.add_conditional_edges(
    "node_llm_call", has_tool_calls, ["node_tool_invoke", "ask_solve_problem"]
)
calculator_graph.add_edge("node_tool_invoke", "node_llm_call")

calculator_agent = calculator_graph.compile(checkpointer=MemorySaver())
Path("calculator-agent.png").write_bytes(
    calculator_agent.get_graph().draw_mermaid_png()
)

config = {"configurable": {"thread_id": "test1"}}
calculator_response = calculator_agent.invoke(input={}, config=config)
while "__interrupt__" in calculator_response:
    if calculator_response["__interrupt__"][0].value["interrupt"] == "ask":
        asked = input(calculator_response["__interrupt__"][0].value["question"])
        calculator_response = calculator_agent.invoke(
            Command(resume=asked), config=config
        )

calculator_response_messages = calculator_response["messages"]
logging.info(f"{len(calculator_response_messages)} messages")
for i, message in enumerate(calculator_response_messages):
    logging.info(f"Message {i}: {message}")
