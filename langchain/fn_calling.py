import os
import openai
from langchain_community.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.tools.render import format_tool_to_openai_function
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser


def load_openai_key():
    """
    Loads the OpenAI API key from environment variables.
    """
    openai.api_key = os.environ.get("OPENAI_API_KEY")


@tool
def get_weather(latitude: float, longitude: float) -> str:
    """
    A tool to get the weather for a given latitude and longitude.
    """
    return f"The weather at coordinates ({latitude}, {longitude}) is sunny with a chance of rain."


@tool
def get_currecy_by_country(country: str) -> str:
    """
    A tool to get the currency used in a given country.
    """
    currency = {
        "USA": "Dollar",
        "Canada": "Canadian Dollar",
        "United Kingdom": "Pound Sterling",
        "EU": "Euro",
    }
    return f"The currency used in {country} is the {currency.get(country, 'Unknown')}."


def main():
    load_openai_key()

    tools = [get_weather, get_currecy_by_country]
    functions = [format_tool_to_openai_function(tool) for tool in tools]
    tools_map = {tool.name: tool for tool in tools}
    model = ChatOpenAI(temperature=0, model="gpt-4o").bind_functions(functions)
    prompt = ChatPromptTemplate.from_messages(
        [
            {"role": "system", "content": "You are a world class assistant"},
            {"role": "user", "content": "{input}"},
        ]
    )
    chain = prompt | model | OpenAIFunctionsAgentOutputParser()
    response = chain.invoke(
        # {"input": "What is the weather like in San Francisco?"})
        # {"input": "What is the weather like in London?"})
        {"input": "What is the currency used in London?"}
    )

    print(f"Type: {type(response)}")
    print(f"Response: {response}")

    invoke_tool = tools_map[response.tool]
    invoke_response = invoke_tool.invoke(response.tool_input)
    print(f"Invoke Tool Response: {invoke_response}")


if __name__ == "__main__":
    main()
