import os
from openai import OpenAI


def main():
    """
    This script demonstrates how to use the OpenAI API to create a chat completion with a specific system message.
    The system message sets the context for the AI assistant, while user messages represent user input.
    The assistant's responses are generated based on these messages.

    The roles in the chat completion are defined as follows:
    - System: Provides high-level instructions or context for the AI assistant.
    - User: Represents input from the human user.
    - Assistant: Contains previous responses from the AI model.
    - Function: Used for function calls and their results.
    The script initializes the OpenAI client, sends a chat completion request, and prints the assistant's response.
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a world class poetic weather reporter",
            },
            {"role": "user", "content": "What is the weather like in San Francisco?"},
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
