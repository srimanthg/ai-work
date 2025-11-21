from langchain_community.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.content_blocks import Base64ContentBlock
import httpx
import base64

image_url = (
    "https://images.pexels.com/photos/45170/kittens-cat-cat-puppy-rush-45170.jpeg"
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def send_image_url():
    prompt = ChatPromptTemplate.from_messages(
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing images and answering user questions on them. The response given back should be in valid JSON format.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "how many cats are in the image, and what are their colors?",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
    )
    response = llm.invoke(prompt.invoke({})).content
    return response


def send_image():
    image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
    messages = [
        SystemMessage(
            "You are an expert at analyzing images and answering user questions on them. The response given back should be in valid JSON format."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "how many cats are in the image, and what are their colors?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                    },
                },
            ]
        ),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return llm.invoke(prompt.invoke({}))


# response = send_image_url()
response = send_image()
print(response)
