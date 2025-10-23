from langchain_community.chat_models.openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages(
    messages=[
        {
            "role": "system",
            "content": "You are a translator that translates from English to German",
        },
        {"role": "user", "content": "Hello"},
    ]
)

response = llm.invoke(prompt.invoke({}))
print(response)
