from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
     model="openai/gpt-oss-20b",
    temperature=0
)

messages=[
    SystemMessage(content="you are a helful assistant"),
    HumanMessage(content="tell me about langchain")
]

result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)