from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
load_dotenv()

model = ChatGroq(
     model="openai/gpt-oss-20b",
    temperature=0
)

chat_History = [
    SystemMessage(content="You are a helpful ai assistance that gives answer in brief")
]

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    chat_History.append(HumanMessage(content=user_input))
    result = model.invoke(chat_History)
    chat_History.append(AIMessage(content=result.content))
    print("AI: ", result.content)
print(chat_History)
