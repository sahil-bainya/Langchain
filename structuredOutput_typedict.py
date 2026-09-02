from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


# schema
class Review(TypedDict):
    summary: str
    sentiment: str


structured_model = model.with_structured_output(Review,method="function_calling")

result = structured_model.invoke("""
The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.
""")

print(result)
print(result['summary'])
print(result['sentiment'])
print(type(result))
