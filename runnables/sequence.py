from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a joke about {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Give the brief explanation of this joke - {joke}",
    input_variables=["joke"],
)

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({"topic": "AI"}))
