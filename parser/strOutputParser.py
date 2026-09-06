from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()


model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


# 1 st prompt -> report
template1 = PromptTemplate(
    template="write a detailed report on {topic}", input_variables=["topic"]
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="write a 5 line summary on the following text.\n {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)