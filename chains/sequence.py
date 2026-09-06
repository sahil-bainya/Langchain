from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", temperature=2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate the 5 pointer summary from the following text - \n {text}",
    input_variables=["text"],
)
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "unemployement in india"})

print(result)
