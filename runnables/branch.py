from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
    RunnableBranch,
)

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="summarize the following text in less than 100 words -\n {text}", input_variables=["text"]
)
report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 100, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({"topic": "vector database"})

print(result)
 