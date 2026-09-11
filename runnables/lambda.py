from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

parser = StrOutputParser()


def word_counter(text):
    return len(text.split())


prompt = PromptTemplate(
    template="Generate a joke about {topic}", input_variables=["topic"]
)


joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {"joke": RunnablePassthrough(), "count": RunnableLambda(word_counter)}
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "study"})

print(result)
