from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

load_dotenv()


model = ChatGroq(model="openai/gpt-oss-20b", temperature=2)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name , age and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({})

print(result)


# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))
