from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", temperature=2)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Generate 5 facts about {topic}", input_variables=["topic"]
)

chain = prompt | model | parser

result = chain.invoke({"topic": "cricket"})

print(result)

chain.get_graph().print_ascii()

#      +-------------+       
#      | PromptInput |       
#      +-------------+       
#             *              
#             *                        
#     +----------------+     
#     | PromptTemplate |     
#     +----------------+     
#             *              
#             *                           
#       +----------+         
#       | ChatGroq |         
#       +----------+         
#             *              
#             *              
#             *              
#    +-----------------+     
#    | StrOutputParser |     
#    +-----------------+     
#             *              
#             *                          
# +-----------------------+  
# | StrOutputParserOutput |  
# +-----------------------+  