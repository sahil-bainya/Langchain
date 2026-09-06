from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b", temperature=2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text: \n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 questions and answres from the following text: \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the following notes and quiz and make a single document \n notes -{notes} \n quiz -{quiz}",
    input_variables=["notes", "quiz"],
)

parallel_chain = RunnableParallel(
    {"notes": prompt1 | model | parser, "quiz": prompt2 | model | parser}
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It tries to find the best boundary known as hyperplane that separates different classes in the data. It is useful when you want to do binary classification like spam vs. not spam or cat vs. dog.
The main goal of SVM is to maximize the margin between the two classes. The larger the margin the better the model performs on new and unseen data.
"""
result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()

#           +---------------------------+            
#           | Parallel<notes,quiz>Input |            
#           +---------------------------+            
#                 ***             ***                
#               **                   **              
#             **                       **            
# +----------------+              +----------------+ 
# | PromptTemplate |              | PromptTemplate | 
# +----------------+              +----------------+ 
#           *                             *          
#           *                             *          
#           *                             *          
#     +----------+                  +----------+     
#     | ChatGroq |                  | ChatGroq |
#     +----------+                  +----------+     
#           *                             *          
#           *                             *          
#           *                             *          
# +-----------------+            +-----------------+ 
# | StrOutputParser |            | StrOutputParser | 
# +-----------------+            +-----------------+ 
#                 ***             ***                
#                    **         **                   
#                      **     **                     
#           +----------------------------+           
#           | Parallel<notes,quiz>Output |           
#           +----------------------------+           
#                          *                         
#                          *                         
#                          *                         
#                 +----------------+                 
#                 | PromptTemplate |                 
#                 +----------------+                 
#                          *                         
#                          *                         
#                          *                         
#                    +----------+                    
#                    | ChatGroq |                    
#                    +----------+                    
#                          *                         
#                          *                         
#                          *                         
#                 +-----------------+                
#                 | StrOutputParser |                
#                 +-----------------+                
#                          *                         
#                          *                         
#                          *                         
#             +-----------------------+              
#             | StrOutputParserOutput |              
#             +-----------------------+          