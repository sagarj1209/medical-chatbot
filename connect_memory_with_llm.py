import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

## Setup llm (Mistral with Huggingface)
HF_TOKEN = os.environ.get('HF_TOKEN')
HF_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(huggingface_repo_id):
  llm = HuggingFaceEndpoint(
    repo_id=huggingface_repo_id,
    temperature=0.5,
    model_kwargs={'token':HF_TOKEN, 'max_length':512}
  )
  return llm

## Connect llm with FAISS and create chain
CUSTOM_PROMPT_TEMPLATE = '''
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {input}

Start the answer directly. No small talk please.
'''

def set_custom_prompt(custom_prompt_template):
  prompt = PromptTemplate(template=custom_prompt_template,
                          input_variables=['context','question'])
  return prompt

## Load Dataset
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

## Create QA chain
retriever = db.as_retriever(search_kwargs={'k': 3})
combine_documents_chain = create_stuff_documents_chain(
  load_llm(HF_REPO_ID),
  prompt=set_custom_prompt(CUSTOM_PROMPT_TEMPLATE))
qa_chain = create_retrieval_chain(retriever, combine_documents_chain)  

# Now invoke with a single query
user_query = input('Write query here: ')
response = qa_chain.invoke({'context': "",'input':user_query})
print("RESULT: ", response["answer"])
print("SOURCE DOCUMENTS: ", response.get("context",'No source doc available'))