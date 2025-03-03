import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

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

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(huggingface_repo_id):
  llm = HuggingFaceEndpoint(
    repo_id=huggingface_repo_id,
    temperature=0.5,
    model_kwargs={'token':HF_TOKEN, 'max_length':512}
  )
  return llm

def main():

  st.markdown("""
      <style>
          .title {
              font-size: 40px;
              font-weight: bold;
              color: #007BFF;
              text-align: center;
              margin-bottom: 5px;
          }
          .subtitle {
              font-size: 20px;
              font-weight: lighter;
              color: #007BFF;
              text-align: center;
              margin-bottom: 20px;
          }
          .logo {
              display: block;
              margin-left: auto;
              margin-right: auto;
              width: 100px;
          }
      </style>
  """, unsafe_allow_html=True)

  # Adding a logo (Optional: Replace with your image URL)
  st.markdown('<img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png" class="logo">', unsafe_allow_html=True)

  # Main Heading
  st.markdown('<p class="title">🩺 AI-Powered Medical Chatbot</p>', unsafe_allow_html=True)

  # Subtitle
  st.markdown('<p class="subtitle">🤖 Your AI assistant for quick & reliable medical information</p>', unsafe_allow_html=True)


  if 'messages' not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

  prompt = st.chat_input("Pass your prompt here")

  if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user','content':prompt})
    try:
      vectorstore = get_vectorstore()
      if vectorstore is None:
        st.error("Failed to load the vector store")
        return
      retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
      combine_documents_chain = create_stuff_documents_chain(
          load_llm(HF_REPO_ID),
          prompt=set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
          )
      qa_chain = create_retrieval_chain(retriever, combine_documents_chain) 
      response = qa_chain.invoke({'context': "",'input':prompt})
      result = response["answer"]
      source_documents =  response.get("context",'No source doc available')

      formatted_sources = "\n".join([f"- {doc.metadata.get('source', 'Unknown Source')} (Page {doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))})" for doc in source_documents])
            
      result_to_show = f"{result}\n\n**Source Docs:**\n{formatted_sources if formatted_sources else 'No source document available.'}"
      # result_to_show=result+"\nSource Docs:\n"+str(source_documents)


      st.chat_message('assistant').markdown(result_to_show)
      st.session_state.messages.append({'role':'assistant','content':result_to_show})

    except Exception as e:
       st.error(f"Error: {str(e)}")

if __name__ == '__main__':
  main()