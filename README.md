# 🏥 AI Medical Chatbot using RAG (MediBot)  

An intelligent chatbot that retrieves answers from medical documents using **Retrieval-Augmented Generation (RAG)**.  
## 🚀 Features  
- **Document Processing**: Loads PDFs, creates text chunks, and generates vector embeddings.  
- **Vector Database**: Uses **FAISS** to store and retrieve embeddings efficiently.  
- **LLM Integration**: Connects with **Mistral LLM (via HuggingFace)** for intelligent responses.  
- **Interactive UI**: Built with **Streamlit** for a smooth chatbot experience.  

## 🛠️ Tech Stack  
- **Python**  
- **Langchain** (AI framework for LLM applications)  
- **HuggingFace** (ML/AI Hub)  
- **FAISS** (Vector database)  
- **Mistral LLM**  
- **Streamlit** (For chatbot UI)  

## 🔧 Future Improvements  
- 🔹 Add **authentication** in the UI  
- 🔹 Enable **multi-document** support  
- 🔹 Implement **self-upload** functionality  
- 🔹 Add **unit testing** for the RAG pipeline  

## 📸 Project Workflow  
### **Phase 1 – Setup Memory for LLM (Vector Database)**  
✅ Load raw PDF(s)  
✅ Create chunks and vector embeddings  
✅ Store embeddings in FAISS  

### **Phase 2 – Connect Memory with LLM**  
✅ Setup Mistral LLM (HuggingFace)  
✅ Connect LLM with FAISS  
✅ Create a retrieval chain  

### **Phase 3 – Setup UI for the Chatbot**  
✅ Develop chatbot UI using **Streamlit**  
✅ Load FAISS vector store in cache  
✅ Implement **RAG-based retrieval**  

## 🎯 Clone this repository:
  
   ```bash
   git clone https://github.com/sagarj1209/medical-chatbot
