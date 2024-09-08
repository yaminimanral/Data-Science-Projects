import streamlit as st
import os
import requests
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
import tempfile
import time

# from dotenv import load_dotenv
# load_dotenv()
# os.environ['USER_AGENT'] = 'myagent'
st.cache_data.clear()
# Load the Groq API key
os.environ['NVIDIA_API_KEY'] = "enter your api key here"
# os.environ['USER_AGENT'] = "PDFBot/2.0"

# headers = {
#     'User-Agent': 'PDFBot/2.0'
# }

def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()

        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())  # Write the uploaded file's contents to the temp file
                temp_file.flush()
                temp_file.seek(0)
                
                # Load the PDF from the temporary file
                loader = PyPDFLoader(temp_file.name)
                documents.extend(loader.load())
# def vector_embedding():
#     if "vectors" not in st.session_state:

#         st.session_state.embeddings = NVIDIAEmbeddings()

#         st.session_state.loader=PyPDFDirectoryLoader("./us_census") ## Data Ingestion
#         st.session_state.docs=st.session_state.loader.load() ## Document Loading
        st.session_state.docs = documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # vector embeddings


st.title("Nvidia NIM Demo")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
"""
)

# Allow the user to upload PDF files
st.write("upload desired PDF files and then hit 'Documents Embedding'. Then go ahead and ask away your questions!")
uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

if st.button("Documents Embedding"):
    vector_embedding(uploaded_files)
    st.write("Vector Store DB is Ready")

prompt1 = st.text_input("Enter Your Question From Documents")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start:.2f} seconds")
    
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
