import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

#loading all the environment variables
from dotenv import load_dotenv
load_dotenv()

#------------------------------------------------------------------------------------------------
# This line sets the LANGSMITH project environment variable for tracking and debuging the project
os.environ["LANGSMITH_PROJECT_RAG"] = os.getenv("LANGSMITH_PROJECT_RAG")
# Setting the environment variable for the LANGSMITH API key
os.environ["LANGSMITH_API_KEY_RAG"] = os.getenv("LANGSMITH_API_KEY_RAG")
# Setting the environment variable for LangChain tracking
os.environ["LANGSMITH_TRACING_RAG"] = os.getenv("LANGSMITH_TRACING_RAG")
# This is the endpoint for LangSmith tracing
os.environ["LANGSMITH_ENDPOINT_RAG"] = os.getenv("LANGSMITH_ENDPOINT_RAG")
#------------------------------------------------------------------------------------------------


## load the GROQ API Key
os.environ['GROQ_API_KEY_RAG']=os.getenv("GROQ_API_KEY_RAG")
groq_api_key=os.getenv("GROQ_API_KEY_RAG")

##LOAD the hugging face api key
os.environ['HUGGING_FACE_API_RAG']=os.getenv("HUGGING_FACE_API_RAG")
hf_api_key=os.getenv("HUGGING_FACE_API_RAG")

## initialize the model
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

## creating promt
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """)

#importing vectore store and embedding
embeddings=HuggingFaceEmbeddings()
vectorstore = FAISS.load_local("faiss_vector_store", embeddings,allow_dangerous_deserialization=True)


st.title("RAG Document Q&A With Groq And Lama3 on Cricket")

user_prompt=st.text_input("Enter your query related to cricket:")

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=vectorstore.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response=retrieval_chain.invoke({'input':user_prompt})

    st.write(response['answer'])

    ## With a streamlit expander
    with st.expander("Document similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')
