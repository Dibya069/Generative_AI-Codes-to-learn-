import streamlit as st
from langchain_groq import ChatGroq
import os
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Embeddings and Vector Store
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

url = "http://localhost:6333"
client = QdrantClient(url=url, prefer_grpc=False)
db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# LLM
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")

# Prompt Template
prompt_template = """
You are an intelligent assistant tasked with answering user queries based on provided context. 
Use the following context to respond to the user's question.

Context:
{context}

Question:
{query}

Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Define Chain
chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit Interface
st.title("Interactive Chatbot with Qdrant and Groq")
st.write("Ask any question, and the chatbot will respond using context from the vector database!")

# Input Box for User Query
user_query = st.text_input("Enter your question here:", value="What qualities did Phileas Fogg display during his journey?")

if st.button("Get Response"):
    with st.spinner("Generating response..."):
        try:
            # Get Response from Chain
            response = chain.invoke(user_query)
            st.success("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
