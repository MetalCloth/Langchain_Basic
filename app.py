from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

import os
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path="c:/Users/rawat/LangChain/.env")
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')


# --- Streamlit UI ---
st.title("📄 Smart PDF Q&A Bot")
pdf_url = st.text_input("Enter the URL of a PDF:")
user_query = st.text_input("Ask a question based on the PDF content:")
pdf_path = "temp.pdf"  # Temporary file path for the downloaded PDF

import requests

if pdf_url and user_query:
    with st.spinner("Downloading and processing PDF..."):
            try:
                response = requests.get(pdf_url)
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

                # Step 2: Load the PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                spiltter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

                splitted=spiltter.split_documents(documents)

                vectordb=FAISS.from_documents(splitted, OpenAIEmbeddings())


                retriever=vectordb.as_retriever()

                llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                qa=RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


                prompt_template = PromptTemplate(
                    input_variables=["context", "input"],
                    template="""
                    You are a helpful assistant. Answer the question based on the provided context.
                    Context: {context}
                    Question: {input}
                    """
                )

                document_chain=create_stuff_documents_chain(
                    llm=llm, prompt=prompt_template)


                retrieval_chain=create_retrieval_chain(
                    retriever,document_chain
                )


                x=retrieval_chain.invoke({"input":"Generate a list of the most important questions based on the given context. For each question, provide a detailed and accurate answer.",})
                st.subheader("Answer:")
                st.write(x["answer"])


            except Exception as e:
                st.error(f"Error: {e}")


            finally:

                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    print("Temporary PDF deleted.")
                else:
                    print("File not found.")
