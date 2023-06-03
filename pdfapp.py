#To load api key create a new file with the name .env and store api key in it(ex: OPENAI_API_KEY=xxxxxxxxxxxxxxxxxx)
#this code supports only PDF files
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
def main():
    load_dotenv()
    uploaded_file = st.file_uploader("Choose a pdf file",type="pdf")
    if uploaded_file is not None:
        pdf=PdfReader(uploaded_file)
        text=""
        for page in pdf.pages:
            text+=page.extract_text()
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text=text)
        embeddings=OpenAIEmbeddings()
        vectorstore=FAISS.from_texts(chunks,embedding=embeddings)
        query=st.text_input("question")
        if query:
            docs=vectorstore.similarity_search(query=query,k=1)
            llm=OpenAI(temperature=0)
            chain=load_qa_chain(llm=llm,chain_type="stuff")
            responce=chain.run(input_documents=docs,question=query)
            st.write(responce)
main()

