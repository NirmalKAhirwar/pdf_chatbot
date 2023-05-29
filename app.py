import os
import pickle
from PyPDF2 import PdfReader
from dotenv import load_dotenv

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space


from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


headers = {
    "OPENAI_API_KEY" : st.secrets["OPENAI_API_KEY"],
    "content-type": "application/json"
}


with st.sidebar:
    st.title("GPT-3 PDF Chatbot")
    st.markdown("This is a simple GPT-3 chatbot that can answer questions from a given PDF file.")
    st.markdown("1] Upload your PDF.")
    st.markdown("2] Ask Questions about your PDF and hit enter.")
    st.markdown("3] You will get the answers below.")
    
    add_vertical_space(5)
    # st.markdown(f"Made by <https://github.com/NirmalKAhirwar>")
    # st.link("Streamlit website", "https://github.com/NirmalKAhirwar")


def main():
    st.write("Chat with your own PDF ")
    load_dotenv()
    
    #upload a PDF File
    pdf = st.file_uploader("Upload a PDF file", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf.name)
        # st.write(pdf_reader)
        
        text = " "
        for page in pdf_reader.pages:
            text += page.extract_text() 
            
        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        
        chunks = text_splitter.split_text(text)
        # st.write(chunks) 
        
        # Word Embeddings Setup
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings loaded from Disk")
            
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            # st.write("Embeddings Computation completed and saved to disk")
        
        # Accept User questions/queries
        query = st.text_input("Ask a question about the PDF file: ")
        st.write(query)

        if query:
            docs = VectorStore.similarity_search(query, k=3)
            # different model to try "gpt-3.5-turbo"
            llm = OpenAI(openai_api_key = headers["OPENAI_API_KEY"] , model_name = "gpt-3.5-turbo",temperature = 0.2)
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            # st.write(docs)
            st.write(response)
        

if __name__ == "__main__":
    main()
