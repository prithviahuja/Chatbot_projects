import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit
st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload PDFs and chat with their content")

# Input the Groq API key
api_key = st.text_input("Enter your Groq API key:", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model="Llama3-8b-8192")
    session_id = st.text_input("Session ID", value="session1")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader("Choose a PDF to upload", type="pdf", accept_multiple_files=False)
    
    if uploaded_file:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_pdf)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = "Provide context-aware retrieval based on the conversation history."
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. Use the retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. Keep responses concise.\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.write(st.session_state.store)
            st.success("Assistant:", response["answer"])
            st.write("Chat History:", session_history.messages)