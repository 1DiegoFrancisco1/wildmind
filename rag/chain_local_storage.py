import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("data/faiss_index"):
        vectorstore = FAISS.load_local(
            "data/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        loader = PyPDFLoader("docs/WWF_Legacy_Report_2026.pdf")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("data/faiss_index")

    return vectorstore


def build_chain():
    vectorstore = load_vectorstore()

    llm = ChatOllama(model="llama3.2")

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain