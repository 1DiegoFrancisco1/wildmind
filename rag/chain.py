import os
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

BUCKET_NAME = "wildmind-docs"
LOCAL_DOCS_PATH = "docs/"


def download_documents_from_s3():
    print("Downloading documents from S3...")
    s3 = boto3.client("s3", region_name="us-east-2")
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME)

    for obj in objects.get("Contents", []):
        filename = obj["Key"]
        local_path = os.path.join(LOCAL_DOCS_PATH, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            s3.download_file(BUCKET_NAME, filename, local_path)
        else:
            print(f"{filename} already exists locally, skipping...")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("data/faiss_index"):
        print("Loading existing index...")
        vectorstore = FAISS.load_local(
            "data/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Building index for the first time...")
        download_documents_from_s3()

        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        for filename in os.listdir(LOCAL_DOCS_PATH):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(LOCAL_DOCS_PATH, filename))
                pages = loader.load()
                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)

        vectorstore = FAISS.from_documents(all_chunks, embeddings)
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