import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load index if it exists, otherwise build it
if os.path.exists("data/faiss_index"):
    print("Loading existing index...")
    vectorstore = FAISS.load_local(
        "data/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Building index for the first time...")
    loader = PyPDFLoader("docs/WWF_Legacy_Report_2026.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(pages)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("data/faiss_index")

# Set up the LLM
llm = ChatOllama(
    model="llama3.2"
)

# Create the prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

# Create the RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Ask a question!
question = "What progress did WWF report in 2026?"
answer = chain.invoke(question)

print(f"Question: {question}")
print(f"Answer: {answer.content}")