import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Load and chunk documents
loader = PyPDFLoader("docs/WWF_Legacy_Report_2026.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(pages)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Set up the LLM
llm = ChatBedrock(
    model_id="us.meta.llama3-2-3b-instruct-v1:0",
    region_name="us-east-2"
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
question = "What species does WWF protect?"
answer = chain.invoke(question)

print(f"Question: {question}")
print(f"Answer: {answer.content}")