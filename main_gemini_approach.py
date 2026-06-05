import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

loader = PyPDFLoader("docs/WWF_Legacy_Report_2026.pdf")
pages = loader.load()

#print(pages[0].page_content[:500])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Set up the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# Ask a question!
question = "What species does WWF protect?"
answer = rag_chain.invoke({"query": question})

print(f"Question: {question}")
print(f"Answer: {answer['result']}")

"""
#test = embeddings.embed_query("What animals does WWF protect?")

#print(f"Total pages: {len(pages)}")
#print(f"Total chunks: {len(chunks)}")
#print(f"Embedding size: {len(test)}")
#print(f"First 5 numbers: {test[:5]}")

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("data/faiss_index")

print("Vector store created and saved!")
print(f"Total vectors stored: {vectorstore.index.ntotal}")

question = "What species does WWF protect?"
results = vectorstore.similarity_search(question, k=2)

print(f"\nQuestion: {question}")
print("\n--- Most relevant chunks ---")
for i, doc in enumerate(results):
    print(f"\nChunk {i+1}:")
    print(doc.page_content)
"""