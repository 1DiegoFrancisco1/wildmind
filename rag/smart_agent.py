import os
import boto3
import requests
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

BUCKET_NAME = "wildmind-docs"
LOCAL_DOCS_PATH = "docs/"
FAISS_INDEX_PATH = "data/faiss_index"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def get_vectorstore():
    embeddings = get_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

@tool
def search_knowledge_base(query: str) -> str:
    """Search the current knowledge base to check if we already have
    information about a topic. Returns relevant chunks if found."""
    vectorstore = get_vectorstore()
    if not vectorstore:
        return "Knowledge base is empty."

    docs = vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found in knowledge base."

    results = "\n\n".join(doc.page_content for doc in docs)
    return f"Found relevant information:\n{results}"


@tool
def evaluate_document(title: str, description: str) -> str:
    """Evaluate if a document is relevant to wildlife conservation,
    animal biology, or environmental topics. Returns 'relevant' or
    'not relevant' with a reason."""
    wildlife_keywords = [
        "animal", "species", "wildlife", "conservation", "habitat",
        "endangered", "biodiversity", "ecosystem", "marine", "forest",
        "fish", "bird", "mammal", "reptile", "amphibian", "insect",
        "tiger", "leopard", "elephant", "shark", "whale", "gorilla"
    ]

    text = (title + " " + description).lower()
    matches = [kw for kw in wildlife_keywords if kw in text]

    if matches:
        return f"relevant — matches wildlife keywords: {', '.join(matches)}"
    return "not relevant — no wildlife keywords found"


@tool
def download_and_index_document(url: str, filename: str) -> str:
    """Downloads a PDF from a URL, uploads it to S3, and adds it
    to the FAISS knowledge base. Use this for relevant documents only."""
    local_path = os.path.join(LOCAL_DOCS_PATH, filename)

    # Download
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return f"Failed to download {filename} — status {response.status_code}"
        with open(local_path, "wb") as f:
            f.write(response.content)

    # Upload to S3
    s3 = boto3.client("s3", region_name="us-east-2")
    s3.upload_file(local_path, BUCKET_NAME, filename)

    # Add to FAISS index
    embeddings = get_embeddings()
    loader = PyPDFLoader(local_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    if os.path.exists(FAISS_INDEX_PATH):
        # load it
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        # if not, create it
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    return f"Successfully added {filename} to knowledge base — {len(chunks)} chunks indexed!"


@tool
def check_document_exists(filename: str) -> str:
    """Check if a document already exists in the local docs folder
    or in the S3 bucket. Returns 'exists' or 'not found'."""
    local_path = os.path.join(LOCAL_DOCS_PATH, filename)

    # Check locally first
    if os.path.exists(local_path):
        return f"exists — {filename} is already downloaded locally"

    # Check in S3
    try:
        s3 = boto3.client("s3", region_name="us-east-2")
        s3.head_object(Bucket=BUCKET_NAME, Key=filename)
        return f"exists — {filename} is already in S3"
    except:
        return f"not found — {filename} is not in local docs or S3"


def run_smart_agent(task: str):
    """Run the smart agent with a given task."""
    llm = ChatOllama(model="llama3.2")
    tools = [search_knowledge_base, evaluate_document, check_document_exists, download_and_index_document]

    agent = create_react_agent(llm, tools)

    print(f"\n🤖 Smart Agent starting...")
    print(f"Task: {task}\n")

    result = agent.invoke({
        "messages": [{"role": "user", "content": task}]
    })

    final_answer = result["messages"][-1].content
    print(f"\n✅ Agent finished!\n{final_answer}")
    return final_answer


if __name__ == "__main__":
    run_smart_agent("""
        You are a knowledge base manager for a wildlife conservation assistant.
        
        A new document is available:
        - Title: "Law of the Tiger - Tiger Legislation Report"
        - URL: https://www.worldwildlife.org/documents/2277/v8_legislation-report_2025_hr-pages.pdf
        - Filename: WWF_Law_of_the_Tiger.pdf

        Follow these steps strictly:
        1. Use check_document_exists to verify the document isn't already downloaded
        2. Use search_knowledge_base to check if we already have enough info on tigers
        3. Use evaluate_document to check if this document is relevant to wildlife
        4. Only use download_and_index_document if ALL are true:
        - Document does not exist yet
        - We don't have sufficient information on this topic
        - The document is relevant to wildlife
        5. Report what you decided and why
    """)