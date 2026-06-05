# 🐾 WildMind — Animal & Wildlife Q&A Assistant

A RAG-powered chatbot that answers questions about animal behavior, biology, and wildlife conservation — built as a hands-on learning project to understand LangChain, RAG, embeddings, and local LLMs.

---

## 🚀 Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | Orchestration framework — connects every piece together |
| HuggingFace Embeddings | Converts text into vectors locally, for free |
| FAISS | Vector store — stores and searches embeddings |
| Ollama + Llama 3.2 | Local LLM — generates answers, no API key needed |
| Streamlit | Web UI — simple chat interface |
| AWS S3 | Cloud storage for source documents |
| Python + dotenv | Core language and secret management |

---

## 📁 Project Structure

```
wildmind/
├── app.py              → Streamlit UI
├── rag/
│   ├── __init__.py
│   └── chain.py        → RAG logic (loading, chunking, retrieval, chain)
├── docs/               → PDF source documents
├── data/               → Saved FAISS index
├── .env                → API keys and secrets (never commit this!)
└── requirements.txt    → Python dependencies
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/wildmind.git
cd wildmind

# 2. Create and activate virtual environment
py -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your documents to docs/

# 5. Make sure Ollama is running with Llama 3.2
ollama pull llama3.2

# 6. Run the app
streamlit run app.py
```

---

## 🧠 Concepts Learned

### What is RAG? (Retrieval Augmented Generation)

RAG is a technique that makes LLMs smarter by giving them access to your own documents before answering a question.

Without RAG → the LLM answers only from its training data (may be outdated or vague).

With RAG → the system first searches your documents for relevant information, then sends that information to the LLM along with the question. The LLM answers based on YOUR documents.

```
User question
     ↓
Convert question to vector
     ↓
Search vector store for closest chunks
     ↓
Send chunks + question to LLM
     ↓
LLM generates grounded answer
```

---

### What are Embeddings?

Embeddings are numerical representations of text — they convert words and sentences into lists of numbers (vectors) that capture their **meaning**.

The model we use (`all-MiniLM-L6-v2`) always produces exactly **384 numbers** for any piece of text, no matter how long or short.

```
"Snow leopards live in the Himalayas" → [0.022, 0.057, 0.005, ... 384 numbers]
"Wolf packs hunt together"            → [0.019, 0.061, 0.008, ... 384 numbers]
"Photosynthesis in plants"            → [0.901, 0.012, 0.743, ... 384 numbers]
```

The key insight: **similar meanings produce similar vectors**. This is what makes semantic search possible — you can find relevant text even when the exact words don't match.

---

### Vector Similarity — How Search Works

When two pieces of text have similar meanings, their vectors point in similar directions in a high-dimensional space. The similarity between vectors is measured using **cosine similarity** — the angle between two vectors.

- Small angle → similar direction → similar meaning → high similarity
- Large angle → different direction → different meaning → low similarity

This is exactly what FAISS uses to find the most relevant chunks when a question is asked.

Think of it like GPS coordinates — but instead of 2 numbers to locate a point on a map, you need 384 numbers to locate a piece of meaning in "meaning space". 🗺️

---

### What is FAISS?

FAISS (Facebook AI Similarity Search) is a library built by Meta for efficiently searching through millions of vectors.

The naive approach — comparing a question vector against every stored vector one by one — works for small datasets but becomes impossibly slow at scale.

FAISS organizes vectors in a smart structure so it can find the closest ones almost instantly, even with millions of entries.

In our project:
- We build the FAISS index once and save it to disk
- Every time the app runs, we load the saved index instead of rebuilding
- This makes the app fast — no reprocessing needed

---

### What is Chunking?

AI models have a **context limit** — there's a maximum amount of text they can process at once. Sending an entire document would be too slow and expensive.

Chunking solves this by splitting documents into small overlapping pieces:

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,    # each chunk is max 500 characters
    chunk_overlap=50   # chunks share 50 characters with neighbors
)
```

The overlap is important — it prevents losing context at the boundaries between chunks. A sentence cut in half between two chunks would lose its meaning without overlap.

---

### What is a Retriever?

A retriever is the component that searches the vector store and brings back the most relevant chunks.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

The `k` parameter controls how many chunks to retrieve. This is one of the most important tuning parameters in RAG:

- **k too small** → LLM doesn't have enough context → incomplete answers
- **k too large** → LLM gets too much noise → confused answers
- **Sweet spot** → depends on your documents and use case

---

### What is LangChain?

LangChain is an orchestration framework — it connects every piece of the pipeline together.

Instead of manually wiring together document loaders, text splitters, embeddings, vector stores, and LLMs, LangChain provides a clean interface using **LCEL** (LangChain Expression Language):

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

The `|` operator passes the output of each step as input to the next — like a pipeline. This makes the RAG flow readable and easy to modify.

---

### What is an LLM?

A Large Language Model (LLM) is a neural network trained on massive amounts of text. It has learned patterns, facts, reasoning, and language from billions of examples.

In our project we use **Llama 3.2** (3 billion parameters) running locally via Ollama — no internet, no API keys, no cost.

**Parameters** are the numbers inside the model that encode its knowledge. More parameters = more knowledge = more memory needed:

| Model size | Parameters | Use case |
|------------|-----------|---------|
| Small | 1B - 3B | Simple Q&A, local use |
| Medium | 7B - 13B | More complex reasoning |
| Large | 70B+ | Near GPT-4 quality |

---

### The Attention Mechanism

Modern LLMs are built on the **Transformer** architecture, whose key innovation is the **attention mechanism**.

When generating each word of an answer, the model doesn't treat all input words equally — it **pays attention** to the most relevant parts.

For example, when answering "What do snow leopards eat?", the model pays high attention to words like "snow leopards", "eat", and "prey" — and low attention to unrelated words.

This is why Transformers are so powerful — they can capture long-range relationships between words, understanding that "it" in "the leopard chased the deer and it escaped" refers to the deer, not the leopard.

The "T" in **ChatGPT** literally stands for Transformer. 🧠

---

### RAG Tuning — Key Parameters

One of the most important skills in RAG engineering is tuning these parameters:

| Parameter | Effect |
|-----------|--------|
| `chunk_size` | Larger = more context per chunk, less precision |
| `chunk_overlap` | Larger = less information lost at boundaries |
| `k` (retrieval) | Larger = more context for LLM, more noise risk |
| Embedding model | Better model = more accurate similarity search |

Finding the right balance is an iterative process — experiment and evaluate.

---

## 📚 Resources

- [LangChain Documentation](https://docs.langchain.com)
- [FAISS Documentation](https://faiss.ai)
- [Ollama](https://ollama.com)
- [HuggingFace Sentence Transformers](https://www.sbert.net)
- [Attention Is All You Need (original Transformer paper)](https://arxiv.org/abs/1706.03762)

---

## 👨‍💻 Author: Diego Francisco Dominguez Aguilar

Built as a learning project to gain hands-on experience with LangChain, RAG, embeddings, vector stores, and local LLMs.
