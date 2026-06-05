import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag.chain import build_chain

load_dotenv()

@st.cache_resource
def get_chain():
    return build_chain()

# UI
st.title("🐾 WildMind")
st.caption("Animal & Wildlife Q&A Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
question = st.chat_input("Ask a question about wildlife...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Build LangChain message history
    chat_history = []
    for msg in st.session_state.messages[:-1]:  # exclude current question
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = get_chain()
            answer = chain({
                "question": question,
                "chat_history": chat_history
            })
            st.write(answer.content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer.content
    })