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

# Show suggested questions when chat is empty
if not st.session_state.messages and "suggested_question" not in st.session_state:
    st.markdown("### 💡 Try asking:")

    suggestions = [
        "What animals does WWF protect?",
        "What threats do tigers face?",
        "How does climate change affect wildlife?",
        "Tell me about freshwater fish conservation",
        "What is human-wildlife conflict?",
        "How does WWF protect snow leopards?"
    ]

    # Display suggestions as clickable buttons in two columns
    col1, col2 = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if i % 2 == 0:
            if col1.button(suggestion):
                st.session_state.suggested_question = suggestion
                st.rerun()
        else:
            if col2.button(suggestion):
                st.session_state.suggested_question = suggestion
                st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
question = st.chat_input("Ask a question about wildlife...")

# Handle suggested question clicks
if "suggested_question" in st.session_state:
    question = st.session_state.pop("suggested_question")

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