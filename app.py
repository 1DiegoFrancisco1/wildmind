import streamlit as st
from dotenv import load_dotenv
from rag.chain import build_chain

load_dotenv()

# @st.cache_resource means this function only runs ONCE
# even if the user asks many questions
# the chain gets built on first question and reused after
@st.cache_resource
def get_chain():
    return build_chain()

# UI header
st.title("🐾 WildMind")
st.caption("Animal & Wildlife Q&A Assistant")

# st.session_state is Streamlit's way of storing data
# that persists between interactions (like a memory)
# without this, every time the user types something
# the whole app reruns and forgets everything
if "messages" not in st.session_state:
    st.session_state.messages = []  # start with empty chat history

# Loop through all previous messages and display them
# so the conversation stays visible as it grows
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # "user" or "assistant" bubble
        st.write(message["content"])

# st.chat_input creates the input box at the bottom of the page
# it returns None if the user hasn't typed anything yet
question = st.chat_input("Ask a question about wildlife...")

if question:
    # Show the user's message in the chat
    with st.chat_message("user"):
        st.write(question)

    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and show the assistant's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chain = get_chain()
            answer = chain.invoke(question)
            st.write(answer.content)

    # Save assistant answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer.content})