import streamlit as st
from rag_engine import build_or_load_index, get_query_engine

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Intelligence")
st.caption("Ask anything about your uploaded documents")

# This loads the index only once — not every time you ask a question
@st.cache_resource
def load_engine():
    index = build_or_load_index()
    return get_query_engine(index)

engine = load_engine()

# Keep track of the conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle new question
if question := st.chat_input("Ask a question about your documents..."):
    
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer and show it
    with st.chat_message("assistant"):
        with st.spinner("Reading documents..."):
            response = engine.query(question)
            answer = str(response)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})