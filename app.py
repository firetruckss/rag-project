import streamlit as st
import tempfile
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv

load_dotenv()

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.chunk_size = 512
Settings.chunk_overlap = 50

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Intelligence")
st.caption("Upload a document and ask anything about it")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    @st.cache_resource
    def load_engine_from_file(file_name, file_bytes):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)
            chroma_client = chromadb.EphemeralClient()
            chroma_collection = chroma_client.get_or_create_collection("documents")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            documents = SimpleDirectoryReader(tmp_dir).load_data()
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            return index.as_query_engine(similarity_top_k=5, response_mode="compact")

    engine = load_engine_from_file(uploaded_file.name, uploaded_file.getvalue())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Reading document..."):
                response = engine.query(question)
                answer = str(response)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Please upload a PDF to get started!")