import streamlit as st
import tempfile
import os
from rag_engine import build_or_load_index, get_query_engine, Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Intelligence")
st.caption("Upload a document and ask anything about it")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    @st.cache_resource
    def load_engine_from_file(file_name, file_bytes):
        # Save uploaded file to a temp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # Build index from the temp directory
            chroma_client = chromadb.EphemeralClient()
            chroma_collection = chroma_client.get_or_create_collection("documents")
            from llama_index.vector_stores.chroma import ChromaVectorStore
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

    # Chat history
    if "messages" not in