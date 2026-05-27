import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

# Load the API key from .env file
load_dotenv()

# Tell LlamaIndex which LLM to use
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

# Tell LlamaIndex which embedding model to use
# This runs locally on your computer — no API needed
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Each chunk is ~512 tokens with 50 token overlap
Settings.chunk_size = 512
Settings.chunk_overlap = 50

DATA_DIR   = "./data"
CHROMA_DIR = "./chroma_db"

def build_or_load_index():
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection("documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0:
        # Index already exists — just load it
        print(f"Loading existing index ({chroma_collection.count()} chunks)...")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
    else:
        # First time — build the index from your documents
        print("Building index from documents...")
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        print(f"Loaded {len(documents)} pages")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print("Index built and saved!")

    return index

def get_query_engine(index):
    return index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact",
    )