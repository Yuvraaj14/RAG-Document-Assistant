# app/core/embeddings.py
# WHY THIS FILE EXISTS:
# Before we can search documents by meaning, we need to convert
# text into vectors (lists of numbers). Similar meaning = similar numbers.
# FAISS then searches these vectors at very high speed.

import os
import faiss
import pickle
from pathlib import Path
from dotenv import load_dotenv

# HuggingFace embeddings run locally — zero API cost
# sentence-transformers is the library, all-MiniLM-L6-v2 is the model
# It converts any text into a 384-dimensional vector
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS vector store from LangChain — wraps raw FAISS with useful methods
from langchain_community.vectorstores import FAISS

# Document loaders — handles PDF files
from langchain_community.document_loaders import PyPDFLoader

# Text splitter — breaks large documents into smaller chunks
# WHY: LLMs have context limits. Chunk size = how much text per piece.
# Overlap = how many characters shared between chunks (prevents losing context)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Path where we save the FAISS index on disk
FAISS_INDEX_PATH = "faiss_index"

# At the top of app/core/embeddings.py, add this after imports:

# Global cache for embeddings model — load once, reuse forever
_cached_embeddings = None

def get_embeddings():
    """
    Returns HuggingFace embedding model.
    Cached globally to avoid reloading on every request.
    """
    global _cached_embeddings
    
    if _cached_embeddings is None:
        print("🔄 Loading embedding model (first time only)...")
        from langchain_huggingface import HuggingFaceEmbeddings
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("✅ Embedding model loaded and cached")
    
    return _cached_embeddings

def load_and_split_pdf(pdf_path: str) -> list:
    """
    Loads a PDF and splits it into chunks.
    
    WHY SPLIT?
    - A 50-page PDF has ~25,000 words
    - LLMs can only process ~2000-4000 words at once
    - We split into chunks and only send RELEVANT chunks to the LLM
    - This is the core idea behind RAG
    
    chunk_size=500: each chunk = ~500 characters
    chunk_overlap=50: 50 characters shared between chunks
    WHY OVERLAP: prevents losing context at chunk boundaries
    """
    print(f"📄 Loading PDF: {pdf_path}")
    
    # PyPDFLoader extracts text from each page
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"   Loaded {len(pages)} pages")
    
    # Split pages into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        # tries to split on these characters in order
        # prefers splitting on paragraphs, then sentences, then words
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_documents(pages)
    print(f"   Split into {len(chunks)} chunks")
    return chunks

def create_faiss_index(chunks: list) -> FAISS:
    """
    Creates FAISS index from document chunks.
    
    WHY FAISS?
    - Pure vector similarity search
    - No database overhead
    - Extremely fast — searches millions of vectors in milliseconds
    - Built by Meta, battle-tested at massive scale
    
    HOW IT WORKS:
    1. Each chunk gets converted to a 384-dim vector via embeddings
    2. FAISS stores all vectors in an optimised index structure
    3. At query time: convert question to vector, find nearest vectors
    """
    print("🔢 Creating embeddings and FAISS index...")
    embeddings = get_embeddings()
    
    # FAISS.from_documents does two things:
    # 1. Calls embeddings.embed_documents() on every chunk
    # 2. Stores all vectors + original text in FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"   Index created with {len(chunks)} vectors")
    return vectorstore

def save_faiss_index(vectorstore: FAISS):
    """Saves FAISS index to disk so we don't rebuild every time."""
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"💾 Index saved to {FAISS_INDEX_PATH}/")

def load_faiss_index() -> FAISS:
    """Loads existing FAISS index from disk."""
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        # required for newer versions of FAISS + LangChain
        allow_dangerous_deserialization=True
    )
    print(f"📂 Index loaded from {FAISS_INDEX_PATH}/")
    return vectorstore

def index_exists() -> bool:
    """Checks if a saved FAISS index exists."""
    return Path(FAISS_INDEX_PATH).exists()