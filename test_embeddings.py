# test_embeddings.py
# Tests our entire vector store pipeline:
# PDF → chunks → embeddings → FAISS index → similarity search

from app.core.embeddings import (
    load_and_split_pdf,
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    index_exists
)

def test_pipeline():
    print("=" * 50)
    print("Day 2 — Testing FAISS Vector Store Pipeline")
    print("=" * 50)

    # Step 1: Load and split PDF
    print("\n📌 Step 1: Load and split PDF")
    chunks = load_and_split_pdf("test_data/sample.pdf")
    print(f"✅ Got {len(chunks)} chunks")
    print(f"   Sample chunk:\n   '{chunks[0].page_content[:200]}...'")

    # Step 2: Create FAISS index
    print("\n📌 Step 2: Create FAISS index")
    vectorstore = create_faiss_index(chunks)
    print("✅ FAISS index created")

    # Step 3: Save index to disk
    print("\n📌 Step 3: Save index")
    save_faiss_index(vectorstore)
    print("✅ Index saved")

    # Step 4: Load index back from disk
    print("\n📌 Step 4: Load index from disk")
    loaded_store = load_faiss_index()
    print("✅ Index loaded")

    # Step 5: Search by meaning
    print("\n📌 Step 5: Similarity search")
    query = "What is this document about?"
    
    # retriever finds top 3 most relevant chunks
    # WHY k=3: we send 3 chunks as context to the LLM
    # more chunks = more context but more tokens = slower + costlier
    results = loaded_store.similarity_search(query, k=3)
    
    print(f"✅ Query: '{query}'")
    print(f"   Found {len(results)} relevant chunks:")
    for i, doc in enumerate(results):
        print(f"\n   Chunk {i+1}:")
        print(f"   '{doc.page_content[:150]}...'")

    print("\n🎉 FAISS pipeline working!")
    print("    PDF → chunks → embeddings → index → search ✅")

if __name__ == "__main__":
    test_pipeline()