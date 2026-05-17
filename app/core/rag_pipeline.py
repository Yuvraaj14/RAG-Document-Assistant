# app/core/rag_pipeline.py
# Using LCEL (LangChain Expression Language) — modern approach
# WHY LCEL over RetrievalQA?
# RetrievalQA is deprecated in newer LangChain
# LCEL is the new standard — cleaner, more flexible, faster

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.llm import get_llm
from app.core.embeddings import (
    load_and_split_pdf,
    create_faiss_index,
    save_faiss_index,
    load_faiss_index,
    index_exists
)

load_dotenv()

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based ONLY on the provided context.

Context from the document:
{context}

User Question: {question}

Instructions:
- Answer ONLY using information from the context above
- If the answer is not in the context, say "I don't have enough information in the document to answer this"
- Be concise and clear
- Cite which part of the context supports your answer

Answer:
"""

def format_docs(docs):
    """
    Joins retrieved chunks into one string.
    WHY: prompt needs context as plain text, not a list of objects
    """
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(pdf_path: str = None, use_local_llm: bool = False):
    """
    Builds RAG chain using LCEL pipe syntax.
    
    LCEL chain reads left to right:
    retriever → format_docs → prompt → llm → output_parser
    
    Each step's output becomes the next step's input.
    This is cleaner and more explicit than RetrievalQA.
    """

    # Step 1: Get or create FAISS index
    if index_exists() and pdf_path is None:
        print("📂 Loading existing FAISS index...")
        vectorstore = load_faiss_index()
    elif pdf_path:
        print(f"📄 Processing PDF: {pdf_path}")
        chunks = load_and_split_pdf(pdf_path)
        vectorstore = create_faiss_index(chunks)
        save_faiss_index(vectorstore)
    else:
        raise ValueError("No PDF provided and no existing index found.")

    # Step 2: Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Step 3: Build prompt template
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    # Step 4: Get LLM
    llm = get_llm(use_local=use_local_llm)

    # Step 5: Build LCEL chain
    # RunnablePassthrough passes the question through unchanged
    # format_docs converts retrieved docs to plain text
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✅ RAG chain built successfully")
    return rag_chain, retriever

def ask_question(chain_tuple, question: str) -> dict:
    """
    Asks a question through the RAG chain.
    Returns answer + source chunks used.
    """
    chain, retriever = chain_tuple

    print(f"\n❓ Question: {question}")

    # Get answer from chain
    answer = chain.invoke(question)

    # Get source chunks separately for citation
    source_docs = retriever.invoke(question)

    print(f"💬 Answer: {answer}")
    print(f"\n📚 Sources used ({len(source_docs)} chunks):")
    for i, doc in enumerate(source_docs):
        print(f"   [{i+1}] Page {doc.metadata.get('page', 'N/A')}: "
              f"'{doc.page_content[:100]}...'")

    return {
        "answer": answer,
        "sources": [
            {
                "page": doc.metadata.get("page", "N/A"),
                "content": doc.page_content[:200]
            }
            for doc in source_docs
        ]
    }