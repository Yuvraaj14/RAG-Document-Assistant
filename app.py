# hf_app.py
# Hugging Face Spaces version — simplified for deployment

import gradio as gr
import requests
import os

# For HF Spaces, we deploy Gradio + FastAPI together
# This version runs everything in one process

# Option 1: Deploy just Gradio, keep FastAPI local (EASIEST)
# Option 2: Deploy Gradio + FastAPI in same space (BETTER)

# We'll do Option 2

# First, let's create a combined version
import sys
sys.path.insert(0, os.path.dirname(__file__))

from app.core.rag_pipeline import build_rag_chain, ask_question
from app.core.llm import get_llm

# Global state
rag_chain_tuple = None
auth_token = "demo_mode"  # Simplified auth for HF Spaces demo

def process_pdf(file):
    """Process uploaded PDF"""
    global rag_chain_tuple
    try:
        if file is None:
            return "❌ No file uploaded"
        
        # Build RAG chain from uploaded file
        rag_chain_tuple = build_rag_chain(pdf_path=file.name, use_local_llm=False)
        
        # Count chunks
        from app.core.embeddings import load_faiss_index
        vectorstore = load_faiss_index()
        chunk_count = vectorstore.index.ntotal
        
        return f"✅ PDF processed!\n📄 File: {os.path.basename(file.name)}\n🔢 Chunks created: {chunk_count}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def chat_with_pdf(message, history):
    """Handle chat messages"""
    global rag_chain_tuple

    if history is None:
        history = []

    if rag_chain_tuple is None:
        history.append([message, "❌ Please upload a PDF first"])
        return history, ""

    try:
        # Get answer
        result = ask_question(rag_chain_tuple, message)

        # Format sources
        sources_md = "### 📚 Sources\n\n"

        for i, source in enumerate(result["sources"], 1):
            sources_md += f"**[{i}] Page {source['page']}**\n"
            sources_md += f"> {source['content'][:150]}...\n\n"

        # Update history
        history.append([message, result["answer"]])

        return history, sources_md

    except Exception as e:
        history.append([message, f"❌ Error: {str(e)}"])
        return history, ""

# Build Gradio interface
with gr.Blocks(title="RAG Document Q&A Assistant") as demo:
    gr.Markdown("# 📄 RAG Document Q&A Assistant")
    gr.Markdown("Upload a PDF and ask questions — get cited answers powered by LangChain + FAISS + Groq")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload PDF")
            pdf_file = gr.File(label="Select PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Process PDF", variant="primary")
            upload_output = gr.Textbox(label="Status", interactive=False, lines=3)
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                label="Your question",
                placeholder="What is this document about?",
                lines=2
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")
            
            gr.Markdown("---")
            sources_display = gr.Markdown("### 📚 Sources will appear here")
    
    # Event handlers
    upload_btn.click(process_pdf, inputs=pdf_file, outputs=upload_output)
    
    def respond(message, chat_history):
        updated_history, sources = chat_with_pdf(message, chat_history)
        return "", updated_history, sources
    
    send_btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, sources_display]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, sources_display]
    )
    
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, sources_display])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )