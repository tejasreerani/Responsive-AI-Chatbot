# ============================================================
# app.py
# Insight AI Assistant Streamlit UI
# ============================================================

import os
import tempfile  # Added to fix: NameError: name 'tempfile' is not defined
import streamlit as st
from backend import (
    chatbot,
    process_uploaded_file,
    get_chat_history,
    clear_chat,
    library_manager,
    system_status,
    MODEL_NAME
)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Insight AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Hide Streamlit Default Menus and Inject Theme CSS
st.markdown(
"""
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
section[data-testid="stSidebar"] { background: #f7f7f8; }

.chat-user {
    background: #f1f1f1;
    color: #000000;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0;
}

.chat-ai {
    background: white;
    color: #000000;
    border: 1px solid #ddd;
    padding: 12px;
    border-radius: 12px;
    margin: 10px 0;
}
</style>
""",
unsafe_allow_html=True
)

# Initialize Session UI States
if "page" not in st.session_state:
    st.session_state.page = "Home"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Navigation
with st.sidebar:
    st.title("🤖 Insight AI")
    page = st.radio("Navigation", ["🏠 Home", "📊 Dashboard"])
    st.session_state.page = "Home" if "Home" in page else "Dashboard"

# ============================================================
# HOME PAGE
# ============================================================
if st.session_state.page == "Home":
    st.title("🤖 Insight AI Assistant")
    st.subheader("Created by Lakshmi Tejasree Kurmapu")
    
    st.markdown(
    """
    ## Project Overview
    Insight AI Assistant is an enterprise multimodal AI chatbot built using advanced NLP and database integrations.

    ### Capabilities
    ✅ AI Conversation & Contextual History  
    ✅ Enterprise Document Question Answering (RAG)  
    ✅ PDF / DOCX / TXT Section Chunking & Search  
    ✅ Multimodal Image & Audio Analysis  
    ✅ Automated Token Monitoring  
    """
    )

# ============================================================
# DASHBOARD PAGE
# ============================================================
else:
    with st.sidebar:
        st.title("🤖 Chat Control")
        if st.button("➕ New Chat"):
            clear_chat()
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("📚 Loaded Files")
        all_docs = library_manager.get_all()
        if all_docs:
            for d in all_docs:
                st.caption(f"📄 {d['filename']}")
        else:
            st.caption("No files uploaded yet.")

        st.divider()
        st.subheader("⚡ System Metrics")
        status = system_status()
        st.write("Session Tokens Used:", status["tokens"])
        st.write("Active Model:", MODEL_NAME)

    st.title("Insight AI Hub")
    st.caption("Engage in conversation, query vector databases, or process media logs.")

    # Ingestion Widget
    uploaded_file = st.file_uploader(
        "Upload Enterprise Data (PDF, DOCX, TXT, PNG, JPG, MP3, WAV)",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "mp3", "wav", "m4a"]
    )

    if uploaded_file:
        if st.button("Process File"):
            with st.spinner("Processing file contents..."):
                res = process_uploaded_file(uploaded_file)
                st.success(f"Successfully loaded {res['type']} asset into session memory!")

    st.divider()

    # Display Chat History
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {message["content"]}</div>', unsafe_allow_html=True)

    # Chat Input Processing with Live Streaming Output
    user_message = st.chat_input("Message Insight AI...")

    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.markdown(f'<div class="chat-user">👤 {user_message}</div>', unsafe_allow_html=True)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Utilize your backend streaming generation
            for chunk in chatbot.chat_stream(user_message):
                full_response += chunk
                response_placeholder.markdown(f'<div class="chat-ai">🤖 {full_response}</div>', unsafe_allow_html=True)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
