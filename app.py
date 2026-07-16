# ============================================================
# app.py
# Insight AI Assistant Streamlit UI
# ============================================================

import os
import streamlit as st

# Import the initialized objects directly from your analysis.py file
from analysis import (
    chatbot,
    token_manager,
    input_validator,
    document_reader,
    chunking_engine,
    embedding_engine,
    faiss_store,
    image_assistant,
    postgres_logger,
    mongodb_memory,
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

# ============================================================
# CSS for Chat Customization
# ============================================================

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

# ============================================================
# Session State Initialization
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files_list" not in st.session_state:
    st.session_state.uploaded_files_list = []

# ============================================================
# Navigation Sidebar
# ============================================================

with st.sidebar:
    st.title("🤖 Insight AI")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Dashboard"]
    )
    
    if page == "🏠 Home":
        st.session_state.page = "Home"
    else:
        st.session_state.page = "Dashboard"

# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":
    st.title("🤖 Insight AI Assistant")
    st.subheader("Created by Lakshmi Tejasree Kurmapu")
    st.write("")
    
    st.markdown(
    """
    ## Project Overview
    Insight AI Assistant is an enterprise multimodal AI chatbot built using advanced NLP and database integrations.

    ### Technologies
    - **Model:** gemini-3.1-flash-lite / gemini-2.5-flash
    - **Language:** Python
    - **Interface:** Streamlit Web Framework
    - **Architecture:** Retrieval Augmented Generation (RAG)
    - **Vector Storage:** FAISS Vector Search
    - **Text Representation:** Sentence Transformer Embeddings

    ### Capabilities
    ✅ AI Conversation & Contextual History  
    ✅ Enterprise Document Question Answering  
    ✅ PDF / DOCX / TXT Section Chunking & Search  
    ✅ Multimodal Image Understanding & Description  
    ✅ Automated Token Monitoring  
    ✅ Relational Database Logging & NoSQL Memory  

    ### Workflow
    User Input ➔ Streamlit Interface ➔ Chatbot Controller ➔ Vector DB Context Lookup ➔ Gemini AI Model ➔ Verified Response Generation
    """
    )

# ============================================================
# DASHBOARD PAGE (The Main Chat Interface)
# ============================================================

else:
    # -------------------------------
    # Dashboard Sidebar Controls
    # -------------------------------
    with st.sidebar:
        st.title("🤖 Chat Control")
        
        # New Chat Button: Clears local screen, token histories, and memory
        if st.button("➕ New Chat"):
            st.session_state.messages = []
            token_manager.clear()
            if mongodb_memory and mongodb_memory.collection is not None:
                mongodb_memory.clear_memory()
            st.rerun()

        st.divider()

        st.subheader("🕘 Recent Dialogues")
        # Pull conversational lines from our token manager log
        history_items = token_manager.get_chat_history()
        if history_items:
            for item in history_items[-6:]:
                st.caption(f"**{item['role']}**: {item['content'][:30]}...")
        else:
            st.caption("No conversations in this session yet.")

        st.divider()

        st.subheader("📚 Loaded Library")
        if st.session_state.uploaded_files_list:
            for fname in st.session_state.uploaded_files_list:
                st.caption(f"📄 {fname}")
        else:
            st.caption("No enterprise files ingested.")

        st.divider()

        st.subheader("⚡ System Metrics")
        st.write("Session Tokens:", token_manager.total_tokens())
        st.write("Active Model:", MODEL_NAME)

    # -------------------------------
    # Main Dashboard Window
    # -------------------------------
    st.title("Insight AI Hub")
    st.caption("Engage in direct conversation, query vector databases, or pass images/documents.")

    # File uploader widget
    uploaded_file = st.file_uploader(
        "🔎 Search / Upload Enterprise Data",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"]
    )

    if uploaded_file:
        if st.button("Process File"):
            with st.spinner("Processing file for ingestion..."):
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                
                # Create a safe temporary directory to save the file stream to disk for reading
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_filepath = os.path.join(tmpdir, uploaded_file.name)
                    with open(tmp_filepath, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # A. Handle Documents (PDF, DOCX, TXT) via RAG Pipeline Ingestion
                    if file_ext in [".pdf", ".docx", ".txt"]:
                        try:
                            # 1. Read document raw text
                            parsed_doc = document_reader.read_document(tmp_filepath)
                            # 2. Extract context chunks
                            text_chunks = chunking_engine.split_text(parsed_doc["text"])
                            # 3. Form numerical embeddings
                            chunk_embeddings = embedding_engine.generate_embeddings(text_chunks)
                            # 4. Construct flat index array in FAISS store
                            faiss_store.create_index(chunk_embeddings, text_chunks)
                            
                            if uploaded_file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(uploaded_file.name)
                            st.success(f"Successfully processed document! Ingested {len(text_chunks)} clean knowledge segments into Vector Store.")
                        except Exception as e:
                            st.error(f"Error parsing document file: {e}")

                    # B. Handle Images via Multi-Modal Image Assistant
                    elif file_ext in [".png", ".jpg", ".jpeg"]:
                        try:
                            # Save path string inside session state so user can ask contextual questions about it
                            st.session_state["last_image_stream"] = uploaded_file.getvalue()
                            st.session_state["last_image_name"] = uploaded_file.name
                            if uploaded_file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(uploaded_file.name)
                            st.image(uploaded_file, caption="Uploaded Image View", width=300)
                            st.success("Multimodal image loaded! Type a prompt below to analyze its contents.")
                        except Exception as e:
                            st.error(f"Error initializing vision matrix: {e}")

    st.divider()

    # Display full dialogue thread from current state using beautiful HTML classes
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {message["content"]}</div>', unsafe_allow_html=True)

    # Chat Input block
    user_message = st.chat_input("Message Insight AI...")

    if user_message:
        # Append message visually to screen state immediately
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.markdown(f'<div class="chat-user">👤 {user_message}</div>', unsafe_allow_html=True)

        # Container setup for response block
        response_box = st.empty()

        with st.spinner("Thinking..."):
            # Determine if this query focuses on analyzing the newly loaded image stream
            if "last_image_stream" in st.session_state and any(w in user_message.lower() for w in ["image", "photo", "describe", "picture", "look"]):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as img_tmp:
                    img_tmp.write(st.session_state["last_image_stream"])
                    img_tmp_path = img_tmp.name
                
                vision_result = image_assistant.analyze_image(img_tmp_path, prompt=user_message)
                
                # Cleanup physical temporary image
                if os.path.exists(img_tmp_path):
                    os.remove(img_tmp_path)
                
                if vision_result["success"]:
                    answer = vision_result["response"]
                    # Log conversation down into database systems
                    if postgres_logger:
                        postgres_logger.log_conversation(user_message, answer, 0, 0, 0.5, source="IMAGE")
                    if mongodb_memory:
                        mongodb_memory.save_conversation(user_message, answer)
                else:
                    answer = f"Error performing image matrix analysis: {vision_result['response']}"
            else:
                # Default behavior: Send message through Orchestrator pipeline
                orchestrator_result = chatbot.ask(user_message)
                answer = orchestrator_result["response"]

                # Log conversation down into relational audit tables
                if orchestrator_result.get("success"):
                    if postgres_logger:
                        postgres_logger.log_conversation(
                            question=user_message,
                            answer=answer,
                            input_tokens=orchestrator_result.get("input_tokens", 0),
                            output_tokens=orchestrator_result.get("output_tokens", 0),
                            response_time=orchestrator_result.get("response_time", 0.0),
                            source=orchestrator_result.get("intent", "CHAT")
                        )
                    if mongodb_memory:
                        mongodb_memory.save_conversation(user_message, answer)

        # Output final string cleanly inside custom div
        response_box.markdown(f'<div class="chat-ai">🤖 {answer}</div>', unsafe_allow_html=True)
        
        # Save assistant message into persistent session display history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
