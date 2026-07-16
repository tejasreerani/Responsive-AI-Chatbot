# ============================================================
# app.py
# Insight AI Assistant Streamlit UI
# ============================================================

import streamlit as st


from backend import (
    chatbot,
    process_uploaded_file,
    get_chat_history,
    clear_chat,
    library_manager,
    system_status
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
# CSS
# ============================================================

st.markdown(
"""
<style>

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}


section[data-testid="stSidebar"]{
    background:#f7f7f8;
}


.chat-user{

    background:#f1f1f1;
    padding:12px;
    border-radius:12px;
    margin:10px 0;

}


.chat-ai{

    background:white;
    border:1px solid #ddd;
    padding:12px;
    border-radius:12px;
    margin:10px 0;

}


</style>
""",
unsafe_allow_html=True
)





# ============================================================
# Session State
# ============================================================


if "page" not in st.session_state:

    st.session_state.page="Home"



if "messages" not in st.session_state:

    st.session_state.messages=[]





# ============================================================
# Navigation Sidebar
# ============================================================


with st.sidebar:


    st.title("🤖 Insight AI")


    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dashboard"
        ]
    )


    if page=="🏠 Home":

        st.session_state.page="Home"

    else:

        st.session_state.page="Dashboard"







# ============================================================
# HOME PAGE
# ============================================================


if st.session_state.page=="Home":


    st.title(
        "🤖 Insight AI Assistant"
    )


    st.subheader(
        "Created by Lakshmi Tejasree Kurmapu"
    )


    st.write("")


    st.markdown(
    """

## Project Overview


Insight AI Assistant is an enterprise multimodal AI chatbot built using:


### Technologies

- Google Gemini 2.5 Flash
- Python
- Streamlit
- Retrieval Augmented Generation (RAG)
- FAISS Vector Search
- Sentence Transformer Embeddings


### Capabilities


✅ AI Conversation

✅ Document Question Answering

✅ PDF / DOCX / TXT Analysis

✅ Image Understanding

✅ Token Monitoring

✅ Chat History Management


### Workflow


User Input

↓

Streamlit Interface

↓

Chatbot Controller

↓

Gemini AI Model

↓

Response Generation



"""
)






# ============================================================
# DASHBOARD PAGE
# ============================================================


else:


    # -------------------------------
    # Dashboard Sidebar
    # -------------------------------


    with st.sidebar:


        st.title(
            "🤖 Insight AI"
        )


        if st.button(
            "➕ New Chat"
        ):

            clear_chat()

            st.session_state.messages=[]



        st.divider()



        st.subheader(
            "🕘 Recent Chats"
        )


        history=get_chat_history()


        if history:


            for item in history[-10:]:


                st.write(

                    item["role"]
                    +
                    ": "
                    +
                    item["content"][:35]
                    +
                    "..."

                )

        else:

            st.caption(
                "No chats yet"
            )



        st.divider()



        st.subheader(
            "📚 Library"
        )


        files=library_manager.get_all()



        if files:


            for file in files:

                st.write(
                    "📄 "
                    +
                    file["filename"]
                )

        else:

            st.caption(
                "No files uploaded"
            )



        st.divider()



        status=system_status()


        st.subheader(
            "⚡ Usage"
        )


        st.write(
            "Tokens Used:",
            status["tokens"]
        )


        st.write(
            "Model:",
            status["model"]
        )







    # -------------------------------
    # Main Dashboard
    # -------------------------------


    st.title(
        "Insight AI"
    )


    st.caption(
        "Ask questions or upload files"
    )



    # Upload button


    uploaded_file = st.file_uploader(

        "🔎 Search / Upload",

        type=[
            "pdf",
            "docx",
            "txt",
            "png",
            "jpg",
            "jpeg"
        ]

    )



    if uploaded_file:


        if st.button(
            "Process File"
        ):


            result = process_uploaded_file(
                uploaded_file
            )


            if result["type"]=="document":


                library_manager.add(
                    uploaded_file.name
                )


                st.success(
                    "Document processed successfully"
                )


            elif result["type"]=="image":


                st.success(
                    "Image uploaded successfully"
                )


            else:


                st.warning(
                    "Unsupported file"
                )





    st.divider()






    # Display current chat only


    for message in st.session_state.messages:


        if message["role"]=="user":


            st.markdown(

            f"""

            <div class="chat-user">

            👤 {message["content"]}

            </div>

            """,

            unsafe_allow_html=True

            )


        else:


            st.markdown(

            f"""

            <div class="chat-ai">

            🤖 {message["content"]}

            </div>

            """,

            unsafe_allow_html=True

            )






    # Chat Input


    user_message = st.chat_input(

        "Message Insight AI..."

    )



    if user_message:



        st.session_state.messages.append(

            {
                "role":"user",
                "content":user_message
            }

        )



        response_box=st.empty()



        answer=""



        with st.spinner(
            "Thinking..."
        ):


            for chunk in chatbot.chat_stream(
                user_message
            ):


                answer += chunk



                response_box.markdown(

                f"""

                <div class="chat-ai">

                🤖 {answer}

                </div>

                """,

                unsafe_allow_html=True

                )



        st.session_state.messages.append(

            {
                "role":"assistant",
                "content":answer
            }

        )