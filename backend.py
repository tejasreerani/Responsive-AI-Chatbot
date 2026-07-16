# ============================================================
# backend.py
# Enterprise AI Assistant Backend
# PART 1/3
# Gemini + Token + Prompt + Client
# ============================================================


import os
import re
import time
import tempfile


from datetime import datetime


from dotenv import load_dotenv


import google.generativeai as genai


import tiktoken




# ============================================================
# Environment Configuration
# ============================================================


load_dotenv()



GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)



MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gemini-2.5-flash"
)



if not GEMINI_API_KEY:

    raise ValueError(
        "GEMINI_API_KEY not found"
    )




genai.configure(

    api_key=GEMINI_API_KEY

)




generation_config = {


    "temperature":0.3,


    "top_p":0.95,


    "top_k":40,


    "max_output_tokens":2048

}




model = genai.GenerativeModel(

    model_name=MODEL_NAME,

    generation_config=generation_config

)






# ============================================================
# Token Counter
# ============================================================


encoding = tiktoken.get_encoding(
    "cl100k_base"
)



def count_tokens(text):

    try:

        return len(
            encoding.encode(text)
        )

    except:

        return 0






# ============================================================
# Token Manager
# ============================================================


class TokenManager:


    def __init__(self,max_history=20):


        self.max_history=max_history


        self.chat_history=[]


        self.total_usage=0





    def add_message(

        self,

        role,

        content

    ):


        self.chat_history.append(

            {

                "role":role,

                "content":content

            }

        )



        self.total_usage += count_tokens(
            content
        )



        self.chat_history = self.chat_history[
            -(self.max_history*2):
        ]






    def get_history(self):


        history=""



        for item in self.chat_history:


            history += (

                item["role"]

                +

                ": "

                +

                item["content"]

                +

                "\n"

            )



        return history






    def get_messages(self):


        return self.chat_history






    def total_tokens(self):


        return self.total_usage






    def clear(self):


        self.chat_history=[]


        self.total_usage=0





token_manager = TokenManager()







# ============================================================
# Prompt Builder
# ============================================================


class PromptBuilder:


    def build_prompt(

        self,

        question,

        history="",

        context=""

    ):



        prompt=f"""


You are Insight AI Assistant.


Rules:

- Give accurate answers.
- Do not hallucinate.
- Use document context when available.
- Answer only what the user asks.
- If information is unavailable say:
  "I don't have enough information."


Conversation History:

{history}



Document Context:

{context}



User Question:

{question}



Assistant:

"""


        return prompt





prompt_builder = PromptBuilder()







# ============================================================
# Gemini Client
# ============================================================


class GeminiClient:


    def generate(

        self,

        question,

        history="",

        context=""

    ):


        try:



            prompt = prompt_builder.build_prompt(

                question,

                history,

                context

            )



            start=time.time()



            response=model.generate_content(

                prompt

            )



            end=time.time()



            return {


                "success":True,


                "response":response.text,


                "time":round(
                    end-start,
                    2
                ),


                "tokens":count_tokens(
                    response.text
                )


            }





        except Exception as e:



            error=str(e).lower()



            if "quota" in error or "rate" in error:


                message=(

                    "⚠️ Gemini API quota exceeded."

                )


            else:


                message=(

                    "⚠️ Unable to process request."

                )



            return {


                "success":False,


                "response":message,


                "time":0,


                "tokens":0


            }








    # ========================================================
    # Streaming Response
    # ========================================================


    def generate_stream(

        self,

        question,

        history="",

        context=""

    ):



        try:



            prompt = prompt_builder.build_prompt(

                question,

                history,

                context

            )




            response=model.generate_content(

                prompt,

                stream=True

            )




            for chunk in response:


                if chunk.text:


                    yield chunk.text






        except Exception as e:



            yield (

                "⚠️ Unable to process request."

            )







gemini_client = GeminiClient()

# ============================================================
# PART 2/3
# Document Processing + RAG + Image + Audio Assistant
# ============================================================


from pypdf import PdfReader

from docx import Document

from PIL import Image

import faiss

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter






# ============================================================
# Global Uploaded File Memory
# ============================================================


uploaded_image_path = None

uploaded_audio_path = None







# ============================================================
# Document Reader
# ============================================================


class DocumentReader:


    def read_document(

        self,

        file_path

    ):



        extension=os.path.splitext(

            file_path

        )[1].lower()



        text=""



        # -------------------------
        # PDF
        # -------------------------


        if extension==".pdf":


            reader=PdfReader(

                file_path

            )


            for page in reader.pages:


                page_text = page.extract_text()



                if page_text:


                    text += page_text + "\n"





        # -------------------------
        # DOCX
        # -------------------------


        elif extension==".docx":



            doc=Document(

                file_path

            )



            for paragraph in doc.paragraphs:


                text += paragraph.text + "\n"







        # -------------------------
        # TXT
        # -------------------------


        elif extension==".txt":



            with open(

                file_path,

                "r",

                encoding="utf-8"

            ) as file:



                text=file.read()






        else:


            raise ValueError(

                "Unsupported document"

            )




        return text






document_reader=DocumentReader()







# ============================================================
# Text Chunking
# ============================================================


class ChunkingEngine:


    def __init__(self):


        self.splitter = RecursiveCharacterTextSplitter(


            chunk_size=800,


            chunk_overlap=150


        )




    def split(

        self,

        text

    ):



        text=re.sub(

            r"\s+",

            " ",

            text

        )



        return self.splitter.split_text(

            text

        )







chunking_engine=ChunkingEngine()







# ============================================================
# Embedding Model
# ============================================================


embedding_model = SentenceTransformer(

    "sentence-transformers/all-MiniLM-L6-v2"

)







# ============================================================
# FAISS Vector Store
# ============================================================


class VectorStore:


    def __init__(self):


        self.index=None


        self.documents=[]





    def create_index(

        self,

        chunks

    ):



        embeddings = embedding_model.encode(

            chunks,

            convert_to_numpy=True

        )



        dimension=embeddings.shape[1]



        self.index = faiss.IndexFlatL2(

            dimension

        )



        self.index.add(

            embeddings.astype(

                "float32"

            )

        )



        self.documents=chunks







    def search(

        self,

        query,

        top_k=3

    ):



        if self.index is None:


            return []




        query_embedding = embedding_model.encode(

            [query],

            convert_to_numpy=True

        )




        distances,indices = self.index.search(

            query_embedding.astype(

                "float32"

            ),

            top_k

        )




        results=[]




        for idx in indices[0]:


            if idx < len(self.documents):


                results.append(

                    self.documents[idx]

                )




        return results








vector_store=VectorStore()







# ============================================================
# RAG Pipeline
# ============================================================


class RAGPipeline:



    def upload_document(

        self,

        file_path

    ):



        text=document_reader.read_document(

            file_path

        )



        if not text.strip():


            return {


                "filename":os.path.basename(file_path),


                "chunks":0,


                "characters":0


            }





        chunks=chunking_engine.split(

            text

        )




        vector_store.create_index(

            chunks

        )




        return {


            "filename":os.path.basename(file_path),


            "chunks":len(chunks),


            "characters":len(text)


        }







    def retrieve(

        self,

        question

    ):



        return vector_store.search(

            question

        )







    def answer(

        self,

        question

    ):



        documents=self.retrieve(

            question

        )



        context="\n\n".join(

            documents

        )



        return gemini_client.generate(

            question,

            token_manager.get_history(),

            context

        )






rag_pipeline=RAGPipeline()








# ============================================================
# Image Assistant
# ============================================================


class ImageAssistant:



    def analyze(

        self,

        image_path,

        prompt

    ):



        try:



            image=Image.open(

                image_path

            )



            response=model.generate_content(

                [

                    prompt,

                    image

                ]

            )



            return {


                "success":True,


                "response":response.text


            }





        except Exception:



            return {


                "success":False,


                "response":"Unable to analyze image."

            }







image_assistant=ImageAssistant()







# ============================================================
# Audio Assistant
# ============================================================


class AudioAssistant:



    def analyze(

        self,

        audio_path

    ):



        try:



            audio_file=genai.upload_file(

                audio_path

            )




            response=model.generate_content(

                [

                    "Transcribe this audio and answer based on it.",

                    audio_file

                ]

            )



            return {


                "success":True,


                "response":response.text


            }





        except Exception as e:



            return {


                "success":False,


                "response":

                "Unable to process audio."

            }







audio_assistant=AudioAssistant()

# ============================================================
# PART 3/3
# Chatbot Controller + File Handling + Library
# ============================================================



# ============================================================
# Intent Recognition
# ============================================================


class IntentRecognizer:


    def detect(

        self,

        text

    ):


        query=text.lower()



        # If document exists
        # always use RAG


        if vector_store.index is not None:


            return "RAG"




        # Image


        if uploaded_image_path:


            return "IMAGE"




        # Audio


        if uploaded_audio_path:


            return "AUDIO"




        # Normal keywords


        if any(

            word in query

            for word in [

                "pdf",

                "document",

                "file",

                "report",

                "resume"

            ]

        ):


            return "RAG"




        return "CHAT"







intent_recognizer=IntentRecognizer()







# ============================================================
# Chatbot
# ============================================================


class Chatbot:



    def chat(

        self,

        message,

        uploaded_context=None

    ):



        intent=intent_recognizer.detect(

            message

        )




        # ----------------------------
        # PDF / RAG
        # ----------------------------


        if intent=="RAG":



            result=rag_pipeline.answer(

                message

            )





        # ----------------------------
        # IMAGE
        # ----------------------------


        elif intent=="IMAGE":



            result=image_assistant.analyze(

                uploaded_image_path,

                message

            )





        # ----------------------------
        # AUDIO
        # ----------------------------


        elif intent=="AUDIO":



            result=audio_assistant.analyze(

                uploaded_audio_path

            )





        # ----------------------------
        # NORMAL CHAT
        # ----------------------------


        else:



            result=gemini_client.generate(

                message,

                token_manager.get_history(),

                uploaded_context or ""

            )






        if result["success"]:



            token_manager.add_message(

                "User",

                message

            )



            token_manager.add_message(

                "Assistant",

                result["response"]

            )







        result["intent"]=intent


        result["total_tokens"]=token_manager.total_tokens()



        return result







    # ========================================================
    # Streaming Chat
    # ========================================================


    def chat_stream(

        self,

        message

    ):



        intent=intent_recognizer.detect(

            message

        )




        # PDF Context


        context=""



        if intent=="RAG":


            documents=rag_pipeline.retrieve(

                message

            )


            context="\n\n".join(

                documents

            )





        history=token_manager.get_history()



        full_response=""




        # Image


        if intent=="IMAGE":



            result=image_assistant.analyze(

                uploaded_image_path,

                message

            )



            yield result["response"]


            full_response=result["response"]





        # Audio


        elif intent=="AUDIO":



            result=audio_assistant.analyze(

                uploaded_audio_path

            )



            yield result["response"]


            full_response=result["response"]





        # Chat + RAG


        else:



            for token in gemini_client.generate_stream(

                message,

                history,

                context

            ):



                full_response += token



                yield token





        token_manager.add_message(

            "User",

            message

        )



        token_manager.add_message(

            "Assistant",

            full_response

        )









chatbot=Chatbot()







# ============================================================
# File Handling
# ============================================================


def save_uploaded_file(

    uploaded_file

):


    temp_dir=tempfile.gettempdir()



    file_path=os.path.join(

        temp_dir,

        uploaded_file.name

    )




    with open(

        file_path,

        "wb"

    ) as file:



        file.write(

            uploaded_file.getbuffer()

        )



    return file_path







def process_uploaded_file(

    uploaded_file

):



    global uploaded_image_path

    global uploaded_audio_path




    file_path=save_uploaded_file(

        uploaded_file

    )




    extension=os.path.splitext(

        file_path

    )[1].lower()





    # ------------------------
    # Documents
    # ------------------------


    if extension in [

        ".pdf",

        ".docx",

        ".txt"

    ]:



        result=rag_pipeline.upload_document(

            file_path

        )



        library_manager.add(

            uploaded_file.name

        )



        return {


            "type":"document",


            "details":result,


            "path":file_path


        }






    # ------------------------
    # Images
    # ------------------------


    elif extension in [


        ".png",

        ".jpg",

        ".jpeg"

    ]:



        uploaded_image_path=file_path



        return {


            "type":"image",


            "path":file_path


        }






    # ------------------------
    # Audio
    # ------------------------


    elif extension in [


        ".mp3",

        ".wav",

        ".m4a"

    ]:



        uploaded_audio_path=file_path



        return {


            "type":"audio",


            "path":file_path


        }






    else:



        return {


            "type":"unknown",


            "path":file_path


        }









# ============================================================
# History Helpers
# ============================================================


def get_chat_history():


    return token_manager.get_messages()







def clear_chat():


    token_manager.clear()







def get_token_usage():


    return token_manager.total_tokens()







# ============================================================
# Library Manager
# ============================================================


class LibraryManager:



    def __init__(self):


        self.documents=[]






    def add(

        self,

        filename

    ):



        self.documents.append(


            {


                "filename":filename,


                "uploaded_at":datetime.now()


            }


        )







    def get_all(self):


        return self.documents








library_manager=LibraryManager()







# ============================================================
# System Status
# ============================================================


def system_status():



    return {


        "model":MODEL_NAME,


        "gemini":"connected",


        "tokens":get_token_usage()


    }