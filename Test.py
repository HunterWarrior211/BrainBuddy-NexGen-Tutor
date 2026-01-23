import os

# --- 1. ROBUST SQLITE FIX FOR STREAMLIT CLOUD ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass 

import shutil
import time
import streamlit as st
import tempfile
import uuid
import json
import asyncio
import hashlib
import edge_tts
import re
import pandas as pd
import altair as alt 
from datetime import datetime

# --- IMPORTS ---
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# ⚙️ SYSTEM CONFIGURATION & PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BOOKS_FOLDER = os.path.join(BASE_DIR, "resources") 
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploaded_books")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
USERS_FILE = os.path.join(BASE_DIR, "users.json")
QUIZ_FILE = os.path.join(BASE_DIR, "quiz_scores.json")

# Ensure critical directories exist
for path in [UPLOAD_DIR, BOOKS_FOLDER, PERSIST_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

# Streamlit Page Setup
st.set_page_config(
    page_title="NexGen Tutor | AI Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 ULTRA-PREMIUM CSS STYLING (GLOWING SCROLLBAR & ROUNDED CORNERS)
# ==========================================
DARK_CSS = """
<style>
    /* --- FONTS --- */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Playfair+Display:ital,wght@0,700;1,400&family=Exo+2:wght@300;400;600&family=Inter:wght@400;600&display=swap');
    
    /* --- ✨ HYPER-GLOW GOLDEN SCROLLBAR ✨ --- */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
        margin-block: 5px;
    }
    
    ::-webkit-scrollbar-corner {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
        border-radius: 20px; /* UPDATED: Rounder Scrollbar */
        border: 2px solid #0A0E14; 
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.5);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FFD700, #FFFACD, #FFD700);
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.9); 
        border: 1px solid #FFF;
    }

    * {
        scrollbar-width: thin;
        scrollbar-color: #D4AF37 #0A0E14;
    }

    /* --- ANIMATIONS --- */
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes goldPulse { 0% { text-shadow: 0 0 10px rgba(212, 175, 55, 0.2); } 50% { text-shadow: 0 0 25px rgba(212, 175, 55, 0.6); } 100% { text-shadow: 0 0 10px rgba(212, 175, 55, 0.2); } }
    @keyframes cosmicDrift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

    /* --- GLOBAL THEME --- */
    .stApp {
        background-color: #0A0E14;
        background-image: radial-gradient(#1B1F28 1px, transparent 1px), linear-gradient(125deg, #0A0E14 0%, #11161F 40%, #0A0E14 100%);
        background-size: 40px 40px, 200% 200%;
        animation: cosmicDrift 20s ease infinite;
        color: #EAEAEA;
    }

    /* --- TYPOGRAPHY --- */
    h1 {
        font-family: 'Cinzel', serif !important;
        font-weight: 900 !important;
        color: #D4AF37 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px #000000;
        animation: goldPulse 3s infinite ease-in-out;
        margin-bottom: 0.5rem;
    }

    h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: #D4AF37 !important;
        font-weight: 700;
        margin-top: 1.5rem;
        border-bottom: 1px solid rgba(212, 175, 55, 0.3);
        padding-bottom: 5px;
    }

    p, div, li, span {
        font-family: 'Exo 2', sans-serif;
        font-size: 16px;
        line-height: 1.7;
        color: #E0E6ED;
    }

    strong { color: #D4AF37 !important; font-weight: 800; }

    /* --- MOBILE --- */
    @media only screen and (max-width: 768px) {
        h1 { font-size: 1.8rem !important; text-align: center; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.2rem !important; }
        p, div, li, span { font-size: 14px !important; line-height: 1.5 !important; }
        .stButton>button { width: 100% !important; margin-bottom: 5px; }
        section[data-testid="stSidebar"] { width: 85% !important; }
        .stChatMessage { padding: 10px !important; border-radius: 15px !important; }
        ::-webkit-scrollbar { width: 6px; }
    }

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
        background-color: #141A24;
        border-right: 1px solid #D4AF37;
        box-shadow: 5px 0 15px rgba(0,0,0,0.5);
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #E0E6ED !important; }

    /* --- BUTTONS (UPDATED: ROUND CORNERS) --- */
    div.stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #B8962E 100%);
        color: #0A0E14;
        font-family: 'Cinzel', serif;
        font-weight: 900;
        border: none;
        border-radius: 12px; /* UPDATED from 6px */
        padding: 0.6rem 1.4rem;
        transition: all 0.3s;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        color: #000;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.6);
    }
    div[data-testid="column"] button { width: 100%; }

    /* --- CHAT BOXES (UPDATED: ROUND CORNERS) --- */
    .stChatMessage {
        background-color: #1B1F28;
        border-radius: 20px; /* UPDATED from 12px */
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        animation: fadeInUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }
    div[data-testid="stChatMessage"]:nth-child(odd) { border-left: 4px solid #8BE9FD; }
    div[data-testid="stChatMessage"]:nth-child(even) { border-left: 4px solid #D4AF37; background-color: #151921; }

    /* --- INPUT FIELDS (UPDATED: ROUND CORNERS) --- */
    .stTextInput > div > div > input {
        background-color: #1B1F28;
        color: #EAEAEA;
        border: 1px solid #D4AF37;
        border-radius: 12px; /* UPDATED from 8px */
    }

    /* --- EXPANDER (UPDATED: ROUND CORNERS) --- */
    [data-testid="stExpander"] {
        background-color: #1B1F28 !important;
        border: 1px solid #D4AF37 !important; 
        border-radius: 16px !important; /* UPDATED from 12px */
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        margin-top: 20px;
    }
    [data-testid="stExpander"] summary { 
        color: #D4AF37 !important;
        font-family: 'Cinzel', serif !important; 
        font-weight: 900 !important; 
    }

    [data-testid="stPopover"] > button {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: inherit !important;
        padding: 0.5rem !important;
        border-radius: 8px !important; /* Added rounding */
    }
    text { fill: #EAEAEA !important; }
</style>
"""

LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Playfair+Display:ital,wght@0,700;1,400&family=Exo+2:wght@400;600&family=Inter:wght@400;600&family=Roboto+Mono&display=swap');

    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes charcoalPulse { 0% { text-shadow: 0 0 0px rgba(43, 46, 52, 0); } 50% { text-shadow: 2px 4px 8px rgba(43, 46, 52, 0.15); } 100% { text-shadow: 0 0 0px rgba(43, 46, 52, 0); } }
    @keyframes silverDrift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

    .stApp {
        background-color: #F8F9FB;
        background-image: radial-gradient(#BFC3C9 1.5px, transparent 1.5px), linear-gradient(120deg, #F8F9FB 0%, #FFFFFF 50%, #E5E7EB 100%);
        background-size: 30px 30px, 200% 200%;
        animation: silverDrift 20s ease infinite;
        color: #5F6368;
    }

    /* --- LIGHT MODE SCROLLBAR --- */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #BFC3C9;
        border-radius: 20px; /* UPDATED: Rounder */
        border: 2px solid #F8F9FB;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #9FA4AA;
    }

    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: #2B2E34 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 900 !important;
        margin-top: 1.5rem;
        animation: charcoalPulse 4s infinite ease-in-out;
    }

    h3 {
        font-size: 1.6rem !important;
        border-bottom: 2px solid #BFC3C9;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }

    p, li, span, div {
        font-family: 'Exo 2', sans-serif;
        color: #5F6368;
        font-size: 18px;
        line-height: 1.6;
    }
    strong { color: #2B2E34; font-weight: 800; }

    @media only screen and (max-width: 768px) {
        h1 { font-size: 1.8rem !important; text-align: center; }
        h2 { font-size: 1.4rem !important; }
        h3 { font-size: 1.2rem !important; }
        p, li, span, div { font-size: 14px !important; }
        .stButton>button { width: 100% !important; margin-bottom: 5px; }
        .stChatMessage { padding: 10px !important; }
        ::-webkit-scrollbar { width: 6px; }
    }

    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
        box-shadow: 5px 0 20px rgba(0,0,0,0.03);
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p {
        color: #2B2E34 !important;
    }

    /* --- CHAT BOXES (UPDATED: ROUND CORNERS) --- */
    .stChatMessage {
        border-radius: 20px; /* UPDATED from 14px */
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        animation: fadeInUp 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }
    div[data-testid="stChatMessage"]:nth-child(odd) { background-color: #F1F3F6; border: 1px solid #E5E7EB; color: #2B2E34; border-left: 4px solid #BFC3C9; }
    div[data-testid="stChatMessage"]:nth-child(even) { background-color: #FFFFFF; border: 1px solid #E5E7EB; color: #374151; border-left: 4px solid #2B2E34; }

    /* --- EXPANDER (UPDATED: ROUND CORNERS) --- */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 2px solid #BFC3C9 !important; 
        border-radius: 16px !important; /* UPDATED from 14px */
        box-shadow: 0 6px 15px rgba(0,0,0,0.05);
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-family: 'Cinzel', serif !important;
        color: #2B2E34 !important; font-weight: 900; background-color: #F8F9FB; border-bottom: 1px solid #E5E7EB;
    }
    [data-testid="stExpander"] div[data-testid="stMarkdownContainer"] p { color: #5F6368 !important; }

    /* --- INPUTS (UPDATED: ROUND CORNERS) --- */
    .stTextInput > div > div > input { background-color: #FFFFFF; color: #2B2E34; border: 2px solid #E5E7EB; border-radius: 12px; } /* UPDATED from 8px */
    .stTextInput > div > div > input:focus { border-color: #BFC3C9; box-shadow: 0 0 10px rgba(191, 195, 201, 0.4); }

    /* --- BUTTONS (UPDATED: ROUND CORNERS) --- */
    .stButton>button {
        background: linear-gradient(135deg, #BFC3C9 0%, #9FA4AA 100%);
        color: #2B2E34;
        font-family: 'Cinzel', serif;
        font-weight: 700;
        border: none;
        border-radius: 12px; /* UPDATED from 6px */
        padding: 0.6rem 1.4rem;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { transform: scale(1.05); background: #2B2E34; color: #FFFFFF; }
    div[data-testid="column"] button { width: 100%; }

    [data-testid="stPopover"] > button {
        background: transparent !important;
        border: 1px solid #E5E7EB !important;
        color: #2B2E34 !important;
        padding: 0.5rem !important;
        border-radius: 8px !important; /* Added Rounding */
    }
</style>
"""


# ==========================================
# 🛠️ DATA MANAGEMENT & AUTH
# ==========================================

def load_json_db(filepath):
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_json_db(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving DB: {e}")


def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def authenticate_user(username, password):
    users = load_json_db(USERS_FILE)
    if username in users and users[username] == hash_password(password):
        return True
    return False


def register_new_user(username, password):
    users = load_json_db(USERS_FILE)
    if username in users:
        return False
    users[username] = hash_password(password)
    save_json_db(USERS_FILE, users)
    return True


# ==========================================
# 🧠 DOCUMENT & AI ENGINE
# ==========================================
class DocumentProcessor:
    def __init__(self):
        self.vectordb_doc = None
        self.splits = []
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        except:
            pass

        if "GOOGLE_API_KEY" not in os.environ:
            local_secrets = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
            if os.path.exists(local_secrets):
                with open(local_secrets, "r") as f:
                    for line in f:
                        if "GOOGLE_API_KEY" in line:
                            os.environ["GOOGLE_API_KEY"] = line.split("=")[1].strip().strip('"').strip("'")

        if "GOOGLE_API_KEY" not in os.environ:
            st.error("⚠️ API Key Missing! Please check .streamlit/secrets.toml")
            st.stop()

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            max_retries=2
        )

    def load_from_disk(self):
        if os.path.exists(PERSIST_DIR):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.vectordb_doc = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            try:
                if self.vectordb_doc._collection.count() > 0:
                    return True
            except:
                return False
        return False

    def process_local_library(self):
        if not os.path.exists(BOOKS_FOLDER): return False
        pdf_files = [f for f in os.listdir(BOOKS_FOLDER) if f.lower().endswith(".pdf")]
        if not pdf_files: return False

        all_docs = []
        bar = st.progress(0, "Scanning Library...")
        for i, f in enumerate(pdf_files):
            try:
                loader = PyPDFLoader(os.path.join(BOOKS_FOLDER, f))
                all_docs.extend(loader.load())
                bar.progress((i + 1) / len(pdf_files))
            except:
                pass
        bar.empty()

        if not all_docs: return False
        self.splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(all_docs)
        return self.create_embeddings_batched()

    def process_uploaded_file(self, uploaded_file):
        temp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            check = self.llm.invoke(
                [HumanMessage(content=f"Is this educational? YES/NO. Text: {docs[0].page_content[:500]}")]).content
            if "NO" in check.upper():
                st.error("🚫 Educational content only.")
                os.remove(temp_path)
                return False

            self.splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(docs)
            os.remove(temp_path)
            return self.create_embeddings_batched()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return False

    def create_embeddings_batched(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if self.vectordb_doc is None:
            if os.path.exists(PERSIST_DIR):
                self.vectordb_doc = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

        batch_size = 20
        total = len(self.splits)
        if self.vectordb_doc is None:
            self.vectordb_doc = Chroma.from_documents(self.splits[:batch_size], embeddings,
                                                      persist_directory=PERSIST_DIR)
            start_index = batch_size
        else:
            start_index = 0

        try:
            bar = st.progress(0, "Memorizing Content...")
            for i in range(start_index, total, batch_size):
                self.vectordb_doc.add_documents(self.splits[i:i + batch_size])
                bar.progress(min((i + batch_size) / total, 1.0))
                time.sleep(0.05)
            bar.empty()
            return True
        except Exception as e:
            st.error(f"DB Error: {e}")
            return False

    def extract_chat_topics(self, chat_history):
        prompt = f"""
        Analyze the chat history.
        Identify specific educational subjects/topics discussed (e.g., Mitosis, Gravity, Algebra).
        Strictly return ONLY a comma-separated list of distinct topics.
        List up to 4 topics.
        Chat History: {chat_history}
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        cleaned = response.content.replace('**', '').replace('.', '').strip()
        topics = [t.strip() for t in cleaned.split(',')]
        return topics[:4]

    def generate_quiz_json(self, topic, grade):
        prompt = f"""
        Create a 3-question Multiple Choice Quiz on '{topic}' suitable for {grade}.
        REQUIREMENTS:
        1. Questions must be relevant to {grade}.
        2. Provide clear options.
        3. Provide a helpful explanation for the correct answer.

        Return ONLY valid JSON format like this:
        [
            {{"question": "Q1 text...", "options": ["A) Opt1", "B) Opt2", "C) Opt3", "D) Opt4"], "answer": "A) Opt1", "explanation": "Why it is correct..."}},
            ...
        ]
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            content = response.content.strip()
            if content.startswith("```json"): content = content[7:-3]
            if content.startswith("```"): content = content[3:-3]
            return json.loads(content)
        except:
            return []


# --- AUDIO FUNCTION (Edge TTS) ---
async def generate_edge_audio(text, filename):
    communicate = edge_tts.Communicate(text, "en-GB-SoniaNeural")
    await communicate.save(filename)


def text_to_audio(text):
    try:
        clean_text = text.replace('*', '').replace('#', '').replace('$', '')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            filename = fp.name
        asyncio.run(generate_edge_audio(clean_text, filename))
        return filename
    except Exception as e:
        return None


# ==========================================
# 🔄 SESSION & STATE
# ==========================================
def load_history_from_disk(username):
    all_data = load_json_db(HISTORY_FILE)
    return all_data.get(username, {})


def save_history_to_disk(username):
    all_data = load_json_db(HISTORY_FILE)
    all_data[username] = st.session_state.saved_chats
    save_json_db(HISTORY_FILE, all_data)


def save_quiz_score(topic, obtained, total):
    username = st.session_state.current_user
    scores = load_json_db(QUIZ_FILE)
    if username not in scores: scores[username] = []
    scores[username].append({
        "topic": topic,
        "obtained": obtained,
        "total": total,
        "date": datetime.now().strftime("%Y-%m-%d")
    })
    save_json_db(QUIZ_FILE, scores)


def init_session():
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    if st.session_state.current_user:
        user = st.session_state.current_user

        if "saved_chats" not in st.session_state:
            st.session_state.saved_chats = load_history_from_disk(user)

        if "processor" not in st.session_state:
            st.session_state.processor = DocumentProcessor()
            st.session_state.db_ready = False
            st.session_state.show_quiz = False
            st.session_state.quiz_data = None

        if "current_chat_id" not in st.session_state:
            if st.session_state.saved_chats:
                st.session_state.current_chat_id = list(st.session_state.saved_chats.keys())[-1]
            else:
                new_chat()

        if "messages" not in st.session_state:
            st.session_state.messages = list(st.session_state.saved_chats.get(st.session_state.current_chat_id, []))


def new_chat():
    uid = str(uuid.uuid4())
    st.session_state.current_chat_id = uid
    st.session_state.saved_chats[uid] = []
    st.session_state.messages = []
    st.session_state.show_quiz = False
    st.session_state.quiz_data = None
    if "detected_topics" in st.session_state:
        del st.session_state.detected_topics
    save_history_to_disk(st.session_state.current_user)


def load_chat(session_id):
    st.session_state.current_chat_id = session_id
    st.session_state.messages = list(st.session_state.saved_chats.get(session_id, []))
    st.session_state.show_quiz = False
    st.session_state.quiz_data = None
    if "detected_topics" in st.session_state:
        del st.session_state.detected_topics


def delete_chat(session_id):
    if session_id in st.session_state.saved_chats:
        del st.session_state.saved_chats[session_id]
        save_history_to_disk(st.session_state.current_user)

    if st.session_state.current_chat_id == session_id:
        if st.session_state.saved_chats:
            new_id = list(st.session_state.saved_chats.keys())[-1]
            load_chat(new_id)
        else:
            new_chat()
    else:
        load_chat(st.session_state.current_chat_id)


def get_chat_title(messages):
    if not messages: return "New Conversation"
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            words = content.split()[:5]
            return " ".join(words)[:25] + "..."
    return "Conversation"


# ==========================================
# 🖥️ MAIN APPLICATION LOGIC
# ==========================================
def main():
    if 'current_user' not in st.session_state or st.session_state.current_user is None:
        st.markdown(DARK_CSS, unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.title("NexGen Tutor Login")
            tab1, tab2 = st.tabs(["Login", "Sign Up"])

            with tab1:
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                if st.button("Log In", use_container_width=True):
                    if authenticate_user(username, password):
                        st.session_state.clear()
                        st.session_state.current_user = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

            with tab2:
                new_user = st.text_input("New Username", key="signup_user")
                new_pass = st.text_input("New Password", type="password", key="signup_pass")
                if st.button("Sign Up", use_container_width=True):
                    if register_new_user(new_user, new_pass):
                        st.success("Account created! You can now log in.")
                    else:
                        st.error("Username already exists.")
        return

    init_session()

    with st.sidebar:
        # --- 1. USER & THEME (TOP) ---
        st.write(f"👤 **Logged in as:** {st.session_state.current_user}")
        if st.button("Logout", type="secondary"):
            st.session_state.clear()
            st.rerun()

        st.header("🎨 Theme")
        theme_choice = st.radio("Style", ["Dark Mode 🌑 (Cosmic Gold)", "Light Mode ☀️ (Platinum Silver)"],
                                horizontal=True, key="theme_selector")

        if "Dark Mode" in theme_choice:
            st.markdown(DARK_CSS, unsafe_allow_html=True)
            chart_bg_color = '#1B1F28' 
            chart_text_color = '#D4AF37'
        else:
            st.markdown(LIGHT_CSS, unsafe_allow_html=True)
            chart_bg_color = '#FFFFFF'
            chart_text_color = '#2B2E34'

        st.divider()

        # --- 2. GRADE & SUBJECT (MOVED TO TOP AS REQUESTED) ---
        st.header("🎓 Academic Settings")
        st.session_state['user_grade'] = st.selectbox("Grade:", [f"Grade {i}" for i in range(6, 11)], index=0)
        st.session_state['user_subject'] = st.selectbox("Subject:",
                                                         ["Mathematics", "Physics", "Biology", "Chemistry", "Science",
                                                          "History", "Geography"])

        st.divider()

        # --- 3. QUIZ GENERATOR (MOVED BELOW SUBJECT) ---
        with st.expander("🧠 Quiz Generator", expanded=True):
            if st.button("🚀 Analyze Chat for Topics", use_container_width=True):
                if len(st.session_state.messages) > 0:
                    hist_str = "\n".join([m['content'] for m in st.session_state.messages])
                    with st.spinner("Detecting topics..."):
                        st.session_state.detected_topics = st.session_state.processor.extract_chat_topics(hist_str)
                else:
                    st.warning("Please chat first!")

            final_quiz_topic = None
            if "detected_topics" in st.session_state and st.session_state.detected_topics:
                st.markdown("##### Select Topic:")
                options = st.session_state.detected_topics + ["Custom Topic..."]
                choice = st.radio("Pick one:", options, label_visibility="collapsed")

                if choice == "Custom Topic...":
                    final_quiz_topic = st.text_input("Enter topic manually:", placeholder="e.g. Gravity")
                else:
                    final_quiz_topic = choice

                if st.button("📝 Start Quiz", type="primary", use_container_width=True):
                    if final_quiz_topic:
                        with st.spinner(f"Generating Quiz on {final_quiz_topic}..."):
                            q_data = st.session_state.processor.generate_quiz_json(final_quiz_topic,
                                                                                    st.session_state.get('user_grade',
                                                                                                         'Grade 8'))
                            st.session_state.quiz_data = q_data
                            st.session_state.current_quiz_topic = final_quiz_topic
                        st.rerun()

        st.divider()

        # --- 4. NEW CHAT / RESET / SYSTEM (MOVED HERE) ---
        st.header("⚙️ Controls")
        
        # System/Setup Buttons (Essential functionality)
        if st.button("🔄 Sync Local Library"):
            with st.spinner("Scanning Library..."):
                if st.session_state.processor.process_local_library():
                    st.session_state.db_ready = True
                    st.success("Library Loaded!")
                    time.sleep(1)
                    st.rerun()

        uploaded_file = st.file_uploader("📂 Upload PDF (Educational Only)", type="pdf")
        if uploaded_file:
            if f"processed_{uploaded_file.name}" not in st.session_state:
                with st.spinner("🧠 Analyzing content..."):
                    if st.session_state.processor.process_uploaded_file(uploaded_file):
                        st.session_state.db_ready = True
                        st.session_state[f"processed_{uploaded_file.name}"] = True
                        st.success("Educational Material Accepted!")
                        time.sleep(1)
                        st.rerun()
        
        # New Chat / Reset Buttons
        col_new, col_reset = st.columns([1, 1])
        with col_new:
            if st.button("✨ New Chat", type="primary", use_container_width=True):
                new_chat()
                st.rerun()

        with col_reset:
            if st.button("🗑️ Reset", use_container_width=True):
                st.session_state.messages = []
                st.session_state.saved_chats[st.session_state.current_chat_id] = []
                save_history_to_disk(st.session_state.current_user)
                st.rerun()

        if st.button("♻️ Reset DB", use_container_width=True):
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
            st.session_state.db_ready = False
            st.rerun()

        st.divider()

        # --- 5. CHAT HISTORY (RECENT TOPICS) ---
        with st.expander("📜 Recent Topics", expanded=True):
            chat_ids = list(st.session_state.saved_chats.keys())
            if not chat_ids:
                st.caption("No history yet.")

            for session_id in reversed(chat_ids):
                msgs = st.session_state.saved_chats[session_id]
                title = get_chat_title(msgs)
                col_chat, col_menu = st.columns([0.8, 0.2])

                with col_chat:
                    btn_type = "primary" if session_id == st.session_state.current_chat_id else "secondary"
                    if st.button(f"🗨️ {title}", key=f"btn_{session_id}", type=btn_type, use_container_width=True):
                        load_chat(session_id)
                        st.rerun()

                with col_menu:
                    with st.popover("⋮"):
                        st.caption("Manage Chat")
                        if st.button("🗑️ Delete", key=f"del_{session_id}", type="primary", use_container_width=True):
                            delete_chat(session_id)
                            st.rerun()

        st.divider()

        # --- 6. DASHBOARD (MOVED TO BOTTOM) ---
        with st.expander("📊 Progress Dashboard"):
            scores = load_json_db(QUIZ_FILE).get(st.session_state.current_user, [])
            if scores:
                df = pd.DataFrame(scores)
                
                # Cleanup Data (handle old records)
                if 'total' not in df.columns: df['total'] = 3
                if 'obtained' not in df.columns and 'score' in df.columns:
                    df = df.rename(columns={"score": "obtained"})
                
                # 1. TABLE
                st.markdown("### 📝 Recent Scores")
                display_cols = ["topic", "obtained", "total", "date"]
                st.dataframe(df[display_cols], use_container_width=True)
                
                # 2. CUSTOM ALTAIR CHART
                st.markdown("### 📈 Performance Visualizer")
                
                # Base: Total Marks (Dotted Line / Hollow Bar)
                total_chart = alt.Chart(df).mark_bar(
                    stroke='#E0E0E0' if "Light" in theme_choice else '#FFFFFF', 
                    strokeWidth=2,
                    strokeDash=[4, 4], # Dotted effect
                    fill=None,
                    opacity=0.6,
                    cornerRadiusEnd=4
                ).encode(
                    x=alt.X('topic:N', title=None, axis=alt.Axis(labels=False)),
                    y=alt.Y('total:Q', title='Score', scale=alt.Scale(domain=[0, df['total'].max()])),
                )

                # Overlay: Obtained Marks (Solid Gold Bar)
                obtained_chart = alt.Chart(df).mark_bar(
                    color='#D4AF37',
                    width=15, 
                    cornerRadiusEnd=4
                ).encode(
                    x=alt.X('topic:N', title='Quiz Topic', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('obtained:Q'),
                    tooltip=['topic', 'obtained', 'total', 'date']
                )

                # Combine
                final_chart = (total_chart + obtained_chart).properties(height=250)
                st.altair_chart(final_chart, use_container_width=True)
                
                st.caption(f"Total Quizzes Taken: {len(scores)}")
            else:
                st.info("Take a quiz to see your progress here!")


        # --- QUIZ DISPLAY (Main Content Area Logic) ---
        if st.session_state.get('quiz_data'):
            st.divider()
            st.subheader(f"📝 Quiz: {st.session_state.get('current_quiz_topic', 'Topic')}")
            quiz_data = st.session_state.quiz_data

            score = 0
            for i, q in enumerate(quiz_data):
                st.markdown(f"**{i + 1}. {q['question']}**")
                choice = st.radio(f"Select Answer for Q{i + 1}", q['options'], key=f"quiz_q_{i}", index=None,
                                  label_visibility="collapsed")

                if choice:
                    if choice == q['answer']:
                        st.success("✅ Correct!")
                        score += 1
                    else:
                        st.error(f"❌ Wrong! The correct answer is: {q['answer']}")
                        st.info(f"💡 **Explanation:** {q['explanation']}")
                st.write("---")

            if st.button("Finish & Save Score", use_container_width=True):
                save_quiz_score(st.session_state.current_quiz_topic, score, len(quiz_data))
                st.success("Score Saved to Dashboard!")
                st.session_state.quiz_data = None
                st.rerun()

    # 4. MAIN CHAT AREA
    st.title("NexGen Tutor")

    if not st.session_state.db_ready:
        if st.session_state.processor.load_from_disk():
            st.session_state.db_ready = True
        else:
            st.info("👈 Please Sync Library or Upload a PDF to begin!")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"):
                if os.path.exists(msg["audio"]):
                    st.audio(msg["audio"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.saved_chats[st.session_state.current_chat_id] = st.session_state.messages
        save_history_to_disk(st.session_state.current_user)

        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.db_ready:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()

                try:
                    retriever = st.session_state.processor.vectordb_doc.as_retriever()
                    docs = retriever.invoke(prompt)

                    grade_str = st.session_state.get('user_grade', 'Grade 6')
                    subject = st.session_state.get('user_subject', 'Science')

                    try:
                        grade_num = int(''.join(filter(str.isdigit, grade_str)))
                    except:
                        grade_num = 6

                    if grade_num <= 7:
                        complexity_instruction = """
                        **Target Audience: Grade 6-7 Student.**
                        - **Definition:** Simple, precise, no jargon.
                        - **Tone:** Friendly and encouraging.
                        """
                    elif grade_num == 8:
                        complexity_instruction = """
                        **Target Audience: Grade 8 Student.**
                        - **Definition:** Standard educational definition with examples.
                        - **Tone:** Informative.
                        """
                    else:
                        complexity_instruction = """
                        **Target Audience: Grade 9-10 Student.**
                        - **Definition:** Academic, technical, exam-focused.
                        - **Tone:** Professional.
                        """

                    chat_history_str = "\n".join(
                        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])

                    template_text = """
                    You are an AI tutor for {grade} {subject}.
                    Instruction: {complexity_instruction}

                    **MANDATORY RESPONSE STRUCTURE:**
                    1. **Core Concept:** Definition.
                    2. **Details:** Purpose/Location/Phases.
                    3. **Comparison:** Markdown Table (if applicable).
                    4. **Real-World Example:** Relatable analogy.
                    5. **MATH & PHYSICS RULE (CRITICAL):**
                    - If the question is mathematical (Maths/Physics), you MUST solve it **Step-by-Step**.
                    - **Structure for Math:**
                      1. **Formula:** State the formula clearly using LaTeX (e.g., $E=mc^2$).
                      2. **Given:** List known values.
                      3. **Substitution:** Show values plugged into the formula.
                      4. **Calculation:** Show steps.
                      5. **Result:** Final Answer.
                    - **LaTeX:** Wrap ALL math equations/symbols in dollar signs ($). Example: $x^2 + y^2 = r^2$.
                    6. **NO IMAGES.** Text only.

                    Context: {context}
                    Chat History: {chat_history}
                    Question: {input}
                    """

                    custom_prompt = ChatPromptTemplate.from_template(template_text)
                    combine_docs_chain = create_stuff_documents_chain(st.session_state.processor.llm, custom_prompt)
                    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

                    response = retrieval_chain.invoke({
                        "input": prompt,
                        "context": docs,
                        "chat_history": chat_history_str,
                        "grade": grade_str,
                        "subject": subject,
                        "complexity_instruction": complexity_instruction
                    })

                    full_response = response["answer"]
                    response_placeholder.markdown(full_response)

                    audio_file = text_to_audio(full_response)
                    if audio_file:
                        st.audio(audio_file)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "audio": audio_file
                    })
                    st.session_state.saved_chats[st.session_state.current_chat_id] = st.session_state.messages
                    save_history_to_disk(st.session_state.current_user)

                except Exception as e:
                    st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
