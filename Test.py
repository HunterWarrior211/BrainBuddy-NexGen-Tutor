import os
import sys
import shutil
import time
import tempfile
import uuid
import json
import asyncio
import hashlib
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

# --- THIRD PARTY IMPORTS ---
import streamlit as st
import pandas as pd
import altair as alt
import edge_tts

# --- LANGCHAIN IMPORTS ---
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"Critical Dependency Error: {e}. Please ensure requirements.txt is installed.")
    st.stop()

# --- 1. ROBUST SQLITE FIX FOR CLOUD ENVIRONMENTS ---
# Essential for running ChromaDB on Linux-based cloud servers (Streamlit Cloud)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

# ==========================================
# ⚙️ CONFIGURATION & CONSTANTS
# ==========================================
# Application Metadata
APP_TITLE = "NexGen Tutor | Enterprise Learning Platform"
APP_ICON = "🎓"
VERSION = "2.5.0-PRO"

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploaded_books")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
USERS_FILE = os.path.join(BASE_DIR, "users.json")
QUIZ_FILE = os.path.join(BASE_DIR, "quiz_scores.json")

# Ensure critical directories exist
for directory in [RESOURCES_DIR, UPLOAD_DIR, DB_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logger Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 🎨 UI & CSS ENGINE
# ==========================================
def inject_custom_css(theme: str = "Dark"):
    """
    Injects high-performance, professional-grade CSS into the Streamlit app.
    Supports dynamic theming and advanced scrollbar effects.
    """
    
    # Common Animations & Fonts
    common_css = """
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Exo+2:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');
    
    /* Global Animations */
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes glowPulse { 0% { box-shadow: 0 0 5px rgba(212, 175, 55, 0.2); } 50% { box-shadow: 0 0 15px rgba(212, 175, 55, 0.6); } 100% { box-shadow: 0 0 5px rgba(212, 175, 55, 0.2); } }
    """

    if theme == "Dark Mode 🌑":
        theme_css = """
        /* --- DARK THEME VARIABLES --- */
        :root {
            --bg-color: #0A0E14;
            --sidebar-bg: #141A24;
            --text-color: #EAEAEA;
            --accent-gold: #D4AF37;
            --accent-glow: rgba(212, 175, 55, 0.5);
            --card-bg: #1B1F28;
            --border-color: rgba(255, 255, 255, 0.1);
        }

        /* App Background */
        .stApp {
            background-color: var(--bg-color);
            background-image: radial-gradient(#1B1F28 1px, transparent 1px);
            background-size: 40px 40px;
            color: var(--text-color);
        }

        /* --- HYPER-GLOW GOLDEN SCROLLBAR --- */
        ::-webkit-scrollbar { width: 12px; height: 12px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-corner { background: transparent; }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #BF953F, #FCF6BA, #B38728, #AA771C);
            border-radius: 12px;
            border: 3px solid var(--bg-color); /* Creates floating effect */
            box-shadow: 0 0 10px var(--accent-gold);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #FFD700, #FFFACD, #FFD700);
            box-shadow: 0 0 20px #FFD700;
            border: 2px solid #FFF;
        }

        /* Typography */
        h1, h2, h3 {
            font-family: 'Cinzel', serif !important;
            color: var(--accent-gold) !important;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            text-shadow: 2px 2px 4px #000;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            border-right: 2px solid var(--accent-gold);
            box-shadow: 10px 0 30px rgba(0,0,0,0.5);
        }
        section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
            color: #FFF !important;
            font-family: 'Exo 2', sans-serif;
            font-weight: 600;
        }

        /* Inputs & Cards */
        .stTextInput > div > div > input, .stSelectbox > div > div {
            background-color: var(--card-bg);
            color: var(--text-color);
            border: 1px solid var(--accent-gold);
            border-radius: 12px;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #D4AF37 0%, #B8962E 100%);
            color: #0A0E14;
            font-family: 'Cinzel', serif;
            font-weight: 800;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px var(--accent-glow);
            color: #000;
        }
        """
    else:
        theme_css = """
        /* --- LIGHT THEME VARIABLES --- */
        :root {
            --bg-color: #F8F9FB;
            --sidebar-bg: #FFFFFF;
            --text-color: #2B2E34;
            --accent-gold: #2B2E34; /* Dark for contrast */
            --card-bg: #FFFFFF;
        }

        .stApp {
            background-color: var(--bg-color);
            background-image: radial-gradient(#BFC3C9 1.5px, transparent 1.5px);
            background-size: 30px 30px;
            color: var(--text-color);
        }

        /* Gold Scrollbar for Light Mode (High Contrast) */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: #D4AF37;
            border-radius: 10px;
            border: 2px solid var(--bg-color);
        }
        ::-webkit-scrollbar-thumb:hover { background: #B8962E; }

        h1, h2, h3 {
            font-family: 'Cinzel', serif !important;
            color: #2B2E34 !important;
            text-transform: uppercase;
            font-weight: 900;
        }

        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #BFC3C9 0%, #9FA4AA 100%);
            color: #2B2E34;
            font-family: 'Cinzel', serif;
            font-weight: 800;
            border-radius: 10px;
            border: none;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .stButton > button:hover { background: #2B2E34; color: #FFF; }
        """

    # Chat Bubble Styling (Rounded & Modern)
    chat_css = """
    .stChatMessage {
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        animation: fadeInUp 0.5s ease-out forwards;
    }
    div[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #8BE9FD;
    }
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: rgba(212, 175, 55, 0.05);
        border-left: 4px solid #D4AF37;
    }
    """

    st.markdown(f"<style>{common_css}{theme_css}{chat_css}</style>", unsafe_allow_html=True)


# ==========================================
# 🔐 AUTHENTICATION & DATA MANAGER
# ==========================================
class DataManager:
    """Handles all JSON file operations securely."""
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Corrupted JSON at {filepath}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return {}

    @staticmethod
    def save_json(filepath: str, data: Dict):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save {filepath}: {e}")

class AuthManager:
    """Manages User Authentication logic."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(str.encode(password)).hexdigest()

    @staticmethod
    def login(username, password) -> bool:
        users = DataManager.load_json(USERS_FILE)
        hashed = AuthManager.hash_password(password)
        return username in users and users[username] == hashed

    @staticmethod
    def register(username, password) -> bool:
        users = DataManager.load_json(USERS_FILE)
        if username in users:
            return False
        users[username] = AuthManager.hash_password(password)
        DataManager.save_json(USERS_FILE, users)
        return True


# ==========================================
# 🧠 AI CORE: PROMPT ENGINEERING & RAG
# ==========================================
class BrainEngine:
    """
    The Core AI Logic. Handles Document Loading, Vector Store,
    Embedding generation, and Advanced Prompt Construction.
    """
    def __init__(self):
        self.vector_db = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self._setup_api_key()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.7,
            max_retries=3
        )

    def _setup_api_key(self):
        """Securely retrieves API Key from environment or secrets."""
        if "GOOGLE_API_KEY" not in os.environ:
            try:
                if "GOOGLE_API_KEY" in st.secrets:
                    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                else:
                    # Attempt to load from local file for dev
                    secrets_path = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
                    if os.path.exists(secrets_path):
                        with open(secrets_path, "r") as f:
                            content = f.read()
                            match = re.search(r'GOOGLE_API_KEY\s*=\s*["\'](.+?)["\']', content)
                            if match:
                                os.environ["GOOGLE_API_KEY"] = match.group(1)
            except Exception:
                pass
        
        if "GOOGLE_API_KEY" not in os.environ:
            st.error("🚨 Critical Error: Google API Key not found. System cannot start.")
            st.stop()

    def load_database(self) -> bool:
        """Loads the existing vector database from disk."""
        if os.path.exists(PERSIST_DIR):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                self.vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
                if self.vector_db._collection.count() > 0:
                    return True
            except Exception as e:
                logger.error(f"DB Load Error: {e}")
        return False

    def ingest_library(self) -> bool:
        """Reads all PDFs in resources folder and rebuilds database."""
        if not os.path.exists(BOOKS_FOLDER):
            return False
        
        pdf_files = [f for f in os.listdir(BOOKS_FOLDER) if f.lower().endswith(".pdf")]
        if not pdf_files:
            return False

        all_documents = []
        progress_bar = st.progress(0, "Scanning Library...")
        
        for idx, file in enumerate(pdf_files):
            file_path = os.path.join(BOOKS_FOLDER, file)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_documents.extend(docs)
                progress_bar.progress((idx + 1) / len(pdf_files), f"Processed {file}")
            except Exception as e:
                logger.warning(f"Skipped {file}: {e}")
        
        progress_bar.empty()
        
        if not all_documents:
            return False

        # Create/Update Vector DB
        splits = self.text_splitter.split_documents(all_documents)
        return self._batch_insert(splits)

    def process_upload(self, uploaded_file) -> bool:
        """Handles user uploaded PDF, verifies content, updates DB."""
        temp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            # Smart Content Verification
            verify_prompt = f"Is the following text educational/academic? Reply ONLY 'YES' or 'NO'.\n\nText: {docs[0].page_content[:500]}"
            check_msg = self.llm.invoke([HumanMessage(content=verify_prompt)])
            
            if "NO" in check_msg.content.upper().strip():
                st.error("🚫 Upload Rejected: Content does not appear to be educational.")
                os.remove(temp_path)
                return False

            splits = self.text_splitter.split_documents(docs)
            os.remove(temp_path)
            return self._batch_insert(splits)
            
        except Exception as e:
            st.error(f"Processing Error: {e}")
            return False

    def _batch_insert(self, splits) -> bool:
        """Inserts documents in batches to avoid timeouts."""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Initialize if needed
        if self.vector_db is None:
            if os.path.exists(PERSIST_DIR):
                self.vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            else:
                # Start with first batch
                self.vector_db = Chroma.from_documents(splits[:20], embeddings, persist_directory=PERSIST_DIR)
                splits = splits[20:]

        # Batch process remaining
        batch_size = 20
        total = len(splits)
        if total > 0:
            bar = st.progress(0, "Memorizing Knowledge...")
            for i in range(0, total, batch_size):
                batch = splits[i : i + batch_size]
                self.vector_db.add_documents(batch)
                bar.progress(min((i + batch_size) / total, 1.0))
                time.sleep(0.05) 
            bar.empty()
        
        return True

    def generate_quiz_json(self, topic: str, grade: str) -> List[Dict]:
        """Generates a structured JSON quiz."""
        prompt = f"""
        Act as an expert exam setter for {grade} level.
        Topic: {topic}
        
        Task: Create a 3-question Multiple Choice Quiz (MCQ).
        
        Constraints:
        1. Output MUST be valid raw JSON. No markdown formatting (```json).
        2. Format: List of dictionaries with keys: "question", "options" (list of 4 strings), "answer" (string), "explanation".
        3. Questions should test conceptual understanding appropriate for {grade}.
        
        JSON:
        """
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            # Clean possible markdown
            clean_json = response.content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            logger.error(f"Quiz Gen Error: {e}")
            return []

    def extract_topics(self, history_text: str) -> List[str]:
        """Extracts teachable topics from conversation history."""
        prompt = f"""
        Analyze the following educational chat history.
        Identify the top 3 specific academic topics discussed (e.g. "Photosynthesis", "Pythagoras Theorem").
        Return ONLY a comma-separated list of strings.
        
        Chat:
        {history_text[-2000:]}
        """
        res = self.llm.invoke([HumanMessage(content=prompt)])
        return [t.strip() for t in res.content.split(',') if t.strip()]


# ==========================================
# 🔊 AUDIO ENGINE
# ==========================================
class AudioEngine:
    @staticmethod
    def text_to_speech(text: str) -> Optional[str]:
        """Converts text to speech using Edge TTS (Async wrapper)."""
        try:
            # Remove Markdown symbols for cleaner reading
            clean_text = re.sub(r'[*_#`]', '', text)
            
            async def _generate():
                communicate = edge_tts.Communicate(clean_text, "en-GB-SoniaNeural")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    await communicate.save(fp.name)
                    return fp.name

            return asyncio.run(_generate())
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            return None


# ==========================================
# 🎮 SESSION & STATE MANAGEMENT
# ==========================================
def init_session_state():
    """Initializes all session variables ensuring robustness."""
    defaults = {
        "current_user": None,
        "saved_chats": {},
        "current_chat_id": None,
        "messages": [],
        "processor": None,
        "db_ready": False,
        "quiz_data": None,
        "current_quiz_topic": None,
        "user_grade": "Grade 9",
        "user_subject": "Physics"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Lazy load processor
    if st.session_state.processor is None:
        st.session_state.processor = BrainEngine()

def load_user_data():
    """Loads specific user data into session."""
    user = st.session_state.current_user
    if user:
        history = DataManager.load_json(HISTORY_FILE)
        st.session_state.saved_chats = history.get(user, {})
        
        # Initialize new chat if empty
        if not st.session_state.saved_chats:
            new_chat()
        
        # Set initial chat pointer if needed
        if not st.session_state.current_chat_id and st.session_state.saved_chats:
            st.session_state.current_chat_id = list(st.session_state.saved_chats.keys())[-1]
            st.session_state.messages = st.session_state.saved_chats[st.session_state.current_chat_id]

def new_chat():
    """Creates a fresh conversation thread."""
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat_id = chat_id
    st.session_state.saved_chats[chat_id] = []
    st.session_state.messages = []
    st.session_state.quiz_data = None
    save_current_state()

def save_current_state():
    """Persists current chat to disk immediately."""
    if st.session_state.current_user and st.session_state.current_chat_id:
        st.session_state.saved_chats[st.session_state.current_chat_id] = st.session_state.messages
        
        all_history = DataManager.load_json(HISTORY_FILE)
        all_history[st.session_state.current_user] = st.session_state.saved_chats
        DataManager.save_json(HISTORY_FILE, all_history)

def get_smart_title(messages):
    """Generates a dynamic title based on the first user message."""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            return " ".join(content.split()[:4]) + "..."
    return "New Conversation"


# ==========================================
# 🖥️ MAIN UI APPLICATION
# ==========================================
def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()

    # --- 1. LOGIN SCREEN ---
    if not st.session_state.current_user:
        inject_custom_css("Dark Mode 🌑") # Login always dark for sleek look
        
        col_L, col_M, col_R = st.columns([1, 2, 1])
        with col_M:
            st.markdown(f"<h1 style='text-align: center;'>{APP_ICON} NexGen Login</h1>", unsafe_allow_html=True)
            
            tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Sign Up"])
            
            with tab_login:
                with st.form("login_form"):
                    u_name = st.text_input("Username")
                    p_word = st.text_input("Password", type="password")
                    if st.form_submit_button("Access Portal", use_container_width=True):
                        if AuthManager.login(u_name, p_word):
                            st.session_state.current_user = u_name
                            st.rerun()
                        else:
                            st.error("Invalid credentials.")

            with tab_signup:
                with st.form("signup_form"):
                    new_u = st.text_input("Choose Username")
                    new_p = st.text_input("Choose Password", type="password")
                    if st.form_submit_button("Create Account", use_container_width=True):
                        if AuthManager.register(new_u, new_p):
                            st.success("Account created! Please log in.")
                        else:
                            st.error("Username taken.")
        return

    # --- LOAD USER DATA IF LOGGED IN ---
    load_user_data()

    # --- SIDEBAR LAYOUT (REORDERED & OPTIMIZED) ---
    with st.sidebar:
        # A. USER PROFILE
        st.markdown(f"### 👤 {st.session_state.current_user}")
        if st.button("Log Out", type="secondary"):
            st.session_state.clear()
            st.rerun()
        
        st.divider()

        # B. STUDY SETTINGS (PRIORITY 1)
        st.markdown("### 📚 Academic Settings")
        
        # Grade Selection
        grade_options = [f"Grade {i}" for i in range(6, 11)]
        st.session_state.user_grade = st.selectbox(
            "Target Level", 
            grade_options, 
            index=grade_options.index(st.session_state.user_grade) if st.session_state.user_grade in grade_options else 0
        )
        
        # Subject Selection
        subject_options = ["Mathematics", "Physics", "Biology", "Chemistry", "History", "Computer Science", "Geography"]
        st.session_state.user_subject = st.selectbox(
            "Current Subject", 
            subject_options,
            index=subject_options.index(st.session_state.user_subject) if st.session_state.user_subject in subject_options else 1
        )

        st.divider()

        # C. INTELLIGENT QUIZ GENERATOR (PRIORITY 2)
        with st.expander("🧠 AI Quiz Generator", expanded=False):
            if st.button("🚀 Analyze Chat for Topics", use_container_width=True):
                if len(st.session_state.messages) > 1:
                    hist_str = "\n".join([m['content'] for m in st.session_state.messages])
                    with st.spinner("Analyzing conversation..."):
                        topics = st.session_state.processor.extract_topics(hist_str)
                        st.session_state.detected_topics = topics
                else:
                    st.warning("Chat deeper to extract topics!")

            if "detected_topics" in st.session_state:
                topic_choice = st.radio("Select Topic:", st.session_state.detected_topics + ["Custom Topic..."])
                
                final_topic = topic_choice
                if topic_choice == "Custom Topic...":
                    final_topic = st.text_input("Enter Topic Name:")
                
                if st.button("Start Quiz", type="primary", use_container_width=True):
                    with st.spinner(f"Generating {st.session_state.user_grade} Quiz..."):
                        q_data = st.session_state.processor.generate_quiz_json(final_topic, st.session_state.user_grade)
                        if q_data:
                            st.session_state.quiz_data = q_data
                            st.session_state.current_quiz_topic = final_topic
                            st.rerun()
                        else:
                            st.error("AI failed to generate quiz. Try again.")

        st.divider()

        # D. CHAT MANAGEMENT
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            if st.button("✨ New", use_container_width=True):
                new_chat()
                st.rerun()
        with col_n2:
            if st.button("🗑️ Wipe", use_container_width=True):
                st.session_state.messages = []
                save_current_state()
                st.rerun()

        # Chat History List
        with st.expander("📜 History Archives"):
            chats = st.session_state.saved_chats
            # Show reversed (newest first)
            for chat_id in list(chats.keys())[::-1]:
                title = get_smart_title(chats[chat_id])
                c_btn, c_del = st.columns([0.8, 0.2])
                with c_btn:
                    style = "primary" if chat_id == st.session_state.current_chat_id else "secondary"
                    if st.button(title, key=chat_id, type=style, use_container_width=True):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.messages = chats[chat_id]
                        st.rerun()
                with c_del:
                    if st.button("✕", key=f"del_{chat_id}"):
                        del st.session_state.saved_chats[chat_id]
                        if st.session_state.current_chat_id == chat_id:
                            new_chat()
                        save_current_state()
                        st.rerun()

        st.divider()

        # E. SYSTEM CONTROL (BOTTOM)
        st.markdown("### ⚙️ System")
        theme_sel = st.radio("Theme Mode", ["Dark Mode 🌑", "Light Mode ☀️"], horizontal=True)
        inject_custom_css(theme_sel)

        if st.button("🔄 Sync Library", use_container_width=True):
            if st.session_state.processor.ingest_library():
                st.session_state.db_ready = True
                st.success("Knowledge Base Updated!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Library Sync Failed (Check folders)")

        # Upload Area
        uploaded = st.file_uploader("Upload Knowledge (PDF)", type="pdf")
        if uploaded:
            key = f"proc_{uploaded.name}"
            if key not in st.session_state:
                with st.spinner("Analyzing Document Structure..."):
                    if st.session_state.processor.process_upload(uploaded):
                        st.session_state.db_ready = True
                        st.session_state[key] = True
                        st.success("Ingested Successfully!")
                        time.sleep(1)
                        st.rerun()

        # Performance Dashboard (Mini)
        with st.expander("📊 Performance Stats"):
            try:
                scores = DataManager.load_json(QUIZ_FILE).get(st.session_state.current_user, [])
                if scores:
                    df = pd.DataFrame(scores)
                    # Normalize columns
                    if 'total' not in df.columns: df['total'] = 3
                    if 'score' in df.columns and 'obtained' not in df.columns: 
                        df.rename(columns={'score': 'obtained'}, inplace=True)
                    
                    # Altair Chart
                    chart = alt.Chart(df[-5:]).mark_bar(
                        color='#D4AF37', 
                        cornerRadiusTopLeft=5, 
                        cornerRadiusTopRight=5
                    ).encode(
                        x=alt.X('topic', axis=None, title="Last 5 Quizzes"),
                        y=alt.Y('obtained', title="Score"),
                        tooltip=['topic', 'obtained', 'total']
                    ).properties(height=150)
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("No quizzes taken yet.")
            except Exception:
                st.caption("Error loading stats.")

    # --- MAIN CONTENT AREA ---
    
    # 1. HEADER
    st.title("NexGen Tutor")
    
    # Check Database Status
    if not st.session_state.db_ready:
        if st.session_state.processor.load_database():
            st.session_state.db_ready = True
        else:
            st.info("👋 Welcome! Please **Sync Library** or **Upload a PDF** in the sidebar to activate the AI Brain.")

    # 2. QUIZ INTERFACE (IF ACTIVE)
    if st.session_state.quiz_data:
        st.markdown(f"## 📝 Active Quiz: {st.session_state.current_quiz_topic}")
        st.markdown("---")
        
        score = 0
        quiz = st.session_state.quiz_data
        
        for idx, q in enumerate(quiz):
            st.markdown(f"**Question {idx+1}:** {q['question']}")
            # Use columns for better layout
            user_choice = st.radio(f"Select Answer for Q{idx+1}:", q['options'], key=f"qz_{idx}", label_visibility="collapsed")
            
            if user_choice:
                # Immediate Feedback UI
                if user_choice == q['answer']:
                    st.success("✅ Correct!")
                    score += 1
                else:
                    st.error(f"❌ Incorrect. The correct answer was: **{q['answer']}**")
                    st.info(f"💡 **Explanation:** {q['explanation']}")
            st.markdown("---")

        if st.button("Finish & Save Results", type="primary"):
            save_quiz_score(st.session_state.current_quiz_topic, score, len(quiz))
            st.balloons()
            st.success("Result Saved to Dashboard!")
            time.sleep(2)
            st.session_state.quiz_data = None
            st.rerun()

    # 3. CHAT DISPLAY
    # Render historical messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Audio player if audio exists
            if msg.get("audio") and os.path.exists(msg.get("audio")):
                st.audio(msg["audio"])

    # 4. CHAT INPUT & PROCESSING
    if prompt := st.chat_input(f"Ask about {st.session_state.user_subject} ({st.session_state.user_grade})..."):
        
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_current_state()
        
        # Render User Message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        if st.session_state.db_ready:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                
                try:
                    # --- ADVANCED PROMPT ENGINEERING LOGIC ---
                    grade = st.session_state.user_grade
                    subject = st.session_state.user_subject
                    
                    # Extract grade number safely
                    grade_num = int(re.search(r'\d+', grade).group()) if re.search(r'\d+', grade) else 9

                    # A. PEDAGOGICAL INSTRUCTIONS (Grade-Specific)
                    if grade_num == 6:
                        style_guide = "Tone: Friendly, encouraging, simple. Structure: Very short paragraphs. Use emojis. Max 3 bullet points. No complex jargon."
                    elif grade_num == 7:
                        style_guide = "Tone: Informative but accessible. Structure: Concise definitions. Focus on the 'Why'. Brief real-world analogy."
                    elif grade_num == 8:
                        style_guide = "Tone: Standard Academic. Structure: Clear definition, 1 paragraph explanation, 1 specific example. Moderate detail."
                    elif grade_num == 9:
                        style_guide = "Tone: Academic & Preparatory. Structure: Detailed theoretical background. Include formulas if Physics/Math. Mention variations."
                    else: # Grade 10+
                        style_guide = "Tone: Examination/Board Level. Structure: Comprehensive deep-dive. Use bullet points for key characteristics. Use technical terminology. Focus on scoring points in exams."

                    # B. SUBJECT GUARDRAIL
                    # Prevents hallucinations or off-topic answers
                    guardrail = f"""
                    CRITICAL INSTRUCTION:
                    You are explicitly a **{subject}** Tutor.
                    The user has asked: "{prompt}"
                    
                    1. If the question is about {subject}, answer it fully.
                    2. If the question is about a DIFFERENT subject (e.g. asking about History while in Math mode), politely REFUSE to answer and tell the user to switch the Subject in the Sidebar.
                    3. If the question is General (e.g. "Hello", "Help"), answer politely as a {subject} tutor.
                    """

                    # C. FORMATTING TEMPLATE
                    # Enforces the specific visual structure requested
                    formatting = """
                    RESPONSE FORMAT (Strict Markdown):
                    
                    1. **Core Concept:** (A bold, one-sentence definition)
                    
                    2. **Main Points:** - (Bullet point 1)
                       - (Bullet point 2 based on grade level depth)
                    
                    3. **Comparison:** (IF the question asks for a difference/comparison, you MUST output a Markdown Table. If not, skip this.)
                    
                    4. **Real-World Example:** (A relatable scenario for a student of this age)
                    
                    5. **Math/Physics Solver:** (Only if calculation is needed. Use LaTeX enclosed in $$ for all math formulas. Show step-by-step substitution.)
                    """

                    # Construct Final Prompt
                    retriever = st.session_state.processor.vector_db.as_retriever()
                    
                    # Get recent context for conversation flow
                    chat_history_txt = ""
                    for m in st.session_state.messages[-4:]:
                        chat_history_txt += f"{m['role'].capitalize()}: {m['content']}\n"

                    # Define the LangChain System Prompt
                    system_prompt_template = f"""
                    You are NexGen, an advanced AI Tutor specialized in {subject} for {grade} students.
                    
                    {guardrail}
                    
                    PEDAGOGY GUIDE:
                    {style_guide}
                    
                    {formatting}
                    
                    CONTEXT FROM TEXTBOOK:
                    {{context}}
                    
                    CHAT HISTORY:
                    {chat_history_txt}
                    """

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt_template),
                        ("human", "{input}")
                    ])

                    # Execution Chain
                    chain = create_stuff_documents_chain(
                        st.session_state.processor.llm, 
                        prompt_template
                    )
                    
                    retrieval_chain = create_retrieval_chain(retriever, chain)
                    
                    # Run Chain
                    response_dict = retrieval_chain.invoke({"input": prompt})
                    full_response = response_dict["answer"]
                    
                    # Display Text
                    placeholder.markdown(full_response)
                    
                    # Generate Audio
                    audio_path = AudioEngine.text_to_speech(full_response)
                    if audio_path:
                        st.audio(audio_path)
                    
                    # Save to History
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "audio": audio_path
                    })
                    save_current_state()

                except Exception as e:
                    logger.error(f"Generation Error: {e}")
                    st.error("I encountered an issue processing that request. Please try again.")

if __name__ == "__main__":
    main()
