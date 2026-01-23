"""
====================================================================================================
APPLICATION:   NexGen Tutor | Titanium Enterprise Edition
VERSION:       8.0.0 (Infinity Build)
ARCHITECT:     Principal AI Systems Engineer
FRAMEWORK:     Streamlit + LangChain + Tailwind CSS + ChromaDB
DESCRIPTION:   A hyper-advanced educational AI platform designed for extreme robustness.
               
               [FEATURES]
               - Real-time Token Streaming (Typewriter Effect)
               - Context-Aware Diagram Generation (

[Image of X]
)
               - Pedagogical Guardrails (Grade 6-10 specific logic)
               - Enterprise-grade Authentication & Logging
               - High-Fidelity UI/UX via Tailwind Injection
               - Persistent Session Management
====================================================================================================
"""

import os
import sys
import time
import json
import uuid
import re
import logging
import asyncio
import hashlib
import shutil
import tempfile
import threading
from enum import Enum
from datetime import datetime
from typing import List, Dict, Optional, Any, Generator, Union, Tuple

# --- CORE EXTERNAL DEPENDENCIES ---
# We wrap imports in try-except to provide user-friendly error messages if libs are missing.
try:
    import streamlit as st
    import pandas as pd
    import altair as alt
    import edge_tts
    import numpy as np
    
    # LangChain Ecosystem
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"❌ CRITICAL BOOT FAILURE: Missing Dependency. {e}")
    st.markdown("### 🛠️ Solution:")
    st.code("pip install -r requirements.txt", language="bash")
    st.stop()

# --- CLOUD COMPATIBILITY LAYER (SQLite Fix) ---
# Essential for running ChromaDB on Linux/Streamlit Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

# ==================================================================================================
# ⚙️ SYSTEM CONFIGURATION & CONSTANTS
# ==================================================================================================

class SystemConfig:
    """
    Centralized configuration controller for the application.
    Manages paths, constants, and environment variables.
    """
    APP_NAME = "NexGen Tutor"
    APP_TAGLINE = "Titanium Edition"
    APP_ICON = "🎓"
    
    # Filesystem Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
    UPLOAD_DIR = os.path.join(BASE_DIR, "temp_ingest")
    DB_DIR = os.path.join(BASE_DIR, "chroma_vector_store")
    
    # Data Persistence
    HISTORY_PATH = os.path.join(BASE_DIR, "data_history.json")
    USERS_PATH = os.path.join(BASE_DIR, "data_users.json")
    STATS_PATH = os.path.join(BASE_DIR, "data_stats.json")
    LOGS_PATH = os.path.join(BASE_DIR, "system.log")

    # AI Model Configurations
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "models/embedding-001"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SEARCH_K = 6 

    @classmethod
    def initialize_filesystem(cls):
        """Creates necessary directories and files if they don't exist."""
        directories = [cls.RESOURCES_DIR, cls.UPLOAD_DIR, cls.DB_DIR]
        for d in directories:
            os.makedirs(d, exist_ok=True)
        
        # Initialize JSON stores if missing
        for f in [cls.HISTORY_PATH, cls.USERS_PATH, cls.STATS_PATH]:
            if not os.path.exists(f):
                with open(f, 'w') as file:
                    json.dump({}, file)

# Initialize System
SystemConfig.initialize_filesystem()

# Configure Enterprise Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("NexGenCore")


# ==================================================================================================
# 🎨 UI FACTORY & TAILWIND CSS ENGINE
# ==================================================================================================

class Theme(Enum):
    DARK = "Dark Mode 🌑 (Cosmic Gold)"
    LIGHT = "Light Mode ☀️ (Platinum Silver)"

class UIEngine:
    """
    Renders high-fidelity UI components using injected CSS and Tailwind.
    Preserves specific fonts (Cinzel, Exo 2) and layouts requested.
    """

    @staticmethod
    def inject_core_styles(theme: str):
        """Injects Tailwind CDN and Custom CSS overrides."""
        
        # 1. TAILWIND CDN INJECTION (For structure)
        tailwind_cdn = """
        <script src="https://cdn.tailwindcss.com"></script>
        """
        st.markdown(tailwind_cdn, unsafe_allow_html=True)

        # 2. CORE CSS (FONTS & ANIMATIONS & SCROLLBAR)
        # This preserves your exact Golden Scrollbar and Font choices.
        core_css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Playfair+Display:ital,wght@0,700;1,400&family=Exo+2:wght@400;600;700;800&family=Inter:wght@400;600;800&family=Roboto+Mono&display=swap');

            /* --- ✨ HYPER-GLOW GOLDEN SCROLLBAR ✨ --- */
            ::-webkit-scrollbar { width: 10px; height: 10px; }
            ::-webkit-scrollbar-track { background: transparent; margin-block: 5px; }
            ::-webkit-scrollbar-corner { background: transparent; }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
                border-radius: 10px;
                border: 2px solid transparent; 
                background-clip: content-box;
                box-shadow: 0 0 10px rgba(212, 175, 55, 0.5); 
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(180deg, #FFD700, #FFFACD, #FFD700);
                box-shadow: 0 0 20px rgba(255, 215, 0, 0.9); 
                border: 1px solid #FFF; 
            }

            /* --- ANIMATIONS --- */
            @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            @keyframes goldPulse { 0% { text-shadow: 0 0 10px rgba(212, 175, 55, 0.2); } 50% { text-shadow: 0 0 25px rgba(212, 175, 55, 0.6); } 100% { text-shadow: 0 0 10px rgba(212, 175, 55, 0.2); } }
            @keyframes cosmicDrift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

            /* --- TYPOGRAPHY --- */
            h1, h2, h3 {
                font-family: 'Cinzel', serif !important;
                font-weight: 900 !important;
                text-transform: uppercase;
                letter-spacing: 1.5px;
            }
            
            p, li, span, div, label {
                font-family: 'Exo 2', sans-serif !important;
                font-weight: 600 !important; /* Bold as requested */
                font-size: 1.05rem;
                line-height: 1.7;
            }
            
            strong {
                font-weight: 900 !important;
            }

            /* --- ROUNDED CORNERS & UI ELEMENTS --- */
            .stButton>button {
                border-radius: 12px !important; /* Rounded */
                font-family: 'Cinzel', serif !important;
                font-weight: 900 !important;
                text-transform: uppercase;
                transition: all 0.3s ease;
                border: none;
            }
            
            .stTextInput>div>div>input, .stSelectbox>div>div {
                border-radius: 12px !important;
                padding: 10px;
            }
            
            /* Chat Bubbles */
            .stChatMessage {
                border-radius: 16px !important;
                padding: 20px;
                margin-bottom: 15px;
                border: 1px solid rgba(255,255,255,0.05);
                animation: fadeInUp 0.5s ease-out forwards;
            }
        </style>
        """

        # 3. THEME SPECIFIC VARIABLES
        if "Dark" in theme:
            theme_vars = """
            <style>
                .stApp {
                    background-color: #0A0E14;
                    background-image: radial-gradient(#1B1F28 1px, transparent 1px), linear-gradient(125deg, #0A0E14 0%, #11161F 40%, #0A0E14 100%);
                    background-size: 40px 40px, 200% 200%;
                    animation: cosmicDrift 20s ease infinite;
                    color: #EAEAEA;
                }
                
                h1, h2, h3 { color: #D4AF37 !important; text-shadow: 2px 2px 4px #000000; }
                strong { color: #D4AF37 !important; }
                
                /* Sidebar Dark */
                section[data-testid="stSidebar"] {
                    background-color: #141A24;
                    border-right: 1px solid #D4AF37;
                    box-shadow: 5px 0 15px rgba(0,0,0,0.5);
                }
                section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #E0E6ED !important; }
                
                /* Buttons Dark */
                .stButton>button {
                    background: linear-gradient(135deg, #D4AF37 0%, #B8962E 100%);
                    color: #0A0E14;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                }
                .stButton>button:hover { transform: scale(1.05); color: #000; box-shadow: 0 0 20px rgba(212, 175, 55, 0.6); }
                
                /* Chat Dark */
                .stChatMessage { background-color: #1B1F28; }
                div[data-testid="stChatMessage"]:nth-child(odd) { border-left: 4px solid #8BE9FD; }
                div[data-testid="stChatMessage"]:nth-child(even) { border-left: 4px solid #D4AF37; background-color: #151921; }
                
                /* Inputs Dark */
                .stTextInput>div>div>input, .stSelectbox>div>div {
                    background-color: #1B1F28;
                    color: #EAEAEA;
                    border: 1px solid #D4AF37;
                }
            </style>
            """
        else:
            theme_vars = """
            <style>
                .stApp {
                    background-color: #F8F9FB;
                    background-image: radial-gradient(#BFC3C9 1.5px, transparent 1.5px), linear-gradient(120deg, #F8F9FB 0%, #FFFFFF 50%, #E5E7EB 100%);
                    background-size: 30px 30px, 200% 200%;
                    animation: silverDrift 20s ease infinite;
                    color: #5F6368;
                }
                
                h1, h2, h3 { color: #2B2E34 !important; }
                strong { color: #2B2E34 !important; }
                
                /* Sidebar Light */
                section[data-testid="stSidebar"] {
                    background-color: #FFFFFF;
                    border-right: 1px solid #E5E7EB;
                    box-shadow: 5px 0 20px rgba(0,0,0,0.03);
                }
                section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label { color: #2B2E34 !important; }
                
                /* Buttons Light */
                .stButton>button {
                    background: linear-gradient(135deg, #BFC3C9 0%, #9FA4AA 100%);
                    color: #2B2E34;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                .stButton>button:hover { transform: scale(1.05); background: #2B2E34; color: #FFFFFF; }
                
                /* Chat Light */
                .stChatMessage { box-shadow: 0 4px 6px rgba(0,0,0,0.04); }
                div[data-testid="stChatMessage"]:nth-child(odd) { background-color: #F1F3F6; border: 1px solid #E5E7EB; color: #2B2E34; border-left: 4px solid #BFC3C9; }
                div[data-testid="stChatMessage"]:nth-child(even) { background-color: #FFFFFF; border: 1px solid #E5E7EB; color: #374151; border-left: 4px solid #2B2E34; }
                
                /* Inputs Light */
                .stTextInput>div>div>input, .stSelectbox>div>div {
                    background-color: #FFFFFF;
                    color: #2B2E34;
                    border: 2px solid #E5E7EB;
                }
            </style>
            """
        
        st.markdown(tailwind_cdn + core_css + theme_vars, unsafe_allow_html=True)


# ==================================================================================================
# 🔐 AUTHENTICATION & SECURITY MANAGER
# ==================================================================================================

class AuthManager:
    """
    Handles User Identity, Session Validation, and Credential Hashing.
    """
    
    @staticmethod
    def _load_db() -> Dict:
        if not os.path.exists(SystemConfig.USERS_PATH):
            return {}
        try:
            with open(SystemConfig.USERS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Auth DB Load Error: {e}")
            return {}

    @staticmethod
    def _save_db(data: Dict):
        try:
            with open(SystemConfig.USERS_PATH, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Auth DB Save Error: {e}")

    @staticmethod
    def hash_secret(secret: str) -> str:
        """SHA-256 Hashing for password storage."""
        return hashlib.sha256(secret.encode()).hexdigest()

    @classmethod
    def verify_login(cls, username, password) -> bool:
        db = cls._load_db()
        return username in db and db[username] == cls.hash_secret(password)

    @classmethod
    def create_user(cls, username, password) -> Tuple[bool, str]:
        db = cls._load_db()
        if username in db:
            return False, "User already exists."
        db[username] = cls.hash_secret(password)
        cls._save_db(db)
        return True, "Account created successfully."


# ==================================================================================================
# 🧠 BRAIN CORE: RAG, PROMPTING, & STREAMING ENGINE
# ==================================================================================================

class BrainCore:
    """
    The Intelligence Unit of NexGen Tutor.
    Manages the Vector Database, Document Processing, and Prompt Engineering.
    """
    
    def __init__(self):
        self.vector_store = None
        self.api_ready = self._init_api_keys()
        
        if self.api_ready:
            self.llm = ChatGoogleGenerativeAI(
                model=SystemConfig.LLM_MODEL,
                temperature=0.7, # Balanced creativity
                max_retries=3,
                streaming=True, # Enable Streaming
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(model=SystemConfig.EMBEDDING_MODEL)
    
    def _init_api_keys(self) -> bool:
        """Securely retrieves API Key from environment or Streamlit secrets."""
        key = os.environ.get("GOOGLE_API_KEY")
        
        if not key:
            try:
                if "GOOGLE_API_KEY" in st.secrets:
                    key = st.secrets["GOOGLE_API_KEY"]
                else:
                    # Dev Fallback
                    secrets_path = os.path.join(SystemConfig.BASE_DIR, ".streamlit", "secrets.toml")
                    if os.path.exists(secrets_path):
                        with open(secrets_path, "r") as f:
                            for line in f:
                                if "GOOGLE_API_KEY" in line:
                                    key = line.split("=")[1].strip().strip('"').strip("'")
            except: pass
        
        if key:
            os.environ["GOOGLE_API_KEY"] = key
            return True
        return False

    def load_vector_db(self) -> bool:
        """Initializes the Chroma Vector Store from disk."""
        if os.path.exists(SystemConfig.DB_DIR):
            try:
                self.vector_store = Chroma(
                    persist_directory=SystemConfig.DB_DIR,
                    embedding_function=self.embeddings
                )
                if self.vector_store._collection.count() > 0:
                    return True
            except Exception as e:
                logger.error(f"Vector DB Init Failed: {e}")
        return False

    def ingest_document(self, file_path: str) -> bool:
        """Reads a PDF, splits it, and embeds into Vector Store."""
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            if not docs: return False
            
            # Text Splitting
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=SystemConfig.CHUNK_SIZE, 
                chunk_overlap=SystemConfig.CHUNK_OVERLAP
            )
            splits = splitter.split_documents(docs)
            
            # Embedding in Batches
            batch_size = 50
            total_splits = len(splits)
            
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    splits[:batch_size], 
                    self.embeddings, 
                    persist_directory=SystemConfig.DB_DIR
                )
                splits = splits[batch_size:]
            
            if splits:
                for i in range(0, len(splits), batch_size):
                    batch = splits[i:i+batch_size]
                    self.vector_store.add_documents(batch)
                    time.sleep(0.1) # Rate limit protection
            
            return True
        except Exception as e:
            logger.error(f"Ingestion Error: {e}")
            return False

    def get_rag_chain(self, grade: str, subject: str) -> Any:
        """
        Constructs the Prompt Chain with Pedagogical Logic.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": SystemConfig.SEARCH_K})
        
        # --- DYNAMIC PROMPT CONSTRUCTION ---
        try:
            grade_num = int(re.search(r'\d+', grade).group())
        except:
            grade_num = 9

        # Pedagogical Strategy Selection
        if grade_num <= 7:
            style = "Tone: Fun, Simple, Encouraging. Structure: Short paragraphs, simple words, emojis. Use analogies."
        elif grade_num == 8:
            style = "Tone: Academic but accessible. Structure: Clear definition, one example, moderate detail."
        elif grade_num == 9:
            style = "Tone: High School Academic. Structure: Theory + Application. Detailed bullets."
        else:
            style = "Tone: Board Exam Preparation. Structure: Technical, Comprehensive, Keyword-focused, Exam-oriented."

        # The MASTER PROMPT (Following your exact requirements)
        template = """
        You are NexGen, an advanced AI Tutor for {subject} at {grade} level.
        
        [STRICT SUBJECT GUARDRAIL]
        You must ONLY answer questions related to {subject}.
        If the user asks about a different field (e.g. Asking History in a Math session), politely REFUSE and guide them to switch subjects.
        
        [PEDAGOGY INSTRUCTION]
        {style}
        
        [DIAGRAM TRIGGERING INSTRUCTION]
        Assess if the user's question would benefit from a visual aid (e.g. parts of a cell, circuit diagram, graph).
        If yes, insert a tag in the format: .
        Example: 

[Image of Plant Cell Structure]
, 

[Image of Pythagorean Theorem Proof]
.
        Place these tags naturally where the image should appear.
        
        [REQUIRED OUTPUT STRUCTURE]
        You must strictly follow this Markdown format:
        
        1. **Core Concept:** (A precise, bold definition of the topic).
        
        2. **Key Points:** - (Bullet Point 1: Detailed explanation suitable for {grade})
           - (Bullet Point 2: Mechanism/Process)
           - (Bullet Point 3: Context/Nuance)
        
        3. **Comparison:** (IF the question involves two concepts, e.g. Mitosis vs Meiosis, YOU MUST GENERATE A MARKDOWN TABLE. If not, output 'N/A').
        
        4. **Real-World Example:** (A relatable application or analogy for a student).
        
        5. **Math/Physics Solver:** (If the query is a problem, solve it step-by-step using LaTeX inside $$...$$. State Formula -> Given -> Substitute -> Result).

        [CONTEXT FROM TEXTBOOK]
        {{context}}
        
        [CHAT HISTORY]
        {history}
        
        [USER QUESTION]
        {{input}}
        """
        
        # Inject History String manually to safely handle curly braces
        history_str = ""
        if "messages" in st.session_state:
            for m in st.session_state.messages[-4:]:
                history_str += f"{m['role'].capitalize()}: {m['content']}\n"

        final_prompt_str = template.format(
            subject=subject,
            grade=grade,
            style=style,
            history=history_str
        )

        prompt = ChatPromptTemplate.from_template(final_prompt_str)
        
        # Build Chain
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        
        return rag_chain

    def analyze_topics(self, history_text: str) -> List[str]:
        """Extracts teachable topics for Quizzes."""
        prompt = f"Analyze this chat history. Return a comma-separated list of the top 3 academic topics discussed. History: {history_text[-1500:]}"
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)])
            return [t.strip() for t in res.content.split(',') if t.strip()]
        except: return []

    def create_quiz(self, topic: str, grade: str) -> List[Dict]:
        """Generates a JSON Quiz."""
        prompt = f"""
        Act as an expert exam setter for {grade}. Topic: {topic}.
        Create 3 Multiple Choice Questions (MCQs).
        
        RETURN ONLY RAW JSON. No Markdown formatting.
        Structure: [{{ "question": "...", "options": ["A","B","C","D"], "answer": "Option Text", "explanation": "..." }}]
        """
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)])
            # Clean text
            clean_text = res.content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Quiz Error: {e}")
            return []


# ==================================================================================================
# 🎮 SESSION STATE MANAGER
# ==================================================================================================

class SessionState:
    """Typed wrapper for Streamlit Session State."""
    
    @staticmethod
    def init():
        defaults = {
            "current_user": None,
            "theme": Theme.DARK,
            "messages": [],
            "saved_chats": {},
            "current_chat_id": None,
            "brain": None,
            "db_ready": False,
            "quiz_data": None,
            "processing_upload": False
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        
        # Lazy Init Brain
        if st.session_state.brain is None:
            st.session_state.brain = BrainCore()

    @staticmethod
    def load_user_data():
        if st.session_state.current_user:
            # Load History
            if os.path.exists(SystemConfig.HISTORY_PATH):
                with open(SystemConfig.HISTORY_PATH, "r") as f:
                    all_data = json.load(f)
                    st.session_state.saved_chats = all_data.get(st.session_state.current_user, {})
            
            # Init first chat if needed
            if not st.session_state.saved_chats:
                SessionState.new_chat()
            
            if not st.session_state.current_chat_id and st.session_state.saved_chats:
                st.session_state.current_chat_id = list(st.session_state.saved_chats.keys())[-1]
                st.session_state.messages = st.session_state.saved_chats[st.session_state.current_chat_id]

    @staticmethod
    def save_state():
        if st.session_state.current_user and st.session_state.current_chat_id:
            st.session_state.saved_chats[st.session_state.current_chat_id] = st.session_state.messages
            
            # Load full DB, update user, save back
            if os.path.exists(SystemConfig.HISTORY_PATH):
                with open(SystemConfig.HISTORY_PATH, "r") as f:
                    all_data = json.load(f)
            else:
                all_data = {}
            
            all_data[st.session_state.current_user] = st.session_state.saved_chats
            
            with open(SystemConfig.HISTORY_PATH, "w") as f:
                json.dump(all_data, f, indent=4)

    @staticmethod
    def new_chat():
        uid = str(uuid.uuid4())
        st.session_state.current_chat_id = uid
        st.session_state.saved_chats[uid] = []
        st.session_state.messages = []
        SessionState.save_state()


# ==================================================================================================
# 🖥️ VIEW CONTROLLER (MAIN APP LOOP)
# ==================================================================================================

def render_login_view():
    """Renders the Login/Signup Screen."""
    UIEngine.inject_core_styles("Dark") # Force Dark for cinematic login
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="text-center p-10">
            <h1 class="text-5xl font-black text-amber-400 mb-2">{SystemConfig.APP_ICON}</h1>
            <h1 class="text-4xl font-cinzel text-white mb-2">{SystemConfig.APP_NAME}</h1>
            <p class="text-slate-400 text-lg">{SystemConfig.APP_TAGLINE}</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_login, tab_signup = st.tabs(["🔐 SECURE LOGIN", "📝 NEW REGISTRATION"])
        
        with tab_login:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("AUTHENTICATE", use_container_width=True):
                    if AuthManager.verify_login(u, p):
                        st.session_state.current_user = u
                        st.rerun()
                    else:
                        st.error("Authentication Failed.")

        with tab_signup:
            with st.form("signup_form"):
                nu = st.text_input("Choose Username")
                np = st.text_input("Choose Password", type="password")
                if st.form_submit_button("REGISTER IDENTITY", use_container_width=True):
                    success, msg = AuthManager.create_user(nu, np)
                    if success: st.success(msg)
                    else: st.error(msg)

def render_main_interface():
    """Renders the primary application dashboard."""
    
    # --- 1. SIDEBAR NAVIGATION (ORDERED AS REQUESTED) ---
    with st.sidebar:
        # 1. Profile Section
        st.markdown(f"""
        <div class="p-4 bg-slate-800/50 rounded-xl border border-amber-500/30 mb-6">
            <div class="flex items-center space-x-3">
                <div class="w-10 h-10 rounded-full bg-amber-500 flex items-center justify-center text-black font-bold">
                    {st.session_state.current_user[0].upper()}
                </div>
                <div>
                    <p class="text-sm text-slate-400 font-bold">LOGGED IN AS</p>
                    <p class="text-white font-bold">{st.session_state.current_user}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("TERMINATE SESSION", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.divider()

        # 2. Academic Controls (Top Priority)
        st.markdown("<h3 class='text-amber-400 font-cinzel mb-2'>📚 ACADEMIC SETTINGS</h3>", unsafe_allow_html=True)
        st.session_state.grade = st.selectbox("TARGET LEVEL", [f"Grade {i}" for i in range(6, 11)])
        st.session_state.subject = st.selectbox("SUBJECT DOMAIN", ["Mathematics", "Physics", "Biology", "Chemistry", "Science", "History", "Computer Science"])

        st.divider()

        # 3. Quiz Module
        with st.expander("🧠 INTELLIGENT ASSESSMENT"):
            if st.button("ANALYZE CONTEXT", use_container_width=True):
                if len(st.session_state.messages) > 1:
                    hist = "\n".join([m['content'] for m in st.session_state.messages])
                    st.session_state.detected_topics = st.session_state.brain.analyze_topics(hist)
                else:
                    st.warning("Insufficient data.")
            
            if st.session_state.get("detected_topics"):
                topic = st.radio("SELECT FOCUS:", st.session_state.detected_topics + ["Custom..."])
                if topic == "Custom...": topic = st.text_input("Enter Topic")
                
                if st.button("GENERATE EXAM", type="primary", use_container_width=True):
                    with st.spinner("Compiling Neural Assessment..."):
                        q = st.session_state.brain.create_quiz(topic, st.session_state.grade)
                        st.session_state.quiz_data = q
                        st.session_state.q_topic = topic
                        st.rerun()

        # 4. Chat Management
        st.divider()
        col_new, col_wipe = st.columns(2)
        with col_new:
            if st.button("✨ NEW", use_container_width=True):
                SessionState.new_chat()
                st.rerun()
        with col_wipe:
            if st.button("🗑️ WIPE", use_container_width=True):
                st.session_state.messages = []
                SessionState.save_state()
                st.rerun()

        # 5. History List
        with st.expander("📜 DATA ARCHIVES"):
            for cid in reversed(list(st.session_state.saved_chats.keys())):
                msgs = st.session_state.saved_chats[cid]
                title = "New Conversation"
                for m in msgs:
                    if m['role'] == 'user':
                        title = " ".join(m['content'].split()[:4]) + "..."
                        break
                
                c1, c2 = st.columns([0.8, 0.2])
                if c1.button(title, key=cid, use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.session_state.messages = msgs
                    st.rerun()
                if c2.button("✕", key=f"del_{cid}"):
                    del st.session_state.saved_chats[cid]
                    SessionState.save_state()
                    st.rerun()

        # 6. System Controls (Bottom)
        st.divider()
        st.markdown("<h3 class='text-amber-400 font-cinzel mb-2'>⚙️ SYSTEM CONTROL</h3>", unsafe_allow_html=True)
        
        # Theme Toggle
        theme_mode = st.radio("VISUAL INTERFACE", [Theme.DARK.value, Theme.LIGHT.value], horizontal=True)
        st.session_state.theme = theme_mode
        
        # Library Sync
        if st.button("SYNC KNOWLEDGE BASE", use_container_width=True):
            if st.session_state.brain.ingest_document(SystemConfig.RESOURCES_DIR): # Simplified call for demo
                st.session_state.db_ready = True
                st.success("Database Synchronized.")
                time.sleep(1); st.rerun()
        
        # Upload
        uploaded_file = st.file_uploader("INGEST DOCUMENT (PDF)", type="pdf")
        if uploaded_file:
            key = f"proc_{uploaded_file.name}"
            if key not in st.session_state:
                with st.spinner("Processing Vector Embeddings..."):
                    path = os.path.join(SystemConfig.UPLOAD_DIR, uploaded_file.name)
                    with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                    
                    if st.session_state.brain.ingest_document(path):
                        st.session_state.db_ready = True
                        st.session_state[key] = True
                        st.success("Ingestion Complete.")
                        os.remove(path)
                        time.sleep(1); st.rerun()

    # --- 2. MAIN CONTENT AREA ---
    UIEngine.inject_core_styles(st.session_state.theme)
    
    st.markdown(f"<h1 class='text-center mb-4'>{SystemConfig.APP_NAME} <span class='text-sm text-slate-500'>{SystemConfig.APP_TAGLINE}</span></h1>", unsafe_allow_html=True)

    # DB Status Check
    if not st.session_state.db_ready:
        if st.session_state.brain.load_vector_db():
            st.session_state.db_ready = True
        else:
            st.warning("⚠️ SYSTEM ALERT: Neural Core Offline. Please Sync Library or Upload Data.")

    # --- 3. QUIZ RENDERER (IF ACTIVE) ---
    if st.session_state.get("quiz_data"):
        st.markdown(f"""
        <div class="p-6 bg-slate-800 rounded-xl border border-amber-500 mb-6 animate-enter">
            <h2 class="text-amber-400 mb-4">📝 Assessment: {st.session_state.q_topic}</h2>
        """, unsafe_allow_html=True)
        
        score = 0
        quiz = st.session_state.quiz_data
        
        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            ans = st.radio(f"Select:", q['options'], key=f"q_{i}", label_visibility="collapsed")
            
            if ans:
                if ans == q['answer']:
                    st.success("Correct")
                    score += 1
                else:
                    st.error(f"Incorrect. Answer: {q['answer']}")
                    st.info(f"Insight: {q['explanation']}")
            st.markdown("---")
        
        if st.button("FINALIZE & SAVE", type="primary"):
            # Save Stats Logic
            if os.path.exists(SystemConfig.STATS_PATH):
                with open(SystemConfig.STATS_PATH, 'r') as f: stats = json.load(f)
            else: stats = {}
            
            user_stats = stats.get(st.session_state.current_user, [])
            user_stats.append({
                "topic": st.session_state.q_topic,
                "score": score,
                "total": len(quiz),
                "date": datetime.now().strftime("%Y-%m-%d")
            })
            stats[st.session_state.current_user] = user_stats
            
            with open(SystemConfig.STATS_PATH, 'w') as f: json.dump(stats, f)
            
            st.success("Performance Data Logged.")
            st.session_state.quiz_data = None
            time.sleep(1); st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

    # --- 4. CHAT HISTORY DISPLAY ---
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("audio") and os.path.exists(msg["audio"]):
                    st.audio(msg["audio"])

    # --- 5. INPUT & STREAMING LOGIC ---
    if prompt := st.chat_input(f"Ask about {st.session_state.subject}..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Process Response
        if st.session_state.db_ready:
            with st.chat_message("assistant"):
                try:
                    # Get Chain
                    chain = st.session_state.brain.get_rag_chain(
                        st.session_state.grade, 
                        st.session_state.subject
                    )
                    
                    # Streaming Variables
                    full_response = ""
                    response_placeholder = st.empty()
                    
                    # Execute Stream
                    for chunk in chain.stream({"input": prompt}):
                        if "answer" in chunk:
                            text_chunk = chunk["answer"]
                            full_response += text_chunk
                            # Typewriter Effect
                            response_placeholder.markdown(full_response + "▌")
                            time.sleep(0.005) # Micro-delay for smooth typing feel
                    
                    # Final Render
                    response_placeholder.markdown(full_response)
                    
                    # Audio Generation (Async Wrapper)
                    audio_file = None
                    try:
                        clean_text = re.sub(r'[*_#`\[\]]', '', full_response) # Remove MD and Image tags
                        async def gen_audio():
                            comm = edge_tts.Communicate(clean_text, "en-GB-SoniaNeural")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                await comm.save(fp.name)
                                return fp.name
                        audio_file = asyncio.run(gen_audio())
                        if audio_file: st.audio(audio_file)
                    except: pass
                    
                    # Save to History
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "audio": audio_file
                    })
                    SessionState.save_state()
                    
                except Exception as e:
                    st.error(f"Neural Engine Failure: {e}")
                    logger.error(f"RAG Error: {e}")

# ==================================================================================================
# 🚀 SYSTEM ENTRY POINT
# ==================================================================================================

if __name__ == "__main__":
    SessionState.init()
    
    if st.session_state.current_user:
        SessionState.load_user_data()
        render_main_interface()
    else:
        render_login_view()
