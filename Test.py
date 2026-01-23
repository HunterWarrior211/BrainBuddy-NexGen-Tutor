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

# --- 1. CORE DEPENDENCY LOADING ---
# We load these natively to ensure the environment is correctly configured.
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
    st.error(f"❌ SYSTEM BOOT FAILURE: Missing Dependency. {e}")
    st.info("Please install required packages via 'pip install -r requirements.txt'")
    st.stop()

# --- 2. CLOUD ENVIRONMENT PATCH (SQLite Fix) ---
# Essential for running ChromaDB on Linux/Streamlit Cloud
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

# ==================================================================================================
# ⚙️ SYSTEM CONFIGURATION CONTROLLER
# ==================================================================================================

class SystemConfig:
    """
    Centralized configuration controller for the application.
    Manages paths, constants, and environment variables.
    """
    APP_NAME = "NexGen Tutor"
    APP_TAGLINE = "Titanium Enterprise Edition"
    APP_ICON = "🎓"
    
    # Filesystem Architecture
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
    UPLOAD_DIR = os.path.join(BASE_DIR, "temp_ingest")
    DB_DIR = os.path.join(BASE_DIR, "chroma_vector_store")
    
    # Data Persistence Paths
    HISTORY_PATH = os.path.join(BASE_DIR, "data_history.json")
    USERS_PATH = os.path.join(BASE_DIR, "data_users.json")
    STATS_PATH = os.path.join(BASE_DIR, "data_stats.json")
    LOGS_PATH = os.path.join(BASE_DIR, "system_audit.log")

    # Neural Configuration
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "models/embedding-001"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SEARCH_K = 6  # Higher context window for better answers

    @classmethod
    def bootstrap_environment(cls):
        """Bootstraps the application environment, creating secure directories."""
        directories = [cls.RESOURCES_DIR, cls.UPLOAD_DIR, cls.DB_DIR]
        for d in directories:
            os.makedirs(d, exist_ok=True)
        
        # Initialize JSON stores if missing
        for f in [cls.HISTORY_PATH, cls.USERS_PATH, cls.STATS_PATH]:
            if not os.path.exists(f):
                with open(f, 'w') as file:
                    json.dump({}, file)

# Initialize System
SystemConfig.bootstrap_environment()

# Configure Enterprise Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
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
    Implements the '100+ Years Experience' aesthetic requirements.
    """

    @staticmethod
    def inject_core_styles(theme: str):
        """
        Injects Tailwind CDN and Custom CSS overrides.
        This function creates the 'Glassmorphism' and 'Neon' effects.
        """
        
        # 1. TAILWIND CDN INJECTION
        tailwind_cdn = """
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            slate: { 850: '#151e2e', 900: '#0f172a' },
                            amber: { 450: '#d4af37', 550: '#b49028' }
                        }
                    }
                }
            }
        </script>
        """
        st.markdown(tailwind_cdn, unsafe_allow_html=True)

        # 2. CUSTOM SCROLLBAR & ANIMATIONS (CSS3)
        core_css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Playfair+Display:ital,wght@0,700;1,400&family=Exo+2:wght@400;600;700;800&family=Inter:wght@400;600;800&display=swap');

            /* --- ✨ HYPER-GLOW GOLDEN SCROLLBAR ✨ --- */
            ::-webkit-scrollbar { width: 12px; height: 12px; }
            ::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); margin-block: 5px; }
            ::-webkit-scrollbar-corner { background: transparent; }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
                border-radius: 10px;
                border: 3px solid transparent; 
                background-clip: content-box;
                box-shadow: 0 0 15px rgba(212, 175, 55, 0.6); 
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(180deg, #FFD700, #FFFACD, #FFD700);
                box-shadow: 0 0 25px rgba(255, 215, 0, 0.9); 
            }

            /* --- ANIMATIONS --- */
            @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            @keyframes goldPulse { 0% { text-shadow: 0 0 10px rgba(212, 175, 55, 0.2); } 50% { text-shadow: 0 0 25px rgba(212, 175, 55, 0.6); } 100% { text-shadow: 0 0 10px rgba(212, 175, 55, 0.2); } }
            
            /* --- TYPOGRAPHY --- */
            h1, h2, h3 {
                font-family: 'Cinzel', serif !important;
                font-weight: 900 !important;
                text-transform: uppercase;
                letter-spacing: 1.5px;
            }
            
            p, li, span, div, label {
                font-family: 'Exo 2', sans-serif !important;
                font-weight: 600 !important; /* Bolder for readability */
                font-size: 1.05rem;
                line-height: 1.7;
            }
            
            strong {
                font-weight: 900 !important;
            }

            /* --- UI COMPONENTS --- */
            .stButton>button {
                border-radius: 12px !important;
                font-family: 'Cinzel', serif !important;
                font-weight: 900 !important;
                text-transform: uppercase;
                transition: all 0.3s ease;
                border: none;
                letter-spacing: 1px;
            }
            
            .stTextInput>div>div>input, .stSelectbox>div>div {
                border-radius: 12px !important;
                padding: 10px;
                font-weight: 600;
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
                    background-image: radial-gradient(#1B1F28 1px, transparent 1px), linear-gradient(135deg, #0A0E14 0%, #11161F 100%);
                    background-size: 30px 30px, 100% 100%;
                    color: #EAEAEA;
                }
                
                h1, h2, h3 { color: #D4AF37 !important; text-shadow: 2px 2px 4px #000000; }
                strong { color: #D4AF37 !important; }
                
                /* Sidebar Dark */
                section[data-testid="stSidebar"] {
                    background-color: #141A24;
                    border-right: 1px solid #D4AF37;
                    box-shadow: 5px 0 20px rgba(0,0,0,0.5);
                }
                
                /* Buttons Dark */
                .stButton>button {
                    background: linear-gradient(135deg, #D4AF37 0%, #B8962E 100%);
                    color: #0A0E14;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                }
                .stButton>button:hover { transform: scale(1.02); color: #000; box-shadow: 0 0 25px rgba(212, 175, 55, 0.6); }
                
                /* Chat Dark */
                .stChatMessage { background-color: #1B1F28; }
                div[data-testid="stChatMessage"]:nth-child(odd) { border-left: 4px solid #8BE9FD; background: linear-gradient(90deg, rgba(139, 233, 253, 0.05) 0%, rgba(0,0,0,0) 100%); }
                div[data-testid="stChatMessage"]:nth-child(even) { border-left: 4px solid #D4AF37; background: linear-gradient(90deg, rgba(212, 175, 55, 0.05) 0%, rgba(0,0,0,0) 100%); }
                
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
                
                /* Buttons Light */
                .stButton>button {
                    background: linear-gradient(135deg, #BFC3C9 0%, #9FA4AA 100%);
                    color: #2B2E34;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                .stButton>button:hover { transform: scale(1.02); background: #2B2E34; color: #FFFFFF; }
                
                /* Chat Light */
                .stChatMessage { box-shadow: 0 4px 6px rgba(0,0,0,0.04); }
                div[data-testid="stChatMessage"]:nth-child(odd) { background-color: #F1F3F6; border: 1px solid #E5E7EB; border-left: 4px solid #BFC3C9; }
                div[data-testid="stChatMessage"]:nth-child(even) { background-color: #FFFFFF; border: 1px solid #E5E7EB; border-left: 4px solid #2B2E34; }
                
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

class AuthEngine:
    """
    Manages User Identity, Session Validation, and Credential Hashing.
    Ensures secure access to the educational platform.
    """
    
    @staticmethod
    def _read_db() -> Dict:
        """Reads user database securely."""
        try:
            with open(SystemConfig.USERS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Auth Read Error: {e}")
            return {}

    @staticmethod
    def _write_db(data: Dict):
        """Writes to user database securely."""
        try:
            with open(SystemConfig.USERS_PATH, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Auth Write Error: {e}")

    @staticmethod
    def hash_token(secret: str) -> str:
        """Applies SHA-256 encryption to user credentials."""
        return hashlib.sha256(secret.encode()).hexdigest()

    @classmethod
    def authenticate(cls, username, password) -> bool:
        """Validates login credentials."""
        db = cls._read_db()
        return username in db and db[username] == cls.hash_token(password)

    @classmethod
    def register_identity(cls, username, password) -> Tuple[bool, str]:
        """Registers a new user identity."""
        db = cls._read_db()
        if username in db:
            return False, "Identity conflict: Username exists."
        db[username] = cls.hash_token(password)
        cls._write_db(db)
        return True, "Identity registered successfully."


# ==================================================================================================
# 📊 ANALYTICS & METRICS ENGINE
# ==================================================================================================

class AnalyticsEngine:
    """
    Tracks user performance, quiz scores, and engagement metrics.
    Stores data for the 'Progress Dashboard'.
    """
    
    @staticmethod
    def log_quiz_result(user: str, topic: str, score: int, total: int):
        """Saves a quiz result to the stats database."""
        if not os.path.exists(SystemConfig.STATS_PATH): return
        
        try:
            with open(SystemConfig.STATS_PATH, 'r') as f: 
                data = json.load(f)
            
            user_data = data.get(user, [])
            user_data.append({
                "id": str(uuid.uuid4())[:8],
                "topic": topic,
                "score": score,
                "total": total,
                "percentage": round((score/total)*100, 1),
                "date": datetime.now().strftime("%Y-%m-%d")
            })
            data[user] = user_data
            
            with open(SystemConfig.STATS_PATH, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logger.error(f"Analytics Error: {e}")

    @staticmethod
    def get_user_stats(user: str) -> pd.DataFrame:
        """Retrieves user stats as a Pandas DataFrame."""
        try:
            with open(SystemConfig.STATS_PATH, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data.get(user, []))
        except:
            return pd.DataFrame()


# ==================================================================================================
# 🎓 PEDAGOGY ENGINE (GRADE LOGIC)
# ==================================================================================================

class PedagogyEngine:
    """
    Determines the tone, complexity, and structure of AI responses
    based on the user's Grade Level (6-10).
    """
    
    @staticmethod
    def get_instruction_set(grade_str: str) -> str:
        """Returns the specific pedagogical prompt instructions."""
        try:
            grade = int(re.search(r'\d+', grade_str).group())
        except:
            grade = 9 # Default

        if grade == 6:
            return """
            TARGET AUDIENCE: Grade 6 (Age 11-12).
            TONE: Enthusiastic, Simple, Story-telling.
            RULES:
            - Use short sentences.
            - Avoid complex jargon; if used, define it immediately.
            - Use emojis to keep it engaging 🌟.
            - Max 3 bullet points per section.
            """
        elif grade == 7:
            return """
            TARGET AUDIENCE: Grade 7 (Age 12-13).
            TONE: Clear, Encouraging, Informative.
            RULES:
            - Focus on the 'Why' and 'How'.
            - Use analogies from daily life.
            - Keep explanations concise.
            """
        elif grade == 8:
            return """
            TARGET AUDIENCE: Grade 8 (Age 13-14).
            TONE: Academic but accessible.
            RULES:
            - Standard textbook definitions.
            - Introduce formal terminology.
            - Moderate detail in explanations.
            """
        elif grade == 9:
            return """
            TARGET AUDIENCE: Grade 9 (Age 14-15).
            TONE: Formal, Preparatory, Scientific.
            RULES:
            - Detailed theoretical background.
            - Explicit mention of formulas and laws.
            - Focus on application of concepts.
            """
        else: # Grade 10
            return """
            TARGET AUDIENCE: Grade 10 (Board Exam Level).
            TONE: Professional, Technical, Comprehensive.
            RULES:
            - Deep dive into mechanisms.
            - Focus on keywords for exam scoring.
            - Point-wise structured answers strictly.
            - High technical density.
            """


# ==================================================================================================
# 🧠 NEURAL CORE (RAG & LLM)
# ==================================================================================================

class NeuralCore:
    """
    The Intelligence Unit of NexGen Tutor.
    Manages the Vector Database, Document Processing, and Prompt Engineering.
    """
    
    def __init__(self):
        self.vector_store = None
        self._init_keys()
        
        # LLM Initialization
        self.llm = ChatGoogleGenerativeAI(
            model=SystemConfig.LLM_MODEL,
            temperature=0.6,
            max_retries=3,
            streaming=True
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model=SystemConfig.EMBEDDING_MODEL)

    def _init_keys(self):
        """Loads API keys from Secrets or Local Config."""
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            try:
                if "GOOGLE_API_KEY" in st.secrets:
                    key = st.secrets["GOOGLE_API_KEY"]
                else:
                    path = os.path.join(SystemConfig.BASE_DIR, ".streamlit", "secrets.toml")
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            for line in f:
                                if "GOOGLE_API_KEY" in line:
                                    key = line.split("=")[1].strip().strip('"').strip("'")
            except: pass
        
        if key:
            os.environ["GOOGLE_API_KEY"] = key
        else:
            st.error("⚠️ CRITICAL: Google API Key Missing. Please add it to secrets.")
            st.stop()

    def connect_memory(self) -> bool:
        """Connects to the Vector Database."""
        if os.path.exists(SystemConfig.DB_DIR):
            try:
                self.vector_store = Chroma(
                    persist_directory=SystemConfig.DB_DIR,
                    embedding_function=self.embeddings
                )
                if self.vector_store._collection.count() > 0:
                    return True
            except Exception as e:
                logger.error(f"DB Connect Fail: {e}")
        return False

    def ingest_data(self, file_path: str) -> bool:
        """Ingests a single PDF into the Neural Memory."""
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if not docs: return False
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=SystemConfig.CHUNK_SIZE,
                chunk_overlap=SystemConfig.CHUNK_OVERLAP
            )
            splits = splitter.split_documents(docs)
            
            # Initialize DB if needed
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    splits[:20], 
                    self.embeddings, 
                    persist_directory=SystemConfig.DB_DIR
                )
                splits = splits[20:]
            
            # Batch Add
            batch_size = 40
            total = len(splits)
            if total > 0:
                bar = st.progress(0, "Neural Encoding...")
                for i in range(0, total, batch_size):
                    batch = splits[i:i+batch_size]
                    self.vector_store.add_documents(batch)
                    bar.progress(min((i+batch_size)/total, 1.0))
                    time.sleep(0.05)
                bar.empty()
            
            return True
        except Exception as e:
            logger.error(f"Ingest Fail: {e}")
            st.error(f"Ingestion Error: {e}")
            return False

    def build_chain(self, grade: str, subject: str) -> Any:
        """Constructs the RAG Chain with Grade-Specific Logic."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": SystemConfig.SEARCH_K})
        
        pedagogy = PedagogyEngine.get_instruction_set(grade)
        
        # --- THE MASTER PROMPT (Following your exact requirements) ---
        template = """
        You are NexGen, an expert AI Tutor for {subject}.
        
        [SYSTEM RULES]
        1. STRICTLY answer only questions related to {subject}. If asked about other topics, politely refuse.
        2. Adopt the following Pedagogical Style:
           {pedagogy}
        
        [DIAGRAM TRIGGERING INSTRUCTION]
        Assess if the user would understand the response better with a diagram. 
        You can insert a diagram by adding the  tag where X is a contextually relevant and domain-specific query to fetch the diagram. 
        Examples: 

[Image of the human digestive system]
, 

[Image of hydrogen fuel cell]
 etc. 
        Avoid triggering images just for visual appeal. Only add if instructive.
        Place the image tag immediately before or after the relevant text without disrupting the flow.
        
        [OUTPUT FORMAT - STRICT MARKDOWN]
        1. **Core Concept:** (Bold definition)
        
        2. **Key Points:** - (Bullet Point 1: Detailed explanation suitable for {grade})
           - (Bullet Point 2: Mechanism/Process)
           - (Bullet Point 3: Context/Nuance)
        
        3. **Comparison:** (IF the question implies a difference between two concepts, YOU MUST GENERATE A MARKDOWN TABLE. If not, output 'N/A').
        
        4. **Real-World Example:** (A relatable application or analogy for a student).
        
        5. **Math/Physics Solver:** (If the query is a problem, solve it step-by-step using LaTeX inside $$...$$. State Formula -> Given -> Substitute -> Result).

        [CONTEXT FROM TEXTBOOK]
        {{context}}
        
        [CHAT HISTORY]
        {history}
        
        [USER QUESTION]
        {{input}}
        """
        
        # History Injection
        hist_txt = ""
        if "messages" in st.session_state:
            for m in st.session_state.messages[-4:]:
                hist_txt += f"{m['role'].capitalize()}: {m['content']}\n"

        final_prompt = template.format(
            subject=subject,
            pedagogy=pedagogy,
            history=hist_txt
        )
        
        prompt = ChatPromptTemplate.from_template(final_prompt)
        doc_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, doc_chain)

    def generate_quiz(self, topic: str, grade: str) -> List[Dict]:
        """Generates a JSON Quiz."""
        prompt = f"""
        Create 3 Multiple Choice Questions (MCQ) on '{topic}' for {grade}.
        Output RAW JSON ONLY. Format:
        [
            {{"question": "...", "options": ["A","B","C","D"], "answer": "Option Text", "explanation": "..."}}
        ]
        """
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)])
            txt = res.content.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except: return []

    def detect_topics(self, text: str) -> List[str]:
        prompt = f"Extract top 3 academic topics from this chat. Comma separated. Chat: {text[-1000:]}"
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)])
            return [t.strip() for t in res.content.split(',') if t.strip()]
        except: return []


# ==================================================================================================
# 🎮 CONTROLLER (SESSION & EVENTS)
# ==================================================================================================

class AppController:
    """
    Main Application Logic Controller.
    """
    
    @staticmethod
    def init_session():
        defaults = {
            "current_user": None,
            "theme": "Dark Mode 🌑 (Cosmic Gold)",
            "messages": [],
            "saved_chats": {},
            "current_chat_id": None,
            "brain": None,
            "db_ready": False,
            "quiz_data": None
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        
        if st.session_state.brain is None:
            st.session_state.brain = NeuralCore()

    @staticmethod
    def load_profile():
        if st.session_state.current_user:
            # In real app, load history from HISTORY_PATH
            if os.path.exists(SystemConfig.HISTORY_PATH):
                with open(SystemConfig.HISTORY_PATH, 'r') as f:
                    hist = json.load(f)
                    st.session_state.saved_chats = hist.get(st.session_state.current_user, {})
            
            if not st.session_state.saved_chats:
                AppController.new_chat()
            
            if not st.session_state.current_chat_id and st.session_state.saved_chats:
                st.session_state.current_chat_id = list(st.session_state.saved_chats.keys())[-1]
                st.session_state.messages = st.session_state.saved_chats[st.session_state.current_chat_id]

    @staticmethod
    def new_chat():
        uid = str(uuid.uuid4())
        st.session_state.current_chat_id = uid
        st.session_state.saved_chats[uid] = []
        st.session_state.messages = []
        AppController.save_state()

    @staticmethod
    def save_state():
        if st.session_state.current_user and st.session_state.current_chat_id:
            st.session_state.saved_chats[st.session_state.current_chat_id] = st.session_state.messages
            
            if os.path.exists(SystemConfig.HISTORY_PATH):
                with open(SystemConfig.HISTORY_PATH, 'r') as f: full = json.load(f)
            else: full = {}
            
            full[st.session_state.current_user] = st.session_state.saved_chats
            with open(SystemConfig.HISTORY_PATH, 'w') as f: json.dump(full, f)


# ==================================================================================================
# 🚀 MAIN EXECUTION LOOP
# ==================================================================================================

def main():
    st.set_page_config(page_title=SystemConfig.APP_NAME, page_icon=SystemConfig.APP_ICON, layout="wide")
    AppController.init_session()

    # --- 1. LOGIN VIEW ---
    if not st.session_state.current_user:
        UIEngine.inject_core_styles("Dark") # Force Dark for cinematic login
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(f"<h1 class='text-center text-5xl text-amber-400 mb-2'>{SystemConfig.APP_ICON}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h1 class='text-center'>{SystemConfig.APP_NAME}</h1>", unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
            with tab1:
                u = st.text_input("Username", key="l1")
                p = st.text_input("Password", type="password", key="l2")
                if st.button("AUTHENTICATE", use_container_width=True):
                    if AuthEngine.authenticate(u, p):
                        st.session_state.current_user = u
                        st.rerun()
                    else: st.error("Invalid Credentials")
            with tab2:
                nu = st.text_input("New User", key="r1")
                np = st.text_input("New Pass", type="password", key="r2")
                if st.button("CREATE ACCOUNT", use_container_width=True):
                    ok, msg = AuthEngine.register_identity(nu, np)
                    if ok: st.success(msg)
                    else: st.error(msg)
        return

    # --- 2. MAIN DASHBOARD ---
    AppController.load_profile()
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.current_user}")
        if st.button("LOGOUT", type="secondary"):
            st.session_state.clear()
            st.rerun()
        
        # 1. Academic Controls
        st.divider()
        st.markdown("### 📚 ACADEMIC SETTINGS")
        st.session_state.grade = st.selectbox("Level", [f"Grade {i}" for i in range(6, 11)])
        st.session_state.subject = st.selectbox("Subject", ["Mathematics", "Physics", "Biology", "Chemistry", "History", "Computer Science"])
        
        # 2. Quiz Generator
        st.divider()
        with st.expander("🧠 INTELLIGENT QUIZ"):
            if st.button("🚀 ANALYZE CONTEXT", use_container_width=True):
                hist = "\n".join([m['content'] for m in st.session_state.messages])
                st.session_state.topics = st.session_state.brain.detect_topics(hist)
            
            if st.session_state.get("topics"):
                topic = st.radio("Focus:", st.session_state.topics + ["Custom..."])
                if topic == "Custom...": topic = st.text_input("Topic")
                
                if st.button("GENERATE EXAM", type="primary", use_container_width=True):
                    with st.spinner("Compiling..."):
                        st.session_state.quiz_data = st.session_state.brain.generate_quiz(topic, st.session_state.grade)
                        st.session_state.q_topic = topic
                        st.rerun()

        # 3. Chat Controls
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("✨ NEW"):
            AppController.new_chat()
            st.rerun()
        if c2.button("🗑️ WIPE"):
            st.session_state.messages = []
            AppController.save_state()
            st.rerun()
            
        with st.expander("📜 ARCHIVES"):
            for cid in reversed(list(st.session_state.saved_chats.keys())):
                msgs = st.session_state.saved_chats[cid]
                label = "New Chat"
                for m in msgs:
                    if m['role'] == 'user':
                        label = " ".join(m['content'].split()[:4]) + "..."
                        break
                if st.button(label, key=cid, use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.session_state.messages = msgs
                    st.rerun()

        # 4. System Controls
        st.divider()
        st.markdown("### ⚙️ SYSTEM")
        theme_choice = st.radio("Theme", ["Dark Mode 🌑 (Cosmic Gold)", "Light Mode ☀️ (Platinum Silver)"], horizontal=True)
        UIEngine.inject_core_styles(theme_choice)
        
        if st.button("🔄 SYNC DATABASE", use_container_width=True):
            if st.session_state.brain.ingest_data(SystemConfig.RESOURCES_DIR): # Bulk ingest
                st.session_state.db_ready = True
                st.success("Synced")
        
        up_file = st.file_uploader("UPLOAD PDF", type="pdf")
        if up_file:
            path = os.path.join(SystemConfig.UPLOAD_DIR, up_file.name)
            with open(path, "wb") as f: f.write(up_file.getbuffer())
            if st.session_state.brain.ingest_data(path):
                st.session_state.db_ready = True
                st.success("Ingested")
                os.remove(path)

        # 5. Dashboard
        with st.expander("📊 METRICS"):
            stats = AnalyticsEngine.get_user_stats(st.session_state.current_user)
            if not stats.empty:
                st.dataframe(stats[['topic', 'score', 'date']], use_container_width=True)
                
                # Chart
                c = alt.Chart(stats).mark_bar(color='#D4AF37').encode(
                    x='topic', y='score', tooltip=['topic', 'score']
                ).properties(height=200)
                st.altair_chart(c, use_container_width=True)
            else:
                st.info("No data yet.")

    # Content
    st.title(SystemConfig.APP_NAME)
    
    if not st.session_state.db_ready:
        if st.session_state.brain.connect_memory():
            st.session_state.db_ready = True
        else:
            st.info("👈 Please **Sync Database** or **Upload PDF** to initialize Neural Core.")

    # Quiz Render
    if st.session_state.get("quiz_data"):
        st.markdown(f"### 📝 {st.session_state.q_topic}")
        score = 0
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            ans = st.radio("Select:", q['options'], key=f"q{i}", label_visibility="collapsed")
            if ans == q['answer']:
                st.success("Correct")
                score += 1
            elif ans:
                st.error("Incorrect")
                st.info(q['explanation'])
            st.markdown("---")
        if st.button("SAVE RESULT", type="primary"):
            AnalyticsEngine.log_quiz_result(st.session_state.current_user, st.session_state.q_topic, score, 3)
            st.success("Saved")
            st.session_state.quiz_data = None
            st.rerun()

    # Chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"): st.audio(msg["audio"])

    if prompt := st.chat_input(f"Ask about {st.session_state.subject}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        if st.session_state.db_ready:
            with st.chat_message("assistant"):
                box = st.empty()
                full_text = ""
                try:
                    chain = st.session_state.brain.build_chain(st.session_state.grade, st.session_state.subject)
                    
                    # Streaming
                    for chunk in chain.stream({"input": prompt}):
                        if "answer" in chunk:
                            full_text += chunk["answer"]
                            box.markdown(full_text + "▌")
                            time.sleep(0.005)
                    
                    box.markdown(full_text)
                    
                    # Audio
                    aud = None
                    try:
                        clean = re.sub(r'[*_#`\[\]]', '', full_text)
                        async def gen_tts():
                            c = edge_tts.Communicate(clean, "en-GB-SoniaNeural")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                                await c.save(f.name)
                                return f.name
                        aud = asyncio.run(gen_tts())
                        if aud: st.audio(aud)
                    except: pass
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_text, "audio": aud})
                    AppController.save_state()
                    
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
