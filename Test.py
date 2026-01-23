"""
================================================================================
APPLICATION: NexGen Tutor | Enterprise AI Learning Platform
VERSION:     4.0.0 (Ultimate Edition)
ARCHITECT:   Senior Systems Engineer (100+ Years Experience Simulation)
DESCRIPTION: A state-of-the-art RAG application featuring:
             - Real-time Streaming (Typewriter Effect)
             - High-Fidelity UI/UX with Neon/Glassmorphism
             - Robust Error Handling & Logging
             - Role-Based Pedagogical Logic
================================================================================
"""

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
import pandas as pd
import altair as alt
import edge_tts
from datetime import datetime
from typing import List, Dict, Optional, Any, Generator

# --- 1. CORE DEPENDENCIES CHECK ---
try:
    import streamlit as st
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except ImportError as e:
    st.error(f"🚨 CRITICAL SYSTEM FAILURE: Missing Dependency - {e}")
    st.info("Please ensure 'requirements.txt' contains: streamlit, langchain, chromadb, google-generativeai, pypdf, edge-tts, pandas, altair")
    st.stop()

# --- 2. CLOUD ENVIRONMENT PATCH (SQLite Fix) ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except (ImportError, KeyError):
    pass

# ==============================================================================
# ⚙️ GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================

class AppConfig:
    """Centralized configuration management for the entire application."""
    
    APP_NAME = "NexGen Tutor"
    APP_VERSION = "v4.0.0 Enterprise"
    APP_ICON = "🎓"
    
    # File System Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
    UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploaded_books")
    DB_DIR = os.path.join(BASE_DIR, "chroma_db")
    
    # Data Persistence Paths
    HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")
    USERS_FILE = os.path.join(BASE_DIR, "users.json")
    QUIZ_FILE = os.path.join(BASE_DIR, "quiz_scores.json")

    # AI Configuration
    MODEL_NAME = "gemini-2.5-flash"
    EMBEDDING_MODEL = "models/embedding-001"
    
    @classmethod
    def init_environment(cls):
        """Initializes the execution environment, creating necessary directories."""
        for directory in [cls.RESOURCES_DIR, cls.UPLOAD_DIR, cls.DB_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Configure Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

# Initialize System
AppConfig.init_environment()
logger = logging.getLogger("NexGenCore")


# ==============================================================================
# 🎨 HIGH-FIDELITY UI ENGINE (CSS & ANIMATIONS)
# ==============================================================================

class UIFactory:
    """
    Manages the visual presentation layer. 
    Implements advanced CSS3 features, animations, and responsive design.
    """

    @staticmethod
    def inject_css(theme_mode: str):
        """Injects the appropriate CSS variables and rules based on the selected theme."""
        
        # Base CSS (Animations & Fonts)
        base_css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Playfair+Display:ital,wght@0,700;1,400&family=Exo+2:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');
            
            /* --- ANIMATIONS --- */
            @keyframes slideUpFade { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            @keyframes glowPulse { 0% { box-shadow: 0 0 5px var(--accent-color); } 50% { box-shadow: 0 0 20px var(--accent-color); } 100% { box-shadow: 0 0 5px var(--accent-color); } }
            
            /* --- SCROLLBAR ENGINE (The "100% Perfection" Golden Glow) --- */
            ::-webkit-scrollbar { width: 12px; height: 12px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-corner { background: transparent; }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
                border-radius: 12px;
                border: 3px solid var(--bg-color); /* Creates 'floating' look */
                box-shadow: 0 0 15px rgba(212, 175, 55, 0.5);
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(180deg, #FFD700, #FFFACD, #FFD700);
                box-shadow: 0 0 25px rgba(255, 215, 0, 0.8);
                border: 2px solid #FFF;
            }
        </style>
        """

        if "Dark" in theme_mode:
            theme_vars = """
            <style>
                :root {
                    --bg-color: #0A0E14;
                    --sidebar-bg: #141A24;
                    --card-bg: #1B1F28;
                    --text-primary: #EAEAEA;
                    --text-secondary: #B0B3B8;
                    --accent-color: #D4AF37;
                    --accent-secondary: #8BE9FD;
                    --border-glow: rgba(212, 175, 55, 0.3);
                }
            </style>
            """
        else:
            theme_vars = """
            <style>
                :root {
                    --bg-color: #F8F9FB;
                    --sidebar-bg: #FFFFFF;
                    --card-bg: #FFFFFF;
                    --text-primary: #2B2E34;
                    --text-secondary: #5F6368;
                    --accent-color: #2B2E34; /* Dark Gold/Graphite for contrast */
                    --accent-secondary: #007AFF;
                    --border-glow: rgba(43, 46, 52, 0.1);
                }
            </style>
            """

        # Component Styling
        components_css = """
        <style>
            /* --- GLOBAL CONTAINER --- */
            .stApp {
                background-color: var(--bg-color);
                background-image: radial-gradient(var(--card-bg) 1px, transparent 1px);
                background-size: 30px 30px;
                color: var(--text-primary);
            }

            /* --- TYPOGRAPHY --- */
            h1, h2, h3 {
                font-family: 'Cinzel', serif !important;
                color: var(--accent-color) !important;
                text-transform: uppercase;
                letter-spacing: 1.5px;
                font-weight: 900 !important;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            
            p, li, div, label, span {
                font-family: 'Exo 2', sans-serif !important;
                color: var(--text-primary);
                font-size: 1.05rem;
                font-weight: 600; /* Bold for visibility */
                line-height: 1.7;
            }

            /* --- SIDEBAR --- */
            section[data-testid="stSidebar"] {
                background-color: var(--sidebar-bg);
                border-right: 2px solid var(--accent-color);
                box-shadow: 10px 0 30px rgba(0,0,0,0.5);
            }
            
            /* --- CHAT MESSAGE BUBBLES --- */
            .stChatMessage {
                background-color: var(--card-bg);
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 15px;
                border: 1px solid rgba(255,255,255,0.05);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                animation: slideUpFade 0.5s ease-out forwards;
            }
            
            /* Assistant Message */
            div[data-testid="stChatMessage"]:nth-child(even) {
                border-left: 4px solid var(--accent-color);
            }
            
            /* User Message */
            div[data-testid="stChatMessage"]:nth-child(odd) {
                border-left: 4px solid var(--accent-secondary);
                background-color: rgba(139, 233, 253, 0.05);
            }

            /* --- INPUT FIELDS --- */
            .stTextInput > div > div > input, .stSelectbox > div > div {
                background-color: var(--card-bg);
                color: var(--text-primary);
                border: 1px solid var(--accent-color);
                border-radius: 12px;
                padding: 10px;
            }
            .stTextInput > div > div > input:focus {
                box-shadow: 0 0 15px var(--border-glow);
            }

            /* --- BUTTONS --- */
            .stButton > button {
                background: linear-gradient(135deg, var(--accent-color) 0%, #AA8C2C 100%);
                color: #FFF; /* Ensure text is white on buttons */
                font-family: 'Cinzel', serif;
                font-weight: 800;
                border: none;
                border-radius: 8px;
                padding: 0.6rem 1.5rem;
                text-transform: uppercase;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            }
            
            /* Specific override for Light Mode Text on Buttons if needed */
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px var(--border-glow);
                color: #FFF;
            }
        </style>
        """
        
        st.markdown(base_css + theme_vars + components_css, unsafe_allow_html=True)


# ==============================================================================
# 🔐 SECURITY & PERSISTENCE LAYER
# ==============================================================================

class AuthSystem:
    """Handles User Authentication and Secure Data Storage."""
    
    @staticmethod
    def _load_db(path: str) -> Dict:
        if not os.path.exists(path): return {}
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return {}

    @staticmethod
    def _save_db(path: str, data: Dict):
        try:
            with open(path, 'w') as f: json.dump(data, f, indent=4)
        except Exception as e: logger.error(f"Save Failed: {e}")

    @staticmethod
    def hash_password(password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    @classmethod
    def login(cls, username, password) -> bool:
        db = cls._load_db(AppConfig.USERS_FILE)
        return username in db and db[username] == cls.hash_password(password)

    @classmethod
    def register(cls, username, password) -> bool:
        db = cls._load_db(AppConfig.USERS_FILE)
        if username in db: return False
        db[username] = cls.hash_password(password)
        cls._save_db(AppConfig.USERS_FILE, db)
        return True


# ==============================================================================
# 🧠 BRAIN CORE: RAG & STREAMING ENGINE
# ==============================================================================

class BrainCore:
    """
    The Intelligence Unit. 
    Handles Document Ingestion, Vector Embeddings, and Streaming LLM Responses.
    """
    
    def __init__(self):
        self.vector_store = None
        self._setup_api()
        
        # Initialize LLM with Streaming Capability
        self.llm = ChatGoogleGenerativeAI(
            model=AppConfig.MODEL_NAME,
            temperature=0.7,
            streaming=True, # <--- KEY FOR TYPING EFFECT
            max_retries=3
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model=AppConfig.EMBEDDING_MODEL)

    def _setup_api(self):
        """Securely loads API Keys."""
        if "GOOGLE_API_KEY" not in os.environ:
            # Try Streamlit Secrets
            if "GOOGLE_API_KEY" in st.secrets:
                os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
            else:
                # Local Fallback
                secrets_path = os.path.join(AppConfig.BASE_DIR, ".streamlit", "secrets.toml")
                if os.path.exists(secrets_path):
                    with open(secrets_path, "r") as f:
                        for line in f:
                            if "GOOGLE_API_KEY" in line:
                                os.environ["GOOGLE_API_KEY"] = line.split("=")[1].strip().strip('"').strip("'")
        
        if "GOOGLE_API_KEY" not in os.environ:
            st.error("⚠️ SYSTEM HALTED: Google API Key Missing.")
            st.stop()

    def load_knowledge_base(self) -> bool:
        """Loads Vector DB from Disk."""
        if os.path.exists(AppConfig.DB_DIR):
            try:
                self.vector_store = Chroma(
                    persist_directory=AppConfig.DB_DIR,
                    embedding_function=self.embeddings
                )
                if self.vector_store._collection.count() > 0:
                    return True
            except Exception as e:
                logger.error(f"DB Load Error: {e}")
        return False

    def ingest_library(self) -> bool:
        """Processes all PDFs in resources folder."""
        if not os.path.exists(AppConfig.RESOURCES_DIR): return False
        
        files = [f for f in os.listdir(AppConfig.RESOURCES_DIR) if f.endswith('.pdf')]
        if not files: return False

        docs = []
        bar = st.progress(0, "Scanning Knowledge Library...")
        
        for i, f in enumerate(files):
            try:
                path = os.path.join(AppConfig.RESOURCES_DIR, f)
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
                bar.progress((i + 1) / len(files), f"Indexing {f}...")
            except: pass
        
        bar.empty()
        
        if not docs: return False
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = splitter.split_documents(docs)
        
        # Batch Embed
        return self._batch_embed(splits)

    def process_upload(self, file_obj) -> bool:
        """Handles user file uploads."""
        path = os.path.join(AppConfig.UPLOAD_DIR, file_obj.name)
        with open(path, "wb") as f: f.write(file_obj.getbuffer())
        
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            
            # Content Verification
            verify = self.llm.invoke([HumanMessage(content=f"Is educational? YES/NO. {docs[0].page_content[:500]}")])
            if "NO" in verify.content.upper():
                st.error("🚫 Non-Educational Content Rejected.")
                os.remove(path)
                return False
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            splits = splitter.split_documents(docs)
            res = self._batch_embed(splits)
            os.remove(path)
            return res
        except Exception as e:
            st.error(f"Error: {e}")
            return False

    def _batch_embed(self, splits) -> bool:
        """Batch processing to prevent API timeouts."""
        batch_size = 40
        total = len(splits)
        
        if self.vector_store is None:
            if os.path.exists(AppConfig.DB_DIR):
                self.vector_store = Chroma(persist_directory=AppConfig.DB_DIR, embedding_function=self.embeddings)
            else:
                self.vector_store = Chroma.from_documents(splits[:batch_size], self.embeddings, persist_directory=AppConfig.DB_DIR)
                splits = splits[batch_size:]
                
        if not splits: return True

        bar = st.progress(0, "Encoding Neural Embeddings...")
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i+batch_size]
            self.vector_store.add_documents(batch)
            bar.progress(min((i + batch_size) / total, 1.0))
            time.sleep(0.05)
        bar.empty()
        return True

    def get_streaming_chain(self, grade_info: str, subject: str):
        """Builds the RAG Chain optimized for Streaming."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # --- PROMPT ENGINEERING LOGIC ---
        # Extract numeric grade
        try: g_num = int(re.search(r'\d+', grade_info).group())
        except: g_num = 9

        # Pedagogical Strategy
        if g_num == 6:
            strategy = "Tone: Fun, Simple, Encouraging. Structure: Very short paragraphs. Use Emojis. Max 3 simple bullet points."
        elif g_num == 7:
            strategy = "Tone: Clear, Concise. Structure: Focus on 'What' and 'Why'. Simple analogies."
        elif g_num == 8:
            strategy = "Tone: Academic Intro. Structure: Formal definition, detailed explanation, 1 concrete example."
        elif g_num == 9:
            strategy = "Tone: High School Academic. Structure: Theory + Application. Detailed bullet points."
        else:
            strategy = "Tone: Board Exam Prep. Structure: Professional, Technical terms, Comprehensive analysis, Keyword focus."

        # --- THE ULTIMATE PROMPT TEMPLATE ---
        template_str = """
        You are NexGen, an elite AI Tutor for {subject} at {grade} level.
        
        [STRICT SUBJECT GUARDRAIL]
        You are ONLY allowed to teach {subject}.
        If the user asks about a different subject (e.g. History in a Math session), politely REFUSE and tell them to switch subjects in the sidebar.
        
        [PEDAGOGY INSTRUCTION]
        {strategy}
        
        [REQUIRED RESPONSE STRUCTURE]
        You MUST follow this format exactly:
        
        1. **Core Concept:** (A bold, precise definition of the topic).
        
        2. **Key Points:** - (Bullet Point 1: Detailed explanation)
           - (Bullet Point 2: Key features/mechanisms)
           - (Bullet Point 3: Important context)
        
        3. **Comparison:** (IF the question implies a difference between X and Y, generate a Markdown Table. If not, output 'N/A').
        
        4. **Real-World Example:** (A relatable scenario for a student of this age).
           
        5. **Math/Physics Solver:** (Only if calculation needed. Use LaTeX enclosed in $$...$$ for formulas. Show step-by-step substitution).

        [CONTEXT FROM TEXTBOOK]
        {{context}}
        
        [CHAT HISTORY]
        {history}
        
        [USER QUESTION]
        {{input}}
        """
        
        # Inject History manually to avoid f-string conflicts with LangChain vars
        history_txt = ""
        if "messages" in st.session_state:
            for m in st.session_state.messages[-4:]:
                history_txt += f"{m['role'].capitalize()}: {m['content']}\n"

        # Format fixed variables, leave {{context}} and {{input}} for chain
        final_prompt = template_str.format(
            subject=subject,
            grade=grade_info,
            strategy=strategy,
            history=history_txt
        )

        prompt = ChatPromptTemplate.from_template(final_prompt)
        
        # Chain Construction
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain

    def generate_quiz(self, topic: str, grade: str):
        """Generates Quiz JSON."""
        prompt = f"""
        Create a 3-question MCQ Quiz on '{topic}' for {grade}.
        Output STRICT VALID JSON only. No Markdown.
        Format: [{{ "question": "...", "options": ["A","B","C","D"], "answer": "Option Text", "explanation": "..." }}]
        """
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)])
            txt = res.content.replace("```json", "").replace("```", "").strip()
            return json.loads(txt)
        except: return []

    def detect_topics(self, text: str):
        prompt = f"Extract top 3 educational topics from this chat. Return comma-separated list. Chat: {text[-1500:]}"
        res = self.llm.invoke([HumanMessage(content=prompt)])
        return [t.strip() for t in res.content.split(',') if t.strip()]


# ==============================================================================
# 🎮 APPLICATION CONTROLLER (MAIN LOOP)
# ==============================================================================

def main():
    st.set_page_config(
        page_title=AppConfig.APP_NAME,
        page_icon=AppConfig.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- SESSION INITIALIZATION ---
    if "current_user" not in st.session_state: st.session_state.current_user = None
    if "messages" not in st.session_state: st.session_state.messages = []
    if "brain" not in st.session_state: st.session_state.brain = BrainCore()
    if "db_ready" not in st.session_state: st.session_state.db_ready = False

    # --- 1. LOGIN SCREEN ---
    if not st.session_state.current_user:
        UIFactory.inject_css("Dark Mode 🌑") # Force Dark for Login
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(f"<h1 style='text-align:center;'>{AppConfig.APP_ICON} {AppConfig.APP_NAME}</h1>", unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
            with tab1:
                u = st.text_input("Username", key="l1")
                p = st.text_input("Password", type="password", key="l2")
                if st.button("Enter Portal", use_container_width=True):
                    if AuthSystem.login(u, p):
                        st.session_state.current_user = u
                        st.rerun()
                    else: st.error("Access Denied")
            with tab2:
                nu = st.text_input("New User", key="r1")
                np = st.text_input("New Pass", type="password", key="r2")
                if st.button("Create Account", use_container_width=True):
                    if AuthSystem.register(nu, np): st.success("Created! Login now.")
                    else: st.error("Username exists.")
        return

    # --- 2. MAIN APP ---
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 **{st.session_state.current_user}**")
        if st.button("Log Out", type="secondary"):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        
        # A. Settings
        st.markdown("### 📚 Academic Settings")
        st.session_state.grade = st.selectbox("Grade Level", [f"Grade {i}" for i in range(6, 11)])
        st.session_state.subject = st.selectbox("Subject", ["Mathematics", "Physics", "Biology", "Chemistry", "History", "Computer Science"])
        
        st.divider()
        
        # B. Quiz
        with st.expander("🧠 Quiz Generator"):
            if st.button("🚀 Detect Topics", use_container_width=True):
                hist = "\n".join([m['content'] for m in st.session_state.messages])
                st.session_state.topics = st.session_state.brain.detect_topics(hist)
            
            if "topics" in st.session_state:
                topic = st.radio("Focus:", st.session_state.topics + ["Custom..."])
                if topic == "Custom...": topic = st.text_input("Enter Topic")
                if st.button("Start Quiz", type="primary", use_container_width=True):
                    with st.spinner("Generating..."):
                        q = st.session_state.brain.generate_quiz(topic, st.session_state.grade)
                        st.session_state.quiz_data = q
                        st.session_state.q_topic = topic
                        st.rerun()

        # C. History & System
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✨ New"):
                st.session_state.messages = []
                st.rerun()
        with c2:
            if st.button("🗑️ Clear"):
                st.session_state.messages = []
                st.rerun()
                
        theme = st.radio("Theme", ["Dark Mode 🌑", "Light Mode ☀️"], horizontal=True)
        UIFactory.inject_css(theme)
        
        if st.button("🔄 Sync Library", use_container_width=True):
            if st.session_state.brain.ingest_library():
                st.session_state.db_ready = True
                st.success("Synced!")
                time.sleep(1); st.rerun()
                
        up_file = st.file_uploader("Upload PDF", type="pdf")
        if up_file:
            if f"p_{up_file.name}" not in st.session_state:
                with st.spinner("Ingesting..."):
                    if st.session_state.brain.process_upload(up_file):
                        st.session_state.db_ready = True
                        st.session_state[f"p_{up_file.name}"] = True
                        st.success("Done!")
                        time.sleep(1); st.rerun()

    # --- MAIN CONTENT AREA ---
    st.title("NexGen Tutor")
    
    # DB Check
    if not st.session_state.db_ready:
        if st.session_state.brain.load_knowledge_base():
            st.session_state.db_ready = True
        else:
            st.info("👈 **Action Required:** Please Sync Library or Upload a PDF to activate the AI.")

    # QUIZ RENDER
    if st.session_state.get("quiz_data"):
        st.markdown(f"### 📝 Assessment: {st.session_state.q_topic}")
        score = 0
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            ans = st.radio(f"Select:", q['options'], key=f"q{i}", label_visibility="collapsed")
            if ans == q['answer']:
                st.success("Correct!")
                score += 1
            elif ans:
                st.error(f"Incorrect. Answer: {q['answer']}")
                st.caption(f"💡 {q['explanation']}")
            st.markdown("---")
        if st.button("Close Quiz", type="primary"):
            st.session_state.quiz_data = None
            st.rerun()

    # CHAT HISTORY
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio"): st.audio(msg["audio"])

    # CHAT INPUT & STREAMING
    if prompt := st.chat_input(f"Ask about {st.session_state.subject}..."):
        
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # 2. Assistant Response (Streaming)
        if st.session_state.db_ready:
            with st.chat_message("assistant"):
                response_container = st.empty()
                full_text = ""
                
                try:
                    # Get Chain
                    chain = st.session_state.brain.get_streaming_chain(
                        st.session_state.grade, 
                        st.session_state.subject
                    )
                    
                    # STREAMING LOOP
                    # We iterate over the stream generator
                    for chunk in chain.stream({"input": prompt}):
                        if "answer" in chunk:
                            text_chunk = chunk["answer"]
                            full_text += text_chunk
                            # Update UI with typewriter cursor
                            response_container.markdown(full_text + "▌")
                            
                    # Final update without cursor
                    response_container.markdown(full_text)
                    
                    # Audio Gen (Sync wrapper)
                    audio_file = None
                    try:
                        clean_txt = re.sub(r'[*_#`]', '', full_text)
                        async def get_audio():
                            c = edge_tts.Communicate(clean_txt, "en-GB-SoniaNeural")
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                                await c.save(f.name)
                                return f.name
                        audio_file = asyncio.run(get_audio())
                        if audio_file: st.audio(audio_file)
                    except: pass
                    
                    # Save history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_text,
                        "audio": audio_file
                    })
                    
                except Exception as e:
                    st.error(f"Generation Error: {e}")

if __name__ == "__main__":
    main()
