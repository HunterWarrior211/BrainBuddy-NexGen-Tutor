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

for path in [UPLOAD_DIR, BOOKS_FOLDER, PERSIST_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

st.set_page_config(
    page_title="NexGen Tutor | AI Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🎨 ULTRA-PREMIUM CSS (ROUNDED & GOLDEN)
# ==========================================
DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Exo+2:wght@400;600;800&family=Inter:wght@400;600&display=swap');

    /* --- ✨ GOLDEN SCROLLBAR (HIGH VISIBILITY) ✨ --- */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
        background: #0A0E14;
    }
    ::-webkit-scrollbar-track {
        background: #11161F;
        border-left: 1px solid #333;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FFD700, #B8860B, #FFD700);
        border-radius: 6px;
        border: 2px solid #0A0E14;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.7);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FFFACD, #FFD700, #FFFACD);
        box-shadow: 0 0 25px rgba(255, 215, 0, 1);
    }

    /* --- GLOBAL THEME --- */
    .stApp {
        background-color: #0A0E14;
        background-image: radial-gradient(#1B1F28 1px, transparent 1px);
        background-size: 30px 30px;
        color: #EAEAEA;
    }

    /* --- SIDEBAR TYPOGRAPHY (PROMINENT) --- */
    section[data-testid="stSidebar"] {
        background-color: #141A24;
        border-right: 2px solid #D4AF37;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #D4AF37 !important;
        font-family: 'Cinzel', serif !important;
        text-shadow: 0px 0px 5px rgba(212, 175, 55, 0.5);
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-size: 1.1rem !important; /* Larger text */
        font-weight: 600 !important;   /* Thicker font */
        font-family: 'Exo 2', sans-serif !important;
    }

    /* --- ROUNDED CORNERS (MINI MILITIA STYLE) --- */
    div.stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #B8962E 100%);
        color: #0A0E14;
        font-family: 'Exo 2', sans-serif;
        font-weight: 800;
        border: none;
        border-radius: 18px !important; /* ROUNDED */
        padding: 0.6rem 1.4rem;
        transition: all 0.3s;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.6);
        color: #000;
    }
    
    /* Inputs & Selectboxes */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #1B1F28;
        color: #EAEAEA;
        border: 1px solid #D4AF37;
        border-radius: 18px !important; /* ROUNDED */
    }

    /* --- CHAT BUBBLES --- */
    .stChatMessage {
        background-color: #1B1F28;
        border-radius: 20px !important; /* ROUNDED */
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    div[data-testid="stChatMessage"]:nth-child(odd) { 
        border-left: 5px solid #8BE9FD; 
        border-radius: 20px 20px 20px 4px !important;
    }
    div[data-testid="stChatMessage"]:nth-child(even) { 
        border-right: 5px solid #D4AF37; 
        border-radius: 20px 20px 4px 20px !important;
        background-color: #151921;
    }

    h1, h2, h3 { font-family: 'Cinzel', serif !important; color: #D4AF37 !important; }
    p, li { font-family: 'Exo 2', sans-serif; font-size: 16px; line-height: 1.7; }
    strong { color: #D4AF37 !important; }
</style>
"""

LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Exo+2:wght@400;600;800&display=swap');

    /* --- LIGHT MODE SCROLLBAR --- */
    ::-webkit-scrollbar { width: 12px; height: 12px; }
    ::-webkit-scrollbar-thumb {
        background: #BFC3C9;
        border-radius: 6px;
        border: 2px solid #F8F9FB;
    }

    .stApp { background-color: #F8F9FB; color: #2B2E34; }

    /* Sidebar Prominence */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #BFC3C9;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {
        color: #2B2E34 !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    /* Rounded UI */
    div.stButton > button {
        background: linear-gradient(135deg, #BFC3C9 0%, #9FA4AA 100%);
        color: #2B2E34;
        border-radius: 18px !important;
        font-weight: 800;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #FFFFFF;
        color: #2B2E34;
        border: 2px solid #E5E7EB;
        border-radius: 18px !important;
    }

    .stChatMessage {
        border-radius: 20px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
    }
    
    h1, h2, h3 { font-family: 'Cinzel', serif !important; color: #2B2E34 !important; }
</style>
"""

# ==========================================
# 🛠️ DATA MANAGEMENT & AUTH
# ==========================================
def load_json_db(filepath):
    if not os.path.exists(filepath): return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_json_db(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f: json.dump(data, f, indent=4)
    except: pass

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def authenticate_user(username, password):
    users = load_json_db(USERS_FILE)
    if username in users and users[username] == hash_password(password): return True
    return False

def register_new_user(username, password):
    users = load_json_db(USERS_FILE)
    if username in users: return False
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
        
        # API Key Handling
        if "GOOGLE_API_KEY" not in os.environ:
            try:
                if "GOOGLE_API_KEY" in st.secrets:
                    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
                else:
                    local_secrets = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
                    if os.path.exists(local_secrets):
                        with open(local_secrets, "r") as f:
                            for line in f:
                                if "GOOGLE_API_KEY" in line:
                                    os.environ["GOOGLE_API_KEY"] = line.split("=")[1].strip().strip('"').strip("'")
            except: pass

        if "GOOGLE_API_KEY" not in os.environ:
            st.error("⚠️ API Key Missing! Check secrets.")
            st.stop()

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_retries=2)

    def load_from_disk(self):
        if os.path.exists(PERSIST_DIR):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.vectordb_doc = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            try:
                if self.vectordb_doc._collection.count() > 0: return True
            except: return False
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
            except: pass
        bar.empty()

        if not all_docs: return False
        self.splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(all_docs)
        return self.create_embeddings_batched()

    def process_uploaded_file(self, uploaded_file):
        temp_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())

        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            check = self.llm.invoke([HumanMessage(content=f"Is this educational? YES/NO. Text: {docs[0].page_content[:500]}")]).content
            if "NO" in check.upper():
                st.error("🚫 Educational content only.")
                os.remove(temp_path)
                return False

            self.splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(docs)
            os.remove(temp_path)
            return self.create_embeddings_batched()
        except Exception as e:
            st.error(f"Error: {e}")
            return False

    def create_embeddings_batched(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        batch_size = 20
        total = len(self.splits)
        
        if self.vectordb_doc is None:
            self.vectordb_doc = Chroma.from_documents(self.splits[:batch_size], embeddings, persist_directory=PERSIST_DIR)
            start = batch_size
        else:
            start = 0

        bar = st.progress(0, "Memorizing...")
        for i in range(start, total, batch_size):
            self.vectordb_doc.add_documents(self.splits[i:i+batch_size])
            bar.progress(min((i + batch_size) / total, 1.0))
            time.sleep(0.05)
        bar.empty()
        return True

    def extract_chat_topics(self, chat_history):
        prompt = f"Analyze chat history: {chat_history}. Return comma-separated list of 4 distinct educational topics."
        res = self.llm.invoke([HumanMessage(content=prompt)])
        return [t.strip() for t in res.content.split(',')[:4]]

    def generate_quiz_json(self, topic, grade):
        prompt = f"""
        Create a 3-question Multiple Choice Quiz on '{topic}' for {grade}.
        Return ONLY valid JSON: 
        [{{ "question": "...", "options": ["A)...", "B)..."], "answer": "A)...", "explanation": "..." }}]
        """
        res = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            content = res.content.strip().replace("```json", "").replace("```", "")
            return json.loads(content)
        except: return []

async def generate_edge_audio(text, filename):
    communicate = edge_tts.Communicate(text, "en-GB-SoniaNeural")
    await communicate.save(filename)

def text_to_audio(text):
    try:
        clean = re.sub(r'[\*\#\$]', '', text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            filename = fp.name
        asyncio.run(generate_edge_audio(clean, filename))
        return filename
    except: return None

# ==========================================
# 🔄 SESSION STATE
# ==========================================
def load_history_from_disk(username):
    return load_json_db(HISTORY_FILE).get(username, {})

def save_history_to_disk(username):
    all_data = load_json_db(HISTORY_FILE)
    all_data[username] = st.session_state.saved_chats
    save_json_db(HISTORY_FILE, all_data)

def save_quiz_score(topic, obtained, total):
    username = st.session_state.current_user
    scores = load_json_db(QUIZ_FILE)
    if username not in scores: scores[username] = []
    scores[username].append({"topic": topic, "obtained": obtained, "total": total, "date": datetime.now().strftime("%Y-%m-%d")})
    save_json_db(QUIZ_FILE, scores)

def init_session():
    if "current_user" not in st.session_state: st.session_state.current_user = None
    if st.session_state.current_user:
        user = st.session_state.current_user
        if "saved_chats" not in st.session_state: st.session_state.saved_chats = load_history_from_disk(user)
        if "processor" not in st.session_state: 
            st.session_state.processor = DocumentProcessor()
            st.session_state.db_ready = False
        if "current_chat_id" not in st.session_state: new_chat()
        if "messages" not in st.session_state: st.session_state.messages = []

def new_chat():
    uid = str(uuid.uuid4())
    st.session_state.current_chat_id = uid
    st.session_state.saved_chats[uid] = []
    st.session_state.messages = []
    st.session_state.show_quiz = False
    st.session_state.quiz_data = None
    if "detected_topics" in st.session_state: del st.session_state.detected_topics
    save_history_to_disk(st.session_state.current_user)

def load_chat(session_id):
    st.session_state.current_chat_id = session_id
    st.session_state.messages = list(st.session_state.saved_chats.get(session_id, []))
    st.session_state.show_quiz = False
    st.session_state.quiz_data = None

def delete_chat(session_id):
    if session_id in st.session_state.saved_chats:
        del st.session_state.saved_chats[session_id]
        save_history_to_disk(st.session_state.current_user)
    new_chat()

def get_chat_title(messages):
    if not messages: return "New Conversation"
    for msg in messages:
        if msg.get("role") == "user": return " ".join(msg.get("content", "").split()[:5])[:25] + "..."
    return "Conversation"

# ==========================================
# 🖥️ MAIN UI
# ==========================================
def main():
    if 'current_user' not in st.session_state or st.session_state.current_user is None:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.title("NexGen Tutor Login")
            tab1, tab2 = st.tabs(["Login", "Sign Up"])
            with tab1:
                u, p = st.text_input("Username"), st.text_input("Password", type="password")
                if st.button("Log In", use_container_width=True):
                    if authenticate_user(u, p):
                        st.session_state.clear()
                        st.session_state.current_user = u
                        st.rerun()
                    else: st.error("Invalid")
            with tab2:
                nu, np = st.text_input("New User"), st.text_input("New Pass", type="password")
                if st.button("Sign Up", use_container_width=True):
                    if register_new_user(nu, np): st.success("Created! Log in.")
                    else: st.error("Exists.")
        return

    init_session()

    # --- SIDEBAR LAYOUT (REORDERED AS REQUESTED) ---
    with st.sidebar:
        st.write(f"👤 **{st.session_state.current_user}**")
        
        # Theme Selector
        theme = st.radio("Theme", ["Dark Mode 🌑", "Light Mode ☀️"], horizontal=True, label_visibility="collapsed")
        if "Dark" in theme:
            st.markdown(DARK_CSS, unsafe_allow_html=True)
            chart_color = '#D4AF37'
        else:
            st.markdown(LIGHT_CSS, unsafe_allow_html=True)
            chart_color = '#2B2E34'

        st.divider()

        # 1. GRADE & SUBJECT (ON TOP)
        st.header("1. Study Settings")
        st.session_state['user_grade'] = st.selectbox("Select Grade:", [f"Grade {i}" for i in range(6, 11)], index=0)
        st.session_state['user_subject'] = st.selectbox("Select Subject:", ["Mathematics", "Physics", "Biology", "Chemistry", "History", "Geography"])

        st.divider()

        # 2. ACTIONS & QUIZ
        st.header("2. Actions")
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("✨ New Chat", use_container_width=True):
                new_chat()
                st.rerun()
        with col_act2:
            if st.button("🗑️ Reset", use_container_width=True):
                st.session_state.messages = []
                st.session_state.saved_chats[st.session_state.current_chat_id] = []
                st.rerun()
        
        # Quiz Section
        with st.expander("🧠 Take a Quiz", expanded=True):
            if st.button("🚀 Identify Topics", use_container_width=True):
                if st.session_state.messages:
                    hist = "\n".join([m['content'] for m in st.session_state.messages])
                    st.session_state.detected_topics = st.session_state.processor.extract_chat_topics(hist)
                    st.rerun()
                else: st.warning("Chat first!")
            
            if "detected_topics" in st.session_state:
                topic = st.radio("Pick Topic:", st.session_state.detected_topics)
                if st.button("Start Quiz", use_container_width=True):
                    st.session_state.quiz_data = st.session_state.processor.generate_quiz_json(topic, st.session_state['user_grade'])
                    st.session_state.current_quiz_topic = topic
                    st.rerun()

        st.divider()

        # 3. CHAT HISTORY
        st.header("3. History")
        with st.container(height=200):
            for sid in reversed(list(st.session_state.saved_chats.keys())):
                title = get_chat_title(st.session_state.saved_chats[sid])
                if st.button(f"🗨️ {title}", key=sid, use_container_width=True):
                    load_chat(sid)
                    st.rerun()

        st.divider()

        # 4. DASHBOARD (BOTTOM)
        st.header("4. Progress")
        with st.expander("📊 View Results"):
            scores = load_json_db(QUIZ_FILE).get(st.session_state.current_user, [])
            if scores:
                df = pd.DataFrame(scores)
                if 'total' not in df.columns: df['total'] = 3
                
                chart = alt.Chart(df).mark_bar(color=chart_color, cornerRadiusEnd=4).encode(
                    x=alt.X('topic:N', title=None),
                    y=alt.Y('obtained:Q', title='Score'),
                    tooltip=['topic', 'obtained', 'total']
                ).properties(height=150)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No quizzes taken yet.")
        
        # Utils
        if st.button("🔄 Sync Library", type="secondary"):
            st.session_state.processor.process_local_library()
            st.rerun()
        
        up_file = st.file_uploader("Upload PDF", type="pdf")
        if up_file:
            if f"proc_{up_file.name}" not in st.session_state:
                st.session_state.processor.process_uploaded_file(up_file)
                st.session_state[f"proc_{up_file.name}"] = True
                st.session_state.db_ready = True
                st.rerun()

    # --- MAIN CONTENT ---
    # Quiz Display
    if st.session_state.get('quiz_data'):
        st.subheader(f"📝 Quiz: {st.session_state.get('current_quiz_topic')}")
        score = 0
        for i, q in enumerate(st.session_state.quiz_data):
            st.write(f"**Q{i+1}: {q['question']}**")
            ans = st.radio(f"Options {i}", q['options'], key=f"q_{i}", index=None)
            if ans:
                if ans == q['answer']: 
                    st.success("Correct!")
                    score += 1
                else: 
                    st.error(f"Wrong. Answer: {q['answer']}")
                    st.info(q['explanation'])
            st.write("---")
        
        if st.button("Save Score"):
            save_quiz_score(st.session_state.current_quiz_topic, score, len(st.session_state.quiz_data))
            st.session_state.quiz_data = None
            st.rerun()
        return

    # Chat Interface
    st.title("NexGen Tutor")
    
    if not st.session_state.db_ready:
        if st.session_state.processor.load_from_disk(): st.session_state.db_ready = True
        else: st.info("👈 Upload a PDF or Sync Library to start learning.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio") and os.path.exists(msg["audio"]): st.audio(msg["audio"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        if st.session_state.db_ready:
            with st.chat_message("assistant"):
                try:
                    retriever = st.session_state.processor.vectordb_doc.as_retriever()
                    docs = retriever.invoke(prompt)
                    
                    # --- PROMPT LOGIC PER GRADE ---
                    grade_str = st.session_state.get('user_grade', 'Grade 6')
                    grade_num = int(re.search(r'\d+', grade_str).group()) if re.search(r'\d+', grade_str) else 6
                    subject = st.session_state.get('user_subject', 'Science')

                    if grade_num == 6:
                        instructions = """
                        **TARGET: Grade 6 Student (Beginner)**
                        - **Length:** SHORT, PRECISE, SIMPLE.
                        - **Structure:** Only Main Points.
                        - **Style:** No complex words. Use bullet points heavily.
                        - **Visuals:** Create a clear distinction between concepts.
                        """
                    elif grade_num == 7:
                        instructions = """
                        **TARGET: Grade 7 Student**
                        - **Length:** Moderate.
                        - **Structure:** Definition + 1 Example.
                        - **Style:** Friendly but educational.
                        - **Detail:** A bit more detail than Grade 6, but still simple.
                        """
                    elif grade_num == 8:
                        instructions = """
                        **TARGET: Grade 8 Student**
                        - **Length:** Standard.
                        - **Structure:** Definition + 2 Examples.
                        - **Style:** Informative.
                        - **Detail:** Include 'How' and 'Why'.
                        """
                    elif grade_num == 9:
                        instructions = """
                        **TARGET: Grade 9 Student (High School)**
                        - **Length:** Detailed.
                        - **Structure:** Proper Definition + Process + Real World Application.
                        - **Style:** Academic/Formal.
                        - **Visuals:** Use Markdown tables for comparisons.
                        """
                    else: # Grade 10+
                        instructions = """
                        **TARGET: Grade 10 Student (Exam Prep)**
                        - **Length:** COMPREHENSIVE & COMPLETE.
                        - **Structure:** In-depth analysis.
                        - **Style:** Professional, Technical, Exam-Oriented.
                        - **Requirement:** Visually distinct depth. Cover exceptions and formulas.
                        """

                    template = """
                    You are an expert AI tutor with 100+ years of experience.
                    Context: {context}
                    Chat History: {chat_history}
                    
                    **USER SETTINGS:**
                    Grade: {grade} | Subject: {subject}
                    Instructions: {complexity_instruction}

                    **MANDATORY RESPONSE STRUCTURE (Follow Strictly):**
                    1. **CORE CONCEPT:** A clear, standout definition suitable for the grade.
                    2. **MAIN CONCEPTS:** Bullet points explaining the key mechanisms.
                    3. **COMPARISON TABLE:** If the user asks for a difference (e.g., Mitosis vs Meiosis), YOU MUST use a Markdown Table.
                    4. **REAL-WORLD EXAMPLE:** An analogy the student can relate to immediately.
                    5. **MATH/PHYSICS RULE:** If math is involved, use LaTeX ($x^2$) and solve Step-by-Step.
                    6. **NO IMAGES.**

                    Question: {input}
                    """
                    
                    hist_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:]])
                    custom_prompt = ChatPromptTemplate.from_template(template)
                    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(st.session_state.processor.llm, custom_prompt))
                    
                    res = chain.invoke({
                        "input": prompt, "context": docs, "chat_history": hist_str,
                        "grade": grade_str, "subject": subject, "complexity_instruction": instructions
                    })
                    
                    full_res = res["answer"]
                    st.markdown(full_res)
                    
                    audio_path = text_to_audio(full_res)
                    if audio_path: st.audio(audio_path)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_res, "audio": audio_path})
                    save_history_to_disk(st.session_state.current_user)
                    
                except Exception as e: st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
