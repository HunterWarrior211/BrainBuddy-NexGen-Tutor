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
# 🎨 ULTRA-PREMIUM CSS STYLING (BOLD SIDEBAR)
# ==========================================
DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Playfair+Display:ital,wght@0,700;1,400&family=Exo+2:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');

    /* --- ✨ HYPER-GLOW GOLDEN SCROLLBAR ✨ --- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; margin-block: 5px; }
    ::-webkit-scrollbar-corner { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #BF953F, #FCF6BA, #B38728, #FBF5B7, #AA771C);
        border-radius: 10px;
        border: 2px solid #0A0E14;
        box-shadow: 0 0 10px rgba(212, 175, 55, 0.5);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FFD700, #FFFACD, #FFD700);
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.9);
        border: 1px solid #FFF;
    }
    * { scrollbar-width: thin; scrollbar-color: #D4AF37 #0A0E14; }

    /* --- GLOBAL THEME --- */
    .stApp {
        background-color: #0A0E14;
        background-image: radial-gradient(#1B1F28 1px, transparent 1px), linear-gradient(125deg, #0A0E14 0%, #11161F 40%, #0A0E14 100%);
        background-size: 40px 40px, 200% 200%;
        color: #EAEAEA;
    }

    /* --- TYPOGRAPHY --- */
    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: #D4AF37 !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 2px 2px 4px #000000;
    }
    h3 { border-bottom: 1px solid rgba(212, 175, 55, 0.3); padding-bottom: 5px; }

    /* --- SIDEBAR VISIBILITY IMPROVEMENTS --- */
    section[data-testid="stSidebar"] {
        background-color: #141A24;
        border-right: 2px solid #D4AF37;
        box-shadow: 5px 0 20px rgba(0,0,0,0.6);
    }
    /* Make Sidebar Text BOLDER and BRIGHTER */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div { 
        color: #FFFFFF !important; 
        font-family: 'Exo 2', sans-serif !important;
        font-weight: 600 !important; /* Bolder text */
        font-size: 1.05rem !important; /* Slightly larger */
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        text-shadow: 0 0 10px rgba(212, 175, 55, 0.4);
    }

    /* --- BUTTONS & INPUTS --- */
    div.stButton > button {
        background: linear-gradient(135deg, #D4AF37 0%, #B8962E 100%);
        color: #0A0E14;
        font-family: 'Cinzel', serif;
        font-weight: 900;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        transition: all 0.3s;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div.stButton > button:hover {
        transform: scale(1.05); color: #000;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.6);
    }
    
    .stTextInput > div > div > input, .stSelectbox > div > div {
        background-color: #1B1F28;
        color: #EAEAEA;
        border: 1px solid #D4AF37;
        font-weight: 500;
    }

    /* --- CHAT BUBBLES --- */
    .stChatMessage {
        background-color: #1B1F28;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    div[data-testid="stChatMessage"]:nth-child(odd) { border-left: 4px solid #8BE9FD; }
    div[data-testid="stChatMessage"]:nth-child(even) { border-left: 4px solid #D4AF37; background-color: #151921; }

    @media only screen and (max-width: 768px) {
        h1 { font-size: 1.8rem !important; text-align: center; }
        section[data-testid="stSidebar"] { width: 85% !important; }
        ::-webkit-scrollbar { width: 6px; }
    }
</style>
"""

LIGHT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700;900&family=Playfair+Display:ital,wght@0,700;1,400&family=Exo+2:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #BFC3C9; border-radius: 10px; border: 2px solid #F8F9FB; }
    ::-webkit-scrollbar-thumb:hover { background: #9FA4AA; }

    .stApp {
        background-color: #F8F9FB;
        background-image: radial-gradient(#BFC3C9 1.5px, transparent 1.5px), linear-gradient(120deg, #F8F9FB 0%, #FFFFFF 50%, #E5E7EB 100%);
        background-size: 30px 30px, 200% 200%;
        color: #5F6368;
    }

    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: #2B2E34 !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* --- SIDEBAR VISIBILITY (LIGHT MODE) --- */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #E5E7EB;
        box-shadow: 5px 0 20px rgba(0,0,0,0.05);
    }
    /* Bold Text for Clarity */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div { 
        color: #2B2E34 !important; 
        font-weight: 700 !important; /* Bold */
        font-size: 1.05rem !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #BFC3C9 0%, #9FA4AA 100%);
        color: #2B2E34;
        font-family: 'Cinzel', serif;
        font-weight: 800;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stButton>button:hover { transform: scale(1.05); background: #2B2E34; color: #FFFFFF; }
    
    div[data-testid="stChatMessage"]:nth-child(odd) { background-color: #F1F3F6; border: 1px solid #E5E7EB; border-left: 4px solid #BFC3C9; }
    div[data-testid="stChatMessage"]:nth-child(even) { background-color: #FFFFFF; border: 1px solid #E5E7EB; border-left: 4px solid #2B2E34; }
    
    @media only screen and (max-width: 768px) {
        h1 { font-size: 1.8rem !important; }
    }
</style>
"""


# ==========================================
# 🛠️ DATA MANAGEMENT
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
# 🧠 AI ENGINE (UPDATED PROMPTING)
# ==========================================
class DocumentProcessor:
    def __init__(self):
        self.vectordb_doc = None
        self.splits = []
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        except: pass

        if "GOOGLE_API_KEY" not in os.environ:
            local_secrets = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
            if os.path.exists(local_secrets):
                with open(local_secrets, "r") as f:
                    for line in f:
                        if "GOOGLE_API_KEY" in line:
                            os.environ["GOOGLE_API_KEY"] = line.split("=")[1].strip().strip('"').strip("'")

        if "GOOGLE_API_KEY" not in os.environ:
            st.error("⚠️ API Key Missing! Check secrets.toml")
            st.stop()

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

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
        bar = st.progress(0, "Scanning...")
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
            self.splits = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(docs)
            os.remove(temp_path)
            return self.create_embeddings_batched()
        except: return False

    def create_embeddings_batched(self):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if self.vectordb_doc is None:
            self.vectordb_doc = Chroma.from_documents(self.splits[:20], embeddings, persist_directory=PERSIST_DIR)
            start = 20
        else: start = 0
        try:
            bar = st.progress(0, "Memorizing...")
            for i in range(start, len(self.splits), 20):
                self.vectordb_doc.add_documents(self.splits[i:i + 20])
                bar.progress(min((i + 20) / len(self.splits), 1.0))
            bar.empty()
            return True
        except: return False

    def extract_chat_topics(self, chat_history):
        prompt = f"Analyze chat. Return strictly comma-separated list of educational topics (max 4). Chat: {chat_history}"
        res = self.llm.invoke([HumanMessage(content=prompt)])
        return [t.strip() for t in res.content.split(',')[:4]]

    def generate_quiz_json(self, topic, grade):
        prompt = f"""Create 3 MCQ quiz on '{topic}' for {grade}. 
        Return strictly JSON: [{{"question": "...", "options": ["A)..."], "answer": "A)...", "explanation": "..."}}]"""
        res = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            t = res.content.strip().replace("```json", "").replace("```", "")
            return json.loads(t)
        except: return []

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
    except: return None

# ==========================================
# 🔄 SESSION STATE
# ==========================================
def init_session():
    if "current_user" not in st.session_state: st.session_state.current_user = None
    if st.session_state.current_user:
        user = st.session_state.current_user
        if "saved_chats" not in st.session_state:
            st.session_state.saved_chats = load_json_db(HISTORY_FILE).get(user, {})
        if "processor" not in st.session_state:
            st.session_state.processor = DocumentProcessor()
            st.session_state.db_ready = False
        if "current_chat_id" not in st.session_state:
            if st.session_state.saved_chats:
                st.session_state.current_chat_id = list(st.session_state.saved_chats.keys())[-1]
            else:
                uid = str(uuid.uuid4())
                st.session_state.current_chat_id = uid
                st.session_state.saved_chats[uid] = []
        if "messages" not in st.session_state:
            st.session_state.messages = st.session_state.saved_chats.get(st.session_state.current_chat_id, [])

def save_chat_state():
    st.session_state.saved_chats[st.session_state.current_chat_id] = st.session_state.messages
    all_hist = load_json_db(HISTORY_FILE)
    all_hist[st.session_state.current_user] = st.session_state.saved_chats
    save_json_db(HISTORY_FILE, all_hist)

def save_quiz_score(topic, obtained, total):
    user = st.session_state.current_user
    scores = load_json_db(QUIZ_FILE)
    if user not in scores: scores[user] = []
    scores[user].append({"topic": topic, "obtained": obtained, "total": total, "date": datetime.now().strftime("%Y-%m-%d")})
    save_json_db(QUIZ_FILE, scores)

# ==========================================
# 🖥️ MAIN APP
# ==========================================
def main():
    if 'current_user' not in st.session_state or st.session_state.current_user is None:
        st.markdown(DARK_CSS, unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.title("NexGen Login")
            t1, t2 = st.tabs(["Login", "Sign Up"])
            with t1:
                u = st.text_input("User", key="l_u")
                p = st.text_input("Pass", type="password", key="l_p")
                if st.button("Login", use_container_width=True):
                    if authenticate_user(u, p):
                        st.session_state.clear()
                        st.session_state.current_user = u
                        st.rerun()
                    else: st.error("Invalid")
            with t2:
                nu = st.text_input("New User", key="s_u")
                np = st.text_input("New Pass", type="password", key="s_p")
                if st.button("Sign Up", use_container_width=True):
                    if register_new_user(nu, np): st.success("Created!")
                    else: st.error("Exists")
        return

    init_session()

    # --- SIDEBAR ---
    with st.sidebar:
        st.write(f"👤 **{st.session_state.current_user}**")
        if st.button("Logout"): st.session_state.clear(); st.rerun()
        
        st.header("🎨 Theme")
        theme = st.radio("Style", ["Dark Mode 🌑", "Light Mode ☀️"], key="theme")
        if "Dark" in theme: st.markdown(DARK_CSS, unsafe_allow_html=True)
        else: st.markdown(LIGHT_CSS, unsafe_allow_html=True)

        st.header("⚙️ Controls")
        if st.button("🔄 Sync Library"):
            if st.session_state.processor.process_local_library():
                st.session_state.db_ready = True
                st.success("Synced!")
                time.sleep(1); st.rerun()

        st.divider()
        if st.button("✨ New Chat"):
            uid = str(uuid.uuid4())
            st.session_state.current_chat_id = uid
            st.session_state.saved_chats[uid] = []
            st.session_state.messages = []
            save_chat_state()
            st.rerun()

        # --- PROGRESS ---
        with st.expander("📊 Dashboard"):
            scores = load_json_db(QUIZ_FILE).get(st.session_state.current_user, [])
            if scores:
                df = pd.DataFrame(scores)
                if 'total' not in df: df['total'] = 3
                if 'obtained' not in df and 'score' in df: df.rename(columns={'score':'obtained'}, inplace=True)
                
                base = alt.Chart(df).mark_bar(strokeDash=[4,4], stroke='white', fill=None).encode(x=alt.X('topic:N', axis=None), y='total:Q')
                bar = alt.Chart(df).mark_bar(color='#D4AF37', width=15).encode(x='topic:N', y='obtained:Q', tooltip=['topic','obtained'])
                st.altair_chart((base+bar).properties(height=200), use_container_width=True)

        st.divider()
        st.session_state['user_grade'] = st.selectbox("Grade:", [f"Grade {i}" for i in range(6, 11)])
        st.session_state['user_subject'] = st.selectbox("Subject:", ["Mathematics", "Physics", "Biology", "Chemistry", "History"])

        # --- QUIZ ---
        with st.expander("🧠 Quiz"):
            if st.button("🚀 Detect Topics"):
                h = "\n".join([m['content'] for m in st.session_state.messages])
                st.session_state.topics = st.session_state.processor.extract_chat_topics(h)
            
            if "topics" in st.session_state:
                topic = st.radio("Topic:", st.session_state.topics + ["Custom..."])
                if topic == "Custom...": topic = st.text_input("Enter Topic:")
                if st.button("Start Quiz"):
                    st.session_state.quiz = st.session_state.processor.generate_quiz_json(topic, st.session_state.user_grade)
                    st.session_state.q_topic = topic
                    st.rerun()

        if st.session_state.get('quiz'):
            st.divider()
            st.write(f"**Quiz: {st.session_state.q_topic}**")
            score = 0
            for i, q in enumerate(st.session_state.quiz):
                st.write(f"**{i+1}. {q['question']}**")
                sel = st.radio(f"Opt {i}", q['options'], key=f"q{i}", label_visibility="collapsed")
                if sel == q['answer']: score+=1; st.success("Correct")
                elif sel: st.error(f"Wrong. {q['explanation']}")
            if st.button("Save Score"):
                save_quiz_score(st.session_state.q_topic, score, len(st.session_state.quiz))
                st.success("Saved!"); st.session_state.quiz=None; st.rerun()

    # --- MAIN CHAT ---
    st.title("NexGen Tutor")
    
    if not st.session_state.db_ready:
        if st.session_state.processor.load_from_disk(): st.session_state.db_ready = True
        else: st.info("Please Sync Library")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio") and os.path.exists(msg["audio"]): st.audio(msg["audio"])

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_state()
        with st.chat_message("user"): st.markdown(prompt)

        if st.session_state.db_ready:
            with st.chat_message("assistant"):
                ph = st.empty()
                try:
                    retriever = st.session_state.processor.vectordb_doc.as_retriever()
                    docs = retriever.invoke(prompt)
                    
                    # --- 🧠 INTELLIGENT GRADE & SUBJECT LOGIC ---
                    grade = st.session_state.user_grade
                    subj = st.session_state.user_subject
                    g_num = int(grade.split()[1])

                    # 1. GRADE PROMPTING
                    if g_num == 6:
                        instr = "Explain like I am 11 years old. Keep it SHORT, FUN, and PRECISE. Max 3 bullet points. No complex words."
                    elif g_num == 7:
                        instr = "Explain like I am 12. Keep it concise. Focus on the main idea. Slightly more detail than Grade 6 but still brief."
                    elif g_num == 8:
                        instr = "Standard Middle School level. Provide a clear definition and 1 or 2 key examples. Moderate detail."
                    elif g_num == 9:
                        instr = "High School level. Detailed explanation required. Include theoretical background and specific examples."
                    else: # 10
                        instr = "Board Exam Prep level. COMPREHENSIVE and HIGHLY DETAILED. Use technical terms. Break down into point-wise explanations for full marks."

                    # 2. SUBJECT GUARD
                    subject_guard = f"""
                    STRICT SUBJECT RULE:
                    - You are a **{subj}** Tutor.
                    - If the user asks a question about a completely different subject (e.g. asking Physics questions while Subject is Math), 
                      REFUSE TO ANSWER and tell them to switch the subject in the sidebar.
                    - Exception: If the question overlaps (e.g. Math in Physics), answer it.
                    """

                    full_prompt = f"""
                    You are an expert AI Tutor for {grade} {subj}.
                    {subject_guard}
                    
                    GRADE INSTRUCTION ({grade}): {instr}

                    REQUIRED STRUCTURE:
                    1. **Core Concept:** (The main definition, properly bolded)
                    2. **Main Data:** (The explanation based on grade level)
                    3. **Comparison:** (Use Markdown Table IF comparing two things)
                    4. **Real-World Example:** (Relatable scenario)
                    5. **Math/Physics Solver:** (IF applicable, use LaTeX $$...$$ for formulas. Step-by-step bullet points).

                    Context: {docs}
                    Chat History: {st.session_state.messages[-4:]}
                    Question: {prompt}
                    """
                    
                    chain = create_stuff_documents_chain(st.session_state.processor.llm, ChatPromptTemplate.from_template(full_prompt))
                    res = chain.invoke({"context": docs})
                    
                    # Output
                    ph.markdown(res)
                    af = text_to_audio(res)
                    if af: st.audio(af)
                    
                    st.session_state.messages.append({"role": "assistant", "content": res, "audio": af})
                    save_chat_state()

                except Exception as e: st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
