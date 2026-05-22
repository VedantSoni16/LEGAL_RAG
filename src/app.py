import os
import sys
import streamlit as st

# 🌟 THE FOOLPROOF PATH FIX: Get the absolute path of the directory containing app.py (the 'src' folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now you can import query_engine directly with absolute zero path issues!
from query_engine import LegalRagEngine

# Initialize the backend engine once across the web session using st.cache_resource
@st.cache_resource
def init_backend_core():
    return LegalRagEngine()

rag_core = init_backend_core()

# --- UI Layout Configuration ---
# 🌟 "centered" layout packs your chat window tightly right in the middle of wide displays
st.set_page_config(page_title="Indian Legal RAG Assistant", layout="centered", page_icon="⚖️")

st.title("⚖️ Indian Legal Framework RAG Assistant")
st.caption("Advanced Domain-Isolated Multi-Bot Retrieval System (BNS, IPC, BNSS, BSA & Land Laws)")
st.write("---")

# 1. Sidebar Configuration Window
st.sidebar.header("🤖 Chatbot Domain Routing")
chosen_domain = st.sidebar.selectbox(
    "Select Active Bot Domain:",
    options=["criminal", "land"],
    format_func=lambda x: "🚨 Criminal Law Assistant" if x == "criminal" else "🏡 Land & Property Assistant"
)

# 2. Main Conversational App Interface
user_query = st.chat_input("Ask a legal question or search a section...")

if user_query:
    # Render user chat bubble instantly
    with st.chat_message("user"):
        st.write(user_query)
        
    # Trigger processing state spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching local vector indices and validating context..."):
            answer, citations = rag_core.generate_answer(user_query, domain=chosen_domain)
            
            # Print the structured text response
            st.markdown(answer)
            
            # If citations were located, render them nicely inside individual UI badges
            if citations:
                st.write("")
                st.markdown("**Verified Source Citations:**")
                cols = st.columns(len(citations) if len(citations) < 5 else 4)
                for idx, cite in enumerate(citations):
                    col_idx = idx % 4
                    cols[col_idx].info(f"📖 {cite}")