import os
import sys
import streamlit as st

# 🌟 THE FOOLPROOF PATH FIX: Get the absolute path of the directory containing app.py (the 'src' folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import your pristine backend engine class
from query_engine import LegalRagEngine

# Initialize the backend engine once across the web session using st.cache_resource
@st.cache_resource
def init_backend_core():
    return LegalRagEngine()

rag_core = init_backend_core()

# --- UI Layout Configuration ---
st.set_page_config(page_title="Indian Legal Assistant", layout="centered", page_icon="⚖️")

st.title("⚖️ Indian Legal Framework Assistant")
st.caption("Advanced Domain-Isolated Multi-Bot Retrieval System (BNS, IPC, BNSS, BSA & Land Laws)")
st.write("---")

# 🧠 --- 1. CHAT HISTORY MEMORY INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. Sidebar Configuration Window ---
st.sidebar.header("Legal Domain")


# Using st.radio instead of st.selectbox so both options remain permanently visible!
chosen_domain = st.sidebar.radio(
    "Select Active Bot Domain:",
    options=["criminal", "land"],
    format_func=lambda x: "🚨 Criminal Law Assistant" if x == "criminal" else "🏡 Land & Property Assistant"
)

# Clear chat history button in sidebar (Great utility feature for users/interviewers)
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()


# 🧠 --- 3. RENDER CONVERSATIONAL HISTORY TRAIL ---
# Automatically loops and displays previous interactions whenever the app triggers a rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Re-render the citation row badges if they exist for that specific historic response
        if "citations" in message and message["citations"]:
            st.write("")
            st.markdown("**Verified Source Citations:**")
            cols = st.columns(len(message["citations"]) if len(message["citations"]) < 5 else 4)
            for idx, cite in enumerate(message["citations"]):
                col_idx = idx % 4
                cols[col_idx].info(f"📖 {cite}")


# 🚀 --- 4. MAIN CONVERSATIONAL INPUT & PROCESSING INTERFACE ---
user_query = st.chat_input("Ask a legal question or search a section...")

if user_query:
    # Append the user's prompt to memory array instantly and print bubble to UI
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Trigger processing state spinner inside assistant container bubble
    with st.chat_message("assistant"):
        with st.spinner("Searching local vector indices and validating context..."):
            answer, citations = rag_core.generate_answer(user_query, domain=chosen_domain)
            
            # Print the structured text response live
            st.markdown(answer)
            
            # If citations were located, render them nicely inside individual UI badges
            if citations:
                st.write("")
                st.markdown("**Verified Source Citations:**")
                cols = st.columns(len(citations) if len(citations) < 5 else 4)
                for idx, cite in enumerate(citations):
                    col_idx = idx % 4
                    cols[col_idx].info(f"📖 {cite}")
            
            # Append assistant response along with its isolated citations to memory arrays
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "citations": citations
            })