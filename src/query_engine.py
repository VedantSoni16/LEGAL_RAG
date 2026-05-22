import os
import re
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types

# --- Configurations ---
CHROMA_DIR = "chroma_db"

class LegalRagEngine:
    def __init__(self):
        print("🤖 Initializing Retrieval Core Framework...")
        
        # 🌟 PRODUCTION METADATA SAFEGUARD: 
        # Checks background cloud systems before initializing API modules
        if "GEMINI_API_KEY" not in os.environ:
            # Fallback for local terminal executions only
            os.environ["GEMINI_API_KEY"] = "AIzaSyDSjWKw03o1mZIq1yOoQca5ILXVlh65xo0"
            
        self.embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Connect back to the persistent Chroma directory
        self.vector_db = Chroma(
            persist_directory=CHROMA_DIR, 
            embedding_function=self.embedding_engine
        )
        
        # Initialize the modern Google GenAI Client
        self.ai_client = genai.Client()
        self.llm_model = "gemini-2.5-flash" 

    def retrieve_context(self, user_query, domain):
        """Searches the local database applying strict metadata routing filters based on chosen domain."""
        print(f"🔍 Searching Vector Store inside [{domain.upper()}] space...")
        search_filter = {"domain": domain}
        docs = self.vector_db.similarity_search(user_query, k=4, filter=search_filter)
        return docs

    def generate_answer(self, user_query, domain):
        # 1. Retrieve the filtered contexts
        matched_chunks = self.retrieve_context(user_query, domain)
        if not matched_chunks:
            return "No matching legal references found inside the database for this domain.", []

        context_str = ""
        citations = []
        for doc in matched_chunks:
            meta = doc.metadata
            citation_tag = f"Act: {meta.get('act')}, Section/Clause ID: {meta.get('section_id')}"
            citations.append(citation_tag)
            context_str += f"\n--- Reference Source [{citation_tag}] ---\n{doc.page_content}\n"

        # System instructions framework for the LLMs
        system_instruction = (
            "You are an expert Indian Legal Advisor AI. Your goal is to provide highly precise counsel "
            "using ONLY the legally valid text blocks provided in the Context below.\n\n"
            "CRITICAL RULES:\n"
            "1. Grounding: Rely exclusively on the provided context source blocks. If the answer cannot be found "
            "definitively inside the context, state clearly: 'I am sorry, but I cannot find a valid direct reference for this query inside the database.'\n"
            "2. Citations: You must explicitly cite the Act name and Section Number for every single claim you write.\n"
            "3. Formatting: Structure your response using clean bullet points and clear, scannable headings."
        )
        user_prompt_content = f"Context Material:\n{context_str}\n\nUser Question: {user_query}"

        # 2. Primary Execution Stream: Gemini Client
        try:
            response = self.ai_client.models.generate_content(
                model=self.llm_model,
                contents=user_prompt_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                )
            )
            return response.text, list(set(citations))

        # 3. 🌟 Automatic Fallback Engine if Gemini hits a 503 or 429 Error
        except Exception as gemini_error:
            print(f"⚠️ Gemini throttled (503). Pausing briefly, then activating fallback Groq core engine...")
            time.sleep(1) # Added defensive delay to allow cloud traffic spikes to settle
            try:
                from groq import Groq
                groq_client = Groq() 
                
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt_content}
                    ],
                    model="llama-3.1-8b-instant", 
                    temperature=0.1
                )
                return chat_completion.choices[0].message.content, list(set(citations))
                
            except Exception as groq_error:
                # 🌟 ULTIMATE OFFLINE FAILOVER: If both API servers crash, 
                # dump the raw text blocks safely onto the web interface directly.
                panic_recovery_text = (
                    "🚨 **[All Cloud AI Providers Saturated]**\n\n"
                    "The cloud models are experiencing temporary load issues. However, your local database successfully "
                    "isolated the corresponding legal text blocks natively without cloud assistance. Please review the raw statutes below:\n\n"
                    f"{context_str}"
                )
                return panic_recovery_text, list(set(citations))

# --- Simple CLI Tester Loop ---
if __name__ == "__main__":
    # Fallback key loaders logic for local verification testing execution routines
    if "GEMINI_API_KEY" not in os.environ or os.environ["GEMINI_API_KEY"] == "":
        os.environ["GEMINI_API_KEY"] = "AIzaSyDSjWKw03o1mZIq1yOoQca5ILXVlh65xo0"
    if "GROQ_API_KEY" not in os.environ or os.environ["GROQ_API_KEY"] == "":
        os.environ["GROQ_API_KEY"] = "gsk_YourActualGroqKeyStringHere..." # Put your real key string here for local testing

    engine = LegalRagEngine()
    
    # Test Question 1: Criminal Domain Check
    q1 = "What happens if a group of 6 people beat someone to death over their caste? Is there a new section for mob lynching?"
    ans, cites = engine.generate_answer(q1, domain="criminal")
    print(f"\n=================================\nQ: {q1}\nAnswer:\n{ans}\nCitations Retrieved: {cites}")
    
    print("\n---------------------------------\n")
    
    # Test Question 2: Land Domain Check
    q2 = "Is it mandatory to register a rent agreement if the lease period is only for 11 months?"
    ans2, cites2 = engine.generate_answer(q2, domain="land")
    print(f"Q: {q2}\nAnswer:\n{ans2}\nCitations Retrieved: {cites2}")