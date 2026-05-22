import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration & Paths ---
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# Grouping our files cleanly into your two distinct chatbot domains
DOMAINS = {
    "criminal": [
        "bns_final_clean.csv",
        "ipc_cleaned.csv",
        "bnss_final_clean.csv",
        "bsa_cleaned.csv"
    ],
    "land": [
        "property_transfer_cleaned.csv",
        "property_registration_cleaned.csv"
    ]
}

def load_csv_to_langchain_docs(file_name, domain_label):
    """Reads a structural CSV file and converts each row into a LangChain Document object."""
    file_path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File {file_path} not found. Skipping...")
        return []
        
    df = pd.read_csv(file_path)
    documents = []
    
    for _, row in df.iterrows():
        # Safeguard against empty data rows
        if pd.isna(row['text_content']):
            continue
            
        # Create a unified structured text block for the vector to read
        text = str(row['text_content']).strip()
        
        # Enriched metadata tracking layout (Crucial for routing and citations later)
        metadata = {
            "act": str(row['act']),
            "section_id": str(row['section_id']),
            "domain": domain_label  # 'criminal' or 'land'
        }
        
        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)
        
    return documents

def build_vector_database():
    print("🚀 Initializing Local Embedding Engine (all-MiniLM-L6-v2)...")
    # Loads sentence-transformers onto your local CPU for completely free, fast embedding generation
    embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Text splitter config: Keeps chunks tightly packed with context overlaps
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    
    all_processed_chunks = []
    
    # Process both domains sequentially
    for domain, file_list in DOMAINS.items():
        print(f"\n📁 Processing Domain Pool: [{domain.upper()}]")
        
        for file_name in file_list:
            # 1. Convert CSV records into raw long documents
            raw_docs = load_csv_to_langchain_docs(file_name, domain)
            if not raw_docs:
                continue
                
            # 2. Slice into sub-chunks while natively duplicating row metadata tags
            chunked_docs = text_splitter.split_documents(raw_docs)
            print(f" -> Parsed {file_name}: Converted {len(raw_docs)} laws into {len(chunked_docs)} vector chunks.")
            all_processed_chunks.extend(chunked_docs)
            
    # 3. Commit chunks into local persistent disk storage via ChromaDB
    print(f"\n📦 Loading {len(all_processed_chunks)} total chunks into ChromaDB at '{CHROMA_DIR}'...")
    
    db = Chroma.from_documents(
        documents=all_processed_chunks,
        embedding=embedding_engine,
        persist_directory=CHROMA_DIR
    )
    
    print("✅ Day 2 complete! Local vector database successfully established and saved to disk.")
    return db

if __name__ == "__main__":
    build_vector_database()