# ⚖️ Indian Legal RAG Assistant

A domain-isolated Retrieval-Augmented Generation (RAG) system that lets you query **1,700+ sections of Indian law** in plain English and get cited, grounded answers — with source references down to the exact Act and Section number.

Built with LangChain, ChromaDB, HuggingFace embeddings, and Streamlit. Deployed free at [legalragassistant.streamlit.app](https://legalragassistant.streamlit.app).

---

## What it does

You type a legal question like *"What is the punishment for murder under BNS 2023?"* and the system:

1. Routes your query to the correct legal domain (Criminal or Land & Property)
2. Searches a local vector database of 1,700+ law sections for the most relevant chunks
3. Sends those chunks along with your question to an LLM
4. Returns a structured answer that cites the exact Act name and Section number
5. Falls back gracefully from Gemini → Groq → raw statute text if APIs are rate-limited

---

## Legal domains covered

**Criminal Law**
- Bharatiya Nyaya Sanhita (BNS) 2023 — 358 sections, replaces IPC
- Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023 — 530 sections, replaces CrPC
- Bharatiya Sakshya Adhiniyam (BSA) 2023 — 170 sections, replaces Evidence Act
- Indian Penal Code (IPC) 1860 — 506 sections (retained for historical reference)

**Land & Property Law**
- Transfer of Property Act 1882 — 122 sections
- Registration Act 1908 — 92 sections

---

## Tech stack

| Layer | Tool | Purpose |
|---|---|---|
| Data extraction | PyPDF, pandas | Parse PDFs, clean CSVs |
| Chunking | LangChain RecursiveCharacterTextSplitter | Split sections into 1200-char chunks |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Convert text to vectors locally |
| Vector store | ChromaDB | Store and retrieve embeddings by domain |
| LLM (primary) | Gemini 2.5 Flash (Google GenAI) | Generate grounded answers |
| LLM (fallback) | Llama 3.1 8B via Groq | Automatic failover on rate limits |
| Framework | LangChain | Retrieval chain orchestration |
| Frontend | Streamlit | Chat UI with domain selector |
| Deployment | Streamlit Community Cloud | Free hosting |
| Evaluation | RAGAS + custom metrics | Faithfulness, hit rate, section grounding |

---

## Project structure

```
LEGAL_RAG/
├── data/
│   ├── bns_final_clean.csv          # BNS 2023 — 358 sections
│   ├── bnss_final_clean.csv         # BNSS 2023 — 530 sections
│   ├── bsa_cleaned.csv              # BSA 2023 — 170 sections
│   ├── ipc_cleaned.csv              # IPC 1860 — 506 sections
│   ├── property_transfer_cleaned.csv    # TPA 1882 — 122 sections
│   └── property_registration_cleaned.csv # Registration Act 1908 — 92 sections
├── src/
│   ├── vector_store.py     # Builds ChromaDB from CSVs
│   ├── query_engine.py     # RAG retrieval + LLM generation + fallback
│   ├── app.py              # Streamlit frontend
│   ├── evaluate.py         # 3-layer evaluation suite (29 Q&A pairs)
│   ├── bns_clean.py        # PDF cleaning pipeline for BNS
│   ├── ingest.py           # General ingestion pipeline
│   └── ingest_*.py         # Per-act ingestion scripts
├── chroma_db/              # Local vector database (gitignored)
├── eval_results/           # Evaluation outputs (gitignored)
├── requirements.txt
└── .env                    # API keys (gitignored)
```

---

## Run locally

**1. Clone the repo**
```bash
git clone https://github.com/VedantSoni16/LEGAL_RAG.git
cd LEGAL_RAG
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
pip install python-dotenv
```

**3. Set up API keys**

Create a `.env` file in the root:
```
GOOGLE_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

Get keys free at:
- Gemini: [aistudio.google.com](https://aistudio.google.com)
- Groq: [console.groq.com](https://console.groq.com)

**4. Build the vector database**
```bash
python src/vector_store.py
```
This reads all 6 CSVs, chunks them, embeds them locally using MiniLM, and saves to `chroma_db/`. Takes 2–3 minutes on first run.

**5. Run the app**
```bash
streamlit run src/app.py
```

---

## Run the evaluation suite

```bash
pip install ragas datasets
python src/evaluate.py
```

Runs 3 layers of evaluation across 29 hand-crafted legal Q&A pairs:

- **Layer 1 — Retrieval**: Hit Rate and MRR (does the correct section land in top-k?)
- **Layer 2 — RAGAS**: Faithfulness, answer relevancy, context precision, context recall
- **Layer 3 — Legal-specific**: Section grounding rate, no-hallucination rate, citation accuracy

Results are saved to `eval_results/` as CSVs and a `summary.txt` report.

---

## How the RAG pipeline works

```
User query
    ↓
Domain filter (criminal / land)
    ↓
ChromaDB similarity search (k=6, metadata-filtered)
    ↓
Top-k chunks + metadata (act, section_id)
    ↓
Prompt: system instruction + context + query
    ↓
Gemini 2.5 Flash  →  (fallback) Groq Llama 3.1  →  (fallback) raw statute text
    ↓
Structured answer + citation badges
```

The key design decision is **domain-isolated retrieval** — each query only searches within its domain's chunks using ChromaDB metadata filtering. A land law question never retrieves criminal law sections, which prevents cross-domain hallucination and significantly improves precision.

---

## Data sources

All statutes sourced from official government repositories:
- [indiacode.nic.in](https://indiacode.nic.in) — BNS, BNSS, BSA, Registration Act
- [legislative.gov.in](https://legislative.gov.in) — Transfer of Property Act, IPC

---

## Disclaimer

This tool is for **educational and research purposes only**. It is not a substitute for professional legal advice. Always consult a qualified lawyer for legal matters.