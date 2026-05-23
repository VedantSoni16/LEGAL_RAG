"""
evaluate.py  —  LEGAL_RAG Evaluation Suite
============================================
Runs three layers of evaluation on your RAG pipeline:
  1. RAGAS metrics  : faithfulness, answer_relevancy, context_precision, context_recall
  2. Hit rate / MRR : checks if correct section lands in top-k retrieved chunks
  3. Custom legal   : citation accuracy, no-hallucination rate, section grounding rate

Usage (from repo root):
    cd LEGAL_RAG
    pip install ragas datasets --quiet
    GOOGLE_API_KEY=your_key python src/evaluate.py

Outputs:
    eval_results/ragas_scores.csv       — per-question RAGAS scores
    eval_results/retrieval_scores.csv   — hit-rate / MRR per question
    eval_results/legal_scores.csv       — citation / hallucination per question
    eval_results/summary.txt            — human-readable final report
"""

import os
import sys
import json
import csv
import time
from pathlib import Path
from datetime import datetime

# ── Make sure src/ imports work when running from repo root ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()
from query_engine import LegalRagEngine

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("eval_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# GOLDEN TEST SET
# 30 hand-crafted Q&A pairs with ground-truth answers and correct section IDs.
# Add more rows here as your corpus grows.
# ═════════════════════════════════════════════════════════════════════════════
GOLDEN_TEST_SET = [
    # ── Criminal domain ───────────────────────────────────────────────────────
    {
        "question": "What is the punishment for murder under BNS 2023?",
        "ground_truth": "Under BNS 2023, whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "101",
    },
    {
        "question": "What is culpable homicide under BNS 2023?",
        "ground_truth": "Culpable homicide is the act of causing death by doing an act with the intention of causing death, or with the intention of causing such bodily injury as is likely to cause death, or with the knowledge that the act is likely to cause death.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "100",
    },
    {
        "question": "What punishment does BNS prescribe for robbery?",
        "ground_truth": "Whoever commits robbery shall be punished with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "309",
    },
    {
        "question": "What is the definition of theft under BNS 2023?",
        "ground_truth": "Whoever, intending to take dishonestly any moveable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "303",
    },
    {
        "question": "What does IPC say about cheating?",
        "ground_truth": "Whoever, by deceiving any person, fraudulently or dishonestly induces the person so deceived to deliver any property to any person, or to consent that any person shall retain any property, or intentionally induces the person so deceived to do or omit to do anything which he would not do or omit if he were not so deceived, is said to cheat.",
        "domain": "criminal",
        "expected_act": "IPC_1860",
        "expected_section": "415",
    },
    {
        "question": "What is the punishment for kidnapping under BNS?",
        "ground_truth": "Whoever kidnaps any person from India or from lawful guardianship shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "137",
    },
    {
        "question": "How does BNS define assault?",
        "ground_truth": "Whoever makes any gesture, or any preparation intending or knowing it to be likely that such gesture or preparation will cause any person present to apprehend that he who makes that gesture or preparation is about to use criminal force to that person, is said to commit an assault.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "130",
    },
    {
        "question": "What is the right of private defence of body under BNS?",
        "ground_truth": "Every person has a right to defend his own body and the body of any other person against any offence affecting the human body. The right of private defence of the body extends to the voluntary causing of death if the offence reasonably apprehended is assault which may cause death or grievous hurt.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "34",
    },
    {
        "question": "What is criminal conspiracy under BNS 2023?",
        "ground_truth": "When two or more persons agree to do, or cause to be done, an illegal act, or an act which is not illegal by illegal means, such an agreement is designated a criminal conspiracy.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "61",
    },
    {
        "question": "What does the Bharatiya Sakshya Adhiniyam say about burden of proof?",
        "ground_truth": "The burden of proof in a suit or proceeding lies on that person who would fail if no evidence at all were given on either side. The burden of proof lies on the person who wishes the court to believe in its existence.",
        "domain": "criminal",
        "expected_act": "BSA_2023",
        "expected_section": "101",
    },
    {
        "question": "What is voluntarily causing grievous hurt under BNS?",
        "ground_truth": "Whoever voluntarily causes hurt, if the hurt which he intends to cause or knows himself to be likely to cause is grievous hurt, and if the hurt which he causes is grievous hurt, is said to have voluntarily caused grievous hurt.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "118",
    },
    {
        "question": "What is the IPC definition of criminal breach of trust?",
        "ground_truth": "Whoever, being in any manner entrusted with property, or with any dominion over property, dishonestly misappropriates or converts to his own use that property, or dishonestly uses or disposes of that property in violation of any direction of law, commits criminal breach of trust.",
        "domain": "criminal",
        "expected_act": "IPC_1860",
        "expected_section": "405",
    },
    {
        "question": "What is the punishment for dacoity under BNS?",
        "ground_truth": "Whoever commits dacoity shall be punished with imprisonment for life, or with rigorous imprisonment for a term which may extend to ten years, and shall also be liable to fine.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "310",
    },
    {
        "question": "What does BNSS say about remand of an accused?",
        "ground_truth": "A Magistrate may authorise the detention of the accused in such custody as the Magistrate thinks fit, for a term not exceeding fifteen days in the whole. If the Magistrate has no jurisdiction to try the case, they may authorise detention not exceeding seven days at a time.",
        "domain": "criminal",
        "expected_act": "BNSS_2023",
        "expected_section": "187",
    },
    {
        "question": "What is abetment of an offence under BNS 2023?",
        "ground_truth": "A person abets the doing of a thing who instigates any person to do that thing, or engages with one or more other person or persons in any conspiracy for the doing of that thing, or intentionally aids, by any act or illegal omission, the doing of that thing.",
        "domain": "criminal",
        "expected_act": "BNS_2023",
        "expected_section": "45",
    },

    # ── Land domain ───────────────────────────────────────────────────────────
    {
        "question": "Is an 11-month lease agreement required to be registered?",
        "ground_truth": "No. Under the Registration Act 1908, leases for a term not exceeding one year, or for yearly rents, are exempt from compulsory registration. Therefore an 11-month lease does not need to be registered.",
        "domain": "land",
        "expected_act": "REGISTRATION_1908",
        "expected_section": "17",
    },
    {
        "question": "What documents must be compulsorily registered under the Registration Act?",
        "ground_truth": "Documents that must be compulsorily registered include instruments of gift of immovable property, leases of immovable property from year to year or for a term exceeding one year or reserving a yearly rent, and non-testamentary instruments which purport or operate to create, declare, assign, limit or extinguish any right, title or interest of the value of one hundred rupees and upwards in immovable property.",
        "domain": "land",
        "expected_act": "REGISTRATION_1908",
        "expected_section": "17",
    },
    {
        "question": "What is a mortgage by deposit of title deeds under Transfer of Property Act?",
        "ground_truth": "Where a person in any of the towns of Calcutta, Madras, and Bombay, or in any other town which the State Government may specify, delivers to a creditor or his agent documents of title to immovable property, with intent to create a security thereon, the transaction is called a mortgage by deposit of title-deeds.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "58",
    },
    {
        "question": "What is a sale deed under Transfer of Property Act?",
        "ground_truth": "Sale is a transfer of ownership in exchange for a price paid or promised or part-paid and part-promised. A contract for the sale of immovable property is a contract that a sale of such property shall take place on terms settled between the parties.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "54",
    },
    {
        "question": "What are the rights of a mortgagor under Transfer of Property Act?",
        "ground_truth": "A mortgagor has the right to redeem the mortgaged property. The mortgagor may require the mortgagee to transfer the mortgaged property to a third party instead of retransferring it to the mortgagor. The mortgagor is also entitled to inspection and production of documents relating to the mortgaged property held by the mortgagee.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "60",
    },
    {
        "question": "What is a lease under Transfer of Property Act?",
        "ground_truth": "A lease of immoveable property is a transfer of a right to enjoy such property, made for a certain time, express or implied, or in perpetuity, in consideration of a price paid or promised, or of money, a share of crops, service or any other thing of value, to be rendered periodically or on specified occasions to the transferor by the transferee.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "105",
    },
    {
        "question": "What is the doctrine of lis pendens under Transfer of Property Act?",
        "ground_truth": "During the pendency in any court of a suit or proceeding in which any right to immoveable property is directly and specifically in question, the property cannot be transferred or otherwise dealt with by any party to the suit or proceeding so as to affect the rights of any other party thereto under any decree or order which may be made therein.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "52",
    },
    {
        "question": "What is fraudulent transfer under Transfer of Property Act?",
        "ground_truth": "Every transfer of immoveable property made with intent to defeat or delay the creditors of the transferor shall be voidable at the option of any creditor so defeated or delayed.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "53",
    },
    {
        "question": "When does ownership pass in a sale of immovable property under TPA?",
        "ground_truth": "In the case of a tangible immoveable property of a value less than one hundred rupees, ownership passes to the buyer when the seller delivers the property. In the case of tangible immoveable property of a value of one hundred rupees and upwards, ownership passes on registration of the instrument of sale.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "54",
    },
    {
        "question": "What documents are not required to be registered under Registration Act 1908?",
        "ground_truth": "Documents not requiring registration include wills, instruments of partition, Government grants of immoveable property, receipts or acknowledgements of payment of rent, and leases of immoveable property for a term not exceeding one year.",
        "domain": "land",
        "expected_act": "REGISTRATION_1908",
        "expected_section": "18",
    },
    {
        "question": "What is an actionable claim under Transfer of Property Act?",
        "ground_truth": "An actionable claim means a claim to any debt, other than a debt secured by mortgage of immoveable property or by hypothecation or pledge of moveable property, or to any beneficial interest in moveable property not in the possession, either actual or constructive, of the claimant, which the civil courts recognise as affording grounds for relief.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "3",
    },
    {
        "question": "What is usufructuary mortgage?",
        "ground_truth": "Where the mortgagor delivers possession or expressly or by implication binds himself to deliver possession of the mortgaged property to the mortgagee, and authorises him to retain such possession until payment of the mortgage-money, and to receive the rents and profits accruing from the property or any part of such rents and profits and to appropriate the same in lieu of interest, or in payment of the mortgage-money, the transaction is called a usufructuary mortgage.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "58",
    },
    {
        "question": "What is the time limit for presenting a document for registration?",
        "ground_truth": "Every document other than a will must be presented for registration within four months from the date of its execution. If it was executed at different times, then within four months from the date of each execution.",
        "domain": "land",
        "expected_act": "REGISTRATION_1908",
        "expected_section": "23",
    },
    {
        "question": "What is a gift deed under Transfer of Property Act?",
        "ground_truth": "Gift is the transfer of certain existing moveable or immoveable property made voluntarily and without consideration by one person called the donor to another called the donee, and accepted by or on behalf of the donee. A gift of immoveable property must be effected by a registered instrument signed by or on behalf of the donor and attested by at least two witnesses.",
        "domain": "land",
        "expected_act": "PROPERTY_1882",
        "expected_section": "122",
    },
]

# ═════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def safe_run(fn, *args, retries=2, delay=3, **kwargs):
    """Call fn with retries — handles Gemini/Groq rate limits gracefully."""
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt < retries:
                print(f"   ⚠️  Attempt {attempt+1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def keyword_overlap_score(text_a: str, text_b: str) -> float:
    """Simple unigram overlap score between two strings (0.0 – 1.0)."""
    stop = {"the", "a", "an", "is", "of", "to", "in", "and", "or", "that",
            "with", "shall", "be", "by", "for", "any", "such", "which",
            "this", "may", "not", "as", "at", "from", "under", "on", "are"}
    def tokens(s):
        return {w.lower().strip(".,;:()[]\"'") for w in s.split()} - stop
    a, b = tokens(text_a), tokens(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1 — HIT RATE & MRR  (no LLM call needed)
# ═════════════════════════════════════════════════════════════════════════════

def run_retrieval_eval(engine: LegalRagEngine) -> dict:
    """
    For each test question, retrieve top-k chunks and check:
      - Hit@k   : does the expected section appear in any of the k chunks?
      - RR      : reciprocal rank of the first correct chunk (for MRR)
    """
    print("\n" + "="*60)
    print("LAYER 1 — RETRIEVAL: Hit Rate & MRR")
    print("="*60)

    rows = []
    hits = 0
    rr_sum = 0.0

    for i, item in enumerate(GOLDEN_TEST_SET, 1):
        q = item["question"]
        expected_section = str(item["expected_section"])
        expected_act = item["expected_act"]
        domain = item["domain"]

        print(f"\n[{i}/{len(GOLDEN_TEST_SET)}] {q[:70]}...")

        try:
            docs = safe_run(engine.retrieve_context, q, domain)
        except Exception as e:
            print(f"   ❌ Retrieval failed: {e}")
            rows.append({**item, "hit": 0, "rank": None, "rr": 0.0, "retrieved_sections": "ERROR"})
            continue

        retrieved = [(d.metadata.get("act",""), str(d.metadata.get("section_id",""))) for d in docs]
        retrieved_str = " | ".join([f"{a}§{s}" for a,s in retrieved])

        hit_rank = None
        for rank, (act, sec) in enumerate(retrieved, 1):
            if sec == expected_section and act == expected_act:
                hit_rank = rank
                break

        is_hit = 1 if hit_rank else 0
        rr = 1.0 / hit_rank if hit_rank else 0.0

        hits += is_hit
        rr_sum += rr

        status = f"✅ Hit@{hit_rank}" if hit_rank else "❌ Miss"
        print(f"   {status}  |  Retrieved: {retrieved_str}")

        rows.append({
            "question": q,
            "domain": domain,
            "expected": f"{expected_act}§{expected_section}",
            "retrieved_sections": retrieved_str,
            "hit": is_hit,
            "rank": hit_rank,
            "rr": round(rr, 4),
        })

    hit_rate = hits / len(GOLDEN_TEST_SET)
    mrr = rr_sum / len(GOLDEN_TEST_SET)

    print(f"\n📊 Hit Rate : {hit_rate:.2%}  ({hits}/{len(GOLDEN_TEST_SET)})")
    print(f"📊 MRR      : {mrr:.4f}")

    # Save CSV
    out_path = OUTPUT_DIR / "retrieval_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"💾 Saved → {out_path}")

    return {"hit_rate": hit_rate, "mrr": mrr, "rows": rows}


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 — RAGAS  (faithfulness, answer_relevancy, context_precision, recall)
# ═════════════════════════════════════════════════════════════════════════════

def run_ragas_eval(engine: LegalRagEngine) -> dict:
    """
    Builds a Dataset of (question, answer, contexts, ground_truth) and
    scores it with RAGAS.  Requires: pip install ragas datasets
    """
    print("\n" + "="*60)
    print("LAYER 2 — RAGAS METRICS")
    print("="*60)

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        print("⚠️  RAGAS not installed. Run: pip install ragas datasets")
        print("   Skipping RAGAS layer.")
        return {}

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(GOLDEN_TEST_SET, 1):
        q = item["question"]
        print(f"\n[{i}/{len(GOLDEN_TEST_SET)}] Generating answer for: {q[:65]}...")

        try:
            # get answer + retrieved docs separately
            docs = safe_run(engine.retrieve_context, q, item["domain"])
            contexts = [d.page_content for d in docs]

            answer, _ = safe_run(engine.generate_answer, q, item["domain"])
            time.sleep(1)  # be polite to API rate limits
        except Exception as e:
            print(f"   ❌ Skipping — {e}")
            continue

        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])

    if not questions:
        print("❌ No questions processed — RAGAS skipped.")
        return {}

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    print("\n⏳ Running RAGAS evaluation (this calls an LLM internally)...")
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return {}

    scores = result.to_pandas()
    out_path = OUTPUT_DIR / "ragas_scores.csv"
    scores.to_csv(out_path, index=False)
    print(f"💾 Saved → {out_path}")

    summary = {
        "faithfulness":       round(float(scores["faithfulness"].mean()), 4),
        "answer_relevancy":   round(float(scores["answer_relevancy"].mean()), 4),
        "context_precision":  round(float(scores["context_precision"].mean()), 4),
        "context_recall":     round(float(scores["context_recall"].mean()), 4),
    }

    print("\n📊 RAGAS Summary:")
    for k, v in summary.items():
        bar = "█" * int(v * 20)
        print(f"   {k:<25} {v:.4f}  {bar}")

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 3 — LEGAL-SPECIFIC METRICS
# ═════════════════════════════════════════════════════════════════════════════

def run_legal_eval(engine: LegalRagEngine) -> dict:
    """
    For each question:
      - Citation accuracy   : do cited sections overlap with expected section text?
      - Section grounding   : does cited section ID match expected_section?
      - Hallucination proxy : keyword overlap between answer and retrieved context.
                              Low overlap → likely hallucinated.
    """
    print("\n" + "="*60)
    print("LAYER 3 — LEGAL-SPECIFIC METRICS")
    print("="*60)

    rows = []
    section_hits = 0
    hallucination_safe = 0
    citation_scores = []

    for i, item in enumerate(GOLDEN_TEST_SET, 1):
        q = item["question"]
        print(f"\n[{i}/{len(GOLDEN_TEST_SET)}] {q[:70]}...")

        try:
            answer, citations = safe_run(engine.generate_answer, q, item["domain"])
            docs = safe_run(engine.retrieve_context, q, item["domain"])
            time.sleep(1)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            rows.append({
                "question": q, "domain": item["domain"],
                "expected_section": item["expected_section"],
                "citations": "ERROR",
                "section_grounded": 0,
                "hallucination_safe": 0,
                "citation_accuracy": 0.0,
            })
            continue

        # --- Section grounding: is the expected section in the citation list? ---
        expected_sec = str(item["expected_section"])
        grounded = int(any(expected_sec in c for c in citations))
        section_hits += grounded

        # --- Citation accuracy: keyword overlap of cited chunk vs answer -------
        full_context = " ".join(d.page_content for d in docs)
        cit_score = keyword_overlap_score(answer, full_context)
        citation_scores.append(cit_score)

        # --- Hallucination proxy: answer overlap with retrieved context ---------
        # If answer contains lots of words NOT in the retrieved context, 
        # it may be hallucinating. Threshold 0.25 is conservative — tune as needed.
        HALLUCINATION_THRESHOLD = 0.25
        is_safe = int(cit_score >= HALLUCINATION_THRESHOLD)
        hallucination_safe += is_safe

        grounded_icon = "✅" if grounded else "❌"
        safe_icon = "✅" if is_safe else "⚠️"
        print(f"   Section grounded: {grounded_icon}  |  Hallucination safe: {safe_icon}  |  Overlap: {cit_score:.2f}")
        print(f"   Citations: {citations}")

        rows.append({
            "question": q,
            "domain": item["domain"],
            "expected_act": item["expected_act"],
            "expected_section": item["expected_section"],
            "citations_returned": " | ".join(citations),
            "section_grounded": grounded,
            "hallucination_safe": is_safe,
            "citation_accuracy_score": round(cit_score, 4),
        })

    n = len(rows)
    section_grounding_rate = section_hits / n if n else 0
    no_hallucination_rate = hallucination_safe / n if n else 0
    avg_citation_accuracy = sum(citation_scores) / len(citation_scores) if citation_scores else 0

    print(f"\n📊 Section grounding rate : {section_grounding_rate:.2%}  ({section_hits}/{n})")
    print(f"📊 No-hallucination rate  : {no_hallucination_rate:.2%}  ({hallucination_safe}/{n})")
    print(f"📊 Avg citation accuracy  : {avg_citation_accuracy:.4f}")

    out_path = OUTPUT_DIR / "legal_scores.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"💾 Saved → {out_path}")

    return {
        "section_grounding_rate": section_grounding_rate,
        "no_hallucination_rate": no_hallucination_rate,
        "avg_citation_accuracy": avg_citation_accuracy,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═════════════════════════════════════════════════════════════════════════════

def write_summary(retrieval: dict, ragas: dict, legal: dict):
    lines = [
        "=" * 60,
        "  LEGAL_RAG — EVALUATION SUMMARY",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Test set : {len(GOLDEN_TEST_SET)} questions "
        f"({sum(1 for x in GOLDEN_TEST_SET if x['domain']=='criminal')} criminal, "
        f"{sum(1 for x in GOLDEN_TEST_SET if x['domain']=='land')} land)",
        "=" * 60,
        "",
        "── LAYER 1: RETRIEVAL ───────────────────────────────────",
        f"  Hit Rate (correct section in top-4) : {retrieval.get('hit_rate', 0):.2%}",
        f"  Mean Reciprocal Rank (MRR)           : {retrieval.get('mrr', 0):.4f}",
        "",
        "── LAYER 2: RAGAS ───────────────────────────────────────",
    ]
    if ragas:
        lines += [
            f"  Faithfulness       : {ragas.get('faithfulness', 'N/A')}",
            f"  Answer relevancy   : {ragas.get('answer_relevancy', 'N/A')}",
            f"  Context precision  : {ragas.get('context_precision', 'N/A')}",
            f"  Context recall     : {ragas.get('context_recall', 'N/A')}",
        ]
    else:
        lines.append("  (RAGAS not run — install ragas package)")

    lines += [
        "",
        "── LAYER 3: LEGAL-SPECIFIC ──────────────────────────────",
        f"  Section grounding rate : {legal.get('section_grounding_rate', 0):.2%}",
        f"  No-hallucination rate  : {legal.get('no_hallucination_rate', 0):.2%}",
        f"  Avg citation accuracy  : {legal.get('avg_citation_accuracy', 0):.4f}",
        "",
        "── INTERPRETATION ───────────────────────────────────────",
        "  Target benchmarks for a strong project:",
        "  Hit Rate          > 70%   | MRR > 0.65",
        "  Faithfulness      > 0.85  | Context precision > 0.80",
        "  Section grounding > 70%   | No-hallucination  > 85%",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    out_path = OUTPUT_DIR / "summary.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n💾 Full report saved → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🏛️  LEGAL_RAG Evaluation Suite")
    print("   Initialising engine (loads ChromaDB + embeddings)...\n")

    engine = LegalRagEngine()

    # Run all three layers
    retrieval_results = run_retrieval_eval(engine)
    ragas_results     = run_ragas_eval(engine)
    legal_results     = run_legal_eval(engine)

    # Print + save the final report
    write_summary(retrieval_results, ragas_results, legal_results)