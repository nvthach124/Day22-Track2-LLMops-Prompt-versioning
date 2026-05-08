"""
Step 3 — RAGAS Evaluation
===========================
TASK:
  1. Run all 50 QA pairs through BOTH prompt versions, capturing answers + contexts
  2. Build EvaluationDataset with SingleTurnSample objects
  3. Evaluate with 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. Print a V1 vs V2 comparison table
  5. Save results to data/ragas_report.json
"""

import os
import json
import warnings
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")   # suppress RAGAS deprecation warnings

# ── 1. Imports ───────────────────────────────────────────────────────────────
load_dotenv()
from config import get_config
cfg = get_config()

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 2. QA pairs ─────────────────────────────────────────────────────────────
from qa_pairs import QA_PAIRS


# ── 3. Prompt templates (same as step 2) ────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {
    "v1": PROMPT_V1,
    "v2": PROMPT_V2,
}


# ── 4. Build vectorstore (reuse logic from step 1) ───────────────────────────
def build_vectorstore():
    embeddings = OpenAIEmbeddings(
        model=cfg["EMBEDDING_MODEL"],
        api_key=cfg["OPENAI_API_KEY"],
        base_url=cfg["OPENAI_ENDPOINT"],
    )
    
    file_path = Path("data/knowledge_base.txt")
    if not file_path.exists():
        print(f"❌ Error: {file_path} not found.")
        return None
        
    text = file_path.read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 5. Run RAG and capture outputs + contexts ────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Run the RAG chain for one question.
    """
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # list of strings
    ctx_str  = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx_str, "question": question})

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, llm, prompt_version: str) -> list:
    """
    Run all 50 QA pairs through the given prompt version.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Running 50 questions with prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}...")

    return results


# ── 6. Build RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list):
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 7. Run RAGAS evaluation ──────────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str, llm_eval, emb_eval) -> dict:
    """
    Evaluate RAG outputs with 4 RAGAS metrics.
    """
    print(f"\n📐 Running RAGAS evaluation for prompt {version} ...")

    dataset = build_ragas_dataset(rag_results)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]           # list of floats
        scores[key] = float(np.mean([v for v in raw if v is not None]))

    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")
    return scores


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    vectorstore = build_vectorstore()
    if not vectorstore: return
    
    llm = ChatOpenAI(
        model=cfg["MODEL"],
        api_key=cfg["OPENAI_API_KEY"],
        base_url=cfg["OPENAI_ENDPOINT"],
    )

    # Collect outputs (this takes some time)
    v1_results = collect_rag_outputs(vectorstore, llm, "v1")
    v2_results = collect_rag_outputs(vectorstore, llm, "v2")

    # Setup RAGAS evaluation models
    llm_eval = ChatOpenAI(model=cfg["MODEL"], api_key=cfg["OPENAI_API_KEY"], base_url=cfg["OPENAI_ENDPOINT"])
    emb_eval = OpenAIEmbeddings(model=cfg["EMBEDDING_MODEL"], api_key=cfg["OPENAI_API_KEY"], base_url=cfg["OPENAI_ENDPOINT"])

    # Run RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1", llm_eval, emb_eval)
    v2_scores = run_ragas_eval(v2_results, "v2", llm_eval, emb_eval)

    # Comparison table
    print("\n" + "=" * 60)
    print("  COMPARISON: V1 vs V2")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"\n⚠️  Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    # Save report
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    Path("data/ragas_report.json").write_text(json.dumps(report, indent=2))
    print("\n💾 Saved data/ragas_report.json")


if __name__ == "__main__":
    main()
