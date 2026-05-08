"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment setup ────────────────────────────────────────────────────
load_dotenv()
from config import get_config
cfg = get_config()

os.environ["LANGCHAIN_TRACING_V2"]  = cfg["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"]     = cfg["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]     = cfg["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
llm = ChatOpenAI(
    model=cfg["MODEL"],
    api_key=cfg["OPENAI_API_KEY"],
    base_url=cfg["OPENAI_ENDPOINT"],
)

embeddings = OpenAIEmbeddings(
    model=cfg["EMBEDDING_MODEL"],
    api_key=cfg["OPENAI_API_KEY"],
    base_url=cfg["OPENAI_ENDPOINT"],
)


# ── 4. Build FAISS vector store ─────────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base, split into chunks, embed and index with FAISS.
    """
    file_path = Path("data/knowledge_base.txt")
    if not file_path.exists():
        print(f"❌ Error: {file_path} not found.")
        return None
        
    text = file_path.read_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"✅ Split into {len(chunks)} chunks")

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer the user question. If the answer is not in the context, say you don't know.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    """
    return chain.invoke(question)


# ── 8. Sample questions ──────────────────────────────────────────────────
from qa_pairs import QA_PAIRS
SAMPLE_QUESTIONS = [q["question"] for q in QA_PAIRS]


# ── 9. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not cfg["LANGCHAIN_API_KEY"]:
        print("❌ Error: LANGCHAIN_API_KEY is missing. Tracing will not work.")
        print("   Please add it to your .env file.")
        return

    vectorstore = build_vectorstore()
    if not vectorstore:
        return
        
    chain, retriever = build_rag_chain(vectorstore)

    print(f"Running {len(SAMPLE_QUESTIONS)} questions...")
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        try:
            answer = ask(chain, question)
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
            # print(f"       A: {answer[:100]}\n")
        except Exception as e:
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] ❌ Error: {e}")

    print("-" * 60)
    print(f"✅ {len(SAMPLE_QUESTIONS)} queries processed.")
    print(f"   Traces sent to LangSmith project: '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
