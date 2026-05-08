"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces
"""

import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# ── 1. Environment / imports ────────────────────────────────────────────────
load_dotenv()
from config import get_config
cfg = get_config()

os.environ["LANGCHAIN_TRACING_V2"] = cfg["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_API_KEY"]    = cfg["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]    = cfg["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_ENDPOINT"]   = "https://api.smith.langchain.com"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
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

# Use the project name as a prefix to ensure uniqueness in the shared Hub
PROJECT_PREFIX = cfg["LANGCHAIN_PROJECT"].replace("-", "_")
PROMPT_V1_NAME = f"{PROJECT_PREFIX}_rag_v1"
PROMPT_V2_NAME = f"{PROJECT_PREFIX}_rag_v2"


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.
    """
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 Push failed: {e}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 Push failed: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for V1: {e}")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception as e:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for V2: {e}")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
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


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version_tag: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    return {"question": question, "answer": answer, "version": version_tag}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    if not cfg["LANGCHAIN_API_KEY"]:
        print("❌ Error: LANGCHAIN_API_KEY is missing.")
        return

    client = Client(api_key=cfg["LANGCHAIN_API_KEY"])

    # Push and Pull
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    # Setup RAG components
    vectorstore = build_vectorstore()
    if not vectorstore: return
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatOpenAI(
        model=cfg["MODEL"],
        api_key=cfg["OPENAI_API_KEY"],
        base_url=cfg["OPENAI_ENDPOINT"],
    )

    # Questions
    from qa_pairs import QA_PAIRS
    SAMPLE_QUESTIONS = [q["question"] for q in QA_PAIRS]
    
    counts = {"v1": 0, "v2": 0}
    
    print(f"Running {len(SAMPLE_QUESTIONS)} questions with A/B routing...")
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]
        
        counts[version_tag] += 1
        
        try:
            result = ask_ab(retriever, llm, prompt, question, version_tag)
            print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")
        except Exception as e:
            print(f"[{i+1:02d}] [prompt-{version_tag}] ❌ Error: {e}")

    print("-" * 60)
    print(f"✅ Summary: V1={counts['v1']}, V2={counts['v2']}")
    print(f"   Traces sent to LangSmith project: '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Prompts managed in LangSmith Prompt Hub.")


if __name__ == "__main__":
    main()
