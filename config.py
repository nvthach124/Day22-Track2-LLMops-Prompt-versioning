import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

def get_config():
    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT", "day22-lab-default"),
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true"),
        "OPENAI_ENDPOINT": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    }
    return config

if __name__ == "__main__":
    config = get_config()
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {config['LANGCHAIN_PROJECT']}")
    print(f"   OpenAI endpoint   : {config['OPENAI_ENDPOINT']}")
    print(f"   Default LLM model : {config['MODEL']}")
    print(f"   Embedding model   : {config['EMBEDDING_MODEL']}")
    
    if not config["LANGCHAIN_API_KEY"]:
        print("⚠️  Warning: LANGCHAIN_API_KEY is missing in .env")
    if not config["OPENAI_API_KEY"]:
        print("❌ Error: OPENAI_API_KEY is missing in .env")
