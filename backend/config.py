import os
from pathlib import Path

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "claude-sonnet-4-6"
LLM_TEMPERATURE = 0.0

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 10

MAX_FILE_SIZE_MB = 50
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
