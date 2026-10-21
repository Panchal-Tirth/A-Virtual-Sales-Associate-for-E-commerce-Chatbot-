from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CATALOG_PATH = ROOT / "data" / "catalog.parquet"
CHROMA_DIR = str(ROOT / "data" / "chroma_db")
COLLECTION = "fashion_catalog"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_SMART = "openai/gpt-oss-120b"   # reasoning and final answers
LLM_FAST = "openai/gpt-oss-20b"     # intent and slot extraction