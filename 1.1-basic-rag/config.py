"""项目配置。所有路径和 RAG 参数统一从此模块读取。"""

from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

# ---- Project paths ----
PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "docs"
TEMP_DIR = PROJECT_ROOT / "temp"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
CHROMA_PERSIST_DIR = str(CHROMA_DB_DIR)
RUNTIME_DIRECTORIES = (
    DOCUMENTS_DIR,
    TEMP_DIR,
    UPLOADS_DIR,
    CHROMA_DB_DIR,
)


def ensure_runtime_directories(
    directories: tuple[Path, ...] = RUNTIME_DIRECTORIES,
) -> None:
    """创建不会提交到 Git 的本地运行目录。"""
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


ensure_runtime_directories()

# ---- Ollama 本地服务 ----
OLLAMA_BASE_URL = "http://localhost:11434"

# ---- LLM Models ----
EMBEDDING_MODEL = "mxbai-embed-large"
CHAT_MODEL = "qwen3:8b"
CHAT_REASONING = False
OLLAMA_KEEP_ALIVE = "30m"

# ---- 文档切分 ----
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", " ", ""]

# ---- Embedding 批量 ----
DB_WRITE_BATCH_SIZE = 10       # 分批写入数据库，避免一次性写入过多

# ---- 检索 ----
RETRIEVER_K = 5                # 相似度搜索返回的文档数

# ---- ChromaDB ----
CHROMA_COLLECTION_NAME = "doc_database"
