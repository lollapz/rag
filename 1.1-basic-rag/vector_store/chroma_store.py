"""Chroma 向量库的创建、加载和写入。"""

from collections.abc import Callable, Sequence
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DB_WRITE_BATCH_SIZE,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)


@lru_cache(maxsize=1)
def get_embedding_model() -> OllamaEmbeddings:
    """创建并缓存 Ollama embedding 模型。"""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """加载或创建项目使用的持久化 Chroma 向量库。"""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_documents(
    documents: Sequence[Document],
    *,
    vector_store: Chroma | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """按配置批量写入文档，并可报告已完成数量。"""
    store = (
        vector_store if vector_store is not None else get_vectorstore()
    )
    total = len(documents)

    for batch_start in range(0, total, DB_WRITE_BATCH_SIZE):
        batch = documents[batch_start:batch_start + DB_WRITE_BATCH_SIZE]
        store.add_documents(batch)
        done = min(batch_start + DB_WRITE_BATCH_SIZE, total)
        if progress_callback is not None:
            progress_callback(done, total)
