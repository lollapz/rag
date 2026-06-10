"""相似度检索。"""

from typing import Any

from langchain_core.documents import Document

from config import RETRIEVER_K
from vector_store.chroma_store import get_vectorstore


def build_retriever(vector_store: Any = None) -> Any:
    """基于 Chroma 构建相似度 retriever。"""
    store = vector_store if vector_store is not None else get_vectorstore()
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )


def retrieve_documents(query: str, retriever: Any = None) -> list[Document]:
    """执行一次相似度检索。"""
    active_retriever = (
        retriever if retriever is not None else build_retriever()
    )
    return active_retriever.invoke(query)
