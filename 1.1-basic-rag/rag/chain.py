"""RAG 问答链路组装。"""

from functools import lru_cache
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from config import (
    CHAT_MODEL,
    CHAT_REASONING,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
)
from prompts.rag_prompt import RAG_PROMPT
from rag.retriever import build_retriever


def format_docs(documents: Iterable[Document]) -> str:
    """将检索到的文档拼接为上下文。"""
    return "\n\n".join(document.page_content for document in documents)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOllama:
    """创建并缓存 Ollama 聊天模型。"""
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=1,
        keep_alive=OLLAMA_KEEP_ALIVE,
        reasoning=CHAT_REASONING,
    )


def build_rag_chain(
    *,
    retriever: Any = None,
    chat_model: Any = None,
) -> Any:
    """组装 retriever、prompt、LLM 和输出解析器。"""
    active_retriever = (
        retriever if retriever is not None else build_retriever()
    )
    active_chat_model = (
        chat_model if chat_model is not None else get_chat_model()
    )

    return (
        {
            "context": active_retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | active_chat_model
        | StrOutputParser()
    )


def run_rag_chain(
    query: str,
    *,
    retriever: Any = None,
    chat_model: Any = None,
) -> str:
    """执行 RAG 问答并返回纯文本结果。"""
    chain = build_rag_chain(
        retriever=retriever,
        chat_model=chat_model,
    )
    return chain.invoke(query)
