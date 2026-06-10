from unittest.mock import MagicMock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag import chain


def test_format_docs_joins_page_content():
    documents = [
        Document(page_content="第一段"),
        Document(page_content="第二段"),
    ]

    assert chain.format_docs(documents) == "第一段\n\n第二段"


def test_get_chat_model_uses_existing_ollama_settings(monkeypatch):
    chain.get_chat_model.cache_clear()
    chat_factory = MagicMock()
    monkeypatch.setattr(chain, "ChatOllama", chat_factory)

    chain.get_chat_model()

    chat_factory.assert_called_once_with(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=1,
        keep_alive="30m",
        reasoning=False,
    )
    chain.get_chat_model.cache_clear()


def test_run_rag_chain_retrieves_context_and_returns_parsed_text():
    seen_queries = []
    retriever = RunnableLambda(
        lambda query: (
            seen_queries.append(query)
            or [Document(page_content="mock context")]
        )
    )
    chat_model = RunnableLambda(
        lambda messages: AIMessage(content="这是回答")
    )

    result = chain.run_rag_chain(
        "核心观点是什么？",
        retriever=retriever,
        chat_model=chat_model,
    )

    assert result == "这是回答"
    assert seen_queries == ["核心观点是什么？"]
