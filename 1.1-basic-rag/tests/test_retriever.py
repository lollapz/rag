from unittest.mock import MagicMock

from rag.retriever import build_retriever, retrieve_documents


def test_build_retriever_uses_similarity_and_configured_top_k():
    vector_store = MagicMock()

    retriever = build_retriever(vector_store)

    assert retriever == vector_store.as_retriever.return_value
    vector_store.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 5},
    )


def test_retrieve_documents_invokes_retriever_with_query():
    retriever = MagicMock()
    retriever.invoke.return_value = ["document"]

    result = retrieve_documents("question", retriever)

    assert result == ["document"]
    retriever.invoke.assert_called_once_with("question")
