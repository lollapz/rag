from unittest.mock import MagicMock, call

from vector_store import chroma_store


def test_get_embedding_model_uses_configured_ollama_settings(monkeypatch):
    chroma_store.get_embedding_model.cache_clear()
    embedding_factory = MagicMock()
    monkeypatch.setattr(chroma_store, "OllamaEmbeddings", embedding_factory)

    chroma_store.get_embedding_model()

    embedding_factory.assert_called_once_with(
        model="mxbai-embed-large",
        base_url="http://localhost:11434",
    )
    chroma_store.get_embedding_model.cache_clear()


def test_get_vectorstore_uses_existing_collection_and_persist_directory(monkeypatch):
    chroma_store.get_vectorstore.cache_clear()
    vectorstore_factory = MagicMock()
    embedding_model = MagicMock()
    monkeypatch.setattr(chroma_store, "Chroma", vectorstore_factory)
    monkeypatch.setattr(
        chroma_store,
        "get_embedding_model",
        MagicMock(return_value=embedding_model),
    )

    chroma_store.get_vectorstore()

    vectorstore_factory.assert_called_once_with(
        collection_name="doc_database",
        embedding_function=embedding_model,
        persist_directory=chroma_store.CHROMA_PERSIST_DIR,
    )
    chroma_store.get_vectorstore.cache_clear()


def test_add_documents_writes_batches_and_reports_progress():
    vector_store = MagicMock()
    progress_callback = MagicMock()
    documents = list(range(23))

    chroma_store.add_documents(
        documents,
        vector_store=vector_store,
        progress_callback=progress_callback,
    )

    assert vector_store.add_documents.call_args_list == [
        call(documents[0:10]),
        call(documents[10:20]),
        call(documents[20:23]),
    ]
    assert progress_callback.call_args_list == [
        call(10, 23),
        call(20, 23),
        call(23, 23),
    ]
