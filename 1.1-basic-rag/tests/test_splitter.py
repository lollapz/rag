from unittest.mock import MagicMock

from langchain_core.documents import Document

from rag import splitter


def test_split_documents_uses_configured_recursive_splitter(monkeypatch):
    chunks = [Document(page_content="chunk")]
    splitter_instance = MagicMock()
    splitter_instance.create_documents.return_value = chunks
    splitter_factory = MagicMock(return_value=splitter_instance)
    monkeypatch.setattr(
        splitter,
        "RecursiveCharacterTextSplitter",
        splitter_factory,
    )
    documents = [
        Document(
            page_content="page content",
            metadata={"source": "doc.pdf", "page": 0},
        )
    ]

    result = splitter.split_documents(documents)

    assert result == chunks
    splitter_factory.assert_called_once_with(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    splitter_instance.create_documents.assert_called_once_with(
        ["page content"],
        [{"source": "doc.pdf", "page": 0}],
    )
