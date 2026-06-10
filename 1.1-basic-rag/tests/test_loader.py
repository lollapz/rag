from unittest.mock import MagicMock, call

import pytest

from rag import loader


def test_load_pdf_documents_filters_blank_pages_and_keeps_metadata(monkeypatch):
    text_page = MagicMock()
    text_page.extract_text.return_value = "  有效内容  "
    blank_page = MagicMock()
    blank_page.extract_text.return_value = "  "
    monkeypatch.setattr(
        loader,
        "PdfReader",
        MagicMock(return_value=MagicMock(pages=[text_page, blank_page])),
    )

    documents = loader.load_pdf_documents("/tmp/example.pdf")

    assert len(documents) == 1
    assert documents[0].page_content == "有效内容"
    assert documents[0].metadata == {
        "source": "/tmp/example.pdf",
        "page": 0,
    }


def test_load_pdf_documents_rejects_pdf_without_extractable_text(monkeypatch):
    page = MagicMock()
    page.extract_text.return_value = None
    monkeypatch.setattr(
        loader,
        "PdfReader",
        MagicMock(return_value=MagicMock(pages=[page])),
    )

    with pytest.raises(loader.NoExtractableTextError, match="OCR"):
        loader.load_pdf_documents("/tmp/scanned.pdf")


def test_load_documents_reads_all_pdfs_from_directory(monkeypatch, tmp_path):
    first_pdf = tmp_path / "a.pdf"
    second_pdf = tmp_path / "b.pdf"
    first_pdf.touch()
    second_pdf.touch()
    (tmp_path / "ignored.txt").touch()
    load_pdf = MagicMock(
        side_effect=lambda path: [str(path)]
    )
    monkeypatch.setattr(loader, "load_pdf_documents", load_pdf)

    documents = loader.load_documents(tmp_path)

    assert documents == [str(first_pdf), str(second_pdf)]
    assert load_pdf.call_args_list == [
        call(first_pdf),
        call(second_pdf),
    ]
