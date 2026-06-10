"""文档加载。"""

from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from config import DOCUMENTS_DIR


class NoExtractableTextError(ValueError):
    """PDF 中没有可用于检索的文本。"""


def load_pdf_documents(file_path: str | Path) -> list[Document]:
    """读取 PDF 中有文本的页面，扫描版 PDF 需要先进行 OCR。"""
    source = str(file_path)
    reader = PdfReader(source)
    documents = []

    for page_number, page in enumerate(reader.pages):
        content = (page.extract_text() or "").strip()
        if not content:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={"source": source, "page": page_number},
            )
        )

    if not documents:
        raise NoExtractableTextError(
            "未检测到可提取文本。该 PDF 可能是扫描版，请先执行 OCR 后重新上传。"
        )

    return documents


def load_documents(
    documents_dir: str | Path = DOCUMENTS_DIR,
) -> list[Document]:
    """加载指定目录下的全部 PDF 文档。"""
    directory = Path(documents_dir)
    documents = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        documents.extend(load_pdf_documents(pdf_path))

    return documents
