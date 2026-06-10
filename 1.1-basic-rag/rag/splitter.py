"""文档切块。"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE


def split_documents(documents: list[Document]) -> list[Document]:
    """使用项目配置切分文档并保留页面元数据。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    return text_splitter.create_documents(
        [document.page_content for document in documents],
        [document.metadata for document in documents],
    )
