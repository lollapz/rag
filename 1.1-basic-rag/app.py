from pathlib import Path

import streamlit as st

from config import (
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TEMP_DIR,
)
from rag.chain import run_rag_chain
from rag.loader import NoExtractableTextError, load_pdf_documents
from rag.splitter import split_documents
from vector_store.chroma_store import add_documents


def get_user_error_message(error: Exception) -> str:
    """将常见运行错误转换为可直接操作的提示。"""
    if isinstance(error, NoExtractableTextError):
        return str(error)

    message = str(error)
    normalized = message.lower()

    if "model" in normalized and (
        "not found" in normalized or "status code: 404" in normalized
    ):
        model = EMBEDDING_MODEL if EMBEDDING_MODEL in message else CHAT_MODEL
        return f"本机缺少 Ollama 模型，请执行：ollama pull {model}"

    connection_markers = (
        "connection refused",
        "failed to connect",
        "connecterror",
        "connection error",
    )
    if any(marker in normalized for marker in connection_markers):
        return "无法连接 Ollama，请先执行：ollama serve"

    return f"操作失败：{message}"


# ---------------------------------------------------------------------------
# 核心流程
# ---------------------------------------------------------------------------
def add_to_db(uploaded_files):
    """解析 PDF → 切分 → 向量化 → 写入 ChromaDB"""
    if not uploaded_files:
        st.error("No files uploaded!")
        return

    total_files = len(uploaded_files)
    for idx, uploaded_file in enumerate(uploaded_files):
        st.write(f"处理中 ({idx + 1}/{total_files}): {uploaded_file.name}")

        safe_filename = Path(uploaded_file.name).name
        temp_file_path = TEMP_DIR / safe_filename
        temp_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with temp_file_path.open("wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())

            data = load_pdf_documents(temp_file_path)
            chunks = split_documents(data)

            progress_bar = st.progress(
                0, text=f"正在生成向量... 0/{len(chunks)} 片段"
            )

            def update_progress(done: int, total: int) -> None:
                progress_bar.progress(
                    done / total,
                    text=f"正在生成向量... {done}/{total} 片段",
                )

            add_documents(chunks, progress_callback=update_progress)
            progress_bar.empty()
        finally:
            # 无论是否出错，确保临时文件被清理
            temp_file_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Streamlit 界面
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="PDF智能问答助手", page_icon="📄")
    st.header("📄 PDF 智能问答助手")

    # ---- 查询区 ----
    query = st.text_area(
        ":bulb: 请输入你的问题：",
        placeholder="例如：这份文档的核心观点是什么？",
    )
    st.caption("首次查询需要加载本地模型，后续查询会更快。")

    if st.button("提交查询"):
        if not query:
            st.warning("请输入问题")
        else:
            try:
                with st.spinner("思考中..."):
                    result = run_rag_chain(query=query)
                    st.write(result)
            except Exception as error:
                st.error(get_user_error_message(error))

    # ---- 侧边栏：文件上传 ----
    with st.sidebar:
        st.markdown("---")
        pdf_docs = st.file_uploader(
            "上传你的 PDF 文档（可选） :memo:",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("上传并处理"):
            if not pdf_docs:
                st.warning("请先上传文件")
            else:
                try:
                    add_to_db(pdf_docs)
                    st.success(":file_folder: 文档已成功添加到知识库！")
                except Exception as error:
                    st.error(get_user_error_message(error))

    # ---- 侧边栏：底部 ----
    st.sidebar.markdown("---")
    st.sidebar.caption("基于 LangChain、ChromaDB 与 Ollama 构建的本地 RAG 知识库问答系统\n支持 PDF 文档上传、向量检索与大模型问答")


if __name__ == "__main__":
    main()
