from unittest.mock import MagicMock

import pytest

import app


@pytest.fixture
def st_mock(monkeypatch):
    streamlit = MagicMock()
    streamlit.sidebar = MagicMock()
    streamlit.sidebar.__enter__.return_value = streamlit.sidebar
    streamlit.sidebar.__exit__.return_value = False
    streamlit.spinner.return_value.__enter__.return_value = None
    streamlit.spinner.return_value.__exit__.return_value = False
    streamlit.text_area.return_value = ""
    streamlit.file_uploader.return_value = []
    streamlit.button.side_effect = lambda label, *args, **kwargs: False
    monkeypatch.setattr(app, "st", streamlit)
    return streamlit


def test_get_user_error_message_maps_ollama_connection_error():
    message = app.get_user_error_message(RuntimeError("Connection refused"))

    assert "ollama serve" in message


def test_get_user_error_message_maps_missing_chat_model():
    message = app.get_user_error_message(
        RuntimeError("model 'qwen3:8b' not found")
    )

    assert "ollama pull qwen3:8b" in message


def test_add_to_db_rejects_empty_upload(st_mock):
    app.add_to_db([])

    st_mock.error.assert_called_once_with("No files uploaded!")


def test_add_to_db_loads_splits_writes_and_removes_temp_file(
    monkeypatch,
    st_mock,
    tmp_path,
):
    uploaded_file = MagicMock()
    uploaded_file.name = "example.pdf"
    uploaded_file.getbuffer.return_value = b"pdf data"
    pages = [MagicMock()]
    chunks = [MagicMock(), MagicMock()]
    load_pdf = MagicMock(return_value=pages)
    split = MagicMock(return_value=chunks)
    write = MagicMock()
    monkeypatch.setattr(app, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(app, "load_pdf_documents", load_pdf)
    monkeypatch.setattr(app, "split_documents", split)
    monkeypatch.setattr(app, "add_documents", write)

    app.add_to_db([uploaded_file])

    temp_path = tmp_path / "example.pdf"
    load_pdf.assert_called_once_with(temp_path)
    split.assert_called_once_with(pages)
    write.assert_called_once()
    assert write.call_args.args == (chunks,)
    assert callable(write.call_args.kwargs["progress_callback"])
    assert not temp_path.exists()
    st_mock.progress.return_value.empty.assert_called_once()


def test_add_to_db_removes_temp_file_when_processing_fails(
    monkeypatch,
    st_mock,
    tmp_path,
):
    uploaded_file = MagicMock()
    uploaded_file.name = "broken.pdf"
    uploaded_file.getbuffer.return_value = b"broken data"
    monkeypatch.setattr(app, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(
        app,
        "load_pdf_documents",
        MagicMock(side_effect=RuntimeError("Cannot read PDF")),
    )

    with pytest.raises(RuntimeError, match="Cannot read PDF"):
        app.add_to_db([uploaded_file])

    assert not (tmp_path / "broken.pdf").exists()


def test_add_to_db_uses_safe_upload_filename(
    monkeypatch,
    st_mock,
    tmp_path,
):
    uploaded_file = MagicMock()
    uploaded_file.name = "../outside.pdf"
    uploaded_file.getbuffer.return_value = b"pdf data"
    load_pdf = MagicMock(return_value=[MagicMock()])
    monkeypatch.setattr(app, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(app, "load_pdf_documents", load_pdf)
    monkeypatch.setattr(app, "split_documents", MagicMock(return_value=[]))
    monkeypatch.setattr(app, "add_documents", MagicMock())

    app.add_to_db([uploaded_file])

    load_pdf.assert_called_once_with(tmp_path / "outside.pdf")
    assert not (tmp_path.parent / "outside.pdf").exists()


def test_main_configures_page_and_renders_inputs(st_mock):
    app.main()

    st_mock.set_page_config.assert_called_once_with(
        page_title="PDF智能问答助手",
        page_icon="📄",
    )
    st_mock.header.assert_called_once_with("📄 PDF 智能问答助手")
    st_mock.text_area.assert_called_once()
    st_mock.file_uploader.assert_called_once_with(
        "上传你的 PDF 文档（可选） :memo:",
        type=["pdf"],
        accept_multiple_files=True,
    )


def test_main_warns_when_query_is_empty(st_mock):
    st_mock.button.side_effect = (
        lambda label, *args, **kwargs: label == "提交查询"
    )

    app.main()

    st_mock.warning.assert_called_once_with("请输入问题")


def test_main_runs_rag_chain_and_displays_result(monkeypatch, st_mock):
    st_mock.text_area.return_value = "核心观点是什么？"
    st_mock.button.side_effect = (
        lambda label, *args, **kwargs: label == "提交查询"
    )
    run_chain = MagicMock(return_value="这是回答")
    monkeypatch.setattr(app, "run_rag_chain", run_chain)

    app.main()

    run_chain.assert_called_once_with(query="核心观点是什么？")
    st_mock.spinner.assert_called_once_with("思考中...")
    st_mock.write.assert_called_once_with("这是回答")


def test_main_processes_uploaded_files(monkeypatch, st_mock):
    uploaded_file = MagicMock()
    st_mock.file_uploader.return_value = [uploaded_file]
    st_mock.button.side_effect = (
        lambda label, *args, **kwargs: label == "上传并处理"
    )
    add_to_db = MagicMock()
    monkeypatch.setattr(app, "add_to_db", add_to_db)

    app.main()

    add_to_db.assert_called_once_with([uploaded_file])
    st_mock.success.assert_called_once_with(
        ":file_folder: 文档已成功添加到知识库！"
    )


def test_main_displays_actionable_error_without_success(
    monkeypatch,
    st_mock,
):
    uploaded_file = MagicMock()
    st_mock.file_uploader.return_value = [uploaded_file]
    st_mock.button.side_effect = (
        lambda label, *args, **kwargs: label == "上传并处理"
    )
    monkeypatch.setattr(
        app,
        "add_to_db",
        MagicMock(side_effect=app.NoExtractableTextError("需要 OCR")),
    )

    app.main()

    st_mock.error.assert_called_once_with("需要 OCR")
    st_mock.success.assert_not_called()
