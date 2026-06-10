from config import ensure_runtime_directories


def test_ensure_runtime_directories_creates_missing_directories(tmp_path):
    directories = (
        tmp_path / "docs",
        tmp_path / "temp",
        tmp_path / "uploads",
        tmp_path / "chroma_db",
    )

    ensure_runtime_directories(directories)

    assert all(directory.is_dir() for directory in directories)
