from __future__ import annotations

from pathlib import Path

from app.config import resolve_default_data_dir


def test_resolve_default_data_dir_prefers_workspace_data_folder(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "OhioT1DM").mkdir()
    monkeypatch.delenv("OHIO_DATA_DIR", raising=False)

    resolved = resolve_default_data_dir(tmp_path)

    assert resolved == (tmp_path / "data").resolve()


def test_resolve_default_data_dir_respects_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom-dataset"
    custom.mkdir()
    monkeypatch.setenv("OHIO_DATA_DIR", str(custom))

    resolved = resolve_default_data_dir(tmp_path / "unused")

    assert resolved == custom.resolve()
