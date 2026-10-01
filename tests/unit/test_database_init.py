"""A fresh install creates the folder of its SQLite database."""

from __future__ import annotations

from backend.database import sqlite_dir


def test_sqlite_folder_is_created(tmp_path):
    db = tmp_path / "fresh" / "db" / "trichome.db"
    sqlite_dir(f"sqlite:///{db}")
    assert db.parent.is_dir()


def test_memory_and_relative_urls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sqlite_dir("sqlite:///:memory:")
    sqlite_dir("sqlite:///./trichome.db")
    assert list(tmp_path.iterdir()) == []
