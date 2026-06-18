"""Tests for the index-version guard (#1) and API auth/rate-limit (#6)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.config import get_settings
from app.retrieval.vector_store import IndexMismatchError, VectorStore

# --- #1 index-version guard ---------------------------------------------------

_M = {
    "embed_provider": "local",
    "embed_model": "BAAI/bge-m3",
    "dim": 1024,
    "chunk_target_chars": 700,
    "chunk_overlap_chars": 120,
}


def _mem_store():
    s = get_settings()
    s.qdrant_mode = "local"
    s.qdrant_path = ":memory:"
    return VectorStore(s)


def test_manifest_match_passes():
    store = _mem_store()
    store.ensure_collection(1024, _M)
    store.assert_manifest(dict(_M))  # identical → ok


def test_manifest_same_dim_different_model_raises():
    # The scary case: voyage-3 and BGE-M3 are both 1024-d.
    store = _mem_store()
    store.ensure_collection(1024, _M)
    with pytest.raises(IndexMismatchError):
        store.assert_manifest({**_M, "embed_model": "voyage-3"})


def test_manifest_missing_raises():
    store = _mem_store()
    store.ensure_collection(1024)  # no manifest written
    with pytest.raises(IndexMismatchError):
        store.assert_manifest(_M)


# --- #6 auth + rate limit -----------------------------------------------------


def test_auth_required_when_token_set():
    client = TestClient(main_mod.app)
    main_mod.settings.api_auth_token = "secret"
    try:
        assert client.get("/strategies").status_code == 401
        ok = client.get("/strategies", headers={"X-API-Key": "secret"})
        assert ok.status_code == 200
        bearer = client.get("/strategies", headers={"Authorization": "Bearer secret"})
        assert bearer.status_code == 200
    finally:
        main_mod.settings.api_auth_token = ""


def test_open_when_no_token():
    client = TestClient(main_mod.app)
    main_mod.settings.api_auth_token = ""
    assert client.get("/strategies").status_code == 200


def test_rate_limit_429():
    client = TestClient(main_mod.app)
    main_mod._hits.clear()
    main_mod.settings.rate_limit_per_min = 3
    try:
        codes = [client.get("/strategies").status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert 429 in codes
    finally:
        main_mod.settings.rate_limit_per_min = 120
        main_mod._hits.clear()
