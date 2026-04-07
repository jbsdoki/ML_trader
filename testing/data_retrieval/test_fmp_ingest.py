"""FMP ingest: API key / payload helpers and HTTP via mocks (no real network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_retrieval import fmp_ingest as fmp


def test_fmp_api_key_uses_explicit_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    assert fmp._fmp_api_key("secret") == "secret"


def test_fmp_api_key_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    assert fmp._fmp_api_key("  abc  ") == "abc"


def test_fmp_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMP_API_KEY", "from-env")
    assert fmp._fmp_api_key(None) == "from-env"


def test_fmp_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FMP_API_KEY"):
        fmp._fmp_api_key(None)


def test_fmp_api_base_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FMP_API_BASE", raising=False)
    assert fmp._fmp_api_base() == fmp._DEFAULT_FMP_BASE


def test_fmp_api_base_from_env_strips_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMP_API_BASE", "https://example.test/stable/")
    assert fmp._fmp_api_base() == "https://example.test/stable"


def test_fmp_parse_payload_raises_on_error_message() -> None:
    with pytest.raises(ValueError, match="FMP error"):
        fmp._fmp_parse_payload({"Error Message": "bad key"}, "profile/AAPL")


def test_fmp_parse_payload_passes_through_list() -> None:
    data = [{"symbol": "AAPL"}]
    assert fmp._fmp_parse_payload(data, "profile/AAPL") == data


def _mock_urlopen_response(body: bytes) -> MagicMock:
    cm = MagicMock()
    cm.read.return_value = body
    cm.__enter__.return_value = cm
    cm.__exit__.return_value = None
    return cm


def test_fmp_get_json_appends_apikey_with_ampersand_when_path_has_query() -> None:
    captured: dict[str, str] = {}

    def fake_urlopen(req: object, timeout: float = 60.0) -> MagicMock:
        captured["url"] = getattr(req, "full_url", "")
        return _mock_urlopen_response(b"[]")

    with patch.object(fmp, "urlopen", side_effect=fake_urlopen):
        fmp.fmp_get_json("ratios/AAPL?limit=1", api_key="k")

    assert "apikey=" in captured["url"]
    assert "ratios/AAPL?limit=1" in captured["url"]
    assert "&apikey=" in captured["url"] or "?limit=1&apikey=" in captured["url"]


def test_fmp_get_json_returns_parsed_json() -> None:
    payload = [{"symbol": "AAPL", "companyName": "Apple Inc."}]
    raw = __import__("json").dumps(payload).encode("utf-8")

    with patch.object(fmp, "urlopen", return_value=_mock_urlopen_response(raw)):
        out = fmp.fmp_get_json("profile/AAPL", api_key="test-key")

    assert out == payload


def test_fetch_company_profile_empty_list_returns_empty_df() -> None:
    with patch.object(fmp, "fmp_get_json", return_value=[]):
        df = fmp.fetch_company_profile("AAPL", api_key="x")

    assert df.empty


def test_fetch_company_profile_list_to_dataframe() -> None:
    rows = [{"symbol": "AAPL", "companyName": "Apple Inc."}]

    with patch.object(fmp, "fmp_get_json", return_value=rows):
        df = fmp.fetch_company_profile("aapl", api_key="x")

    assert len(df) == 1
    assert df["symbol"].iloc[0] == "AAPL"


def test_fmp_ingestor_delegates_get_json() -> None:
    with patch.object(fmp, "fmp_get_json", return_value={"ok": True}) as m:
        ing = fmp.FMPIngestor(api_key="my-key")
        out = ing.get_json("profile/MSFT")

    assert out == {"ok": True}
    m.assert_called_once()
    assert m.call_args.kwargs["api_key"] == "my-key"


def test_fmp_ingestor_get_json_explicit_key_overrides_instance() -> None:
    with patch.object(fmp, "fmp_get_json", return_value=[]) as m:
        ing = fmp.FMPIngestor(api_key="default")
        ing.get_json("x", api_key="override")

    assert m.call_args.kwargs["api_key"] == "override"
