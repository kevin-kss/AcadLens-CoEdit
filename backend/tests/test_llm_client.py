"""OpenAI 兼容客户端单测。

不发真实网络请求：所有用例都打桩 urlopen 或只测纯函数部分。响应解析是
最容易在生产中静默出错的地方（不同服务的响应体细节有差异），所以这里
重点覆盖畸形响应必须抛 LLMError 而不是返回空串 —— 返回空串会让管线把
「接口坏了」误判成「模型输出了空草稿」。
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.services.llm.client import (
    ChatClient,
    ChatMessage,
    LLMError,
    OpenAICompatClient,
    ScriptedClient,
    build_default_client,
)


def _ok_payload(text: str = "hello") -> dict[str, object]:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


class _FakeResponse(io.BytesIO):
    """urlopen 的上下文管理器桩。"""

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class _Captured:
    """承接 urlopen 桩捕获的请求，带类型以免取值退化为 object。"""

    request: urllib.request.Request | None = None
    body: dict[str, Any] = field(default_factory=dict)
    timeout: float = 0.0


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    captured: _Captured,
) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
        captured.request = request
        # Request.data 的声明类型是联合类型，但我们的桩只会收到 bytes
        raw = request.data
        assert isinstance(raw, bytes)
        captured.body = json.loads(raw.decode("utf-8"))
        captured.timeout = timeout
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


class TestResponseExtraction:
    def test_extracts_content(self):
        assert OpenAICompatClient._extract_text(_ok_payload("结果")) == "结果"

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"choices": []},
            {"choices": "not a list"},
            {"choices": ["not a dict"]},
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": None}}]},
            {"choices": [{"message": "not a dict"}]},
            {"error": {"message": "model not found"}},
        ],
    )
    def test_malformed_payload_raises(self, payload: dict[str, object]):
        """必须抛错而非返回空串：空串会被管线误判成「模型输出了空草稿」。"""
        with pytest.raises(LLMError):
            OpenAICompatClient._extract_text(payload)

    def test_empty_string_content_is_valid(self):
        """内容为空串是模型的合法输出（与字段缺失不同），不应抛错。"""
        assert OpenAICompatClient._extract_text(_ok_payload("")) == ""


class TestRequestConstruction:
    def test_request_body_shape(self, monkeypatch: pytest.MonkeyPatch):
        captured = _Captured()
        _patch_urlopen(monkeypatch, _ok_payload(), captured)
        client = OpenAICompatClient(base_url="http://localhost:11434/v1", api_key="k", model="m")
        client.complete([ChatMessage("user", "hi")], temperature=0.3)

        body = captured.body
        assert body["model"] == "m"
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["temperature"] == 0.3
        assert body["stream"] is False
        assert "max_tokens" not in body
        assert "response_format" not in body

    def test_optional_fields_included_when_set(self, monkeypatch: pytest.MonkeyPatch):
        captured = _Captured()
        _patch_urlopen(monkeypatch, _ok_payload(), captured)
        client = OpenAICompatClient(base_url="http://x/v1", api_key="k", model="m")
        client.complete([], max_tokens=256, response_format_json=True)

        body = captured.body
        assert body["max_tokens"] == 256
        assert body["response_format"] == {"type": "json_object"}

    def test_trailing_slash_in_base_url_normalized(self, monkeypatch: pytest.MonkeyPatch):
        captured = _Captured()
        _patch_urlopen(monkeypatch, _ok_payload(), captured)
        client = OpenAICompatClient(base_url="http://x/v1/", api_key="k", model="m")
        client.complete([])
        assert captured.request is not None
        assert captured.request.full_url == "http://x/v1/chat/completions"

    def test_authorization_header_sent(self, monkeypatch: pytest.MonkeyPatch):
        captured = _Captured()
        _patch_urlopen(monkeypatch, _ok_payload(), captured)
        client = OpenAICompatClient(base_url="http://x/v1", api_key="secret", model="m")
        client.complete([])
        assert captured.request is not None
        headers = {k.lower(): v for k, v in captured.request.headers.items()}
        assert headers["authorization"] == "Bearer secret"
        assert headers["content-type"] == "application/json"

    def test_defaults_come_from_settings(self):
        """未显式传参时应落到 .env 配置，便于本地 Ollama 与云端切换。"""
        from app.config import get_settings

        settings = get_settings()
        client = OpenAICompatClient()
        assert client.base_url == settings.llm_base_url.rstrip("/")
        assert client.model == settings.llm_model_name


class TestRetryBehaviour:
    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch):
        calls = {"n": 0}

        def flaky(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError("connection refused")
            return _FakeResponse(json.dumps(_ok_payload("ok")).encode("utf-8"))

        monkeypatch.setattr("urllib.request.urlopen", flaky)
        monkeypatch.setattr("time.sleep", lambda _: None)
        client = OpenAICompatClient(base_url="http://x/v1", api_key="k", model="m")
        assert client.complete([]) == "ok"
        assert calls["n"] == 2

    def test_exhausted_retries_raise_llm_error(self, monkeypatch: pytest.MonkeyPatch):
        def always_fail(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
            raise urllib.error.URLError("down")

        monkeypatch.setattr("urllib.request.urlopen", always_fail)
        monkeypatch.setattr("time.sleep", lambda _: None)
        client = OpenAICompatClient(base_url="http://x/v1", api_key="k", model="m", max_retries=2)
        with pytest.raises(LLMError, match="3 次尝试"):
            client.complete([])

    def test_invalid_json_response_retried(self, monkeypatch: pytest.MonkeyPatch):
        calls = {"n": 0}

        def bad_then_good(request: urllib.request.Request, timeout: float = 0) -> _FakeResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResponse(b"not json at all")
            return _FakeResponse(json.dumps(_ok_payload("ok")).encode("utf-8"))

        monkeypatch.setattr("urllib.request.urlopen", bad_then_good)
        monkeypatch.setattr("time.sleep", lambda _: None)
        client = OpenAICompatClient(base_url="http://x/v1", api_key="k", model="m")
        assert client.complete([]) == "ok"

    def test_timeout_is_passed_through(self, monkeypatch: pytest.MonkeyPatch):
        captured = _Captured()
        _patch_urlopen(monkeypatch, _ok_payload(), captured)
        client = OpenAICompatClient(base_url="http://x/v1", api_key="k", model="m", timeout=7.5)
        client.complete([])
        assert captured.timeout == 7.5


class TestProtocolConformance:
    def test_both_clients_satisfy_protocol(self):
        """管线只依赖 ChatClient 协议，两个实现都必须符合。"""
        assert isinstance(ScriptedClient(), ChatClient)
        assert isinstance(OpenAICompatClient(base_url="http://x/v1"), ChatClient)

    def test_build_default_client_returns_http_client(self):
        assert isinstance(build_default_client(), OpenAICompatClient)
