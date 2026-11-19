"""大模型接口抽象。

管线只依赖 ChatClient 协议，不依赖具体 SDK —— 这样 Stage 1 / Stage 2 的逻辑
可以在没有可用接口的环境里完整测透，也便于在云端接口与本地 Ollama / vLLM
之间切换（未发表论文的隐私要求决定了本地部署必须是一等选项）。

HTTP 实现用标准库 urllib 而非 openai SDK：这里只需要 /chat/completions 一个
端点，为此引入一个大依赖并不划算，且 OpenAI 兼容服务（Ollama、vLLM、
LM Studio）的请求体都是同一套。
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """模型调用失败。管线据此降级，不让单个策略的失败拖垮整次请求。"""


@dataclass
class ChatMessage:
    role: str
    content: str


@runtime_checkable
class ChatClient(Protocol):
    """管线唯一依赖的模型接口。"""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        """返回模型的文本输出。失败时抛 LLMError。"""
        ...


@dataclass
class OpenAICompatClient:
    """OpenAI 兼容的 /chat/completions 客户端（云端或本地 Ollama / vLLM）。"""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: float = 120.0
    max_retries: int = 2

    def __post_init__(self) -> None:
        settings = get_settings()
        self.base_url = (self.base_url or settings.llm_base_url).rstrip("/")
        self.api_key = self.api_key or settings.llm_api_key
        self.model = self.model or settings.llm_model_name

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                return self._extract_text(parsed)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    # 指数退避：本地 Ollama 冷启动加载模型常需若干秒
                    time.sleep(2**attempt)
                    continue

        raise LLMError(f"模型调用失败（{self.max_retries + 1} 次尝试）: {last_error}")

    @staticmethod
    def _extract_text(payload: dict[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(f"响应缺少 choices: {payload}")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMError(f"choices[0] 格式异常: {first}")
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        raise LLMError(f"响应缺少 message.content: {first}")


@dataclass
class ScriptedClient:
    """测试与离线演示用的脚本化客户端。

    按调用顺序返回预设响应，并记录收到的 messages，便于断言「提示词里确实
    带上了锁定术语与槽位值」—— 提示词内容是管线正确性的一部分。
    """

    responses: list[str] = field(default_factory=list)
    calls: list[list[ChatMessage]] = field(default_factory=list)
    default_response: str = ""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format_json: bool = False,
    ) -> str:
        self.calls.append(list(messages))
        if self.responses:
            return self.responses.pop(0)
        if self.default_response:
            return self.default_response
        raise LLMError("ScriptedClient 已无预设响应")

    @property
    def last_prompt(self) -> str:
        """最近一次调用的全部消息文本，便于做包含性断言。"""
        if not self.calls:
            return ""
        return "\n".join(m.content for m in self.calls[-1])


def build_default_client() -> ChatClient:
    return OpenAICompatClient()
