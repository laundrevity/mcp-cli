from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol

from .models import ContentBlock, SamplingRequest, SamplingResponse

LOG = logging.getLogger("mcp_cli.sampling")


class SamplingProvider(Protocol):
    async def create_message(
        self,
        request: SamplingRequest,
    ) -> SamplingResponse:
        ...


@dataclass
class SamplingConfig:
    base_url: str = "http://127.0.0.1:8080"
    path: str = "/v1/chat/completions"
    model_name: str = "local-llm"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 60.0


class LocalLLMSamplingProvider:
    """Sampling provider that targets a llama.cpp-compatible server."""

    def __init__(self, config: Optional[SamplingConfig] = None) -> None:
        self.config = config or SamplingConfig()

    async def create_message(
        self,
        request: SamplingRequest,
    ) -> SamplingResponse:
        payload = self._build_payload(request)
        LOG.debug("Submitting sampling request to local LLM: %s", payload)

        try:
            response_data = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._execute_http(payload),
            )
            LOG.debug("Received response from local LLM: %s", response_data)
            return self._parse_response(response_data)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Local LLM sampling failed: %s", exc)
            return SamplingResponse(
                role="assistant",
                content=ContentBlock(
                    type="text",
                    text=f"[sampling error] {exc}",
                ),
                model=self.config.model_name,
                stop_reason="error",
            )

    def _build_payload(self, request: SamplingRequest) -> dict:
        messages = []
        if request.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )
        for message in request.messages:
            content_payload = message.content.to_payload()
            text = content_payload.get("text") or ""
            messages.append(
                {
                    "role": message.role,
                    "content": text,
                }
            )

        max_tokens = request.max_tokens or self.config.max_tokens

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        return payload

    def _execute_http(self, payload: dict) -> dict:
        url = f"{self.config.base_url.rstrip('/')}{self.config.path}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as http_error:
            body = http_error.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {http_error.code}: {body}") from http_error

    def _parse_response(self, response: dict) -> SamplingResponse:
        choice = None

        if isinstance(response, dict):
            if "choices" in response:
                choices = response.get("choices") or []
                if choices:
                    choice = choices[0]
            elif "content" in response:
                choice = response

        if not isinstance(choice, dict):
            return SamplingResponse(
                role="assistant",
                content=ContentBlock(
                    type="text",
                    text="[sampling error] Invalid response payload",
                ),
                model=self.config.model_name,
                stop_reason="error",
            )

        message = choice.get("message", choice)
        role = message.get("role", "assistant")
        content_text = message.get("content")

        if isinstance(content_text, list):
            text_parts = [
                part.get("text", "")
                for part in content_text
                if isinstance(part, dict)
            ]
            content_text = "".join(text_parts)

        if not isinstance(content_text, str):
            content_text = str(content_text)

        stop_reason = choice.get("finish_reason") or choice.get("stopReason")
        model = response.get("model") or choice.get("model") or self.config.model_name

        return SamplingResponse(
            role=role,
            content=ContentBlock(
                type="text",
                text=content_text.strip(),
            ),
            model=model,
            stop_reason=stop_reason,
        )
