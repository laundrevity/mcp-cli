from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _filter_none(mapping: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


@dataclass(frozen=True)
class ServerCapabilities:
    logging: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    tools: Optional[Dict[str, Any]] = None
    completions: Optional[Dict[str, Any]] = None
    experimental: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        return _filter_none(
            {
                "logging": self.logging,
                "prompts": self.prompts,
                "resources": self.resources,
                "tools": self.tools,
                "completions": self.completions,
                "experimental": self.experimental,
            }
        )

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ServerCapabilities":
        return cls(
            logging=payload.get("logging"),
            prompts=payload.get("prompts"),
            resources=payload.get("resources"),
            tools=payload.get("tools"),
            completions=payload.get("completions"),
            experimental=payload.get("experimental"),
        )


@dataclass(frozen=True)
class ClientCapabilities:
    sampling: Optional[Dict[str, Any]] = None
    roots: Optional[Dict[str, Any]] = None
    elicitation: Optional[Dict[str, Any]] = None
    experimental: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        return _filter_none(
            {
                "sampling": self.sampling,
                "roots": self.roots,
                "elicitation": self.elicitation,
                "experimental": self.experimental,
            }
        )

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ClientCapabilities":
        return cls(
            sampling=payload.get("sampling"),
            roots=payload.get("roots"),
            elicitation=payload.get("elicitation"),
            experimental=payload.get("experimental"),
        )


@dataclass(frozen=True)
class ServerInfo:
    name: str
    version: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "version": self.version,
        }
        if self.title is not None:
            payload["title"] = self.title
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ServerInfo":
        return cls(
            name=str(payload.get("name", "")),
            version=str(payload.get("version", "")),
            title=payload.get("title"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ClientInfo:
    name: str
    version: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "version": self.version,
        }
        if self.title is not None:
            payload["title"] = self.title
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ClientInfo":
        return cls(
            name=str(payload.get("name", "")),
            version=str(payload.get("version", "")),
            title=payload.get("title"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class HandshakeResult:
    client_capabilities: ClientCapabilities
    server_capabilities: ServerCapabilities
    protocol_version: str
    request_id: int
    server_info: ServerInfo
    client_info: ClientInfo
    instructions: Optional[str] = None
