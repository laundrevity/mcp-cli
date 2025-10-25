from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ServerCapabilities:
    resources: bool
    prompts: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    sampling: bool = False
    roots: bool = False
    elicitation: bool = False
    protocol_version: str = "2025-06-18"

    def to_payload(self) -> Dict[str, object]:
        return {
            "resources": self.resources,
            "prompts": list(self.prompts),
            "tools": list(self.tools),
            "sampling": self.sampling,
            "roots": self.roots,
            "elicitation": self.elicitation,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "ServerCapabilities":
        return cls(
            resources=bool(payload.get("resources", False)),
            prompts=list(payload.get("prompts", [])),
            tools=list(payload.get("tools", [])),
            sampling=bool(payload.get("sampling", False)),
            roots=bool(payload.get("roots", False)),
            elicitation=bool(payload.get("elicitation", False)),
            protocol_version=str(payload.get("protocol_version", "2025-06-18")),
        )


@dataclass(frozen=True)
class ClientCapabilities:
    sampling: bool = False
    roots: bool = False
    elicitation: bool = False
    protocol_version: str = "2025-06-18"

    def to_payload(self) -> Dict[str, object]:
        return {
            "sampling": self.sampling,
            "roots": self.roots,
            "elicitation": self.elicitation,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "ClientCapabilities":
        return cls(
            sampling=bool(payload.get("sampling", False)),
            roots=bool(payload.get("roots", False)),
            elicitation=bool(payload.get("elicitation", False)),
            protocol_version=str(payload.get("protocol_version", "2025-06-18")),
        )


@dataclass(frozen=True)
class HandshakeResult:
    client_capabilities: ClientCapabilities
    server_capabilities: ServerCapabilities
