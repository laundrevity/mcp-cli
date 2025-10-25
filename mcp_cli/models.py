from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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


@dataclass(frozen=True)
class ContentBlock:
    type: str
    text: Optional[str] = None
    data: Optional[str] = None
    mime_type: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": self.type}
        if self.text is not None:
            payload["text"] = self.text
        if self.data is not None:
            payload["data"] = self.data
        if self.mime_type is not None:
            payload["mimeType"] = self.mime_type
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ContentBlock":
        return cls(
            type=str(payload.get("type", "")),
            text=payload.get("text"),
            data=payload.get("data"),
            mime_type=payload.get("mimeType"),
        )


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    title: Optional[str] = None
    input_schema: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.title is not None:
            payload["title"] = self.title
        if self.input_schema:
            payload["inputSchema"] = self.input_schema
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ToolDefinition":
        return cls(
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            title=payload.get("title"),
            input_schema=dict(payload.get("inputSchema", {})),
        )


@dataclass(frozen=True)
class ToolCallResult:
    content: List[ContentBlock]
    is_error: bool = False

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "content": [block.to_payload() for block in self.content],
            "isError": self.is_error,
        }
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ToolCallResult":
        content_payload = payload.get("content", [])
        return cls(
            content=[
                ContentBlock.from_payload(block)
                for block in content_payload
            ],
            is_error=bool(payload.get("isError", False)),
        )


@dataclass
class ResourceDescriptor:
    uri: str
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    mime_type: Optional[str] = None
    size: Optional[int] = None
    annotations: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "uri": self.uri,
            "name": self.name,
        }
        if self.title is not None:
            payload["title"] = self.title
        if self.description is not None:
            payload["description"] = self.description
        if self.mime_type is not None:
            payload["mimeType"] = self.mime_type
        if self.size is not None:
            payload["size"] = self.size
        if self.annotations:
            payload["annotations"] = self.annotations
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ResourceDescriptor":
        return cls(
            uri=str(payload.get("uri", "")),
            name=str(payload.get("name", "")),
            title=payload.get("title"),
            description=payload.get("description"),
            mime_type=payload.get("mimeType"),
            size=payload.get("size"),
            annotations=payload.get("annotations"),
        )


@dataclass
class ResourceContent:
    uri: str
    name: str
    title: Optional[str] = None
    mime_type: Optional[str] = None
    text: Optional[str] = None
    blob: Optional[str] = None
    annotations: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "uri": self.uri,
            "name": self.name,
        }
        if self.title is not None:
            payload["title"] = self.title
        if self.mime_type is not None:
            payload["mimeType"] = self.mime_type
        if self.text is not None:
            payload["text"] = self.text
        if self.blob is not None:
            payload["blob"] = self.blob
        if self.annotations:
            payload["annotations"] = self.annotations
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ResourceContent":
        return cls(
            uri=str(payload.get("uri", "")),
            name=str(payload.get("name", "")),
            title=payload.get("title"),
            mime_type=payload.get("mimeType"),
            text=payload.get("text"),
            blob=payload.get("blob"),
            annotations=payload.get("annotations"),
        )


@dataclass(frozen=True)
class SamplingMessage:
    role: str
    content: ContentBlock

    def to_payload(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SamplingMessage":
        return cls(
            role=str(payload.get("role", "")),
            content=ContentBlock.from_payload(payload.get("content", {})),
        )


@dataclass(frozen=True)
class SamplingRequest:
    messages: List[SamplingMessage]
    model_preferences: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [message.to_payload() for message in self.messages],
        }
        if self.model_preferences:
            payload["modelPreferences"] = self.model_preferences
        if self.system_prompt is not None:
            payload["systemPrompt"] = self.system_prompt
        if self.max_tokens is not None:
            payload["maxTokens"] = self.max_tokens
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SamplingRequest":
        return cls(
            messages=[
                SamplingMessage.from_payload(message)
                for message in payload.get("messages", [])
            ],
            model_preferences=payload.get("modelPreferences"),
            system_prompt=payload.get("systemPrompt"),
            max_tokens=payload.get("maxTokens"),
        )


@dataclass(frozen=True)
class SamplingResponse:
    role: str
    content: ContentBlock
    model: Optional[str] = None
    stop_reason: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "role": self.role,
            "content": self.content.to_payload(),
        }
        if self.model is not None:
            payload["model"] = self.model
        if self.stop_reason is not None:
            payload["stopReason"] = self.stop_reason
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SamplingResponse":
        return cls(
            role=str(payload.get("role", "")),
            content=ContentBlock.from_payload(payload.get("content", {})),
            model=payload.get("model"),
            stop_reason=payload.get("stopReason"),
        )


@dataclass
class ResourceTemplate:
    uri_template: str
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    mime_type: Optional[str] = None
    annotations: Optional[Dict[str, Any]] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "uriTemplate": self.uri_template,
            "name": self.name,
        }
        if self.title is not None:
            payload["title"] = self.title
        if self.description is not None:
            payload["description"] = self.description
        if self.mime_type is not None:
            payload["mimeType"] = self.mime_type
        if self.annotations:
            payload["annotations"] = self.annotations
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ResourceTemplate":
        return cls(
            uri_template=str(payload.get("uriTemplate", "")),
            name=str(payload.get("name", "")),
            title=payload.get("title"),
            description=payload.get("description"),
            mime_type=payload.get("mimeType"),
            annotations=payload.get("annotations"),
        )


@dataclass(frozen=True)
class RootDescriptor:
    uri: str
    name: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"uri": self.uri}
        if self.name is not None:
            payload["name"] = self.name
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RootDescriptor":
        return cls(
            uri=str(payload.get("uri", "")),
            name=payload.get("name"),
        )


@dataclass(frozen=True)
class PromptArgument:
    name: str
    description: Optional[str] = None
    required: bool = False
    type: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "required": self.required,
        }
        if self.description is not None:
            payload["description"] = self.description
        if self.type is not None:
            payload["type"] = self.type
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "PromptArgument":
        return cls(
            name=str(payload.get("name", "")),
            description=payload.get("description"),
            required=bool(payload.get("required", False)),
            type=payload.get("type"),
        )


@dataclass(frozen=True)
class PromptDefinition:
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    arguments: List[PromptArgument] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
        }
        if self.title is not None:
            payload["title"] = self.title
        if self.description is not None:
            payload["description"] = self.description
        if self.arguments:
            payload["arguments"] = [arg.to_payload() for arg in self.arguments]
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "PromptDefinition":
        return cls(
            name=str(payload.get("name", "")),
            title=payload.get("title"),
            description=payload.get("description"),
            arguments=[
                PromptArgument.from_payload(arg)
                for arg in payload.get("arguments", [])
            ],
        )


@dataclass(frozen=True)
class PromptRenderResult:
    description: Optional[str]
    messages: List[SamplingMessage]

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [message.to_payload() for message in self.messages],
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "PromptRenderResult":
        return cls(
            description=payload.get("description"),
            messages=[
                SamplingMessage.from_payload(message)
                for message in payload.get("messages", [])
            ],
        )
