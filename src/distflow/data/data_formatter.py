from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from distflow.data.types import DatasetProcessOutputItem, MessageData


@runtime_checkable
class FormatterProtocol(Protocol):
    def format(self, raw_item: dict[str, Any]) -> DatasetProcessOutputItem: ...


class AlpacaFormatter:
    def __init__(self, *, user_key: str, assistant_key: str) -> None:
        self.user_key = user_key
        self.assistant_key = assistant_key

    def format(self, raw_item: dict[str, Any]) -> DatasetProcessOutputItem:
        assert (
            self.user_key in raw_item
        ), f"User key '{self.user_key}' not found in raw item"
        assert (
            self.assistant_key in raw_item
        ), f"Assistant key '{self.assistant_key}' not found in raw item"
        user_content = raw_item[self.user_key]
        assert isinstance(
            user_content, str
        ), f"User content must be a string, got {type(user_content).__name__}: {user_content}"
        assistant_content = raw_item[self.assistant_key]
        assert isinstance(
            assistant_content, str
        ), f"Assistant content must be a string, got {type(assistant_content).__name__}: {assistant_content}"

        return DatasetProcessOutputItem(
            messages=[
                cast(MessageData, {"role": "user", "content": user_content}),
                cast(MessageData, {"role": "assistant", "content": assistant_content}),
            ],
            meta={
                "user_key": self.user_key,
                "assistant_key": self.assistant_key,
                "raw_item": raw_item,
            },
        )


from typing import Any, cast


class ShareGptFormatter:
    def __init__(self, *, conversations_key: str) -> None:
        self.conversations_key = conversations_key

    def format(self, raw_item: dict[str, Any]) -> DatasetProcessOutputItem:
        assert (
            self.conversations_key in raw_item
        ), f"Conversations key '{self.conversations_key}' not found in raw item"

        conversations = raw_item[self.conversations_key]
        assert isinstance(
            conversations, list
        ), f"Conversations must be a list, got {type(conversations).__name__}: {conversations}"

        messages: list[MessageData] = []

        for conv in conversations:
            if not isinstance(conv, dict):
                continue

            # 检测格式类型并提取字段
            role = None
            content = None

            # 标准格式: {"role": "user", "content": "..."}
            if "role" in conv and "content" in conv:
                role = conv.get("role")
                content = conv.get("content")

            # ShareGPT 格式: {"from": "human", "value": "..."}
            elif "from" in conv and "value" in conv:
                from_field = conv.get("from")
                content = conv.get("value")

                assert isinstance(from_field, str) and isinstance(
                    content, str
                ), "from和content必须都是str类型"

                role_mapping = {
                    "human": "user",
                    "gpt": "assistant",
                    "system": "system",
                    "user": "user",
                    "assistant": "assistant",
                }
                role = role_mapping.get(from_field, from_field)

            # 添加到 messages
            if role is not None and content is not None:
                messages.append(cast(MessageData, {"role": role, "content": content}))

        return DatasetProcessOutputItem(
            messages=messages,
            meta={
                "conversations_key": self.conversations_key,
                "raw_item": raw_item,
                "detected_format": (
                    "sharegpt"
                    if any(isinstance(c, dict) and "from" in c for c in conversations)
                    else "standard"
                ),
            },
        )
