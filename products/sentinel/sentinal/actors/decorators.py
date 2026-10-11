from __future__ import annotations

from typing import Type

from sentinal.core.message import Message


def handles(message_type: Type[Message]):
    """
    Marks a coroutine as the handler for a Message type.
    """

    def decorator(fn):
        fn.__sentinel_message_type__ = message_type
        return fn

    return decorator