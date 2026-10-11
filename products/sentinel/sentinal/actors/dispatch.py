from __future__ import annotations

import inspect
from typing import Awaitable, Callable

from sentinal.core.message import Message

Handler = Callable[[Message], Awaitable[None]]


def discover_handlers(instance) -> dict[type[Message], Handler]:
    """
    Discover all @handles methods on an actor.
    """

    handlers: dict[type[Message], Handler] = {}

    for _, member in inspect.getmembers(instance):

        message_type = getattr(
            member,
            "__sentinel_message_type__",
            None,
        )

        if message_type is not None:
            handlers[message_type] = member

    return handlers