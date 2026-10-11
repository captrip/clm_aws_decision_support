from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Type

from sentinal.core.actor_ref import ActorRef
from sentinal.core.message import Message
from .envelop import Envelope
from .mailbox import Mailbox


class DefaultDispatcher:
    """
    Default in-memory dispatcher.

    Responsible for routing messages between actors.

    It performs no business logic.
    """

    def __init__(self) -> None:

        self._mailboxes: dict[ActorRef, Mailbox] = {}

        self._subscriptions: DefaultDict[
            Type[Message],
            list[ActorRef],
        ] = defaultdict(list)

    def register(
        self,
        actor: ActorRef,
        mailbox: Mailbox,
    ) -> None:

        self._mailboxes[actor] = mailbox

    def subscribe(
        self,
        actor: ActorRef,
        message_type: Type[Message],
    ) -> None:

        self._subscriptions[message_type].append(actor)

    async def publish(
        self,
        message: Message,
    ) -> None:

        subscribers = self._subscriptions.get(
            type(message),
            [],
        )

        for actor in subscribers:

            mailbox = self._mailboxes.get(actor)

            if mailbox is None:
                continue

            await mailbox.put(
                Envelope(
                    message=message,
                    recipient=actor,
                )
            )