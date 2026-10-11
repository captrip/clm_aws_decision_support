from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Capabilities:
    """
    Declares features supported by an adapter.

    Runtime behaviour is adapted automatically based
    on these capabilities.
    """

    streaming: bool = False

    checkpointing: bool = False

    cancellation: bool = False

    replay: bool = False

    tool_events: bool = False

    node_events: bool = False

    prompt_mutation: bool = False

    gpu_capable: bool = False