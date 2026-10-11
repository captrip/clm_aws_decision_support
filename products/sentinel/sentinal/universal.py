"""
Universal Sentinel - Main Entry Point

Import this to use Sentinel with any framework.

Usage:
    from sentinal.universal import supervise

    # Works with any framework
    supervised = supervise(your_agent)
    result = await supervised.execute(input_data)
"""

# Re-export from sentinel_integration for backward compatibility
from sentinal.adapters.sentinel_integration import (
    supervise,
    SupervisedAgent,
    detect_framework,
)

__all__ = [
    "supervise",
    "SupervisedAgent",
    "detect_framework",
]
