"""RunSift Upstream Subsystem.

High-throughput client routing cache-optimized requests to Nebius Token Factory.
Allowed dependencies per wiring.yaml: ["core"].
"""

from src.upstream.client import NebiusClient

__all__ = ["NebiusClient"]
