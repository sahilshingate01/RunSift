"""RunSift Proxy Subsystem.

FastAPI / OpenAI-compatible reverse proxy ingestion gateway.
Allowed dependencies per wiring.yaml: ["core", "canonicalizer", "compact", "state", "upstream"].
"""

from src.proxy.gateway import app, create_app

__all__ = ["app", "create_app"]
