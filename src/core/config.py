"""Configuration management for RunSift.

Loads environment variables (including repo-root .env) for Nebius Token Factory and RunSift gateways.
Strictly leaf module in wiring.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

_PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    {
        "",
        "your-token-factory-key",
        "your-api-key",
        "your_api_key_here",
        "changeme",
    }
)


def candidate_dotenv_paths() -> list[Path]:
    """Returns candidate .env file locations in priority order across cwd, home, Downloads, Desktop, and repo."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    home = Path.home()
    candidates = [
        Path.cwd() / ".env",
        home / ".runsift.env",
        home / ".cacheforge.env",
        home / ".env",
        home / "Downloads" / ".env",
        home / "Desktop" / ".env",
        repo_root / ".env",
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    return ordered


def _load_dotenv_if_present(override: bool = False) -> Path | None:
    """Loads key=value pairs from candidate .env files into os.environ in priority order.

    Higher-priority files (e.g. Path.cwd() / '.env') take precedence over lower-priority
    fallback locations (e.g. Downloads / '.env'), while skipping placeholder values like
    'your-token-factory-key'.
    """
    loaded_from: Path | None = None
    set_in_this_pass: set[str] = set()
    for env_path in candidate_dotenv_paths():
        if not env_path.is_file():
            continue
        try:
            lines = env_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if not key or val.lower() in _PLACEHOLDER_KEYS or key in set_in_this_pass:
                continue
            current = os.environ.get(key, "")
            if override or not current or current.lower() in _PLACEHOLDER_KEYS:
                os.environ[key] = val
                set_in_this_pass.add(key)
                if loaded_from is None:
                    loaded_from = env_path
    return loaded_from


_load_dotenv_if_present()

SUPPORTED_NVIDIA_MODELS: tuple[str, ...] = (
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
)


class RunSiftConfig(BaseModel):
    """Runtime configuration for RunSift reverse proxy and gateway."""

    nebius_api_key: str = Field(default_factory=lambda: os.getenv("NEBIUS_API_KEY", ""))
    nebius_base_url: str = Field(
        default_factory=lambda: os.getenv(
            "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"
        )
    )
    default_model: str = Field(
        default_factory=lambda: os.getenv(
            "UPSTREAM_MODEL",
            os.getenv(
                "RUNSIFT_DEFAULT_MODEL",
                os.getenv("CACHEFORGE_DEFAULT_MODEL", SUPPORTED_NVIDIA_MODELS[0]),
            ),
        )
    )
    api_key: str = Field(
        default_factory=lambda: os.getenv(
            "RUNSIFT_API_KEY", os.getenv("CACHEFORGE_API_KEY", "")
        )
    )
    # Listening host and port used when running programmatically or via python -m src.proxy.gateway.
    # Note: When starting with `uvicorn src.proxy:app --host ... --port ...`, CLI flags take precedence.
    host: str = Field(
        default_factory=lambda: os.getenv(
            "RUNSIFT_HOST", os.getenv("CACHEFORGE_HOST", "127.0.0.1")
        )
    )
    port: int = Field(
        default_factory=lambda: int(
            os.getenv("RUNSIFT_PORT", os.getenv("CACHEFORGE_PORT", "8000"))
        )
    )
    cors_allowed_origins: list[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "RUNSIFT_CORS_ORIGINS",
                os.getenv(
                    "CACHEFORGE_CORS_ORIGINS",
                    "https://runsift.fieldsmith.pro,https://cacheforge.fieldsmith.pro,http://localhost:8765,http://127.0.0.1:8765,http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000",
                ),
            ).split(",")
            if origin.strip()
        ]
    )

    @property
    def gateway_port(self) -> int:
        """Alias for port to support agent frameworks expecting gateway_port attribute."""
        return self.port

    # Feature Flags
    enable_canonicalizer: bool = Field(
        default_factory=lambda: (
            os.getenv("ENABLE_CANONICALIZER", "true").lower() in ("1", "true", "yes")
        )
    )
    enable_tool_suspension: bool = Field(
        default_factory=lambda: (
            os.getenv("ENABLE_TOOL_SUSPENSION", "true").lower() in ("1", "true", "yes")
        )
    )
    enable_negative_compaction: bool = Field(
        default_factory=lambda: (
            os.getenv("ENABLE_NEGATIVE_COMPACTION", "true").lower() in ("1", "true", "yes")
        )
    )

    # Invariant Bounds
    max_tool_output_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOOL_OUTPUT_TOKENS", "1000"))
    )
    tool_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_SECONDS", "60.0"))
    )
    condenser_version: str = Field(
        default_factory=lambda: os.getenv(
            "RUNSIFT_CONDENSER", os.getenv("CACHEFORGE_CONDENSER", "v2")
        ).lower().strip()
    )


# Backward compatibility alias
CacheForgeConfig = RunSiftConfig

# Global singleton configuration loader
_config: RunSiftConfig | None = None


def get_config() -> RunSiftConfig:
    global _config
    if _config is None:
        _load_dotenv_if_present()
        _config = RunSiftConfig()
    return _config
