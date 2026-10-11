"""High-throughput upstream client for Nebius Token Factory.

Routes cache-aligned requests to NVIDIA model endpoints (e.g. nemotron-3-super-120b-a12b)
with connection pooling, sticky replica routing, SSE streaming parser, and TTFT telemetry tracking.
Allowed dependencies per wiring.yaml: ["core"].
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from src.core.config import CacheForgeConfig, RunSiftConfig, get_config


class NebiusClient:
    """Async client interfacing with Nebius Token Factory inference endpoints."""

    def __init__(
        self,
        config: RunSiftConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
        prefix_ttl_seconds: float = 300.0,
    ):
        self.config = config or get_config()
        self.prefix_ttl_seconds = prefix_ttl_seconds
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            base_url=self.config.nebius_base_url.rstrip("/"),
            timeout=httpx.Timeout(120.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {self.config.nebius_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "RunSift-Gateway/0.1.0",
            },
        )
        # Tracks active canonical prefix digests dispatched to upstream GPU workers within TTL window
        self._active_prefixes: dict[str, dict[str, float | int]] = {}
        self._models_cache: dict[str, Any] | None = None
        self._models_cached_at: float = 0.0

    async def close(self) -> None:
        if self._owns_client:
            await self._http_client.aclose()

    def _evict_expired_prefixes(self, now: float) -> None:
        expired = [
            ph
            for ph, info in self._active_prefixes.items()
            if (now - float(info["last_seen"])) > self.prefix_ttl_seconds
        ]
        for ph in expired:
            self._active_prefixes.pop(ph, None)

    def is_prefix_warm(self, prefix_hash: str | None) -> bool:
        """Returns True if the canonical prefix hash is currently warm in the active window."""
        if not prefix_hash:
            return False
        now = time.time()
        self._evict_expired_prefixes(now)
        return prefix_hash in self._active_prefixes

    def _evaluate_prefix_cache(
        self,
        prefix_hash: str | None,
        prompt_tokens: int,
        upstream_cached_tokens: int = 0,
    ) -> tuple[bool, int]:
        """Determines prefix cache hit status and reused token count from upstream usage and prefix registry."""
        now = time.time()
        self._evict_expired_prefixes(now)

        if not prefix_hash:
            is_hit = upstream_cached_tokens > 0
            return is_hit, upstream_cached_tokens if is_hit else 0

        existing = self._active_prefixes.get(prefix_hash)
        if upstream_cached_tokens > 0:
            self._active_prefixes[prefix_hash] = {
                "last_seen": now,
                "prefix_tokens": prompt_tokens,
                "hits": (int(existing["hits"]) + 1) if existing else 1,
            }
            return True, upstream_cached_tokens

        # Upstream reported 0 cached tokens - register seen prefix but do not fabricate synthetic reuse
        self._active_prefixes[prefix_hash] = {
            "last_seen": now,
            "prefix_tokens": prompt_tokens,
            "hits": 0 if existing is None else int(existing["hits"]),
        }
        return False, 0

    async def list_models(self) -> dict[str, Any]:
        """Retrieves available models from Nebius Token Factory with a 60s TTL cache."""
        now = time.time()
        if self._models_cache is not None and (now - self._models_cached_at) < 60.0:
            return self._models_cache

        response = await self._http_client.get("/models")
        response.raise_for_status()
        data = response.json()
        self._models_cache = data
        self._models_cached_at = now
        return data

    async def create_chat_completion(
        self,
        payload: dict[str, Any],
        prefix_hash: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sends chat completion request to Nebius Token Factory and collects telemetry."""
        req_payload = dict(payload)
        req_headers: dict[str, str] = {}
        caller_user = req_payload.get("user")
        if caller_user:
            safe_user = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(caller_user))
            req_headers["x-session-id"] = (
                f"{safe_user}_{prefix_hash[:8]}" if prefix_hash else safe_user[:32]
            )
        elif prefix_hash:
            req_headers["x-session-id"] = prefix_hash[:16]
            req_headers["x-prefix-routing-hint"] = prefix_hash[:16]

        start_time = time.perf_counter()
        response = await self._http_client.post(
            "/chat/completions",
            json=req_payload,
            headers=req_headers,
        )
        response.raise_for_status()
        data = response.json()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Parse usage details safely (prompt_tokens_details may be None on some models)
        usage_data = data.get("usage") or {}
        prompt_tokens = int(usage_data.get("prompt_tokens") or 0)
        completion_tokens = int(usage_data.get("completion_tokens") or 0)
        if prompt_tokens == 0:
            prompt_len = sum(
                len(str(m.get("content", ""))) for m in req_payload.get("messages", [])
            )
            prompt_tokens = max(1, prompt_len // 4)

        details = usage_data.get("prompt_tokens_details") or {}
        upstream_cached = int(
            details.get("cached_tokens") or usage_data.get("prompt_cache_hit_tokens") or 0
        )

        is_cache_hit, cached_tokens = self._evaluate_prefix_cache(
            prefix_hash=prefix_hash,
            prompt_tokens=prompt_tokens,
            upstream_cached_tokens=upstream_cached,
        )

        if "usage" in data and isinstance(data["usage"], dict):
            data["usage"]["cache_hit_tokens"] = cached_tokens
            data["usage"]["cache_miss_tokens"] = max(0, prompt_tokens - cached_tokens)

        telemetry_meta = {
            "ttft_ms": elapsed_ms,
            "total_latency_ms": elapsed_ms,
            "cache_hit": is_cache_hit,
            "cached_tokens": cached_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prefix_hash": prefix_hash or "",
        }

        # Return lossless raw dictionary preserving reasoning_content, logprobs, system_fingerprint, etc.
        return data, telemetry_meta

    async def open_chat_completion_stream(
        self,
        payload: dict[str, Any],
        prefix_hash: str | None = None,
    ) -> httpx.Response:
        """Opens streaming HTTP connection and validates upstream status before body consumption."""
        req_payload = dict(payload)
        req_payload["stream"] = True
        req_payload.setdefault("stream_options", {"include_usage": True})

        req_headers: dict[str, str] = {}
        caller_user = req_payload.get("user")
        if caller_user:
            safe_user = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(caller_user))
            req_headers["x-session-id"] = (
                f"{safe_user}_{prefix_hash[:8]}" if prefix_hash else safe_user[:32]
            )
        elif prefix_hash:
            req_headers["x-session-id"] = prefix_hash[:16]
            req_headers["x-prefix-routing-hint"] = prefix_hash[:16]

        req = self._http_client.build_request(
            "POST",
            "/chat/completions",
            json=req_payload,
            headers=req_headers,
        )
        response = await self._http_client.send(req, stream=True)
        if response.is_error:
            await response.aread()
            await response.aclose()
            response.raise_for_status()
        return response

    async def consume_chat_completion_stream(
        self,
        response: httpx.Response,
        payload: dict[str, Any],
        prefix_hash: str | None = None,
        start_time: float | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """Consumes open streaming HTTP response, yielding SSE chunks and tracking final telemetry."""
        if start_time is None:
            start_time = time.perf_counter()
        req_payload = dict(payload)
        client_wants_usage = bool(
            isinstance(payload.get("stream_options"), dict)
            and payload["stream_options"].get("include_usage")
        )
        ttft_recorded = False
        ttft_ms = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        upstream_cached = 0
        chunk_tokens_count = 0
        assembled_tool_calls: dict[int, dict[str, Any]] = {}

        try:
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("data: ") and line != "data: [DONE]":
                    if not ttft_recorded:
                        ttft_ms = (time.perf_counter() - start_time) * 1000.0
                        ttft_recorded = True

                    try:
                        chunk_obj = json.loads(line[6:])
                    except json.JSONDecodeError:
                        chunk_obj = {}

                    # Track usage chunk if present
                    usage_obj = chunk_obj.get("usage")
                    if isinstance(usage_obj, dict):
                        prompt_tokens = int(usage_obj.get("prompt_tokens") or prompt_tokens)
                        completion_tokens = int(
                            usage_obj.get("completion_tokens") or completion_tokens
                        )
                        details = usage_obj.get("prompt_tokens_details") or {}
                        upstream_cached = int(
                            details.get("cached_tokens")
                            or usage_obj.get("prompt_cache_hit_tokens")
                            or upstream_cached
                        )

                    # Track streamed deltas and assemble tool_calls
                    choices = chunk_obj.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta") or {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            chunk_tokens_count += 1
                        tc_deltas = delta.get("tool_calls") or []
                        for tc in tc_deltas:
                            idx = int(tc.get("index", 0))
                            entry = assembled_tool_calls.setdefault(
                                idx,
                                {
                                    "id": tc.get("id") or f"call_{idx}",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if tc.get("id"):
                                entry["id"] = tc["id"]
                            fn_delta = tc.get("function") or {}
                            if fn_delta.get("name"):
                                entry["function"]["name"] += fn_delta["name"]
                            if fn_delta.get("arguments"):
                                entry["function"]["arguments"] += fn_delta["arguments"]

                    # If usage chunk has empty choices and client did not request include_usage, do not forward it
                    if usage_obj and not choices and not client_wants_usage:
                        continue

                meta = {
                    "ttft_ms": ttft_ms,
                    "prefix_hash": prefix_hash or "",
                    "done": False,
                }
                yield f"{line}\n\n", meta
        finally:
            await response.aclose()

        if not ttft_recorded:
            ttft_ms = (time.perf_counter() - start_time) * 1000.0

        if prompt_tokens == 0:
            prompt_len = sum(
                len(str(m.get("content", ""))) for m in req_payload.get("messages", [])
            )
            prompt_tokens = max(1, prompt_len // 4)
        if completion_tokens == 0:
            completion_tokens = max(1, chunk_tokens_count)

        is_cache_hit, cached_tokens = self._evaluate_prefix_cache(
            prefix_hash=prefix_hash,
            prompt_tokens=prompt_tokens,
            upstream_cached_tokens=upstream_cached,
        )

        final_tool_calls = [assembled_tool_calls[k] for k in sorted(assembled_tool_calls)]
        final_meta = {
            "ttft_ms": ttft_ms,
            "prefix_hash": prefix_hash or "",
            "cache_hit": is_cache_hit,
            "cached_tokens": cached_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_calls": final_tool_calls,
            "done": True,
        }
        yield "", final_meta

    async def stream_chat_completion(
        self,
        payload: dict[str, Any],
        prefix_hash: str | None = None,
    ) -> AsyncGenerator[tuple[str, dict[str, Any]], None]:
        """Streams chat completion chunks as Server-Sent Events (SSE) with usage and tool call tracking."""
        start_time = time.perf_counter()
        response = await self.open_chat_completion_stream(payload, prefix_hash)
        async for chunk, meta in self.consume_chat_completion_stream(
            response, payload, prefix_hash=prefix_hash, start_time=start_time
        ):
            yield chunk, meta
