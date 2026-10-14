"""FastAPI OpenAI-compatible reverse proxy gateway for RunSift.

Canonicalizes prompt prefixes, decouples tool execution,
and routes cache-optimized requests to Nebius Token Factory.
Allowed dependencies per wiring.yaml: ["core", "canonicalizer", "compact", "state", "upstream"].
"""

from __future__ import annotations

import asyncio
import copy
import hmac
import importlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.canonicalizer import PrefixCanonicalizer
from src.compact import CondenserV2, ToolOutputCompactor
from src.core import (
    SUPPORTED_NVIDIA_MODELS,
    CacheForgeConfig,
    RunSiftConfig,
    _load_dotenv_if_present,
    candidate_dotenv_paths,
    get_config,
    telemetry,
)
from src.proxy import dashboard as _dashboard_mod
from src.state import StateManager
from src.upstream import NebiusClient

logger = logging.getLogger(__name__)
request_logger = logging.getLogger("runsift.requests")


def _configure_request_logger() -> None:
    """Attaches a single stderr StreamHandler to `runsift.requests` unless disabled."""
    mode = os.environ.get(
        "RUNSIFT_REQUEST_LOG", os.environ.get("CACHEFORGE_REQUEST_LOG", "")
    ).strip().lower()
    for h in list(request_logger.handlers):
        if getattr(h, "_runsift_request_handler", False) or getattr(
            h, "_cacheforge_request_handler", False
        ):
            if mode == "off":
                request_logger.removeHandler(h)
            return
    if mode == "off":
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._runsift_request_handler = True  # type: ignore[attr-defined]
    handler._cacheforge_request_handler = True  # type: ignore[attr-defined]
    request_logger.addHandler(handler)
    request_logger.setLevel(logging.INFO)


def _format_request_log_line(
    turn_id: str,
    model_name: str,
    condenser: str,
    tool_chars_before: int,
    tool_chars_after: int,
    tokens_in: int,
    cached_tokens: int,
    ttft_ms: float,
    fallback: bool = False,
    upstream_error_code: int | None = None,
) -> str:
    if upstream_error_code is not None:
        return f"{turn_id}  {model_name}  {condenser}  upstream HTTP {upstream_error_code}"
    if fallback:
        return (
            f"{turn_id}  {model_name}  {condenser}  "
            f"prompt {tokens_in:,}  cached {cached_tokens:,}  first token {round(ttft_ms)} ms  "
            "FALLBACK: condenser error, request sent unchanged"
        )
    if tool_chars_before > 0:
        pct = round(((tool_chars_after - tool_chars_before) / tool_chars_before) * 100)
        pct_str = f"{pct:+d}%" if pct != 0 else "0%"
    else:
        pct_str = "0%"
    return (
        f"{turn_id}  {model_name}  {condenser}  "
        f"tool output {tool_chars_before:,} → {tool_chars_after:,} chars ({pct_str})  "
        f"prompt {tokens_in:,}  cached {cached_tokens:,}  first token {round(ttft_ms)} ms"
    )


_DEMO_SYSTEM_CONSTITUTION = (
    "You are an autonomous Principal GPU Cluster SRE Agent operating across NVIDIA H100 "
    "SuperPOD nodes on Nebius Token Factory.\n"
    "Constitutional Operating Rules:\n"
    "1. Prefix Locality: Maintain deterministic ordering of system instructions "
    "and tool declarations to support prefix caching across turns.\n"
    "2. Async Connection Decoupling: Decouple external diagnostic I/O from active GPU batch slots.\n"
    "3. Root-Cause Synthesis: Isolate exact hardware metrics, NVLink lane health, and test "
    "verification status in 1-2 concise sentences."
)

_DEMO_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_gpu_cluster_telemetry",
            "description": "Queries live node temperature, HBM3 VRAM utilization, and NVLink bandwidth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Target GPU node identifier"},
                    "include_fabric": {
                        "type": "boolean",
                        "description": "Include NVSwitch counters",
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audit_kv_cache_pages",
            "description": "Audits vLLM / TensorRT-LLM prefix cache page allocations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pool_id": {"type": "string", "description": "Page pool identifier"},
                },
                "required": ["pool_id"],
            },
        },
    },
]


def _reject_json_constant(c: str) -> None:
    raise ValueError(f"Non-standard JSON constant '{c}' not allowed")


# MOCK(M-011): Simulated PCIe telemetry log for interactive demo UI
def _collect_live_diagnostic_log(node_id: str, turn_idx: int) -> str:
    """Generates a simulated diagnostic log for the dashboard walkthrough (MOCK M-011)."""
    header = {
        "node_id": node_id,
        "turn": turn_idx,
        "host_os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_cores": os.cpu_count() or 1,
        "radix_cache_status": "ALIGNED",
    }
    lines = [
        f"[DIAG-PASS-{i:02d}] Node {node_id} PCIe Gen5 / NVLink 4.0 lane {i:02d} integrity check ... PASSED (0 CRC errors, 900 GB/s)"
        for i in range(1, 55)
    ]
    return (
        f"TELEMETRY_HEADER: {json.dumps(header)}\n"
        + "\n".join(lines)
        + f"\nSUMMARY_STATUS: Node {node_id} turn {turn_idx} verified healthy; zero thermal throttling."
    )


def _write_local_dotenv(env_lines: list[str]) -> None:
    """Writes .env in the current working directory if writable."""
    env_file = os.path.join(os.getcwd(), ".env")
    try:
        with open(env_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(env_lines) + "\n")
    except OSError:
        pass


def _ensure_and_open_local_dotenv() -> str:
    """Creates .env from .env.example if missing and opens it in the local OS text editor."""
    env_file = os.path.join(os.getcwd(), ".env")
    if not os.path.exists(env_file):
        try:
            with open(env_file, "w", encoding="utf-8") as fh:
                fh.write("NEBIUS_API_KEY=your-token-factory-key\nRUNSIFT_CONDENSER=v2\n")
        except OSError:
            pass
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(["notepad.exe", env_file])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-t", env_file])
        else:
            subprocess.Popen(["xdg-open", env_file])
    except OSError:
        pass
    return env_file


def _capture_gateway_git_commit() -> str:
    """Captures git commit of the gateway at startup."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except (subprocess.SubprocessError, OSError, UnicodeDecodeError):
        return "unknown"


_GATEWAY_GIT_COMMIT: str = _capture_gateway_git_commit()


def create_app(
    config: RunSiftConfig | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Factory creating configured FastAPI gateway instance."""
    cfg = config or get_config()
    _configure_request_logger()

    # Initialize gateway subsystems
    canonicalizer = PrefixCanonicalizer()
    compactor = ToolOutputCompactor(max_char_threshold=cfg.max_tool_output_tokens * 4)
    condenser_v2 = CondenserV2(max_char_threshold=cfg.max_tool_output_tokens * 4)
    condenser_v2_1 = CondenserV2(
        max_char_threshold=cfg.max_tool_output_tokens * 4,
        enable_stale_rewrite=False,
    )
    state_manager = StateManager(compactor=compactor)
    upstream_client = NebiusClient(config=cfg, http_client=http_client)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await upstream_client.close()

    app = FastAPI(
        title="RunSift Gateway",
        description="Inference-aware reverse proxy eliminating the Agentic Inference Tax on Nebius & NVIDIA open models",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _verify_auth(req: Request) -> None:
        if not cfg.api_key:
            return
        auth_header = req.headers.get("authorization", "")
        x_api_key = req.headers.get("x-api-key", "")
        expected_bearer = f"Bearer {cfg.api_key}"
        valid = (
            hmac.compare_digest(auth_header, expected_bearer)
            or hmac.compare_digest(x_api_key, cfg.api_key)
            or hmac.compare_digest(auth_header, cfg.api_key)
        )
        if not valid:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Missing or invalid RunSift API Key",
            )

    def _sync_dotenv_if_needed() -> None:
        """Reloads .env from candidate paths if NEBIUS_API_KEY was saved after startup."""
        if not cfg.nebius_api_key or cfg.nebius_api_key.lower() in (
            "your-token-factory-key",
            "your-api-key",
            "your_api_key_here",
            "changeme",
        ):
            _load_dotenv_if_present(override=True)
            new_key = os.getenv("NEBIUS_API_KEY", "").strip()
            if new_key and new_key.lower() not in (
                "your-token-factory-key",
                "your-api-key",
                "your_api_key_here",
                "changeme",
            ):
                cfg.nebius_api_key = new_key
                upstream_client._http_client.headers["Authorization"] = f"Bearer {new_key}"
            new_cond = os.getenv("RUNSIFT_CONDENSER", os.getenv("CACHEFORGE_CONDENSER", "")).strip().lower()
            if new_cond in ("v1", "v2", "v2.1"):
                cfg.condenser_version = new_cond

    @app.get("/health")
    async def health_check():
        _sync_dotenv_if_needed()
        return {
            "status": "healthy",
            "version": "0.1.0",
            "git_commit": _GATEWAY_GIT_COMMIT,
            "spec": "SDCS SPEC-001 v2.0.0",
            "upstream_base_url": cfg.nebius_base_url,
            "default_model": cfg.default_model,
            "supported_models": list(SUPPORTED_NVIDIA_MODELS),
            "upstream_configured": bool(cfg.nebius_api_key),
            "dotenv_search_paths": [str(p) for p in candidate_dotenv_paths()[:3]],
            "uptime_seconds": time.time() - telemetry.start_time,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    async def get_dashboard():
        """Serves the live interactive telemetry dashboard."""
        importlib.reload(_dashboard_mod)
        return HTMLResponse(content=_dashboard_mod.render_dashboard_html())

    @app.get("/api/telemetry")
    async def get_telemetry(request: Request):
        """Structured telemetry metrics endpoint consumed by the dashboard UI and external monitors."""
        _verify_auth(request)
        _sync_dotenv_if_needed()
        active_sessions = len(state_manager._sessions)
        hit_rate = round(telemetry.cache_hit_rate * 100, 1)
        upstream_status = (
            "CONFIGURED (Nebius Token Factory)"
            if cfg.nebius_api_key
            else "UNCONFIGURED (Missing NEBIUS_API_KEY)"
        )

        return {
            "radix_cache_hit_rate": hit_rate,
            "mean_ttft_cold_ms": round(telemetry.mean_ttft_cold_ms, 1),
            "mean_ttft_warm_ms": round(telemetry.mean_ttft_warm_ms, 1),
            "ttft_reduction_pct": round(telemetry.ttft_reduction_pct, 1),
            "speedup_ratio": round(telemetry.speedup_ratio, 2),
            "total_suspended_tool_seconds": round(telemetry.total_tool_wait_seconds_suspended, 2),
            "active_sessions_count": active_sessions,
            "cumulative_prompt_tokens_original": telemetry.prompt_tokens_original,
            "cumulative_prompt_tokens": telemetry.prompt_tokens_canonical,
            "cumulative_tokens_saved": telemetry.tokens_saved_by_compaction,
            "cumulative_tool_chars_removed": telemetry.tool_chars_removed_total,
            "cached_tokens_reused": telemetry.cached_tokens_reused,
            "estimated_input_cost_avoided_usd": telemetry.estimated_input_cost_avoided_usd,
            "condenser_version": cfg.condenser_version,
            "condenser_char_threshold": cfg.max_tool_output_tokens * 4,
            "upstream_model": cfg.default_model,
            "supported_models": list(SUPPORTED_NVIDIA_MODELS),
            "upstream_base_url": cfg.nebius_base_url,
            "upstream_status": upstream_status,
            "spec_version": "SDCS v2.0.0",
            "requests_total": telemetry.requests_total,
            "cache_hits_total": telemetry.cache_hits_total,
            "uptime_seconds": round(time.time() - telemetry.start_time, 1),
            "recent_requests": telemetry.recent_requests,
        }

    @app.post("/api/model/select")
    async def select_default_model(request: Request):
        """Dynamically switches the gateway's active NVIDIA model target."""
        _verify_auth(request)
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

        requested_model = str(body.get("model", "")).strip()
        if not requested_model:
            raise HTTPException(status_code=400, detail="Model identifier is required")

        cfg.default_model = requested_model
        return {
            "status": "updated",
            "active_model": cfg.default_model,
            "supported_models": list(SUPPORTED_NVIDIA_MODELS),
        }

    @app.post("/api/config/env")
    async def configure_local_env(request: Request):
        """Loopback-only endpoint allowing local users to save .env and hot-load NEBIUS_API_KEY."""
        _verify_auth(request)
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            raise HTTPException(
                status_code=403,
                detail="Forbidden: .env configuration is restricted to localhost loopback.",
            )
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

        nebius_key = str(body.get("nebius_api_key", "")).strip()
        condenser_ver = str(body.get("condenser_version", "v2")).strip().lower()
        if condenser_ver not in ("v1", "v2", "v2.1"):
            condenser_ver = "v2"

        if nebius_key:
            cfg.nebius_api_key = nebius_key
            os.environ["NEBIUS_API_KEY"] = nebius_key
            upstream_client._http_client.headers["Authorization"] = f"Bearer {nebius_key}"

        cfg.condenser_version = condenser_ver
        os.environ["RUNSIFT_CONDENSER"] = condenser_ver
        os.environ["CACHEFORGE_CONDENSER"] = condenser_ver

        if bool(body.get("write_dotenv", True)) and nebius_key:
            env_lines = [
                f"NEBIUS_API_KEY={nebius_key}",
                f"RUNSIFT_CONDENSER={condenser_ver}",
            ]
            await asyncio.to_thread(_write_local_dotenv, env_lines)

        return {
            "status": "configured",
            "upstream_configured": bool(cfg.nebius_api_key),
            "condenser_version": cfg.condenser_version,
            "default_model": cfg.default_model,
        }

    @app.post("/api/config/open-env")
    async def open_local_env_in_editor(request: Request):
        """Loopback-only endpoint that opens the local .env file in the user's OS editor if unconfigured."""
        _verify_auth(request)
        client_host = request.client.host if request.client else ""
        if client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            raise HTTPException(
                status_code=403,
                detail="Forbidden: opening .env is restricted to localhost loopback.",
            )
        _sync_dotenv_if_needed()
        force_open = request.query_params.get("force", "").lower() in ("1", "true", "yes")
        if cfg.nebius_api_key and not force_open:
            return {
                "status": "already_configured",
                "path": os.path.join(os.getcwd(), ".env"),
            }
        env_path = await asyncio.to_thread(_ensure_and_open_local_dotenv)
        return {"status": "opened", "path": env_path}

    @app.post("/api/telemetry/reset")
    async def reset_telemetry(request: Request):
        """Resets gateway telemetry counters, active state sessions, and prefix cache window."""
        _verify_auth(request)
        telemetry.reset()
        state_manager._sessions.clear()
        upstream_client._active_prefixes.clear()
        return {"status": "reset", "requests_total": 0}

    @app.post("/api/demo/run")
    async def run_live_demo_loop(request: Request):
        """Executes a live 3-turn autonomous SRE agent loop through CacheForge against Nebius Token Factory."""
        _verify_auth(request)
        override_model = ""
        try:
            raw_bytes = await request.body()
            if raw_bytes:
                body = json.loads(raw_bytes)
                if isinstance(body, dict) and body.get("model"):
                    override_model = str(body["model"]).strip()
        except ValueError:
            override_model = ""

        session_id = f"demo_{int(time.time())}"
        model_name = override_model or cfg.default_model
        history: list[dict[str, Any]] = []
        turns_summary: list[dict[str, Any]] = []

        session = state_manager.get_or_create_session(
            session_id=session_id,
            model=model_name,
            messages=[],
            prefix_hash="",
        )

        for turn_idx in range(1, 4):
            # Volatile timestamp & session header at top of system prompt (hoisted by PrefixCanonicalizer)
            volatile_header = (
                f"Current Time: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                f"Session-ID: {session_id}\n"
            )
            sys_prompt = volatile_header + _DEMO_SYSTEM_CONSTITUTION

            # Tools passed in reversed order on even turns to exercise deterministic tool sorting
            raw_tools = list(reversed(_DEMO_TOOLS)) if turn_idx % 2 == 0 else list(_DEMO_TOOLS)

            raw_messages: list[dict[str, Any]] = [
                {"role": "system", "content": sys_prompt},
                *history,
                {
                    "role": "user",
                    "content": (
                        f"Turn {turn_idx}: Verify gpu-node-0{turn_idx} telemetry and summarize status in 1 sentence."
                    ),
                },
            ]

            # Compact any verbose tool messages in raw_messages
            demo_tool_chars_before = sum(
                len(str(m.get("content", ""))) for m in raw_messages if m.get("role") == "tool"
            )
            compacted_messages, saved_chars = compactor.compact_messages(raw_messages)
            demo_tool_chars_after = sum(
                len(str(m.get("content", "")))
                for m in compacted_messages
                if m.get("role") == "tool"
            )

            raw_payload = {
                "model": model_name,
                "messages": compacted_messages,
                "tools": raw_tools,
                "max_tokens": 32,
                "temperature": 0.0,
            }
            canonical_payload, canonical_metrics = canonicalizer.canonicalize_request(raw_payload)
            prefix_hash = canonical_metrics.prefix_hash
            session.prefix_hash = prefix_hash

            turn_id = f"req-{telemetry.requests_total + 1:02d}"
            final_meta: dict[str, Any] = {}
            try:
                async for _chunk_str, meta in upstream_client.stream_chat_completion(
                    canonical_payload, prefix_hash=prefix_hash
                ):
                    final_meta = meta
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Live demo turn {turn_idx} failed against upstream Nebius endpoint: {exc}",
                ) from exc

            # Simulate external tool suspension between turns 1->2 and 2->3
            turn_suspended_sec = 0.0
            if turn_idx < 3:
                synthetic_tc_id = f"call_diag_{turn_idx}"
                state_manager.suspend_for_tools(
                    session_id,
                    [
                        {
                            "id": synthetic_tc_id,
                            "type": "function",
                            "function": {
                                "name": "get_gpu_cluster_telemetry",
                                "arguments": json.dumps({"node_id": f"gpu-node-0{turn_idx}"}),
                            },
                        }
                    ],
                )
                t_susp = time.perf_counter()
                # MOCK(M-012): Simulated tool latency delay for demo turn suspension
                await asyncio.sleep(0.45)
                raw_tool_log = _collect_live_diagnostic_log(f"gpu-node-0{turn_idx}", turn_idx)
                state_manager.record_tool_return(session_id, synthetic_tc_id, raw_tool_log)
                state_manager.resume_continuation(session_id)
                turn_suspended_sec = time.perf_counter() - t_susp

                # Append turn messages (including verbose tool output so next turn compacts it)
                history.append(raw_messages[-1])
                history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": synthetic_tc_id,
                                "type": "function",
                                "function": {
                                    "name": "get_gpu_cluster_telemetry",
                                    "arguments": json.dumps({"node_id": f"gpu-node-0{turn_idx}"}),
                                },
                            }
                        ],
                    }
                )
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": synthetic_tc_id,
                        "name": "get_gpu_cluster_telemetry",
                        "content": raw_tool_log,
                    }
                )

            ttft_ms = float(final_meta.get("ttft_ms", 0.0))
            cache_hit = bool(final_meta.get("cache_hit", False))
            prompt_tokens = int(final_meta.get("prompt_tokens", 0))
            completion_tokens = int(final_meta.get("completion_tokens", 0))
            cached_tokens = int(final_meta.get("cached_tokens", 0))

            telemetry.record_request_telemetry(
                turn_id=turn_id,
                model=model_name,
                prefix_hash=prefix_hash,
                cache_hit=cache_hit,
                ttft_ms=ttft_ms,
                tokens_in=prompt_tokens,
                tokens_out=completion_tokens,
                cached_tokens=cached_tokens,
                suspended_seconds=turn_suspended_sec,
                compacted_saved_chars=saved_chars,
                tool_chars_before=demo_tool_chars_before,
                tool_chars_after=demo_tool_chars_after,
                condenser="v1",
            )

            turns_summary.append(
                {
                    "turn": turn_idx,
                    "turn_id": turn_id,
                    "cache_hit": cache_hit,
                    "ttft_ms": round(ttft_ms, 1),
                    "prompt_tokens": prompt_tokens,
                    "tokens_saved": saved_chars // 4,
                    "prefix_hash": prefix_hash,
                }
            )

        return {
            "status": "completed",
            "session_id": session_id,
            "model": model_name,
            "turns": turns_summary,
            "radix_cache_hit_rate": round(telemetry.cache_hit_rate * 100, 1),
            "speedup_ratio": round(telemetry.speedup_ratio, 2),
        }

    @app.get("/metrics")
    async def get_metrics():
        return {
            "requests_total": telemetry.requests_total,
            "cache_hits_total": telemetry.cache_hits_total,
            "cache_hit_rate": round(telemetry.cache_hit_rate, 4),
            "tokens_saved_by_compaction": telemetry.tokens_saved_by_compaction,
            "total_tool_wait_seconds_suspended": round(
                telemetry.total_tool_wait_seconds_suspended, 2
            ),
            "uptime_seconds": round(time.time() - telemetry.start_time, 1),
        }

    @app.get("/v1/models")
    async def list_models(request: Request):
        _verify_auth(request)
        try:
            return await upstream_client.list_models()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Upstream Nebius /models request failed: {exc.response.text}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Upstream Nebius /models request failed: {exc}"
            ) from exc

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str, request: Request):
        if request.headers.get("authorization") or request.headers.get("x-api-key"):
            _verify_auth(request)
        try:
            models_data = await upstream_client.list_models()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Upstream Nebius /models request failed: {exc.response.text}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Upstream Nebius /models request failed: {exc}"
            ) from exc

        data = models_data.get("data", []) if isinstance(models_data, dict) else []
        for m in data:
            if isinstance(m, dict) and m.get("id") == model_id:
                return m
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    @app.post("/v1/embeddings")
    async def create_embeddings(request: Request):
        _verify_auth(request)
        max_bytes = 10 * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Payload Too Large: Maximum allowed request size is 10MB",
                    )
            except ValueError:
                pass

        body_bytes = bytearray()
        async for chunk in request.stream():
            body_bytes.extend(chunk)
            if len(body_bytes) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Payload Too Large: Maximum allowed request size is 10MB",
                )

        try:
            body = json.loads(
                body_bytes.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        try:
            resp = await upstream_client._http_client.post("/embeddings", json=body)
            if resp.is_error:
                try:
                    return JSONResponse(status_code=resp.status_code, content=resp.json())
                except (json.JSONDecodeError, ValueError):
                    return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
            return resp.json()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Upstream embeddings failed: {exc}"
            ) from exc

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_runsift_session_id: str | None = Header(default=None),
        x_cacheforge_session_id: str | None = Header(default=None),
    ):
        _verify_auth(request)
        _sync_dotenv_if_needed()

        # Enforce 10MB payload size limit against memory exhaustion attacks
        max_bytes = 10 * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail="Payload Too Large: Maximum allowed request size is 10MB",
                    )
            except ValueError:
                pass

        body_bytes = bytearray()
        async for chunk in request.stream():
            body_bytes.extend(chunk)
            if len(body_bytes) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Payload Too Large: Maximum allowed request size is 10MB",
                )

        try:
            body = json.loads(
                body_bytes.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON payload: {exc}",
            ) from exc

        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        messages = body.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="'messages' field must be a list")

        for idx, m in enumerate(messages):
            if not isinstance(m, dict):
                raise HTTPException(
                    status_code=400, detail=f"messages[{idx}] must be a JSON object"
                )

        # Fall back to configured default model if omitted or None
        if not body.get("model"):
            body["model"] = cfg.default_model
        model_name = body["model"]

        # 1. State Session Tracking & Tool Return Resolution (before full-message compaction)
        session = None
        eff_session_id = x_runsift_session_id or x_cacheforge_session_id
        if not eff_session_id:
            user_tag = body.get("user") if isinstance(body, dict) else None
            if isinstance(user_tag, str) and user_tag.strip():
                sanitized_tag = re.sub(r"[^a-zA-Z0-9_\-]", "_", user_tag.strip())
                eff_session_id = f"sess_{sanitized_tag}"
            else:
                # Prevent deterministic collision on generic first messages
                eff_session_id = f"sess_{uuid.uuid4().hex[:12]}"
        else:
            eff_session_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", eff_session_id)

        if cfg.enable_tool_suspension and eff_session_id:
            session = state_manager.get_or_create_session(
                session_id=eff_session_id,
                model=model_name,
                messages=body.get("messages", []),
                prefix_hash="",
            )

        if session and "messages" in body:
            for m in body["messages"]:
                if m.get("role") == "tool":
                    t_id = m.get("tool_call_id", "")
                    if t_id and t_id in session.pending_tool_calls:
                        state_manager.record_tool_return(
                            session.session_id, t_id, str(m.get("content", ""))
                        )
            if session.is_all_tools_resolved() and session.pending_tool_calls:
                state_manager.resume_continuation(session.session_id)

        # 2. Negative Memory Compaction on incoming tool returns & 3. Deterministic Prefix Canonicalization
        original_body = copy.deepcopy(body)
        fallback_used = False
        saved_chars = 0
        tool_chars_before = 0
        tool_chars_after = 0
        prefix_hash = ""
        env_condenser = os.getenv(
            "RUNSIFT_CONDENSER", os.getenv("CACHEFORGE_CONDENSER", "")
        ).lower().strip()
        active_condenser = (
            env_condenser if env_condenser in ("v1", "v2", "v2.1") else cfg.condenser_version
        )
        if active_condenser not in ("v1", "v2", "v2.1"):
            active_condenser = "v2"

        disable_compaction = (
            request.headers.get("x-runsift-disable-compaction") == "true"
            or request.headers.get("x-cacheforge-disable-compaction") == "true"
            or request.headers.get("x-runsift-ablation") in ("canonicalize-only", "none")
            or request.headers.get("x-cacheforge-ablation") in ("canonicalize-only", "none")
        )
        disable_canonicalizer = (
            request.headers.get("x-runsift-disable-canonicalizer") == "true"
            or request.headers.get("x-cacheforge-disable-canonicalizer") == "true"
            or request.headers.get("x-runsift-ablation") in ("compact-only", "none")
            or request.headers.get("x-cacheforge-ablation") in ("compact-only", "none")
        )
        try:
            if "messages" in body:
                if active_condenser in ("v2", "v2.1"):
                    target_condenser = (
                        condenser_v2_1 if active_condenser == "v2.1" else condenser_v2
                    )
                    compacted_messages, v2_saved, tool_chars_before, v2_after = (
                        target_condenser.compact_messages(body["messages"])
                    )
                    if not disable_compaction and cfg.enable_negative_compaction:
                        body["messages"] = compacted_messages
                        saved_chars = v2_saved
                        tool_chars_after = v2_after
                    else:
                        tool_chars_after = tool_chars_before
                else:
                    # v1 path
                    _, _, tool_chars_before, _ = condenser_v2.compact_messages(body["messages"])
                    if not disable_compaction and cfg.enable_negative_compaction:
                        compacted_messages, saved_chars = compactor.compact_messages(
                            body["messages"]
                        )
                        body["messages"] = compacted_messages
                        tool_chars_after = max(0, tool_chars_before - saved_chars)
                    else:
                        tool_chars_after = tool_chars_before

            if not disable_canonicalizer and cfg.enable_canonicalizer:
                canonical_payload, canonical_metrics = canonicalizer.canonicalize_request(body)
                body = canonical_payload
                prefix_hash = canonical_metrics.prefix_hash
                if session:
                    session.prefix_hash = prefix_hash
        except (ValueError, TypeError, RecursionError):
            logger.exception(
                "RunSift condensing/canonicalizing failed; failing open with original request"
            )
            body = original_body
            fallback_used = True
            saved_chars = 0
            tool_chars_after = tool_chars_before
            prefix_hash = ""

        def _format_upstream_http_error(exc: httpx.HTTPStatusError, req_id: str) -> JSONResponse:
            status_code = exc.response.status_code
            request_logger.info(
                _format_request_log_line(
                    turn_id=req_id,
                    model_name=model_name,
                    condenser=active_condenser,
                    tool_chars_before=tool_chars_before,
                    tool_chars_after=tool_chars_after,
                    tokens_in=0,
                    cached_tokens=0,
                    ttft_ms=0.0,
                    fallback=fallback_used,
                    upstream_error_code=status_code,
                )
            )
            status_hint = ""
            if status_code == 402:
                status_hint = "Token Factory payment/quota required (HTTP 402): "
            elif status_code == 429:
                status_hint = "Token Factory rate limit exceeded (HTTP 429): "
            try:
                err_json = exc.response.json()
                if isinstance(err_json, dict) and status_hint:
                    err_obj = err_json.get("error")
                    if isinstance(err_obj, dict):
                        msg = str(err_obj.get("message") or exc.response.text)
                        if status_hint.strip() not in msg:
                            err_obj["message"] = f"{status_hint}{msg}"
                    elif isinstance(err_obj, str):
                        if status_hint.strip() not in err_obj:
                            err_json["error"] = f"{status_hint}{err_obj}"
                    else:
                        err_json.setdefault("message", f"{status_hint}{exc.response.text}")
            except (json.JSONDecodeError, ValueError):
                err_json = {
                    "error": {
                        "message": f"{status_hint}{exc.response.text}",
                        "type": "upstream_error",
                        "code": status_code,
                    }
                }
            return JSONResponse(status_code=status_code, content=err_json)

        is_streaming = bool(body.get("stream", False))
        turn_id = telemetry.next_turn_id()

        # 4. Stream Handling
        if is_streaming:
            warm_before_stream = upstream_client.is_prefix_warm(prefix_hash)
            start_time = time.perf_counter()
            try:
                upstream_resp = await upstream_client.open_chat_completion_stream(
                    body, prefix_hash=prefix_hash
                )
            except httpx.HTTPStatusError as exc:
                return _format_upstream_http_error(exc, turn_id)
            except httpx.ConnectError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Bad Gateway: Unable to connect to upstream inference server: {exc}",
                ) from exc
            except httpx.TimeoutException as exc:
                raise HTTPException(
                    status_code=504,
                    detail=f"Gateway Timeout: Upstream inference server timed out: {exc}",
                ) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Upstream Nebius streaming connection failed: {exc}",
                ) from exc

            async def event_generator():
                async for chunk_str, meta in upstream_client.consume_chat_completion_stream(
                    upstream_resp, body, prefix_hash=prefix_hash, start_time=start_time
                ):
                    if meta.get("done"):
                        ttft_ms = float(meta.get("ttft_ms", 0.0))
                        cache_hit = bool(meta.get("cache_hit", False))
                        suspended_sec = session.total_suspended_seconds if session else 0.0
                        s_tokens_in = int(meta.get("prompt_tokens", 0))
                        s_tokens_out = int(meta.get("completion_tokens", 0))
                        s_cached = int(meta.get("cached_tokens", 0))
                        telemetry.record_request_telemetry(
                            turn_id=turn_id,
                            model=model_name,
                            prefix_hash=prefix_hash,
                            cache_hit=cache_hit,
                            ttft_ms=ttft_ms,
                            tokens_in=s_tokens_in,
                            tokens_out=s_tokens_out,
                            cached_tokens=s_cached,
                            suspended_seconds=suspended_sec,
                            compacted_saved_chars=saved_chars,
                            tool_chars_before=tool_chars_before,
                            tool_chars_after=tool_chars_after,
                            condenser=active_condenser,
                            fallback=fallback_used,
                        )
                        request_logger.info(
                            _format_request_log_line(
                                turn_id=turn_id,
                                model_name=model_name,
                                condenser=active_condenser,
                                tool_chars_before=tool_chars_before,
                                tool_chars_after=tool_chars_after,
                                tokens_in=s_tokens_in,
                                cached_tokens=s_cached,
                                ttft_ms=ttft_ms,
                                fallback=fallback_used,
                            )
                        )
                        streamed_tool_calls = meta.get("tool_calls") or []
                        if streamed_tool_calls and cfg.enable_tool_suspension and session:
                            state_manager.suspend_for_tools(session.session_id, streamed_tool_calls)
                    elif chunk_str:
                        yield chunk_str

            tokens_compacted = saved_chars // 4
            stream_headers = {
                "x-runsift-prefix-hash": prefix_hash,
                "x-runsift-model": model_name,
                "x-runsift-prefix-warm": str(warm_before_stream).lower(),
                "x-runsift-tokens-compacted": str(tokens_compacted),
                "x-runsift-condenser": active_condenser,
                "x-runsift-tool-chars-before": str(tool_chars_before),
                "x-runsift-tool-chars-after": str(tool_chars_after),
                "x-runsift-session-id": eff_session_id,
                # Backward compatibility aliases
                "x-cacheforge-prefix-hash": prefix_hash,
                "x-cacheforge-model": model_name,
                "x-cacheforge-prefix-warm": str(warm_before_stream).lower(),
                "x-cacheforge-tokens-compacted": str(tokens_compacted),
                "x-cacheforge-condenser": active_condenser,
                "x-cacheforge-tool-chars-before": str(tool_chars_before),
                "x-cacheforge-tool-chars-after": str(tool_chars_after),
                "x-cacheforge-session-id": eff_session_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            if fallback_used:
                stream_headers["x-runsift-fallback"] = "true"
                stream_headers["x-cacheforge-fallback"] = "true"
            if warm_before_stream and telemetry.speedup_ratio > 1.0:
                # Session cache-miss-to-hit TTFT ratio (cold mean / warm mean in this session)
                stream_headers["x-runsift-ttft-speedup"] = f"{telemetry.speedup_ratio:.1f}x"
                stream_headers["x-runsift-ttft-miss-to-hit-ratio"] = (
                    f"{telemetry.speedup_ratio:.1f}x"
                )
                stream_headers["x-cacheforge-ttft-speedup"] = f"{telemetry.speedup_ratio:.1f}x"
                stream_headers["x-cacheforge-ttft-miss-to-hit-ratio"] = (
                    f"{telemetry.speedup_ratio:.1f}x"
                )

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers=stream_headers,
            )

        # 5. Non-Streaming Dispatch
        try:
            resp_dict, meta = await upstream_client.create_chat_completion(
                body, prefix_hash=prefix_hash
            )
        except httpx.HTTPStatusError as exc:
            return _format_upstream_http_error(exc, turn_id)
        except httpx.ConnectError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Bad Gateway: Unable to connect to upstream inference server: {exc}",
            ) from exc
        except httpx.TimeoutException as exc:
            raise HTTPException(
                status_code=504,
                detail=f"Gateway Timeout: Upstream inference server timed out: {exc}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Upstream Nebius inference request failed: {exc}"
            ) from exc

        ttft_ms = float(meta.get("ttft_ms", 0.0))
        cache_hit = bool(meta.get("cache_hit", False))
        cached_tokens = int(meta.get("cached_tokens", 0))
        suspended_sec = session.total_suspended_seconds if session else 0.0

        usage_dict = resp_dict.get("usage") or {}
        tokens_in = int(usage_dict.get("prompt_tokens") or 0)
        tokens_out = int(usage_dict.get("completion_tokens") or 0)

        # Record into gateway telemetry ledger (single source of truth)
        telemetry.record_request_telemetry(
            turn_id=turn_id,
            model=model_name,
            prefix_hash=prefix_hash,
            cache_hit=cache_hit,
            ttft_ms=ttft_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens=cached_tokens,
            suspended_seconds=suspended_sec,
            compacted_saved_chars=saved_chars,
            tool_chars_before=tool_chars_before,
            tool_chars_after=tool_chars_after,
            condenser=active_condenser,
            fallback=fallback_used,
        )
        request_logger.info(
            _format_request_log_line(
                turn_id=turn_id,
                model_name=model_name,
                condenser=active_condenser,
                tool_chars_before=tool_chars_before,
                tool_chars_after=tool_chars_after,
                tokens_in=tokens_in,
                cached_tokens=cached_tokens,
                ttft_ms=ttft_ms,
                fallback=fallback_used,
            )
        )

        # 6. Check if assistant returned tool calls requiring async state suspension
        choices = resp_dict.get("choices", [])
        if choices and cfg.enable_tool_suspension and session:
            msg = choices[0].get("message", {})
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                state_manager.suspend_for_tools(session.session_id, tool_calls)

        tokens_compacted = saved_chars // 4
        prefill_avoided = (
            cached_tokens if cached_tokens > 0 else (tokens_compacted if cache_hit else 0)
        )
        cold_ref = telemetry.mean_ttft_cold_ms
        if cold_ref <= 0:
            speedup_str = "n/a"
        elif cache_hit and ttft_ms > 0:
            speedup_str = f"{max(1.0, cold_ref / max(1.0, ttft_ms)):.1f}x"
        else:
            speedup_str = "1.0x"

        headers = {
            "x-runsift-prefix-hash": prefix_hash,
            "x-runsift-model": model_name,
            "x-runsift-cache-hit": str(cache_hit).lower(),
            "x-runsift-kv-hit": str(cache_hit).lower(),
            "x-runsift-ttft-ms": f"{ttft_ms:.2f}",
            "x-runsift-ttft-speedup": speedup_str,
            "x-runsift-ttft-miss-to-hit-ratio": speedup_str,
            "x-runsift-prefill-tokens-avoided": str(prefill_avoided),
            "x-runsift-tokens-compacted": str(tokens_compacted),
            "x-runsift-condenser": active_condenser,
            "x-runsift-tool-chars-before": str(tool_chars_before),
            "x-runsift-tool-chars-after": str(tool_chars_after),
            "x-runsift-session-id": eff_session_id,
            # Backward compatibility aliases
            "x-cacheforge-prefix-hash": prefix_hash,
            "x-cacheforge-model": model_name,
            "x-cacheforge-cache-hit": str(cache_hit).lower(),
            "x-cacheforge-kv-hit": str(cache_hit).lower(),
            "x-cacheforge-ttft-ms": f"{ttft_ms:.2f}",
            "x-cacheforge-ttft-speedup": speedup_str,
            "x-cacheforge-ttft-miss-to-hit-ratio": speedup_str,
            "x-cacheforge-prefill-tokens-avoided": str(prefill_avoided),
            "x-cacheforge-tokens-compacted": str(tokens_compacted),
            "x-cacheforge-condenser": active_condenser,
            "x-cacheforge-tool-chars-before": str(tool_chars_before),
            "x-cacheforge-tool-chars-after": str(tool_chars_after),
            "x-cacheforge-session-id": eff_session_id,
        }
        if fallback_used:
            headers["x-runsift-fallback"] = "true"
            headers["x-cacheforge-fallback"] = "true"

        return JSONResponse(content=resp_dict, headers=headers)

    return app


# Module-level ASGI app entrypoint
app = create_app()
