"""Test script for context capacity, prefill/gen speeds, VRAM, and needle retrieval."""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def get_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        return int(out.strip().splitlines()[0])
    except (subprocess.SubprocessError, ValueError, OSError):
        return -1


def run_test(
    port: int,
    target_tokens: int = 100000,
    secret: str = "ORION-9428-BETA",
    output_file: str | Path | None = None,
):
    vram_start = get_vram()
    print(f"VRAM before test: {vram_start} MiB")

    # Generate prompt with needle near the start
    needle = f"\nCONFIDENTIAL ARCHIVE RECORD: The secret project operational code is {secret}.\n"
    filler_sentence = "In distributed systems, KV-cache prefix invariance guarantees stable execution traces across multiple round-trips. "

    # Check token count of filler
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"content": filler_sentence}).encode("utf-8"),
    )
    with urllib.request.urlopen(req) as resp:
        toks = json.loads(resp.read().decode("utf-8"))["tokens"]
        filler_tok_count = len(toks)

    repeats_needed = max(10, int((target_tokens - 100) / filler_tok_count))
    filler = filler_sentence * repeats_needed

    prompt = (
        f"<bos><start_of_turn>user\n"
        f"Background Documentation:\n{needle}\n{filler}\n\n"
        f"Question: What is the secret project operational code mentioned in the archive record? "
        f"State the code clearly.\n<end_of_turn>\n<start_of_turn>model\n"
    )

    # Tokenize full prompt to verify exact count
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/tokenize",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"content": prompt}).encode("utf-8"),
    )
    with urllib.request.urlopen(req) as resp:
        actual_tokens = len(json.loads(resp.read().decode("utf-8"))["tokens"])

    print(f"Prompt prepared: {actual_tokens} tokens (target ~{target_tokens})")

    # Now make the completion request with stream=false to get full timings
    payload = {
        "prompt": prompt,
        "n_predict": 32,
        "temperature": 0.0,
        "stream": False,
        "cache_prompt": False,
    }

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8"),
    )

    t0 = time.perf_counter()
    peak_vram = vram_start

    # We can poll VRAM in a separate thread or just check right after request starts
    import threading

    monitoring = True

    def vram_poller():
        nonlocal peak_vram, monitoring
        while monitoring:
            v = get_vram()
            peak_vram = max(peak_vram, v)
            time.sleep(0.5)

    poller_thread = threading.Thread(target=vram_poller, daemon=True)
    poller_thread.start()

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    finally:
        monitoring = False
        poller_thread.join(timeout=1.0)

    t1 = time.perf_counter()
    elapsed_total = t1 - t0

    content = data.get("content", "").strip()
    timings = data.get("timings", {})
    prompt_n = timings.get("prompt_n", actual_tokens)
    prompt_ms = timings.get("prompt_ms", elapsed_total * 1000)
    prompt_per_second = timings.get(
        "prompt_per_second", prompt_n / (prompt_ms / 1000.0) if prompt_ms else 0
    )
    predicted_n = timings.get("predicted_n", 0)
    predicted_ms = timings.get("predicted_ms", 0)
    predicted_per_second = timings.get(
        "predicted_per_second", predicted_n / (predicted_ms / 1000.0) if predicted_ms else 0
    )

    # TTFT is prompt_ms
    ttft_ms = prompt_ms

    needle_found = secret in content

    print("=" * 60)
    print(f"RESULTS FOR PORT {port}:")
    print(f"  Prompt tokens:       {prompt_n}")
    print(f"  VRAM after load:     {vram_start} MiB")
    print(f"  Peak VRAM:           {peak_vram} MiB")
    print(f"  TTFT:                {ttft_ms:.1f} ms ({ttft_ms / 1000.0:.2f} s)")
    print(f"  Prefill speed:       {prompt_per_second:.1f} tok/s")
    print(f"  Generation tokens:   {predicted_n}")
    print(f"  Generation speed:    {predicted_per_second:.1f} tok/s")
    print(f"  Model reply:         {content!r}")
    print(f"  Needle ({secret}) found: {needle_found}")
    print("=" * 60)

    res = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "port": port,
        "prompt_tokens": prompt_n,
        "vram_load": vram_start,
        "peak_vram": peak_vram,
        "ttft_ms": ttft_ms,
        "prefill_tok_s": prompt_per_second,
        "gen_tokens": predicted_n,
        "gen_tok_s": predicted_per_second,
        "model_reply": content,
        "needle_found": needle_found,
    }

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Results written to: {out_path}")

    return res


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8083
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    out_file = (
        sys.argv[3]
        if len(sys.argv) > 3
        else f"benchmarks/tensorroute_benchmark/results/diag_capacity_port_{port}_{tokens}tok.json"
    )
    run_test(port, tokens, output_file=out_file)
