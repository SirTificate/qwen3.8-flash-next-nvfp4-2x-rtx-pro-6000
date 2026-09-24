#!/usr/bin/env python3
"""Check Anthropic Messages API streams served through an OpenAI-compatible proxy.

Sends N streaming /v1/messages requests with prompts that make a reasoning model think (the
script sends no reasoning parameter; it relies on the model thinking by default) and validates
every server-sent event sequence against the Anthropic streaming rules:
  - a delta belongs to the block that is currently open,
  - the delta type matches the block type (thinking_delta only in thinking blocks, ...),
  - every started block is stopped before the next one starts.
This is the path Claude Code uses. A violation here is what Claude Code reports as
"Content block is not a thinking block".

Usage:
  API_KEY=... check_messages_stream.py --base-url http://localhost:<port> --model qwen3.8-flash-next -n 30
Exit code: 0 = no violations, 1 = violations or request errors, 2 = usage error.
Standard library only.
"""
import argparse
import http.client
import json
import os
import sys
import urllib.error
import urllib.request

PROMPTS = [
    "What is {a}*{b}? Think briefly, then give only the result.",
    "Write a Python function that checks whether {a} is divisible by {b}. Code only.",
    "A train travels {a} km in {b} minutes. What is its speed in km/h? Short answer.",
]
DELTA_BLOCK = {"thinking_delta": "thinking", "signature_delta": "thinking",
               "text_delta": "text", "input_json_delta": "tool_use"}


def parse_sse(lines):
    events = []
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def check_events(events):
    errors, blocks, open_idx, kinds = [], {}, None, []
    for ev in events:
        t = ev.get("type")
        if t == "content_block_start":
            idx, btype = ev["index"], ev["content_block"]["type"]
            if open_idx is not None:
                errors.append(f"block {idx} started while block {open_idx} is open")
            blocks[idx], open_idx = btype, idx
            kinds.append(btype)
        elif t == "content_block_delta":
            idx, dtype = ev["index"], ev["delta"]["type"]
            want = DELTA_BLOCK.get(dtype)
            if idx not in blocks:
                errors.append(f"{dtype} for block {idx} that was never started")
            elif want and blocks[idx] != want:
                errors.append(f"{dtype} in {blocks[idx]} block {idx}")
            if idx != open_idx:
                errors.append(f"{dtype} for block {idx}, but open block is {open_idx}")
        elif t == "content_block_stop":
            if ev["index"] != open_idx:
                errors.append(f"stop for block {ev['index']}, but open block is {open_idx}")
            open_idx = None
        elif t == "error":
            errors.append(f"error event: {ev.get('error')}")
    if open_idx is not None:
        errors.append(f"block {open_idx} never stopped")
    if not kinds:
        errors.append("no content blocks in stream")
    return kinds, errors


def stream(base_url, api_key, model, prompt, max_tokens, timeout):
    body = {"model": model, "max_tokens": max_tokens, "stream": True,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/messages", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + api_key, "x-api-key": api_key,
                 "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return parse_sse(resp)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", required=True, help="proxy base URL, without /v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", "--count", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args(argv)
    api_key = os.environ.get("API_KEY")
    if not api_key:
        print("set API_KEY in the environment")
        return 2
    bad = 0
    saw_thinking = False
    for i in range(args.count):
        prompt = PROMPTS[i % len(PROMPTS)].format(a=17 + i, b=3 + i % 7)
        try:
            kinds, errors = check_events(stream(args.base_url, api_key, args.model, prompt,
                                                args.max_tokens, args.timeout))
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException,
                KeyError, TypeError, AttributeError) as exc:
            kinds, errors = [], [f"{type(exc).__name__}: {exc}"]
        if "thinking" in kinds:
            saw_thinking = True
        bad += bool(errors)
        status = "OK  " if not errors else "FAIL"
        print(f"{i + 1:3d} {status} blocks={kinds} {'; '.join(errors)[:200]}")
    if not saw_thinking:
        print("WARNING: no stream contained a thinking block, so the reasoning/answer boundary "
              "was not exercised.")
    print(f"RESULT {args.model}: {bad}/{args.count} streams with violations or errors")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
