#!/usr/bin/env python3
"""Single-stream comparison without sparkDash; run against an idle server.

Source/export .env first if authentication is required. Results include full
responses for review and cumulative speculation-counter deltas. Throughput
includes prefill and HTTP overhead; it is not a pure decode measurement.
"""
import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    "prose": "Write a 350-word travel guide to Lisbon.",
    "code": "Write a Python LRU cache class using OrderedDict, with get and put methods, and explain its implementation with example tests.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--require-speculation", action="store_true",
                        help="fail if a measured request drafts or accepts no tokens")
    args = parser.parse_args()
    base = "http://127.0.0.1:" + os.environ.get("PORT", "8888")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["API_KEY"]

    def metrics():
        request = urllib.request.Request(base + "/metrics", headers=headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode()
        counters = {}
        for line in raw.splitlines():
            match = re.match(r'^(vllm:spec_decode_\w+_total)(?:\{[^}]*\})? ([0-9.eE+-]+)$', line)
            if match:
                key, value = match.groups()
                counters[key] = counters.get(key, 0) + float(value)
        return counters

    def generate(prompt, max_tokens):
        body = {
            "model": os.environ.get("SERVED_MODEL_NAME", "qwen3.8-flash-next"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "top_p": 1, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(base + "/v1/chat/completions",
                                         json.dumps(body).encode(), headers)
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.load(response)
        elapsed = time.monotonic() - start
        if not result["usage"]["completion_tokens"] or not result["choices"][0]["message"].get("content"):
            raise RuntimeError("Empty generation")
        return result, elapsed

    generate("Reply with OK.", 16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as output:
        for name, prompt in PROMPTS.items():
            for repeat in range(args.repeats):
                before = metrics()
                response, elapsed = generate(prompt, 400)
                after = metrics()
                delta = {key: value - before.get(key, 0) for key, value in after.items()}
                tokens = response["usage"]["completion_tokens"]
                row = {"tag": args.tag, "prompt": name, "repeat": repeat + 1,
                       "completion_tokens": tokens, "elapsed_s": elapsed,
                       "end_to_end_tps": tokens / elapsed, "speculation": delta,
                       "response": response}
                drafts = delta.get("vllm:spec_decode_num_drafts_total", 0)
                if drafts:
                    row["accepted_tokens_per_draft"] = delta.get("vllm:spec_decode_num_accepted_tokens_total", 0) / drafts
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(json.dumps({key: value for key, value in row.items() if key != "response"}), flush=True)
                if args.require_speculation and (drafts <= 0 or delta.get("vllm:spec_decode_num_accepted_tokens_total", 0) <= 0):
                    raise RuntimeError("MTP did not draft and accept tokens for this request")


if __name__ == "__main__":
    main()
