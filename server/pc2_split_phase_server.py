#!/usr/bin/env python3
"""Split-Phase `llama.cpp` (`llama-server`) Gateway for Xiaofeiyu (12B / 26B-A4B MoE / E4B).

Architecture:
- Spawns or connects to `llama-server` (`-ngl 99 -fa on --cache-ram 0 --special --lora-init-without-apply`)
  with strictly 5 thin upper-layer non-QKV GGUF LoRA cartridges preloaded at scale 0.0:
    0: `s0_xiaofeiyu_backbone.gguf`
    1: `c1_neko_clingy.gguf`
    2: `c2_older_tease.gguf`
    3: `c3_brat_mesugaki.gguf`
    4: `c4_fragile_porcelain.gguf`
- Runs two-phase inference per turn:
    * Phase A (100% pure base Gemma 4 Brain — all 5 LoRAs forced to explicit `scale: 0.0`):
      Natural `<|channel>thought` reasoning, isolated `calc_python` execution, live-streamed
      `send_code_or_doc` fenced blocks, and external OpenAI tool calls.
    * Phase B (5-Slider Voice Core active exclusively inside `send_message{bubbles:[<|"|>...<|"|>]}`):
      Streams spoken IM bubbles live with logit-bias protocol/newline guards.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
from datetime import datetime, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterator

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from gemma4_bridge import (  # noqa: E402
    CHANNEL_CLOSE_ID,
    CHANNEL_CLOSE_STR,
    CHANNEL_OPEN_ID,
    DEFAULT_WEIGHTS,
    EOS_ID,
    MULTIMODAL_IDS,
    NEWLINE_ID,
    QUOTE_DELIM_ID,
    QUOTE_STR,
    THINK_FLAG_ID,
    TOOL_CALL_CLOSE_ID,
    TOOL_CALL_CLOSE_STR,
    TOOL_CALL_OPEN_ID,
    TOOL_CALL_OPEN_STR,
    TOOL_RESP_CLOSE_ID,
    TOOL_RESP_OPEN_ID,
    TOOL_RESP_OPEN_STR,
    TURN_END_ID,
    TURN_END_STR,
    TURN_OPEN_ID,
    BridgeStreamStep,
    ExternalToolRegistry,
    OpenAIToolCallOut,
    StreamStep,
    build_external_tool_registry,
    make_chunk_payload,
    make_completion_payload,
    make_error_payload,
    make_openai_tool_chunk_delta,
    make_openai_tool_completion_payload,
    parse_gemma4_tool_calls,
    partition_pc2_and_external_calls,
    render_gemma4_v4_prompt_with_tools,
    run_calc_python_sandbox,
)

FIVE_CARTRIDGE_FILENAMES: tuple[str, ...] = (
    "s0_xiaofeiyu_backbone.gguf",
    "c1_neko_clingy.gguf",
    "c2_older_tease.gguf",
    "c3_brat_mesugaki.gguf",
    "c4_fragile_porcelain.gguf",
)

FLAVOR_PRESETS: dict[str, tuple[float, float, float, float, float]] = {
    "pure": (1.0, 0.0, 0.0, 0.0, 0.0),
    "neko": (1.0, 0.85, 0.0, 0.0, 0.0),
    "tease": (1.0, 0.0, 0.85, 0.0, 0.0),
    "brat": (1.0, 0.0, 0.0, 0.85, 0.0),
    "fragile": (1.0, 0.0, 0.0, 0.0, 0.85),
}

MODEL_FAMILY_DEFAULTS: dict[str, str] = {
    "12b": "google/gemma-4-12B-it",
    "26b-a4b": "google/gemma-4-26B-A4B-it",
    "e4b": "google/gemma-4-E4B-it",
}

SEND_MSG_STOP_VARIANTS = (
    'call:send_message{bubbles:[<|"|>',
    'call:send_message{bubbles: [<|"|>',
)
PHASE_A_STOPS = [*SEND_MSG_STOP_VARIANTS, TOOL_RESP_OPEN_STR, TURN_END_STR]
PHASE_B_STOPS = ["]}<tool_call|>", "] }<tool_call|>", TOOL_CALL_CLOSE_STR, TOOL_RESP_OPEN_STR, TURN_END_STR]

CODE_CONTENT_OPEN_RE = re.compile(
    r'<\|tool_call>call:send_code_or_doc\{(?:language:<\|"\|>([a-zA-Z0-9_+#.-]*)<\|"\|>,\s*)?content:<\|"\|>(.*)$',
    re.S,
)

PUBLIC_SAFE_MAX_BODY_BYTES = 256 * 1024
PUBLIC_SAFE_MAX_MESSAGES = 24
PUBLIC_SAFE_MAX_MSG_CHARS = 4000
PUBLIC_SAFE_MIN_MAX_TOKENS = 64
PUBLIC_SAFE_MAX_MAX_TOKENS = 1280


class SlotPoolExhaustedError(RuntimeError):
    """Raised when all parallel inference slots remain busy past `slot_acquire_timeout`."""


def load_fast_tokenizer(tokenizer_path: Path | None = None, model_family: str = "12b"):
    from tokenizers import Tokenizer

    if tokenizer_path is not None and Path(tokenizer_path).is_file():
        return Tokenizer.from_file(str(tokenizer_path))

    hf_repo = MODEL_FAMILY_DEFAULTS.get(model_family, "google/gemma-4-12B-it")
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=hf_repo, filename="tokenizer.json")
    return Tokenizer.from_file(str(downloaded))


def parse_model_weights(model_field: str | None) -> tuple[str, tuple[float, float, float, float, float]]:
    """Parse strictly 5-dim `model` tag (`xiaofeiyu-v4|w0,w1,w2,w3,w4` or `:flavor` preset)."""
    raw = (model_field or "").strip()
    if not raw:
        return "xiaofeiyu-v4|1.000,0.000,0.000,0.000,0.000", DEFAULT_WEIGHTS

    if "|" in raw:
        prefix, weight_spec = raw.split("|", 1)
        parts = [p.strip() for p in weight_spec.split(",")]
        if len(parts) != 5:
            raise ValueError(f'Expected exactly 5 comma-separated weights after "|", got {len(parts)} in {raw!r}')
        parsed: list[float] = []
        for idx, item in enumerate(parts):
            val = float(item)
            if not math.isfinite(val) or val < 0.0 or val > 2.0:
                raise ValueError(f"Slider weight w{idx}={val} out of safe range [0.0, 2.0]")
            parsed.append(round(val, 4))
        w_tuple = (parsed[0], parsed[1], parsed[2], parsed[3], parsed[4])
        clean_prefix = prefix.strip() or "xiaofeiyu-v4"
        return f"{clean_prefix}|{','.join(f'{w:.3f}' for w in w_tuple)}", w_tuple

    if ":" in raw:
        prefix, flavor = raw.split(":", 1)
        f_key = flavor.strip()
        if f_key in FLAVOR_PRESETS:
            w_tuple = FLAVOR_PRESETS[f_key]
            clean_prefix = prefix.strip() or "xiaofeiyu-v4"
            return f"{clean_prefix}|{','.join(f'{w:.3f}' for w in w_tuple)}", w_tuple

    return f"{raw}|1.000,0.000,0.000,0.000,0.000", DEFAULT_WEIGHTS


def _hold_partial_tag_suffix(text: str, tag: str = QUOTE_STR) -> tuple[str, str]:
    max_k = min(len(text), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if tag.startswith(text[-k:]):
            return text[:-k], text[-k:]
    return text, ""


class PhaseBBubbleStreamParser:
    """Incremental `<|"|>...<|"|>` bubble streamer starting right after `bubbles:[<|"|>`."""

    def __init__(self, initial_visible_blocks: int = 0) -> None:
        self.in_quote = True
        self.pending = ""
        self.bubble_started = False
        self.visible_blocks_emitted = int(initial_visible_blocks)
        self.bubbles_emitted = 0
        self._cur_bubble_parts: list[str] = []
        self.completed_bubbles: list[str] = []

    def _close_current_bubble(self) -> None:
        if self.bubble_started:
            self.visible_blocks_emitted += 1
            self.bubbles_emitted += 1
            text_b = "".join(self._cur_bubble_parts).strip()
            if text_b:
                self.completed_bubbles.append(text_b)
        self._cur_bubble_parts = []
        self.in_quote = False
        self.bubble_started = False

    def feed(self, chunk: str) -> list[str]:
        if not chunk:
            return []
        cleaned = (
            chunk.replace("<|channel>", "")
            .replace("<channel|>", "")
            .replace("<|turn>", "")
            .replace("<turn|>", "")
            .replace("\r", "")
            .replace("\n", "")
        )
        buf = self.pending + cleaned
        self.pending = ""
        out_deltas: list[str] = []

        while QUOTE_STR in buf:
            head, buf = buf.split(QUOTE_STR, 1)
            if self.in_quote:
                if head:
                    if not self.bubble_started:
                        lead = head.lstrip()
                        if lead:
                            prefix = "\n" if self.visible_blocks_emitted > 0 else ""
                            out_deltas.append(prefix + lead)
                            self._cur_bubble_parts.append(lead)
                            self.bubble_started = True
                    else:
                        out_deltas.append(head)
                        self._cur_bubble_parts.append(head)
                self._close_current_bubble()
            else:
                self.in_quote = True
                self.bubble_started = False
                self._cur_bubble_parts = []

        if self.in_quote:
            safe, held = _hold_partial_tag_suffix(buf, QUOTE_STR)
            self.pending = held
            if safe:
                if not self.bubble_started:
                    lead = safe.lstrip()
                    if lead:
                        prefix = "\n" if self.visible_blocks_emitted > 0 else ""
                        out_deltas.append(prefix + lead)
                        self._cur_bubble_parts.append(lead)
                        self.bubble_started = True
                else:
                    out_deltas.append(safe)
                    self._cur_bubble_parts.append(safe)
        else:
            _, held = _hold_partial_tag_suffix(buf, QUOTE_STR)
            self.pending = held

        return out_deltas

    def finalize(self) -> list[str]:
        out: list[str] = []
        if self.in_quote and self.pending:
            rem = self.pending.replace("<|", "").replace("|>", "").strip()
            if rem:
                if not self.bubble_started:
                    prefix = "\n" if self.visible_blocks_emitted > 0 else ""
                    out.append(prefix + rem)
                    self._cur_bubble_parts.append(rem)
                    self.bubble_started = True
                else:
                    out.append(rem)
                    self._cur_bubble_parts.append(rem)
        self.pending = ""
        if self.in_quote and self.bubble_started:
            self._close_current_bubble()
        return out


@dataclass
class LlamaCompletionResult:
    content: str
    stopping_word: str
    stop_type: str
    tokens_predicted: int
    prompt_n: int
    prompt_ms: float
    predicted_ms: float
    predicted_per_second: float


class LlamaCppHTTPClient:
    """Minimal zero-dependency HTTP/SSE client for local `llama-server`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8082, timeout_sec: float = 600.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_sec = float(timeout_sec)

    def get_json(self, path: str, timeout: float = 5.0) -> tuple[int, dict | list]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request("GET", path, headers={"Connection": "close"})
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip() else {})
        finally:
            conn.close()

    def complete_sync(self, payload: dict) -> LlamaCompletionResult:
        req_body = json.dumps({**payload, "stream": False}, ensure_ascii=False).encode("utf-8")
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_sec)
        try:
            conn.request(
                "POST",
                "/completion",
                body=req_body,
                headers={"Content-Type": "application/json", "Connection": "close"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", errors="replace")
            if resp.status != 200:
                raise RuntimeError(f"llama-server /completion returned {resp.status}: {raw[:400]}")
            data = json.loads(raw)
        finally:
            conn.close()

        timings = data.get("timings") or {}
        return LlamaCompletionResult(
            content=str(data.get("content") or ""),
            stopping_word=str(data.get("stopping_word") or ""),
            stop_type=str(data.get("stop_type") or ""),
            tokens_predicted=int(data.get("tokens_predicted") or timings.get("predicted_n") or 0),
            prompt_n=int(timings.get("prompt_n") or 0),
            prompt_ms=float(timings.get("prompt_ms") or 0.0),
            predicted_ms=float(timings.get("predicted_ms") or 0.0),
            predicted_per_second=float(timings.get("predicted_per_second") or 0.0),
        )

    def complete_stream(self, payload: dict) -> Iterator[tuple[str, LlamaCompletionResult | None]]:
        req_body = json.dumps({**payload, "stream": True}, ensure_ascii=False).encode("utf-8")
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_sec)
        try:
            conn.request(
                "POST",
                "/completion",
                body=req_body,
                headers={"Content-Type": "application/json", "Connection": "close"},
            )
            resp = conn.getresponse()
            if resp.status != 200:
                err_txt = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"llama-server streaming /completion returned {resp.status}: {err_txt[:400]}")

            pieces: list[str] = []
            final_res: LlamaCompletionResult | None = None
            while True:
                raw_line = resp.fp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                obj = json.loads(data_str)
                piece = str(obj.get("content") or "")
                if piece:
                    pieces.append(piece)
                    yield piece, None
                if obj.get("stop"):
                    timings = obj.get("timings") or {}
                    final_res = LlamaCompletionResult(
                        content="".join(pieces),
                        stopping_word=str(obj.get("stopping_word") or ""),
                        stop_type=str(obj.get("stop_type") or ""),
                        tokens_predicted=int(obj.get("tokens_predicted") or timings.get("predicted_n") or 0),
                        prompt_n=int(timings.get("prompt_n") or 0),
                        prompt_ms=float(timings.get("prompt_ms") or 0.0),
                        predicted_ms=float(timings.get("predicted_ms") or 0.0),
                        predicted_per_second=float(timings.get("predicted_per_second") or 0.0),
                    )
                    yield "", final_res
                    break

            if final_res is None:
                yield "", LlamaCompletionResult(
                    content="".join(pieces),
                    stopping_word="",
                    stop_type="eof",
                    tokens_predicted=0,
                    prompt_n=0,
                    prompt_ms=0.0,
                    predicted_ms=0.0,
                    predicted_per_second=0.0,
                )
        finally:
            conn.close()


class _SlotLease:
    def __init__(self, engine: LlamaCppPC2Engine, slot_id: int) -> None:
        self.engine = engine
        self.slot_id = int(slot_id)
        self._lock = threading.Lock()
        self._released = False

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        with self.engine._counter_lock:
            self.engine._running_count = max(0, self.engine._running_count - 1)
        self.engine._slot_pool.put_nowait(self.slot_id)


class _ManagedStreamIterator(Iterator[StreamStep | BridgeStreamStep]):
    def __init__(
        self,
        gen: Iterator[StreamStep | BridgeStreamStep],
        lease: _SlotLease,
        telemetry: dict[str, Any],
        *,
        sanitized_messages: list[dict[str, Any]],
        effective_weights: tuple[float, float, float, float, float],
        effective_max_tokens: int,
    ) -> None:
        self._gen = gen
        self.lease = lease
        self.slot_id = lease.slot_id
        self.telemetry = telemetry
        self.sanitized_messages = sanitized_messages
        self.effective_weights = effective_weights
        self.effective_max_tokens = effective_max_tokens

    def __iter__(self) -> _ManagedStreamIterator:
        return self

    def __next__(self) -> StreamStep | BridgeStreamStep:
        try:
            return next(self._gen)
        except StopIteration:
            self.lease.release()
            raise
        except BaseException:
            self.lease.release()
            raise

    def close(self) -> None:
        try:
            close_fn = getattr(self._gen, "close", None)
            if callable(close_fn):
                close_fn()
        finally:
            self.lease.release()

    def __del__(self) -> None:
        self.lease.release()


def sanitize_public_safe_messages(
    messages: list[dict],
    *,
    max_messages: int = PUBLIC_SAFE_MAX_MESSAGES,
    max_chars_per_msg: int = PUBLIC_SAFE_MAX_MSG_CHARS,
) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    sys_parts: list[str] = []
    conv_msgs: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            raise ValueError("Each message must be a JSON object")
        role = str(m.get("role") or "user").strip().lower()
        if role in ("tool", "function"):
            continue
        if role not in ("system", "developer", "user", "assistant", "model"):
            raise ValueError(f"Unsupported message role {role!r} in public-safe mode")
        raw_c = m.get("content")
        txt = raw_c if isinstance(raw_c, str) else ""
        if len(txt) > max_chars_per_msg:
            raise ValueError(f"Message content length ({len(txt)}) exceeds {max_chars_per_msg} chars limit")
        if role in ("system", "developer"):
            if txt.strip():
                sys_parts.append(txt.strip())
        else:
            conv_msgs.append({"role": "assistant" if role == "model" else role, "content": txt})

    sys_msg = {"role": "system", "content": "\n".join(sys_parts)[:max_chars_per_msg]} if sys_parts else None
    budget = (max_messages - 1) if sys_msg is not None else max_messages
    if len(conv_msgs) > budget:
        conv_msgs = conv_msgs[-budget:]
        while conv_msgs and conv_msgs[0]["role"] == "assistant":
            conv_msgs.pop(0)
    if not conv_msgs or not any(cm["role"] == "user" for cm in conv_msgs):
        raise ValueError("Conversation must contain at least one user message")
    return [sys_msg, *conv_msgs] if sys_msg is not None else conv_msgs


class LlamaCppPC2Engine:
    """Thread-safe Multi-Slot Split-Phase Engine backed by `llama-server` (strictly 5 LoRA slots)."""

    def __init__(
        self,
        llama_client: LlamaCppHTTPClient,
        tokenizer,
        *,
        n_parallel: int = 1,
        slot_acquire_timeout: float = 60.0,
        public_safe: bool = False,
        ctx_size: int = 16384,
        min_thought_tokens: int = 12,
        max_thought_tokens: int = 2048,
        max_thought_and_tool_tokens: int = 4096,
        max_tool_hops: int = 4,
        lora_slot_count: int = 5,
    ) -> None:
        if int(n_parallel) < 1:
            raise ValueError("n_parallel must be >= 1")
        if int(lora_slot_count) != 5:
            raise ValueError(f"Exactly 5 LoRA slots supported (s0 + c1..c4), got {lora_slot_count}")

        self.llama = llama_client
        self.tok = tokenizer
        self.n_parallel = int(n_parallel)
        self.slot_acquire_timeout = float(slot_acquire_timeout)
        self.public_safe = bool(public_safe)
        self.ctx_size = int(ctx_size)
        self.slot_ctx_size = max(2048, self.ctx_size // self.n_parallel)
        self.min_thought_tokens = int(min_thought_tokens)
        self.max_thought_tokens = int(max_thought_tokens)
        self.max_thought_and_tool_tokens = int(max_thought_and_tool_tokens)
        self.max_tool_hops = int(max_tool_hops)
        self.lora_slot_count = 5
        self._zero_lora_payload: list[dict] = [{"id": i, "scale": 0.0} for i in range(5)]

        self._slot_pool: queue.Queue[int] = queue.Queue(maxsize=self.n_parallel)
        for sid in range(self.n_parallel):
            self._slot_pool.put_nowait(sid)

        self._counter_lock = threading.Lock()
        self._waiting_count = 0
        self._running_count = 0
        self.ready = True
        self.last_telemetry: dict[str, Any] = {}

        for sym, exp_id in (
            ('<|"|>', QUOTE_DELIM_ID),
            ("<|think|>", THINK_FLAG_ID),
            ("<|channel>", CHANNEL_OPEN_ID),
            ("<channel|>", CHANNEL_CLOSE_ID),
            ("<|tool_call>", TOOL_CALL_OPEN_ID),
            ("<tool_call|>", TOOL_CALL_CLOSE_ID),
            ("<|tool_response>", TOOL_RESP_OPEN_ID),
            ("<turn|>", TURN_END_ID),
            ("\n", NEWLINE_ID),
        ):
            actual = self.tok.token_to_id(sym)
            if actual != exp_id:
                raise ValueError(f"Tokenizer mismatch for {sym!r}: expected {exp_id}, got {actual}")

        nl_set: set[int] = {NEWLINE_ID}
        get_vocab = getattr(self.tok, "get_vocab", None)
        if callable(get_vocab):
            for _, tid in get_vocab().items():
                if 0 <= int(tid) < 1024:
                    s = self.tok.decode([int(tid)], skip_special_tokens=False)
                    if "\n" in s or "\r" in s:
                        nl_set.add(int(tid))
        self._all_newline_ids = sorted(nl_set)

        self._kickoff_logit_bias: list[list] = [
            [tid, False]
            for tid in (CHANNEL_CLOSE_ID, TURN_END_ID, EOS_ID, TOOL_RESP_OPEN_ID, CHANNEL_OPEN_ID, *MULTIMODAL_IDS)
        ]
        self._thought_logit_bias: list[list] = [
            [tid, False]
            for tid in (TURN_END_ID, EOS_ID, TOOL_RESP_OPEN_ID, CHANNEL_OPEN_ID, *MULTIMODAL_IDS)
        ]
        self._phase_a_logit_bias: list[list] = [
            [tid, False] for tid in (CHANNEL_OPEN_ID, CHANNEL_CLOSE_ID, *MULTIMODAL_IDS)
        ]
        self._phase_b_logit_bias: list[list] = [
            [tid, False]
            for tid in (
                CHANNEL_OPEN_ID,
                CHANNEL_CLOSE_ID,
                TOOL_CALL_OPEN_ID,
                TOOL_RESP_OPEN_ID,
                TOOL_RESP_CLOSE_ID,
                TURN_OPEN_ID,
                TURN_END_ID,
                EOS_ID,
                *self._all_newline_ids,
                *MULTIMODAL_IDS,
            )
        ]

    def check_backend_health(self) -> bool:
        if not self.ready:
            return False
        try:
            st, obj = self.llama.get_json("/health", timeout=3.0)
            return st == 200 and isinstance(obj, dict) and obj.get("status") == "ok"
        except Exception:
            return False

    def activity_snapshot(self) -> dict[str, int | bool]:
        with self._counter_lock:
            waiting = self._waiting_count
            running = self._running_count
        return {"ready": bool(self.ready), "busy": bool(waiting > 0 or running > 0), "running": running, "waiting": waiting}

    def format_llama_metrics(self) -> str:
        snap = self.activity_snapshot()
        return (
            "# HELP llamacpp:requests_processing Active inference requests.\n"
            "# TYPE llamacpp:requests_processing gauge\n"
            f'llamacpp:requests_processing {snap["running"]}\n'
            "# HELP llamacpp:requests_deferred Queued inference requests.\n"
            "# TYPE llamacpp:requests_deferred gauge\n"
            f'llamacpp:requests_deferred {snap["waiting"]}\n'
        )

    def _build_lora_payload(self, weights: tuple[float, float, float, float, float]) -> list[dict]:
        return [
            {"id": idx, "scale": (round(float(weights[idx]), 4) if abs(float(weights[idx])) > 1e-6 else 0.0)}
            for idx in range(5)
        ]

    def generate_stream(
        self,
        messages: list[dict],
        model_field: str | None = None,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_tokens: int = 1536,
        seed: int | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[str, int, _ManagedStreamIterator]:
        canonical_tag, user_weights = parse_model_weights(model_field)
        phase_a_lora_payload = self._zero_lora_payload
        phase_b_lora_payload = self._build_lora_payload(user_weights)

        effective_tools = None if self.public_safe else tools
        effective_messages: list[dict] = sanitize_public_safe_messages(messages) if self.public_safe else list(messages)

        tool_registry = build_external_tool_registry(effective_tools)
        prompt_text, tool_registry = render_gemma4_v4_prompt_with_tools(
            effective_messages,
            slider_weights=user_weights,
            registry=tool_registry,
        )
        prompt_ids = list(self.tok.encode(prompt_text, add_special_tokens=False).ids)

        if len(prompt_ids) < 3 or prompt_ids[-1] != NEWLINE_ID:
            raise ValueError("Encoded Gemma 4 prompt does not end with <|channel>thought\\n")
        if len(prompt_ids) > self.slot_ctx_size - 256:
            raise ValueError(
                f"prompt_tokens ({len(prompt_ids)}) exceeds slot context window ({self.slot_ctx_size} - 256); context_length_exceeded"
            )

        max_visible = (
            max(PUBLIC_SAFE_MIN_MAX_TOKENS, min(int(max_tokens or PUBLIC_SAFE_MAX_MAX_TOKENS), PUBLIC_SAFE_MAX_MAX_TOKENS))
            if self.public_safe
            else max(1, min(int(max_tokens or 2048), 8192))
        )
        temp = float(temperature) if temperature is not None else 0.85
        tp = float(top_p) if top_p is not None else 0.95

        t_wait0 = time.perf_counter()
        with self._counter_lock:
            self._waiting_count += 1
        try:
            slot_id = self._slot_pool.get(block=True, timeout=self.slot_acquire_timeout)
        except queue.Empty as exc:
            with self._counter_lock:
                self._waiting_count = max(0, self._waiting_count - 1)
            raise SlotPoolExhaustedError("All inference slots are currently busy, please retry shortly") from exc
        slot_wait_ms = round((time.perf_counter() - t_wait0) * 1000.0, 2)

        with self._counter_lock:
            self._waiting_count = max(0, self._waiting_count - 1)
            self._running_count += 1
        lease = _SlotLease(self, slot_id)
        req_telemetry: dict[str, Any] = {
            "slot_id": slot_id,
            "slot_wait_ms": slot_wait_ms,
            "model_tag": canonical_tag,
            "slider_weights": list(user_weights),
        }

        def _iterator() -> Iterator[StreamStep | BridgeStreamStep]:
            try:
                t_start = time.perf_counter()
                ttft_ms: float | None = None
                prefill_ms_total = 0.0
                decode_ms_total = 0.0
                predicted_tokens_total = 0
                cur_prompt_text = prompt_text
                cur_ids = list(prompt_ids)
                allow_cache_prompt = True
                visible_blocks_emitted = 0
                send_message_bubbles_total = 0
                last_calc_stdout = ""
                visible_chars_emitted = 0

                def _record_first_visible() -> None:
                    nonlocal ttft_ms
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0

                def _finalize_telemetry() -> None:
                    total_ms = (time.perf_counter() - t_start) * 1000.0
                    dec_tok_s = (predicted_tokens_total / (decode_ms_total / 1000.0)) if decode_ms_total > 0 else 0.0
                    req_telemetry.update(
                        {
                            "prompt_tokens": len(prompt_ids),
                            "predicted_tokens": predicted_tokens_total,
                            "visible_chars": visible_chars_emitted,
                            "prefill_ms": round(prefill_ms_total, 2),
                            "decode_ms": round(decode_ms_total, 2),
                            "ttft_ms": round(ttft_ms if ttft_ms is not None else total_ms, 2),
                            "total_ms": round(total_ms, 2),
                            "decode_tok_s": round(dec_tok_s, 2),
                        }
                    )
                    with self._counter_lock:
                        self.last_telemetry = dict(req_telemetry)

                for hop_idx in range(self.max_tool_hops):
                    hop_seed = (int(seed) + hop_idx * 17) if seed is not None else -1
                    kickoff_text = ""
                    if visible_blocks_emitted == 0 and self.min_thought_tokens > 0:
                        a0 = self.llama.complete_sync(
                            {
                                "id_slot": slot_id,
                                "prompt": cur_ids,
                                "n_predict": self.min_thought_tokens,
                                "temperature": min(temp, 0.65),
                                "top_p": tp,
                                "seed": hop_seed,
                                "lora": phase_a_lora_payload,
                                "cache_prompt": allow_cache_prompt,
                                "special": True,
                                "logit_bias": self._kickoff_logit_bias,
                            }
                        )
                        allow_cache_prompt = True
                        kickoff_text = a0.content
                        if kickoff_text:
                            cur_prompt_text += kickoff_text
                            cur_ids = list(self.tok.encode(cur_prompt_text, add_special_tokens=False).ids)
                        prefill_ms_total += a0.prompt_ms
                        decode_ms_total += a0.predicted_ms
                        predicted_tokens_total += a0.tokens_predicted

                    t_final: LlamaCompletionResult | None = None
                    for _, t_done in self.llama.complete_stream(
                        {
                            "id_slot": slot_id,
                            "prompt": cur_ids,
                            "n_predict": self.max_thought_tokens,
                            "temperature": min(temp, 0.65),
                            "top_p": tp,
                            "seed": (hop_seed + 1) if hop_seed >= 0 else -1,
                            "lora": phase_a_lora_payload,
                            "cache_prompt": allow_cache_prompt,
                            "special": True,
                            "stop": [CHANNEL_CLOSE_STR],
                            "logit_bias": self._thought_logit_bias,
                        }
                    ):
                        allow_cache_prompt = True
                        if t_done is not None:
                            t_final = t_done
                    assert t_final is not None
                    prefill_ms_total += t_final.prompt_ms
                    decode_ms_total += t_final.predicted_ms
                    predicted_tokens_total += t_final.tokens_predicted

                    cur_prompt_text += t_final.content + CHANNEL_CLOSE_STR + TOOL_CALL_OPEN_STR
                    seg_open_prefix = TOOL_CALL_OPEN_STR
                    cur_ids = list(self.tok.encode(cur_prompt_text, add_special_tokens=False).ids)

                    hop_completed_calls: list[tuple[str, dict[str, Any]]] = []
                    hop_spoken_chunks: list[str] = []
                    hop_hit_length_limit = False

                    for seg_idx in range(6):
                        a1_chunks: list[str] = []
                        a1_final: LlamaCompletionResult | None = None
                        code_opened = False
                        code_wrapped = False
                        code_emitted = 0
                        code_closed = False

                        for piece, done_res in self.llama.complete_stream(
                            {
                                "id_slot": slot_id,
                                "prompt": cur_ids,
                                "n_predict": self.max_thought_and_tool_tokens,
                                "temperature": min(temp, 0.60),
                                "top_p": tp,
                                "seed": (hop_seed + 2 + seg_idx) if hop_seed >= 0 else -1,
                                "lora": phase_a_lora_payload,
                                "cache_prompt": allow_cache_prompt,
                                "special": True,
                                "stop": PHASE_A_STOPS,
                                "logit_bias": self._phase_a_logit_bias,
                            }
                        ):
                            allow_cache_prompt = True
                            if piece:
                                a1_chunks.append(piece)
                                post_now = seg_open_prefix + "".join(a1_chunks)
                                if not code_closed:
                                    m_code = CODE_CONTENT_OPEN_RE.search(post_now)
                                    if m_code:
                                        lang_tag = (m_code.group(1) or "").strip()
                                        tail_body = m_code.group(2)
                                        if not code_opened and (len(tail_body) >= 4 or QUOTE_STR in tail_body):
                                            prefix = "\n" if visible_blocks_emitted > 0 else ""
                                            code_wrapped = bool(lang_tag) and not tail_body.lstrip().startswith("```")
                                            open_delta = f"{prefix}```{lang_tag}\n" if code_wrapped else prefix
                                            if open_delta:
                                                _record_first_visible()
                                                visible_chars_emitted += len(open_delta)
                                                yield StreamStep(delta_text=open_delta, token_id=QUOTE_DELIM_ID)
                                            code_opened = True
                                        if code_opened:
                                            if QUOTE_STR in tail_body:
                                                code_complete, _ = tail_body.split(QUOTE_STR, 1)
                                                rem = code_complete[code_emitted:]
                                                nl_end = "" if code_complete.endswith("\n") else "\n"
                                                close_delta = f"{rem}{nl_end}```" if code_wrapped else rem.rstrip("\r\n")
                                                if close_delta:
                                                    _record_first_visible()
                                                    visible_chars_emitted += len(close_delta)
                                                    yield StreamStep(delta_text=close_delta, token_id=TOOL_CALL_CLOSE_ID)
                                                code_closed = True
                                                visible_blocks_emitted += 1
                                            else:
                                                safe_code, _ = _hold_partial_tag_suffix(tail_body, QUOTE_STR)
                                                delta_code = safe_code[code_emitted:]
                                                if delta_code:
                                                    _record_first_visible()
                                                    code_emitted = len(safe_code)
                                                    visible_chars_emitted += len(delta_code)
                                                    yield StreamStep(delta_text=delta_code, token_id=NEWLINE_ID)
                            if done_res is not None:
                                a1_final = done_res

                        assert a1_final is not None
                        prefill_ms_total += a1_final.prompt_ms
                        decode_ms_total += a1_final.predicted_ms
                        predicted_tokens_total += a1_final.tokens_predicted

                        raw_a1 = a1_final.content
                        stop_word = a1_final.stopping_word
                        seg_post = seg_open_prefix + raw_a1
                        seg_open_prefix = ""

                        seg_calls = parse_gemma4_tool_calls(seg_post, tool_registry)
                        for t_name, t_args in seg_calls:
                            hop_completed_calls.append((t_name, t_args))
                            if t_name == "send_code_or_doc" and not code_opened:
                                lang = re.sub(r"[^a-zA-Z0-9_+#.-]", "", str(t_args.get("language") or "").strip())
                                body = str(t_args.get("content") or "").rstrip("\r\n")
                                prefix = "\n" if visible_blocks_emitted > 0 else ""
                                blk = f"{prefix}```{lang}\n{body}\n```" if lang else f"{prefix}{body}"
                                _record_first_visible()
                                visible_chars_emitted += len(blk)
                                yield StreamStep(delta_text=blk, token_id=TOOL_CALL_CLOSE_ID)
                                visible_blocks_emitted += 1
                            elif t_name == "send_message":
                                for b_txt in t_args.get("bubbles") or []:
                                    prefix = "\n" if visible_blocks_emitted > 0 else ""
                                    _record_first_visible()
                                    visible_chars_emitted += len(prefix) + len(str(b_txt))
                                    yield StreamStep(delta_text=prefix + str(b_txt), token_id=QUOTE_DELIM_ID)
                                    visible_blocks_emitted += 1
                                    send_message_bubbles_total += 1

                        if stop_word in SEND_MSG_STOP_VARIANTS:
                            hop_completed_calls.append(("send_message", {}))
                            cur_prompt_text += raw_a1 + stop_word
                            cur_ids = list(self.tok.encode(cur_prompt_text, add_special_tokens=False).ids)
                            b_parser = PhaseBBubbleStreamParser(initial_visible_blocks=visible_blocks_emitted)
                            b_chunks: list[str] = []
                            b_final: LlamaCompletionResult | None = None

                            for b_piece, b_done in self.llama.complete_stream(
                                {
                                    "id_slot": slot_id,
                                    "prompt": cur_ids,
                                    "n_predict": max_visible,
                                    "temperature": temp,
                                    "top_p": tp,
                                    "seed": (hop_seed + seg_idx * 7 + 3) if hop_seed >= 0 else -1,
                                    "lora": phase_b_lora_payload,
                                    "cache_prompt": allow_cache_prompt,
                                    "special": True,
                                    "stop": PHASE_B_STOPS,
                                    "logit_bias": self._phase_b_logit_bias,
                                }
                            ):
                                if b_piece:
                                    b_chunks.append(b_piece)
                                    for delta_str in b_parser.feed(b_piece):
                                        _record_first_visible()
                                        visible_chars_emitted += len(delta_str)
                                        hop_spoken_chunks.append(delta_str)
                                        yield StreamStep(delta_text=delta_str, token_id=QUOTE_DELIM_ID)
                                if b_done is not None:
                                    b_final = b_done

                            for tail_str in b_parser.finalize():
                                _record_first_visible()
                                visible_chars_emitted += len(tail_str)
                                hop_spoken_chunks.append(tail_str)
                                yield StreamStep(delta_text=tail_str, token_id=QUOTE_DELIM_ID)

                            visible_blocks_emitted = b_parser.visible_blocks_emitted
                            send_message_bubbles_total += b_parser.bubbles_emitted
                            if b_final is not None:
                                prefill_ms_total += b_final.prompt_ms
                                decode_ms_total += b_final.predicted_ms
                                predicted_tokens_total += b_final.tokens_predicted

                            b_closing = b_final.stopping_word if (b_final and b_final.stopping_word) else "]}<tool_call|>"
                            if not b_closing.endswith(TOOL_CALL_CLOSE_STR):
                                b_closing = "]}<tool_call|>"
                            cur_prompt_text += "".join(b_chunks) + b_closing
                            cur_ids = list(self.tok.encode(cur_prompt_text, add_special_tokens=False).ids)
                            allow_cache_prompt = not any(abs(w) > 1e-6 for w in user_weights)
                            if b_final and b_final.stop_type == "limit":
                                hop_hit_length_limit = True
                                break
                            continue

                        cur_prompt_text += raw_a1
                        cur_ids = list(self.tok.encode(cur_prompt_text, add_special_tokens=False).ids)
                        if a1_final.stop_type == "limit":
                            hop_hit_length_limit = True
                        break

                    _, external_calls = partition_pc2_and_external_calls(hop_completed_calls, tool_registry)
                    if self.public_safe:
                        external_calls = []
                    if external_calls:
                        _finalize_telemetry()
                        yield BridgeStreamStep(
                            delta_text="",
                            token_id=TURN_END_ID,
                            finish_reason="tool_calls",
                            external_tool_calls=external_calls,
                        )
                        return

                    resp_blocks: list[str] = []
                    called_calc = False
                    for t_name, t_args in hop_completed_calls:
                        if t_name in ("calc_python", "run_python"):
                            called_calc = True
                            calc_out = run_calc_python_sandbox(str(t_args.get("code") or ""))
                            last_calc_stdout = calc_out
                            resp_blocks.append(
                                f'<|tool_response>response:{t_name}{{stdout:<|"|>{calc_out}<|"|>}}<tool_response|>'
                            )
                        elif t_name in ("send_code_or_doc", "send_message"):
                            resp_blocks.append(f"<|tool_response>response:{t_name}{{delivered:true}}<tool_response|>")

                    if (hop_idx + 1) < self.max_tool_hops and not hop_hit_length_limit and called_calc:
                        cur_prompt_text += "".join(resp_blocks) + "<|channel>thought\n"
                        cur_ids = list(self.tok.encode(cur_prompt_text, add_special_tokens=False).ids)
                        continue

                    # Safety net: if Phase A planned an external tool call in `<|channel>thought`,
                    # emitted an opening `send_message` preamble bubble first, and stopped at
                    # `<|tool_response>` waiting for `send_message` acknowledgment, feed the
                    # synthetic delivery receipt once so Phase A immediately emits the external tool call.
                    if (
                        not self.public_safe
                        and hop_idx == 0
                        and (hop_idx + 1) < self.max_tool_hops
                        and not hop_hit_length_limit
                        and bool(tool_registry.tools)
                        and resp_blocks
                    ):
                        spoken_txt = "".join(hop_spoken_chunks)
                        is_midtask_promise = bool(
                            re.search(r"(?:这就去|先跑|先看|我先|先帮你|我看看咯|等我先|先去查)", spoken_txt)
                        ) and not bool(re.search(r"(?:全绿|通过|修好|改好|搞定|完成|加好|写好|弄好)", spoken_txt))
                        if is_midtask_promise:
                            cur_prompt_text += "".join(resp_blocks) + "<|channel>thought\n"
                            cur_ids = list(self.tok.encode(cur_prompt_text, add_special_tokens=False).ids)
                            continue

                    if visible_blocks_emitted == 0 and last_calc_stdout:
                        _record_first_visible()
                        visible_chars_emitted += len(last_calc_stdout)
                        yield StreamStep(delta_text=last_calc_stdout, token_id=TURN_END_ID)

                    _finalize_telemetry()
                    yield StreamStep(
                        delta_text="",
                        token_id=TURN_END_ID,
                        finish_reason="length" if hop_hit_length_limit else "stop",
                    )
                    return

                _finalize_telemetry()
                yield StreamStep(delta_text="", token_id=TURN_END_ID, finish_reason="stop")
            finally:
                if "total_ms" not in req_telemetry:
                    _finalize_telemetry()
                lease.release()

        managed = _ManagedStreamIterator(
            _iterator(),
            lease,
            req_telemetry,
            sanitized_messages=effective_messages,
            effective_weights=user_weights,
            effective_max_tokens=max_visible,
        )
        return canonical_tag, len(prompt_ids), managed


def create_pc2_http_server(engine: LlamaCppPC2Engine, host: str = "127.0.0.1", port: int = 8081) -> ThreadingHTTPServer:
    class GatewayHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/health", "/ready"):
                ok = engine.check_backend_health()
                self._send_json(200 if ok else 503, {"status": "ok" if ok else "starting", "slots": 5})
                return
            if path == "/metrics":
                txt = engine.format_llama_metrics().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(txt)))
                self.end_headers()
                self.wfile.write(txt)
                return
            if path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "xiaofeiyu-v4", "object": "model", "owned_by": "xiaofeiyu"},
                            {"id": "xiaofeiyu-26b-a4b", "object": "model", "owned_by": "xiaofeiyu"},
                            {"id": "xiaofeiyu-e4b", "object": "model", "owned_by": "xiaofeiyu"},
                        ],
                    },
                )
                return
            self._send_json(404, make_error_payload(f"Unknown endpoint: {path}", code="not_found"))

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path != "/v1/chat/completions":
                self._send_json(404, make_error_payload(f"Unsupported endpoint: {path}", code="not_found"))
                return
            try:
                clen = int(self.headers.get("Content-Length") or "0")
                req = json.loads(self.rfile.read(clen).decode("utf-8"))
                if not isinstance(req, dict):
                    raise ValueError("JSON body must be an object")
            except Exception as exc:
                self._send_json(400, make_error_payload(f"Malformed request: {exc}"))
                return

            raw_max_tok = (
                req.get("max_tokens")
                if req.get("max_tokens") is not None
                else req.get("max_completion_tokens")
            )
            try:
                model_tag, prompt_tokens, step_iter = engine.generate_stream(
                    messages=req.get("messages") or [],
                    model_field=req.get("model"),
                    temperature=req.get("temperature", 0.85),
                    top_p=req.get("top_p", 0.95),
                    max_tokens=raw_max_tok if raw_max_tok is not None else 2048,
                    seed=req.get("seed"),
                    tools=None if engine.public_safe else (req.get("tools") if isinstance(req.get("tools"), list) else None),
                )
            except SlotPoolExhaustedError as exc:
                self._send_json(429, make_error_payload(str(exc), err_type="rate_limit_error", code="rate_limit"))
                return
            except ValueError as exc:
                self._send_json(400, make_error_payload(str(exc)))
                return

            req_id = "chatcmpl_" + secrets.token_hex(12)
            created = int(time.time())
            stream = bool(req.get("stream", False))

            try:
                if not stream:
                    pieces: list[str] = []
                    ext_calls: list[OpenAIToolCallOut] | None = None
                    finish_reason = "stop"
                    comp_tokens = 0
                    for step in step_iter:
                        if step.delta_text:
                            pieces.append(step.delta_text)
                            comp_tokens += 1
                        if getattr(step, "external_tool_calls", None):
                            ext_calls = step.external_tool_calls
                        if step.finish_reason:
                            finish_reason = step.finish_reason
                    full_reply = "".join(pieces)
                    if ext_calls:
                        resp = make_openai_tool_completion_payload(
                            req_id=req_id,
                            created=created,
                            model_tag=model_tag,
                            content_text=full_reply,
                            external_tool_calls=ext_calls,
                            finish_reason=finish_reason,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=comp_tokens,
                        )
                    else:
                        resp = make_completion_payload(
                            req_id=req_id,
                            created=created,
                            model_tag=model_tag,
                            content_text=full_reply,
                            finish_reason=finish_reason,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=comp_tokens,
                        )
                    self._send_json(200, resp)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                role_chunk = make_chunk_payload(req_id, created, model_tag, {"role": "assistant", "content": ""}, None)
                self.wfile.write(f"data: {json.dumps(role_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

                final_reason = "stop"
                streamed_chunks = 0
                for step in step_iter:
                    if step.delta_text:
                        streamed_chunks += 1
                        c_chunk = make_chunk_payload(req_id, created, model_tag, {"content": step.delta_text}, None)
                        self.wfile.write(f"data: {json.dumps(c_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    step_ext = getattr(step, "external_tool_calls", None)
                    if step_ext:
                        streamed_chunks += len(step_ext)
                        tc_chunk = make_chunk_payload(req_id, created, model_tag, make_openai_tool_chunk_delta(step_ext), None)
                        self.wfile.write(f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    if step.finish_reason is not None:
                        final_reason = step.finish_reason

                pred_n = int(step_iter.telemetry.get("predicted_tokens") or streamed_chunks)
                usage_obj = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": pred_n,
                    "total_tokens": prompt_tokens + pred_n,
                }
                stop_chunk = make_chunk_payload(req_id, created, model_tag, {}, final_reason, usage=usage_obj)
                self.wfile.write(f"data: {json.dumps(stop_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            finally:
                step_iter.close()

    server = ThreadingHTTPServer((host, port), GatewayHandler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--llama-host", default="127.0.0.1")
    ap.add_argument("--llama-port", type=int, default=8082)
    ap.add_argument("--model-family", choices=["12b", "26b-a4b", "e4b"], default="12b")
    ap.add_argument("--tokenizer-path", type=Path, default=None)
    ap.add_argument("--n-parallel", type=int, default=1)
    ap.add_argument("--ctx-size", type=int, default=16384)
    ap.add_argument("--max-thought-tokens", type=int, default=2048)
    ap.add_argument("--max-thought-and-tool-tokens", type=int, default=4096)
    ap.add_argument("--public-safe", action="store_true")
    args = ap.parse_args(argv)

    tokenizer = load_fast_tokenizer(args.tokenizer_path, model_family=args.model_family)
    engine = LlamaCppPC2Engine(
        llama_client=LlamaCppHTTPClient(host=args.llama_host, port=args.llama_port),
        tokenizer=tokenizer,
        n_parallel=args.n_parallel,
        public_safe=args.public_safe,
        ctx_size=args.ctx_size,
        max_thought_tokens=args.max_thought_tokens,
        max_thought_and_tool_tokens=args.max_thought_and_tool_tokens,
    )
    srv = create_pc2_http_server(engine, host=args.host, port=args.port)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=srv.shutdown, daemon=True).start())
    signal.signal(signal.SIGINT, lambda *_: threading.Thread(target=srv.shutdown, daemon=True).start())
    sys.stderr.write(f"[xiaofeiyu-server] Listening on http://{args.host}:{args.port} (upstream llama-server {args.llama_host}:{args.llama_port})\n")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
