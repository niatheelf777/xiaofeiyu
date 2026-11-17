"""Turnkey local Web UI static & `/api/{status,cluster,chat}` handler for `pc2_split_phase_server.py`.

Mounts `web/public/` (the 2D Persona Orb + 5-Slider IM Chat Web UI) directly onto the Python
Split-Phase Gateway so `python server/launch_local.py --preset 12b` serves both the browser UI
(`http://127.0.0.1:8081/`) and the OpenAI-compatible API (`http://127.0.0.1:8081/v1`) with zero
Node/Vercel/Cloudflare setup.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pc2_split_phase_server import LlamaCppPC2Engine

MAX_WEB_BODY_BYTES = 128 * 1024
MAX_WEB_MESSAGES = 20
MAX_WEB_MESSAGE_CHARS = 3000
WEB_WEIGHT_MIN = 0.0
WEB_WEIGHT_MAX = 1.5

WEB_WEIGHT_KEYS: tuple[str, str, str, str, str] = (
    "xiaofeiyu_v4",
    "neko_clingy",
    "older_tease",
    "brat_mesugaki",
    "fragile_porcelain",
)

FIVE_SLOT_CARTRIDGES: tuple[str, str, str, str, str] = (
    "s0_xiaofeiyu_backbone",
    "c1_neko_clingy",
    "c2_older_tease",
    "c3_brat_mesugaki",
    "c4_fragile_porcelain",
)

MIME_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".xml": "application/xml; charset=utf-8",
}

_FENCE_OPEN_RE = re.compile(r"^\s*```([^\r\n`]*)$")
_FENCE_CLOSE_RE = re.compile(r"^\s*```\s*$")
_ID_CLEAN_RE = re.compile(r"[^a-zA-Z0-9_-]")
_CTRL_CLEAN_RE = re.compile(r"[\x00-\x1f\x7f]")


class WebChatRequestError(ValueError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def default_web_public_dir() -> Path | None:
    candidate = Path(__file__).resolve().parent.parent / "web" / "public"
    return candidate if (candidate / "index.html").is_file() else None


def resolve_static_asset(web_public_dir: Path | None, url_path: str) -> tuple[int, str, bytes] | None:
    """Resolve a static file inside `web_public_dir` with strict path-traversal containment."""
    if web_public_dir is None:
        return None
    root = web_public_dir.resolve()
    if not root.is_dir():
        return None

    clean_rel = "index.html" if url_path in ("", "/") else url_path.lstrip("/")
    if not clean_rel or "\x00" in clean_rel:
        return None
    target = (root / clean_rel).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return (403, "text/plain; charset=utf-8", b"Forbidden")

    if not target.is_file():
        return None
    mime = MIME_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return (200, mime, target.read_bytes())


def _clamp_web_weight(val: Any, fallback: float = 0.0) -> float:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return fallback
    if num != num or num in (float("inf"), float("-inf")):
        return fallback
    clamped = min(WEB_WEIGHT_MAX, max(WEB_WEIGHT_MIN, num))
    return round(clamped, 3)


def sanitize_web_weights(raw_weights: Any) -> tuple[dict[str, float], list[float]]:
    if isinstance(raw_weights, (list, tuple)) and len(raw_weights) == 5:
        defaults = (1.0, 0.0, 0.0, 0.0, 0.0)
        obj = {
            WEB_WEIGHT_KEYS[idx]: _clamp_web_weight(raw_weights[idx], defaults[idx])
            for idx in range(5)
        }
        return obj, [obj[k] for k in WEB_WEIGHT_KEYS]

    src = raw_weights if isinstance(raw_weights, dict) else {}
    base_raw = src.get("xiaofeiyu_v4") if "xiaofeiyu_v4" in src else src.get("xiaofeiyu_v2")
    obj = {
        "xiaofeiyu_v4": _clamp_web_weight(base_raw, 1.0),
        "neko_clingy": _clamp_web_weight(src.get("neko_clingy"), 0.0),
        "older_tease": _clamp_web_weight(src.get("older_tease"), 0.0),
        "brat_mesugaki": _clamp_web_weight(src.get("brat_mesugaki"), 0.0),
        "fragile_porcelain": _clamp_web_weight(src.get("fragile_porcelain"), 0.0),
    }
    return obj, [obj[k] for k in WEB_WEIGHT_KEYS]


def _sanitize_id(raw: Any, fallback: str, max_len: int = 64) -> str:
    s = _ID_CLEAN_RE.sub("", str(raw or "").strip())[:max_len]
    return s or fallback


def _sanitize_nickname(raw: Any, fallback: str = "访客") -> str:
    s = _CTRL_CLEAN_RE.sub("", str(raw or "")).strip()[:40]
    return s or fallback


def sanitize_web_chat_payload(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise WebChatRequestError(400, "INVALID_BODY", "请求格式须为 JSON 对象")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) == 0:
        raise WebChatRequestError(400, "EMPTY_MESSAGES", "messages 不能为空")

    cleaned: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        trimmed = content.strip()
        if trimmed:
            cleaned.append({"role": role, "content": trimmed})

    sliced = cleaned[-MAX_WEB_MESSAGES:]
    if not sliced:
        raise WebChatRequestError(400, "EMPTY_MESSAGES", "请先输入有效对话内容")

    last_msg = sliced[-1]
    if last_msg["role"] != "user":
        raise WebChatRequestError(400, "INVALID_TURN_ORDER", "最后一条消息须为用户输入")
    if len(last_msg["content"]) > MAX_WEB_MESSAGE_CHARS:
        raise WebChatRequestError(
            400,
            "MESSAGE_TOO_LONG",
            f"单条消息请控制在 {MAX_WEB_MESSAGE_CHARS} 字符以内",
        )

    bounded = [{"role": m["role"], "content": m["content"][:MAX_WEB_MESSAGE_CHARS]} for m in sliced]
    weights_obj, weight_tuple = sanitize_web_weights(body.get("weights"))
    model_tag = "xiaofeiyu-v4|" + ",".join(f"{w:.3f}" for w in weight_tuple)

    visitor_id = _sanitize_id(body.get("visitorId"), "v_local")
    return {
        "visitorId": visitor_id,
        "profileId": _sanitize_id(body.get("profileId") or visitor_id, "p_local"),
        "visitorNickname": _sanitize_nickname(body.get("visitorNickname"), "访客"),
        "conversationId": _sanitize_id(body.get("conversationId"), "conv_local"),
        "messages": bounded,
        "weights": weights_obj,
        "weightTuple": weight_tuple,
        "modelTag": model_tag,
        "stream": body.get("stream", True) is not False,
    }


def split_reply_bubbles(raw_content: str) -> list[str]:
    normalized = (raw_content or "").replace("\r\n", "\n").replace("\r", "\n")
    bubbles: list[str] = []
    in_fence = False
    fence_lines: list[str] = []

    for line in normalized.split("\n"):
        if not in_fence:
            m_open = _FENCE_OPEN_RE.match(line)
            if m_open:
                in_fence = True
                fence_lines = [f"```{m_open.group(1).strip()}"]
            else:
                trimmed = line.strip()
                if trimmed:
                    bubbles.append(trimmed)
        elif _FENCE_CLOSE_RE.match(line):
            fence_lines.append("```")
            bubbles.append("\n".join(fence_lines))
            in_fence = False
            fence_lines = []
        else:
            fence_lines.append(line)

    if in_fence and fence_lines:
        fence_lines.append("```")
        bubbles.append("\n".join(fence_lines))
    return bubbles


class IncrementalBubbleEmitter:
    """Fence-aware incremental bubble splitter matching `web/api/_lib.js`."""

    def __init__(self) -> None:
        self.line_buffer = ""
        self.in_fence = False
        self.fence_lines: list[str] = []
        self.emitted_count = 0

    def _consume_line(self, line: str, out: list[dict[str, Any]]) -> None:
        if not self.in_fence:
            m_open = _FENCE_OPEN_RE.match(line)
            if m_open:
                self.in_fence = True
                self.fence_lines = [f"```{m_open.group(1).strip()}"]
            else:
                trimmed = line.strip()
                if trimmed:
                    out.append({"index": self.emitted_count, "text": trimmed})
                    self.emitted_count += 1
        elif _FENCE_CLOSE_RE.match(line):
            self.fence_lines.append("```")
            out.append({"index": self.emitted_count, "text": "\n".join(self.fence_lines)})
            self.emitted_count += 1
            self.in_fence = False
            self.fence_lines = []
        else:
            self.fence_lines.append(line)

    def push(self, chunk: str) -> list[dict[str, Any]]:
        if not chunk:
            return []
        self.line_buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        parts = self.line_buffer.split("\n")
        self.line_buffer = parts.pop()
        out: list[dict[str, Any]] = []
        for line in parts:
            self._consume_line(line, out)
        return out

    def flush(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.line_buffer:
            self._consume_line(self.line_buffer, out)
            self.line_buffer = ""
        if self.in_fence and self.fence_lines:
            self.fence_lines.append("```")
            out.append({"index": self.emitted_count, "text": "\n".join(self.fence_lines)})
            self.emitted_count += 1
            self.in_fence = False
            self.fence_lines = []
        return out


def format_sse_event(event_name: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def build_local_status_payload(engine: LlamaCppPC2Engine, model_family: str = "12b") -> dict[str, Any]:
    healthy = engine.check_backend_health()
    snap = engine.activity_snapshot()
    capacity = max(1, int(engine.n_parallel))
    in_flight = int(snap.get("running", 0)) + int(snap.get("waiting", 0))
    available = max(0, capacity - in_flight)
    ewma_raw = engine.last_telemetry.get("ttft_ms") or engine.last_telemetry.get("first_visible_ms")
    ewma_ms = int(round(float(ewma_raw))) if isinstance(ewma_raw, (int, float)) and ewma_raw > 0 else 320
    family_upper = str(model_family or "12b").upper()
    node_id = f"LOCAL-{family_upper}-1"

    return {
        "ok": True,
        "mode": "local-turnkey",
        "model": f"xiaofeiyu-{model_family}",
        "model_family": model_family,
        "ready": healthy,
        "slots": {
            "count": 5,
            "budget_limit": 1.15,
            "cartridges": list(FIVE_SLOT_CARTRIDGES),
        },
        "cluster": {
            "nodes": 1,
            "healthyNodes": 1 if healthy else 0,
            "totalCapacity": capacity,
            "totalInFlight": in_flight,
            "gpuSummary": f"Local {family_upper}",
            "items": [
                {
                    "id": node_id,
                    "gpu": "LOCAL",
                    "capacity": capacity,
                    "inFlight": in_flight,
                    "available": available,
                    "healthy": healthy,
                    "ewmaFirstByteMs": ewma_ms,
                }
            ],
        },
    }


def handle_web_chat_post(
    handler: BaseHTTPRequestHandler,
    engine: LlamaCppPC2Engine,
    raw_bytes: bytes,
    *,
    model_family: str = "12b",
    slot_exhausted_cls: type[Exception] = RuntimeError,
) -> None:
    if len(raw_bytes) > MAX_WEB_BODY_BYTES:
        _send_web_error(handler, 413, "PAYLOAD_TOO_LARGE", "请求体过大（上限 128 KiB）")
        return

    text_body = raw_bytes.decode("utf-8", errors="replace").strip()
    try:
        parsed_json = json.loads(text_body) if text_body else {}
    except json.JSONDecodeError:
        _send_web_error(handler, 400, "INVALID_JSON", "JSON 格式无效")
        return

    try:
        chat_req = sanitize_web_chat_payload(parsed_json)
    except WebChatRequestError as exc:
        _send_web_error(handler, exc.status, exc.code, exc.message)
        return

    started_at = time.perf_counter()
    node_id = f"LOCAL-{str(model_family or '12b').upper()}-1"

    try:
        canonical_tag, _prompt_tokens, step_iter = engine.generate_stream(
            messages=chat_req["messages"],
            model_field=chat_req["modelTag"],
            temperature=0.85,
            top_p=0.95,
            max_tokens=1536,
            tools=None,
        )
    except slot_exhausted_cls:
        _send_web_error(handler, 429, "UPSTREAM_BUSY", "本地推理槽位正忙，请稍等几秒再发一次。")
        return
    except ValueError as exc:
        _send_web_error(handler, 400, "BAD_REQUEST", str(exc))
        return
    except Exception:
        _send_web_error(
            handler,
            503,
            "UPSTREAM_UNAVAILABLE",
            "本地 llama-server 尚未就绪或连接断开，请确认后端已启动。",
        )
        return

    if not chat_req["stream"]:
        try:
            pieces: list[str] = []
            for step in step_iter:
                if step.delta_text:
                    pieces.append(step.delta_text)
            final_raw = "".join(pieces).strip() or "信号刚晃了一下，你再跟我说一遍嘛"
            final_bubbles = split_reply_bubbles(final_raw)
            latency_ms = int(round((time.perf_counter() - started_at) * 1000.0))
            _send_web_json(
                handler,
                200,
                {
                    "ok": True,
                    "text": final_raw,
                    "bubbles": final_bubbles,
                    "weights": chat_req["weights"],
                    "weight_tuple": chat_req["weightTuple"],
                    "latency_ms": latency_ms,
                    "model_tag": canonical_tag,
                    "node_id": node_id,
                    "node_gpu": "LOCAL",
                },
            )
        finally:
            step_iter.close()
        return

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache, no-transform")
    handler.send_header("X-Xfy-Cluster-Node", node_id)
    handler.send_header("Connection", "close")
    handler.end_headers()

    accumulated = ""
    emitter = IncrementalBubbleEmitter()
    collected_bubbles: list[str] = []

    try:
        handler.wfile.write(
            format_sse_event(
                "meta",
                {
                    "model": "xiaofeiyu-v4",
                    "model_tag": canonical_tag,
                    "weights": chat_req["weights"],
                    "weight_tuple": chat_req["weightTuple"],
                    "node_id": node_id,
                    "node_gpu": "LOCAL",
                },
            )
        )
        handler.wfile.flush()

        for step in step_iter:
            delta = step.delta_text
            if delta:
                accumulated += delta
                handler.wfile.write(format_sse_event("delta", {"delta": delta, "text": accumulated}))
                for bubble_evt in emitter.push(delta):
                    collected_bubbles.append(bubble_evt["text"])
                    handler.wfile.write(format_sse_event("bubble", bubble_evt))
                handler.wfile.flush()

        for tail_evt in emitter.flush():
            collected_bubbles.append(tail_evt["text"])
            handler.wfile.write(format_sse_event("bubble", tail_evt))

        final_raw = accumulated.strip() or "信号刚晃了一下，你再跟我说一遍嘛"
        final_bubbles = collected_bubbles if collected_bubbles else split_reply_bubbles(final_raw)
        latency_ms = int(round((time.perf_counter() - started_at) * 1000.0))

        handler.wfile.write(
            format_sse_event(
                "done",
                {
                    "text": final_raw,
                    "bubbles": final_bubbles,
                    "weights": chat_req["weights"],
                    "weight_tuple": chat_req["weightTuple"],
                    "latency_ms": latency_ms,
                    "model_tag": canonical_tag,
                    "node_id": node_id,
                    "node_gpu": "LOCAL",
                },
            )
        )
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
        return
    except Exception:
        try:
            handler.wfile.write(
                format_sse_event(
                    "error",
                    {"code": "STREAM_INTERRUPTED", "message": "本地推理连接中断，请确认 llama-server 运行正常后重试。"},
                )
            )
            handler.wfile.flush()
        except OSError:
            pass
    finally:
        step_iter.close()


def _send_web_json(handler: BaseHTTPRequestHandler, status: int, obj: dict[str, Any]) -> None:
    body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)


def _send_web_error(handler: BaseHTTPRequestHandler, status: int, code: str, message: str) -> None:
    _send_web_json(handler, status, {"error": {"code": code, "message": message}})
