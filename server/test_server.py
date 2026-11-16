#!/usr/bin/env python3
"""Unit tests for the open-source 5-slider Split-Phase server and local launcher."""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gemma4_bridge as bridge  # noqa: E402
import launch_local  # noqa: E402
import pc2_split_phase_server as srv  # noqa: E402


class _FakeTokenizer:
    """Minimal deterministic tokenizer matching Gemma 4 special token IDs for offline unit tests."""

    _SPECIAL = {
        '<|"|>': bridge.QUOTE_DELIM_ID,
        "<|think|>": bridge.THINK_FLAG_ID,
        "<|channel>": bridge.CHANNEL_OPEN_ID,
        "<channel|>": bridge.CHANNEL_CLOSE_ID,
        "<|tool_call>": bridge.TOOL_CALL_OPEN_ID,
        "<tool_call|>": bridge.TOOL_CALL_CLOSE_ID,
        "<|tool_response>": bridge.TOOL_RESP_OPEN_ID,
        "<turn|>": bridge.TURN_END_ID,
        "\n": bridge.NEWLINE_ID,
    }

    def token_to_id(self, sym: str):
        return self._SPECIAL.get(sym)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._SPECIAL)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        rev = {v: k for k, v in self._SPECIAL.items()}
        return "".join(rev.get(i, " ") for i in ids)

    class _Enc:
        def __init__(self, ids: list[int]) -> None:
            self.ids = ids

    def encode(self, text: str, add_special_tokens: bool = False):
        # Ensure prompt ends with NEWLINE_ID when ending with `\n`
        ids = [bridge.BOS_ID, 200, 201]
        if text.endswith("\n"):
            ids.append(bridge.NEWLINE_ID)
        return self._Enc(ids)


class _MockLlamaServer:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.script: list[dict] = []
        parent = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/health":
                    body = b'{"status":"ok"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or "0")
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                parent.requests.append(req)
                spec = parent.script.pop(0)

                if not req.get("stream"):
                    raw = json.dumps(
                        {
                            "content": spec.get("content", ""),
                            "stopping_word": spec.get("stopping_word", ""),
                            "stop_type": spec.get("stop_type", "limit"),
                            "tokens_predicted": spec.get("tokens_predicted", 8),
                            "timings": {"prompt_n": 10, "prompt_ms": 5.0, "predicted_n": 8, "predicted_ms": 40.0, "predicted_per_second": 200.0},
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.end_headers()
                for ch in spec.get("chunks") or []:
                    self.wfile.write(f"data: {json.dumps({'content': ch, 'stop': False}, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                done = {
                    "content": "",
                    "stop": True,
                    "stopping_word": spec.get("stopping_word", ""),
                    "stop_type": spec.get("stop_type", "word"),
                    "tokens_predicted": spec.get("tokens_predicted", 16),
                    "timings": {"prompt_n": 10, "prompt_ms": 5.0, "predicted_n": 16, "predicted_ms": 80.0, "predicted_per_second": 200.0},
                }
                self.wfile.write(f"data: {json.dumps(done, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class TestSplitPhaseServer(unittest.TestCase):
    def test_strict_five_slider_model_parsing(self) -> None:
        tag, weights = srv.parse_model_weights("xiaofeiyu-v4|1.0,0.85,0,0,0")
        self.assertEqual(tag, "xiaofeiyu-v4|1.000,0.850,0.000,0.000,0.000")
        self.assertEqual(weights, (1.0, 0.85, 0.0, 0.0, 0.0))

        tag_preset, w_preset = srv.parse_model_weights("xiaofeiyu-26b-a4b:tease")
        self.assertEqual(tag_preset, "xiaofeiyu-26b-a4b|1.000,0.000,0.850,0.000,0.000")
        self.assertEqual(w_preset, (1.0, 0.0, 0.85, 0.0, 0.0))

        with self.assertRaises(ValueError):
            srv.parse_model_weights("xiaofeiyu-v4|1,0,0,0,0,1")

    def test_split_phase_zero_lora_brain_and_five_slider_voice(self) -> None:
        mock = _MockLlamaServer()
        tok = _FakeTokenizer()
        engine = srv.LlamaCppPC2Engine(
            llama_client=srv.LlamaCppHTTPClient(host="127.0.0.1", port=mock.port),
            tokenizer=tok,
            min_thought_tokens=12,
        )
        gw = srv.create_pc2_http_server(engine, host="127.0.0.1", port=0)
        gw_port = gw.server_address[1]
        th = threading.Thread(target=gw.serve_forever, daemon=True)
        th.start()

        zero_lora = [{"id": i, "scale": 0.0} for i in range(5)]
        try:
            mock.script = [
                # Phase A kickoff (zero LoRA)
                {"content": "Let me calculate ", "tokens_predicted": 12},
                # Phase A thought (zero LoRA)
                {"chunks": ["split bill."], "stopping_word": "<channel|>", "tokens_predicted": 4},
                # Phase A calc_python (zero LoRA)
                {"chunks": ['call:calc_python{code:<|"|>print(150/3)<|"|>}<tool_call|>'], "stopping_word": "<|tool_response>", "tokens_predicted": 12},
                # Hop 1 Phase A kickoff & thought
                {"content": "Result is 50.0, ", "tokens_predicted": 12},
                {"chunks": ["reply warmly."], "stopping_word": "<channel|>", "tokens_predicted": 4},
                # Hop 1 Phase A tool stop at send_message quote
                {"chunks": [], "stopping_word": 'call:send_message{bubbles:[<|"|>', "tokens_predicted": 5},
                # Hop 1 Phase B (5-slider LoRA active!)
                {"chunks": ['算好啦<|"|>,<|"|>每人50.0元<|"|>'], "stopping_word": "]}<tool_call|>", "tokens_predicted": 10},
                # Hop 1 Phase A trailing check -> <turn|>
                {"chunks": [""], "stopping_word": "<turn|>", "tokens_predicted": 1},
            ]

            conn = http.client.HTTPConnection("127.0.0.1", gw_port, timeout=10.0)
            body = json.dumps(
                {
                    "model": "xiaofeiyu-v4|1.0,0.65,0.0,0.0,0.0",
                    "stream": False,
                    "messages": [{"role": "user", "content": "150块三个人平摊是多少"}],
                }
            )
            conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = json.loads(resp.read().decode("utf-8"))
            conn.close()

            self.assertEqual(resp.status, 200)
            self.assertEqual(data["choices"][0]["message"]["content"], "算好啦\n每人50.0元")

            # Verify Phase A had all 5 LoRAs at 0.0 and Phase B had exact 5-slider weights
            phase_b_reqs = [r for r in mock.requests if any(item["scale"] > 0 for item in r["lora"])]
            self.assertEqual(len(phase_b_reqs), 1)
            self.assertEqual(
                phase_b_reqs[0]["lora"],
                [
                    {"id": 0, "scale": 1.0},
                    {"id": 1, "scale": 0.65},
                    {"id": 2, "scale": 0.0},
                    {"id": 3, "scale": 0.0},
                    {"id": 4, "scale": 0.0},
                ],
            )
            for req in mock.requests:
                self.assertEqual(len(req["lora"]), 5)
            self.assertEqual(mock.requests[0]["lora"], zero_lora)
        finally:
            gw.shutdown()
            gw.server_close()
            mock.close()

    def test_external_send_message_collision_avoidance(self) -> None:
        reg = bridge.build_external_tool_registry(
            [{"type": "function", "function": {"name": "send_message", "parameters": {"type": "object"}}}]
        )
        self.assertEqual(reg.to_gemma_name("send_message"), "agent_send_message")
        calls = bridge.parse_gemma4_tool_calls('<|tool_call>call:agent_send_message{to:<|"|>worker<|"|>}<tool_call|>', reg)
        _, ext = bridge.partition_pc2_and_external_calls(calls, reg)
        self.assertEqual(len(ext), 1)
        self.assertEqual(ext[0].wire_name, "send_message")

    def test_dsh_without_send_message_multihop_tools_and_nameless_tool_role(self) -> None:
        """Verify vanilla DSH (only fs/pwsh tools, no send_message, nameless role=tool) works end-to-end."""
        dsh_tools = [
            {
                "type": "function",
                "function": {
                    "name": "fs_list",
                    "description": "List files in directory",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs_read",
                    "description": "Read file content",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            },
        ]
        reg = bridge.build_external_tool_registry(dsh_tools)
        messages = [
            {"role": "user", "content": "帮我看看当前目录有什么，再读一下 package.json"},
            {
                "role": "assistant",
                "content": "等我翻一下目录哈",
                "tool_calls": [
                    {"id": "call_hop1", "type": "function", "function": {"name": "fs_list", "arguments": '{"path":"."}'}}
                ],
            },
            # Standard OpenAI/DSH tool result with `tool_call_id` and NO `name` field:
            {"role": "tool", "tool_call_id": "call_hop1", "content": "package.json\nsrc/"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_hop2",
                        "type": "function",
                        "function": {"name": "fs_read", "arguments": '{"path":"package.json"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_hop2", "content": '{"name":"demo","version":"1.0.0"}'},
        ]
        prompt_text, _ = bridge.render_gemma4_v4_prompt_with_tools(
            messages, slider_weights=(1.0, 0.0, 0.85, 0.0, 0.0), registry=reg
        )
        # 1) Causality preserved: hop1 call -> hop1 response -> thought -> hop2 call -> hop2 response -> open thought
        idx_c1 = prompt_text.index("call:fs_list{")
        idx_r1 = prompt_text.index("response:fs_list{")
        idx_c2 = prompt_text.index("call:fs_read{")
        idx_r2 = prompt_text.index("response:fs_read{")
        self.assertLess(idx_c1, idx_r1)
        self.assertLess(idx_r1, idx_c2)
        self.assertLess(idx_c2, idx_r2)
        self.assertNotIn("response:tool{", prompt_text)
        # 2) Plain-text assistant reply from DSH is automatically re-wrapped as virtual `send_message`
        self.assertIn('call:send_message{bubbles:[<|"|>等我翻一下目录哈<|"|>]}', prompt_text)
        self.assertTrue(prompt_text.endswith("<|channel>thought\n"))

    def test_launch_local_dry_run_presets(self) -> None:
        for preset in ("26b-a4b", "12b", "e4b"):
            plan = launch_local.build_launch_plan(
                preset_name=preset,
                models_dir=Path("./models"),
                loras_dir=Path("./loras"),
                llama_bin="llama-server",
                host="127.0.0.1",
                port=8081,
                llama_port=8082,
                n_parallel=1,
                ctx_size=None,
                n_gpu_layers=99,
            )
            self.assertEqual(len(plan["lora_files"]), 5)
            self.assertTrue(plan["adapters_hf_repo"].startswith("niatheelf777/"))


if __name__ == "__main__":
    unittest.main()
