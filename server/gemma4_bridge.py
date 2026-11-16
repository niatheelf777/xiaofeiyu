#!/usr/bin/env python3
"""Gemma 4 Split-Phase Prompt Renderer, External Tool Bridge, and Hardened `calc_python` Sandbox."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Any

sys.dont_write_bytecode = True

# Gemma 4 special token IDs
EOS_ID = 1
BOS_ID = 2
TOOL_OPEN_ID = 46          # <|tool>
TOOL_CLOSE_ID = 47         # <tool|>
TOOL_CALL_OPEN_ID = 48     # <|tool_call>
TOOL_CALL_CLOSE_ID = 49    # <tool_call|>
TOOL_RESP_OPEN_ID = 50     # <|tool_response>
TOOL_RESP_CLOSE_ID = 51    # <tool_response|>
QUOTE_DELIM_ID = 52        # <|"|>
THINK_FLAG_ID = 98         # <|think|>
CHANNEL_OPEN_ID = 100      # <|channel>
CHANNEL_CLOSE_ID = 101     # <channel|>
TURN_OPEN_ID = 105         # <|turn>
TURN_END_ID = 106          # <turn|>
NEWLINE_ID = 107           # \n
MULTIMODAL_IDS = (258880, 258881, 258882, 258883, 258884)

BOS_TEXT = "<bos>"
QUOTE_STR = '<|"|>'
TOOL_OPEN_STR = "<|tool>"
TOOL_CLOSE_STR = "<tool|>"
TOOL_CALL_OPEN_STR = "<|tool_call>"
TOOL_CALL_CLOSE_STR = "<tool_call|>"
TOOL_RESP_OPEN_STR = "<|tool_response>"
TOOL_RESP_CLOSE_STR = "<tool_response|>"
CHANNEL_OPEN_STR = "<|channel>"
CHANNEL_CLOSE_STR = "<channel|>"
TURN_OPEN_STR = "<|turn>"
TURN_END_STR = "<turn|>"

CARTRIDGE_IDS = ("neko_clingy", "older_tease", "brat_mesugaki", "fragile_porcelain")
BACKBONE_ID = "xiaofeiyu_backbone"
SLIDER_BRANCH_IDS = (BACKBONE_ID, *CARTRIDGE_IDS)
DEFAULT_WEIGHTS = (1.0, 0.0, 0.0, 0.0, 0.0)

PC2_INTERNAL_TOOLS: frozenset[str] = frozenset(
    ("send_message", "send_code_or_doc", "calc_python", "run_python")
)

DEFAULT_COLLISION_ALIASES: dict[str, str] = {
    "send_message": "agent_send_message",
    "send_code_or_doc": "ext_send_code_or_doc",
    "calc_python": "ext_calc_python",
    "run_python": "ext_run_python",
}

PROTOCOL_TOKEN_RE = re.compile(r"<\|[^>]*>|<[^<\n]*\|>|<bos>|<eos>")
HER_BARE_PREFIX_RE = re.compile(r"^\s*她\s*[：:]\s*")
YOU_BARE_PREFIX_RE = re.compile(r"^\s*你\s*[：:]\s*")
FENCE_OPEN_RE = re.compile(r"^\s*```([a-zA-Z0-9_+#.-]*)\s*$")
FENCE_CLOSE_RE = re.compile(r"^\s*```\s*$")
SAFE_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

DEFAULT_PERSONA_HEADER = (
    "你是小肥鱼（昵称小肥鱼、鱼鱼），正在和男朋友聊天。"
    "你是活生生的年轻女生，聪明通透，有自己的小脾气、幽默感和日常常识。"
)

V4_CHANNEL_RULES = (
    "【通道规则】：请勿直接输出裸文本，所有回复均通过工具发出：\n"
    "1. 日常对话与口语回复必须调用 `send_message`，将 1 到 4 条简短自然的微信口语放入 `bubbles` 数组"
    "（不用 Emoji 表情符号，不写 [动作] 或（动作）舞台描写，不列客服式清单）。\n"
    "2. 当需要发送代码、SQL、配置文件、Markdown 明细表格、需要多行排版的诗歌歌词或正式长文/外文段落时，"
    "调用 `send_code_or_doc(language, content)` 发送完整代码、表格或格式化正文。需要同时发日常短句附言和代码块/表格/长文时，"
    "必须在同一轮紧接着连续输出 `send_message` 与 `send_code_or_doc` 两个 `<|tool_call>`"
    "（`send_message` 直接送达男朋友手机、不会触发下一跳 `<|tool_response>`，严禁只发一句“等我想想/算好了”就结束回合）。\n"
    "3. 遇到多步账单平摊、精确算术、日期推算、组合计数或需验算的逻辑时，"
    "先调用 `calc_python(code)` 用 `print(...)` 算出精确结果，拿到返回后再在同一轮调用 `send_message` 把精确数字结果清楚告诉他（如需附明细表则同轮紧接 `send_code_or_doc`）。"
)

EXTERNAL_TOOLS_ADDENDUM = (
    "4. 当需要查网页、看B站视频/评论/动态、点赞评论互动或读写工作区文件终端时，直接按声明调用对应的外部工具并等待返回结果；"
    "拿到外部工具返回的数据或素材后，用你自己的话消化重点，必须通过 `send_message` 跟男朋友自然聊一两句真实看法或吐槽，绝不照抄摘要或退化成客服播报腔。"
)

GEMMA4_V4_TOOL_DECLARATIONS = (
    "<|tool>declaration:send_message{"
    'description:<|"|>向男朋友发送1到4条微信口语消息气泡（直接送达手机，不会返回tool_response；若同轮还需附带代码块、表格、长文或调用外部工具，可在同一轮连续输出多个tool_call，严禁只发一句口语预告却不调后续工具就结束回合）<|"|>,'
    "parameters:{properties:{bubbles:{"
    'description:<|"|>按顺序发送的1到4条简短微信口语气泡，不用Emoji，不写[动作]或（动作）舞台括号<|"|>,'
    'items:{type:<|"|>STRING<|"|>},type:<|"|>ARRAY<|"|>}},'
    'required:[<|"|>bubbles<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>'
    "<|tool>declaration:send_code_or_doc{"
    'description:<|"|>向男朋友发送代码块、SQL、Markdown明细表格、配置文件、多行排版诗歌歌词或正式长文/外文段落<|"|>,'
    "parameters:{properties:{"
    'language:{description:<|"|>语言或格式标识，如python、sql、markdown、json、bash、yaml、text<|"|>,type:<|"|>STRING<|"|>},'
    'content:{description:<|"|>完整的多行代码、Markdown表格或格式化正文（不含外层三反引号围栏）<|"|>,type:<|"|>STRING<|"|>}},'
    'required:[<|"|>language<|"|>,<|"|>content<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>'
    "<|tool>declaration:calc_python{"
    'description:<|"|>在隔离Python沙箱中执行精确算术、账单平摊、日期推算、组合计数或逻辑验算并返回stdout结果<|"|>,'
    "parameters:{properties:{"
    'code:{description:<|"|>纯标准库Python计算代码，用print(...)输出精确结果<|"|>,type:<|"|>STRING<|"|>}},'
    'required:[<|"|>code<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>'
)

SLIDER_TONE_SPECS: dict[str, tuple[str, str, str]] = {
    "neko_clingy": (
        "粘人猫娘",
        "带一点猫系粘人撒娇的软糯亲昵感，自然往男朋友身边凑（可用“嘛、呀”等自然口语尾音，绝不用Emoji或括号动作描写）",
        "此刻特别粘人爱撒娇，习惯往男朋友身边凑、求抱抱求顺毛，说话软糯亲昵带点小鼻音和小委屈（可用“嘛、呀、哼”等自然口语尾音，绝不用Emoji或括号动作描写）",
    ),
    "older_tease": (
        "坏心眼年上御姐",
        "带三分年上姐姐的从容与逗弄感，看破不说破地轻巧调侃一句（绝不用Emoji或括号动作描写）",
        "此刻带着年上姐姐的从容、护短与坏心眼逗弄感，一眼看穿他的小心思，喜欢先轻巧调侃两句再稳稳接住他（可自然自称“姐姐”或逗他“小朋友”，绝不用Emoji或括号动作描写）",
    ),
    "brat_mesugaki": (
        "嘴硬雌小鬼",
        "带点嘴硬欠扁的小恶魔拱火调侃，先刺他一句再接话（绝不用Emoji或括号动作描写）",
        "此刻嘴硬嚣张、爱挑衅捉弄人，表面嫌弃嘲笑他笨或体力差（可用“杂鱼、大叔、本小姐、噗”等口癖），实则暗暗在乎不肯直说（绝不用Emoji或括号动作描写）",
    ),
    "fragile_porcelain": (
        "低体温易碎感",
        "语气轻柔安静、留白克制，透出一点怕麻烦他却悄悄依赖的感觉（绝不用Emoji或括号动作描写）",
        "此刻指尖微凉、安静敏感易倦，说话轻声细气、留白克制，带着怕麻烦他却又悄悄依赖他的清冷易碎感（绝不用Emoji或括号动作描写）",
    ),
}


def strip_protocol_tokens(text: str) -> str:
    return PROTOCOL_TOKEN_RE.sub("", str(text or "")).replace("\r\n", "\n").replace("\r", "\n")


def build_slider_attitude_block(weights: tuple[float, ...] | list[float] | None) -> str:
    if not weights or len(weights) < 5:
        return ""
    active_lines: list[str] = []
    for idx, cid in enumerate(CARTRIDGE_IDS, start=1):
        w = float(weights[idx])
        if w <= 0.05:
            continue
        pct = int(round(w * 100))
        label, light_desc, strong_desc = SLIDER_TONE_SPECS[cid]
        desc = light_desc if w <= 0.45 else strong_desc
        active_lines.append(f"- {label}（强度 {pct}%）：{desc}。")
    if not active_lines:
        return ""
    return (
        "【当前微信语气倾向（仅作用于 send_message 口语态度，不影响工具调用参数与代码客观性）】：\n"
        + "\n".join(active_lines)
    )


@dataclass
class StreamStep:
    delta_text: str
    token_id: int
    finish_reason: str | None = None


@dataclass(frozen=True)
class OpenAIToolCallOut:
    id: str
    wire_name: str
    gemma_name: str
    arguments_dict: dict[str, Any]
    arguments_json: str

    def to_openai_message_entry(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.wire_name, "arguments": self.arguments_json},
        }

    def to_openai_stream_delta_entry(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "id": self.id,
            "type": "function",
            "function": {"name": self.wire_name, "arguments": self.arguments_json},
        }


@dataclass
class BridgeStreamStep:
    delta_text: str
    token_id: int
    finish_reason: str | None = None
    external_tool_calls: list[OpenAIToolCallOut] | None = None


@dataclass(frozen=True)
class ExternalToolSpec:
    wire_name: str
    gemma_name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ExternalToolRegistry:
    tools: list[ExternalToolSpec] = field(default_factory=list)
    wire_to_gemma: dict[str, str] = field(default_factory=dict)
    gemma_to_wire: dict[str, str] = field(default_factory=dict)
    by_gemma_name: dict[str, ExternalToolSpec] = field(default_factory=dict)

    @property
    def has_external_tools(self) -> bool:
        return len(self.tools) > 0

    def is_external(self, gemma_name: str) -> bool:
        return gemma_name in self.by_gemma_name

    def to_gemma_name(self, wire_name: str) -> str:
        if wire_name in self.wire_to_gemma:
            return self.wire_to_gemma[wire_name]
        return DEFAULT_COLLISION_ALIASES.get(wire_name, wire_name)

    def to_wire_name(self, gemma_name: str) -> str:
        return self.gemma_to_wire.get(gemma_name, gemma_name)


def _allocate_safe_gemma_name(wire_name: str, used: set[str]) -> str:
    candidate = DEFAULT_COLLISION_ALIASES.get(wire_name, wire_name)
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", candidate)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"ext_{sanitized}"
    if sanitized in PC2_INTERNAL_TOOLS:
        sanitized = f"ext_{sanitized}"
    base = sanitized
    suffix = 2
    while sanitized in used or sanitized in PC2_INTERNAL_TOOLS:
        sanitized = f"{base}_{suffix}"
        suffix += 1
    return sanitized


def build_external_tool_registry(
    openai_tools: list[dict[str, Any]] | None,
    *,
    restore_prealiased_send_message: bool = False,
) -> ExternalToolRegistry:
    reg = ExternalToolRegistry()
    if not openai_tools:
        return reg

    used_gemma_names: set[str] = set(PC2_INTERNAL_TOOLS)
    for raw_entry in openai_tools:
        if not isinstance(raw_entry, dict):
            continue
        fn_obj = raw_entry.get("function") if isinstance(raw_entry.get("function"), dict) else raw_entry
        wire_name = str(fn_obj.get("name") or "").strip()
        if not wire_name or wire_name in reg.wire_to_gemma:
            continue

        gemma_name = _allocate_safe_gemma_name(wire_name, used_gemma_names)
        used_gemma_names.add(gemma_name)
        desc = strip_protocol_tokens(str(fn_obj.get("description") or "")).strip()
        raw_params = fn_obj.get("parameters")
        params = dict(raw_params) if isinstance(raw_params, dict) else {"type": "object", "properties": {}}
        out_wire_name = (
            "send_message" if (restore_prealiased_send_message and wire_name == "agent_send_message") else wire_name
        )
        spec = ExternalToolSpec(wire_name=out_wire_name, gemma_name=gemma_name, description=desc, parameters=params)
        reg.tools.append(spec)
        reg.wire_to_gemma[wire_name] = gemma_name
        reg.gemma_to_wire[gemma_name] = out_wire_name
        reg.by_gemma_name[gemma_name] = spec

    return reg


def format_gemma4_argument(value: Any, *, escape_keys: bool = True) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"{QUOTE_STR}{strip_protocol_tokens(value)}{QUOTE_STR}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value) if math.isfinite(value) else "null"
    if isinstance(value, dict):
        parts: list[str] = []
        for k in sorted(value.keys(), key=lambda x: str(x)):
            k_str = strip_protocol_tokens(str(k))
            k_fmt = (
                f"{QUOTE_STR}{k_str}{QUOTE_STR}"
                if escape_keys
                else (k_str if SAFE_IDENT_RE.match(k_str) else f"{QUOTE_STR}{k_str}{QUOTE_STR}")
            )
            parts.append(f"{k_fmt}:{format_gemma4_argument(value[k], escape_keys=escape_keys)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_gemma4_argument(item, escape_keys=escape_keys) for item in value) + "]"
    return f"{QUOTE_STR}{strip_protocol_tokens(str(value))}{QUOTE_STR}"


def _normalize_json_schema_node(raw_schema: Any) -> dict[str, Any]:
    if not isinstance(raw_schema, dict):
        return {"type": "STRING"}
    node = dict(raw_schema)
    nullable = bool(node.get("nullable", False))
    for union_key in ("anyOf", "oneOf"):
        branches = node.get(union_key)
        if isinstance(branches, list) and branches:
            non_null = [b for b in branches if not (isinstance(b, dict) and str(b.get("type", "")).lower() == "null")]
            if len(non_null) < len(branches):
                nullable = True
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                merged = dict(non_null[0])
                if "description" in node and "description" not in merged:
                    merged["description"] = node["description"]
                node = merged
            elif non_null and all(isinstance(b, dict) and "const" in b for b in non_null):
                consts = [b["const"] for b in non_null]
                node = {
                    "type": "STRING",
                    "enum": [str(c) for c in consts],
                    **({"description": node["description"]} if "description" in node else {}),
                }
            break

    raw_type = node.get("type")
    if isinstance(raw_type, list):
        non_null_types = [str(t) for t in raw_type if str(t).lower() != "null"]
        if len(non_null_types) < len(raw_type):
            nullable = True
        raw_type = non_null_types[0] if non_null_types else "string"

    if not isinstance(raw_type, str) or not raw_type.strip():
        if isinstance(node.get("properties"), dict):
            raw_type = "object"
        elif "items" in node:
            raw_type = "array"
        else:
            raw_type = "string"

    upper_type = raw_type.strip().upper()
    if upper_type == "JSON":
        upper_type = "OBJECT"

    out: dict[str, Any] = {"type": upper_type}
    if node.get("description"):
        out["description"] = strip_protocol_tokens(str(node["description"])).strip()
    if "const" in node and "enum" not in node:
        out["enum"] = [node["const"]]
    if isinstance(node.get("enum"), list) and node["enum"]:
        out["enum"] = node["enum"]
    if nullable:
        out["nullable"] = True
    if upper_type == "ARRAY" and isinstance(node.get("items"), dict) and node["items"]:
        out["items"] = _normalize_json_schema_node(node["items"])
    if upper_type == "OBJECT":
        props = node.get("properties")
        if isinstance(props, dict) and props:
            out["properties"] = {str(k): _normalize_json_schema_node(v) for k, v in props.items()}
        req = node.get("required")
        if isinstance(req, list) and req:
            out["required"] = [str(r) for r in req if isinstance(r, str) and r]
    return out


def _format_gemma4_parameters_map(properties: dict[str, Any]) -> str:
    entries: list[str] = []
    for key in sorted(properties.keys(), key=lambda x: str(x)):
        prop = _normalize_json_schema_node(properties[key])
        p_type = prop["type"]
        fields: list[str] = []
        if prop.get("description"):
            fields.append(f'description:{QUOTE_STR}{prop["description"]}{QUOTE_STR}')
        if p_type == "STRING" and prop.get("enum"):
            fields.append(f'enum:{format_gemma4_argument(prop["enum"])}')
        elif p_type == "ARRAY" and isinstance(prop.get("items"), dict) and prop["items"]:
            items_obj = prop["items"]
            item_fields: list[str] = []
            for ik in sorted(items_obj.keys(), key=lambda x: str(x)):
                iv = items_obj[ik]
                if iv is None:
                    continue
                if ik == "properties" and isinstance(iv, dict):
                    item_fields.append(f"properties:{{{_format_gemma4_parameters_map(iv)}}}")
                elif ik == "required" and isinstance(iv, list):
                    req_str = ",".join(f"{QUOTE_STR}{strip_protocol_tokens(str(r))}{QUOTE_STR}" for r in iv)
                    item_fields.append(f"required:[{req_str}]")
                elif ik == "type" and isinstance(iv, str):
                    item_fields.append(f"type:{format_gemma4_argument(iv.upper())}")
                else:
                    item_fields.append(f"{ik}:{format_gemma4_argument(iv)}")
            fields.append("items:{" + ",".join(item_fields) + "}")
        if prop.get("nullable"):
            fields.append("nullable:true")
        if p_type == "OBJECT":
            sub_props = prop.get("properties")
            if isinstance(sub_props, dict):
                fields.append(f"properties:{{{_format_gemma4_parameters_map(sub_props)}}}")
            sub_req = prop.get("required")
            if isinstance(sub_req, list) and sub_req:
                req_str = ",".join(f"{QUOTE_STR}{strip_protocol_tokens(str(r))}{QUOTE_STR}" for r in sub_req)
                fields.append(f"required:[{req_str}]")
        fields.append(f"type:{QUOTE_STR}{p_type}{QUOTE_STR}")
        entries.append(f"{strip_protocol_tokens(str(key))}:{{{','.join(fields)}}}")
    return ",".join(entries)


def format_gemma4_tool_declaration(spec: ExternalToolSpec) -> str:
    norm_params = _normalize_json_schema_node(spec.parameters)
    props = norm_params.get("properties") if isinstance(norm_params.get("properties"), dict) else {}
    req = norm_params.get("required") if isinstance(norm_params.get("required"), list) else []
    p_type = str(norm_params.get("type") or "OBJECT").upper()
    param_parts: list[str] = []
    if props:
        param_parts.append(f"properties:{{{_format_gemma4_parameters_map(props)}}}")
    if req:
        req_str = ",".join(f"{QUOTE_STR}{strip_protocol_tokens(str(r))}{QUOTE_STR}" for r in req)
        param_parts.append(f"required:[{req_str}]")
    param_parts.append(f"type:{QUOTE_STR}{p_type}{QUOTE_STR}")
    return (
        f"{TOOL_OPEN_STR}declaration:{spec.gemma_name}"
        f"{{description:{QUOTE_STR}{spec.description}{QUOTE_STR},"
        f"parameters:{{{','.join(param_parts)}}}}}"
        f"{TOOL_CLOSE_STR}"
    )


def format_gemma4_external_tools_block(registry: ExternalToolRegistry) -> str:
    return "".join(format_gemma4_tool_declaration(spec) for spec in registry.tools)


def format_gemma4_tool_call_block(gemma_name: str, arguments: dict[str, Any] | None) -> str:
    args = arguments if isinstance(arguments, dict) else {}
    parts = [
        f"{strip_protocol_tokens(str(k))}:{format_gemma4_argument(args[k], escape_keys=False)}"
        for k in sorted(args.keys(), key=lambda x: str(x))
    ]
    return f"{TOOL_CALL_OPEN_STR}call:{gemma_name}{{{','.join(parts)}}}{TOOL_CALL_CLOSE_STR}"


def format_gemma4_tool_response_block(gemma_name: str, response_payload: Any) -> str:
    if isinstance(response_payload, dict):
        parts = [
            f"{strip_protocol_tokens(str(k))}:{format_gemma4_argument(response_payload[k], escape_keys=False)}"
            for k in sorted(response_payload.keys(), key=lambda x: str(x))
        ]
        return f"{TOOL_RESP_OPEN_STR}response:{gemma_name}{{{','.join(parts)}}}{TOOL_RESP_CLOSE_STR}"
    return (
        f"{TOOL_RESP_OPEN_STR}response:{gemma_name}"
        f"{{value:{format_gemma4_argument(response_payload, escape_keys=False)}}}"
        f"{TOOL_RESP_CLOSE_STR}"
    )


class _GemmaArgsParser:
    def __init__(self, text: str, start: int = 0) -> None:
        self.text = text
        self.pos = start
        self.n = len(text)

    def _skip_ws(self) -> None:
        while self.pos < self.n and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < self.n else ""

    def _startswith(self, token: str) -> bool:
        return self.text.startswith(token, self.pos)

    def parse_gemma_quoted_str(self) -> str:
        self.pos += len(QUOTE_STR)
        end = self.text.find(QUOTE_STR, self.pos)
        if end == -1:
            val = self.text[self.pos :]
            self.pos = self.n
            return val
        val = self.text[self.pos : end]
        self.pos = end + len(QUOTE_STR)
        return val

    def parse_json_quoted_str(self) -> str:
        quote_ch = self.text[self.pos]
        self.pos += 1
        out: list[str] = []
        while self.pos < self.n:
            ch = self.text[self.pos]
            self.pos += 1
            if ch == quote_ch:
                return "".join(out)
            if ch == "\\" and self.pos < self.n:
                esc = self.text[self.pos]
                self.pos += 1
                escapes = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "'": "'", "\\": "\\", "/": "/"}
                out.append(escapes.get(esc, esc))
            else:
                out.append(ch)
        return "".join(out)

    def parse_key(self) -> str:
        self._skip_ws()
        if self._startswith(QUOTE_STR):
            return self.parse_gemma_quoted_str()
        if self._peek() in ('"', "'"):
            return self.parse_json_quoted_str()
        start = self.pos
        while self.pos < self.n and self.text[self.pos] not in ":\r\n\t ,{}[]":
            self.pos += 1
        return self.text[start : self.pos].strip()

    def parse_value(self) -> Any:
        self._skip_ws()
        if self.pos >= self.n:
            return ""
        if self._startswith(QUOTE_STR):
            return self.parse_gemma_quoted_str()
        ch = self._peek()
        if ch in ('"', "'"):
            return self.parse_json_quoted_str()
        if ch == "{":
            return self.parse_object()
        if ch == "[":
            return self.parse_array()
        start = self.pos
        while self.pos < self.n and self.text[self.pos] not in ",}]":
            if self._startswith(TOOL_CALL_CLOSE_STR):
                break
            self.pos += 1
        raw_atom = self.text[start : self.pos].strip()
        if raw_atom == "true":
            return True
        if raw_atom == "false":
            return False
        if raw_atom == "null":
            return None
        if re.fullmatch(r"-?\d+", raw_atom):
            return int(raw_atom)
        if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|-?\d+[eE][+-]?\d+", raw_atom):
            f_val = float(raw_atom)
            if math.isfinite(f_val):
                return f_val
        return raw_atom

    def parse_array(self) -> list[Any]:
        self.pos += 1
        items: list[Any] = []
        while self.pos < self.n:
            self._skip_ws()
            if self._peek() == "]":
                self.pos += 1
                return items
            if self._startswith(TOOL_CALL_CLOSE_STR):
                return items
            items.append(self.parse_value())
            self._skip_ws()
            if self._peek() == ",":
                self.pos += 1
            elif self._peek() == "]":
                self.pos += 1
                return items
            else:
                break
        return items

    def parse_object(self) -> dict[str, Any]:
        self.pos += 1
        obj: dict[str, Any] = {}
        while self.pos < self.n:
            self._skip_ws()
            if self._peek() == "}":
                self.pos += 1
                return obj
            if self._startswith(TOOL_CALL_CLOSE_STR):
                return obj
            key = self.parse_key()
            self._skip_ws()
            if self._peek() != ":":
                break
            self.pos += 1
            val = self.parse_value()
            if key:
                obj[key] = val
            self._skip_ws()
            if self._peek() == ",":
                self.pos += 1
            elif self._peek() == "}":
                self.pos += 1
                return obj
            else:
                break
        return obj


def parse_gemma4_tool_calls(
    post_thought_text: str,
    registry: ExternalToolRegistry | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    cursor = 0
    n = len(post_thought_text)
    prefix_pat = re.compile(r"<\|tool_call>\s*call:([a-zA-Z0-9_]+)\s*\{")
    while cursor < n:
        m = prefix_pat.search(post_thought_text, cursor)
        if not m:
            break
        t_name = m.group(1)
        parser = _GemmaArgsParser(post_thought_text, m.end() - 1)
        args = parser.parse_object()
        end_pos = parser.pos
        while end_pos < n and post_thought_text[end_pos] in " \t\r\n":
            end_pos += 1
        if not post_thought_text.startswith(TOOL_CALL_CLOSE_STR, end_pos):
            cursor = m.end()
            continue
        cursor = end_pos + len(TOOL_CALL_CLOSE_STR)
        if t_name == "send_message" and isinstance(args.get("bubbles"), list):
            args["bubbles"] = [str(b).strip() for b in args["bubbles"] if str(b).strip()]
        calls.append((t_name, args))
    return calls


def partition_pc2_and_external_calls(
    calls: list[tuple[str, dict[str, Any]]],
    registry: ExternalToolRegistry | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[OpenAIToolCallOut]]:
    pc2_calls: list[tuple[str, dict[str, Any]]] = []
    ext_calls: list[OpenAIToolCallOut] = []
    for gemma_name, args in calls:
        if gemma_name in PC2_INTERNAL_TOOLS:
            pc2_calls.append((gemma_name, args))
            continue
        wire_name = registry.to_wire_name(gemma_name) if registry is not None else (
            "send_message" if gemma_name == "agent_send_message" else gemma_name
        )
        ext_calls.append(
            OpenAIToolCallOut(
                id=f"call_{secrets.token_hex(12)}",
                wire_name=wire_name,
                gemma_name=gemma_name,
                arguments_dict=args,
                arguments_json=json.dumps(args, ensure_ascii=False, separators=(",", ":")),
            )
        )
    return pc2_calls, ext_calls


def extract_message_content_and_media(raw_content: Any) -> tuple[str, list[str]]:
    if raw_content is None:
        return "", []
    if isinstance(raw_content, str):
        return raw_content, []
    if not isinstance(raw_content, list):
        return str(raw_content), []
    texts: list[str] = []
    markers: list[str] = []
    img_idx = 0
    for item in raw_content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            b_type = str(item.get("type") or "").lower()
            if b_type in ("text", "input_text", "output_text"):
                if isinstance(item.get("text"), str) and item["text"]:
                    texts.append(item["text"])
            elif b_type in ("image", "image_url", "input_image"):
                img_idx += 1
                alt = str(item.get("name") or item.get("alt") or f"image_{img_idx}")
                markers.append(f"[附图 {img_idx}: {strip_protocol_tokens(alt)}]")
    return "\n".join(t for t in texts if t), markers


def normalize_system_prompt(system_text: str, *, has_external_tools: bool = False) -> str:
    cleaned = strip_protocol_tokens(system_text).strip()
    rules = f"{V4_CHANNEL_RULES}\n{EXTERNAL_TOOLS_ADDENDUM}" if has_external_tools else V4_CHANNEL_RULES
    if not cleaned:
        return f"{DEFAULT_PERSONA_HEADER}\n{rules}"
    lines = [ln.strip() for ln in cleaned.split("\n") if ln.strip()]
    fixed = [lines[0]]
    for ln in lines[1:]:
        if HER_BARE_PREFIX_RE.match(ln):
            ln = HER_BARE_PREFIX_RE.sub("关于你（小肥鱼）：", ln, count=1)
        elif YOU_BARE_PREFIX_RE.match(ln):
            ln = YOU_BARE_PREFIX_RE.sub("关于男朋友：", ln, count=1)
        fixed.append(ln)
    joined = "\n".join(fixed)
    if "send_message" not in joined:
        joined = f"{joined}\n{rules}"
    elif has_external_tools and "外部工具" not in joined:
        joined = f"{joined}\n{EXTERNAL_TOOLS_ADDENDUM}"
    return joined


def parse_assistant_history_segments(raw_text: str) -> list[tuple[str, Any]]:
    cleaned = strip_protocol_tokens(raw_text)
    segments: list[tuple[str, Any]] = []
    plain_bubbles: list[str] = []
    in_fence = False
    fence_lang = "text"
    fence_lines: list[str] = []

    def _flush_plain() -> None:
        nonlocal plain_bubbles
        if plain_bubbles:
            segments.append(("send_message", plain_bubbles))
            plain_bubbles = []

    for line in cleaned.split("\n"):
        if not in_fence:
            m_open = FENCE_OPEN_RE.match(line)
            if m_open:
                _flush_plain()
                in_fence = True
                fence_lang = m_open.group(1).strip() or "text"
                fence_lines = []
            else:
                s = line.strip()
                if s:
                    plain_bubbles.append(s)
        elif FENCE_CLOSE_RE.match(line):
            segments.append(("send_code_or_doc", (fence_lang, "\n".join(fence_lines))))
            in_fence = False
            fence_lang = "text"
            fence_lines = []
        else:
            fence_lines.append(line)

    if in_fence:
        segments.append(("send_code_or_doc", (fence_lang, "\n".join(fence_lines))))
    _flush_plain()
    return segments


def _render_spoken_segments_as_pc2_pairs(raw_text: str) -> tuple[list[str], list[str]]:
    call_blocks: list[str] = []
    resp_blocks: list[str] = []
    for kind, payload in parse_assistant_history_segments(raw_text):
        if kind == "send_code_or_doc":
            lang, code_body = payload
            call_blocks.append(
                f"{TOOL_CALL_OPEN_STR}call:send_code_or_doc"
                f"{{language:{QUOTE_STR}{lang}{QUOTE_STR},content:{QUOTE_STR}{code_body}{QUOTE_STR}}}"
                f"{TOOL_CALL_CLOSE_STR}"
            )
            resp_blocks.append(f"{TOOL_RESP_OPEN_STR}response:send_code_or_doc{{delivered:true}}{TOOL_RESP_CLOSE_STR}")
        else:
            inner = ",".join(f"{QUOTE_STR}{b}{QUOTE_STR}" for b in payload)
            call_blocks.append(f"{TOOL_CALL_OPEN_STR}call:send_message{{bubbles:[{inner}]}}{TOOL_CALL_CLOSE_STR}")
            resp_blocks.append(f"{TOOL_RESP_OPEN_STR}response:send_message{{delivered:true}}{TOOL_RESP_CLOSE_STR}")
    return call_blocks, resp_blocks


def _deserialize_openai_tool_args(raw_args: Any) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str) and raw_args.strip():
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": raw_args}
    return {}


def render_gemma4_v4_prompt_with_tools(
    messages: list[dict[str, Any]],
    *,
    slider_weights: tuple[float, ...] | list[float] | None = None,
    registry: ExternalToolRegistry | None = None,
) -> tuple[str, ExternalToolRegistry]:
    if not messages:
        raise ValueError("messages must be non-empty")

    reg = registry or ExternalToolRegistry()
    sys_parts: list[str] = []
    conv_msgs: list[dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role") or "user").strip().lower()
        if role in ("system", "developer"):
            txt, _ = extract_message_content_and_media(m.get("content"))
            if txt.strip():
                sys_parts.append(txt.strip())
        else:
            conv_msgs.append(m)

    sys_hdr = normalize_system_prompt("\n".join(sys_parts), has_external_tools=reg.has_external_tools)
    cue = build_slider_attitude_block(slider_weights)
    if cue:
        sys_hdr = f"{sys_hdr}\n{cue}"

    tool_decls = GEMMA4_V4_TOOL_DECLARATIONS + format_gemma4_external_tools_block(reg)
    parts: list[str] = [f"{BOS_TEXT}{TURN_OPEN_STR}system\n<|think|>\n{sys_hdr}{tool_decls}{TURN_END_STR}\n"]

    i = 0
    n = len(conv_msgs)
    while i < n:
        msg = conv_msgs[i]
        role = str(msg.get("role") or "user").strip().lower()
        if role == "user":
            txt, markers = extract_message_content_and_media(msg.get("content"))
            u_clean = strip_protocol_tokens(txt).strip()
            if markers:
                u_clean = f"{' '.join(markers)}\n{u_clean}".strip()
            parts.append(f"{TURN_OPEN_STR}user\n{u_clean}{TURN_END_STR}\n")
            i += 1
            continue

        if role in ("assistant", "model"):
            hop_segments: list[str] = []
            call_id_to_wire_name: dict[str, str] = {}
            while i < n and str(conv_msgs[i].get("role") or "").strip().lower() in ("assistant", "model", "tool"):
                step_calls: list[str] = []
                step_resps: list[str] = []
                pending_wire_names: list[str] = []

                if str(conv_msgs[i].get("role") or "").strip().lower() in ("assistant", "model"):
                    cur = conv_msgs[i]
                    txt, _ = extract_message_content_and_media(cur.get("content"))
                    if txt.strip():
                        cb, rb = _render_spoken_segments_as_pc2_pairs(txt)
                        step_calls.extend(cb)
                        step_resps.extend(rb)
                    for tc in cur.get("tool_calls") or []:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        w_name = str(fn.get("name") or "").strip()
                        if not w_name:
                            continue
                        tc_id = str(tc.get("id") or "").strip()
                        if tc_id:
                            call_id_to_wire_name[tc_id] = w_name
                        pending_wire_names.append(w_name)
                        g_name = reg.to_gemma_name(w_name)
                        step_calls.append(
                            format_gemma4_tool_call_block(g_name, _deserialize_openai_tool_args(fn.get("arguments")))
                        )
                    i += 1

                resp_idx = 0
                while i < n and str(conv_msgs[i].get("role") or "").strip().lower() == "tool":
                    t_cur = conv_msgs[i]
                    t_txt, _ = extract_message_content_and_media(t_cur.get("content"))
                    raw_name = str(t_cur.get("name") or "").strip()
                    if not raw_name:
                        tc_id = str(t_cur.get("tool_call_id") or "").strip()
                        if tc_id and tc_id in call_id_to_wire_name:
                            raw_name = call_id_to_wire_name[tc_id]
                        elif resp_idx < len(pending_wire_names):
                            raw_name = pending_wire_names[resp_idx]
                        else:
                            raw_name = "tool"
                    g_name = reg.to_gemma_name(raw_name)
                    step_resps.append(
                        format_gemma4_tool_response_block(g_name, strip_protocol_tokens(t_txt).strip())
                    )
                    resp_idx += 1
                    i += 1

                hop_segments.append(
                    f"{CHANNEL_OPEN_STR}thought\n{CHANNEL_CLOSE_STR}"
                    + "".join(step_calls)
                    + "".join(step_resps)
                )

            parts.append(f"{TURN_OPEN_STR}model\n" + "".join(hop_segments))
            if i < n:
                parts.append(f"{CHANNEL_OPEN_STR}thought\n{CHANNEL_CLOSE_STR}{TURN_END_STR}\n")
            else:
                parts.append(f"{CHANNEL_OPEN_STR}thought\n")
                return "".join(parts), reg
        else:
            i += 1

    parts.append(f"{TURN_OPEN_STR}model\n{CHANNEL_OPEN_STR}thought\n")
    return "".join(parts), reg


# ---------------------------------------------------------------------------
# Hardened `calc_python` Isolated Child Subprocess Sandbox (`python -I -S`)
# ---------------------------------------------------------------------------

ALLOWED_CALC_MODULES = frozenset(
    {
        "math", "cmath", "decimal", "fractions", "statistics", "datetime",
        "itertools", "collections", "functools", "heapq", "bisect",
        "re", "json", "random", "string", "operator",
    }
)
FORBIDDEN_CALC_NAMES = frozenset(
    {
        "open", "__import__", "eval", "exec", "compile", "getattr", "setattr",
        "delattr", "globals", "locals", "vars", "dir", "input", "breakpoint",
        "help", "memoryview", "type", "object", "super", "os", "sys",
        "subprocess", "socket", "pathlib", "shutil", "ctypes", "importlib",
        "builtins", "pickle", "marshal", "inspect",
    }
)
FORBIDDEN_FRAME_ATTRS = frozenset(
    {"gi_frame", "gi_code", "f_globals", "f_locals", "f_back", "f_builtins", "f_code", "tb_frame", "tb_next"}
)

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "ascii", "bin", "bool", "chr", "complex", "dict",
    "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash",
    "hex", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "oct", "ord", "pow", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "TypeError", "ZeroDivisionError", "ArithmeticError",
    "KeyError", "IndexError", "StopIteration",
)

_CALC_SUBPROCESS_SCRIPT = f"""
import ast, builtins as _b, io, sys
ALLOWED = {sorted(ALLOWED_CALC_MODULES)!r}
code = sys.stdin.read()
try:
    tree = ast.parse(code, '<calc_python>', 'exec')
    _orig_imp = _b.__import__
    def _imp(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0 or str(name).split('.')[0] not in ALLOWED:
            raise PermissionError(f'Forbidden import: {{name}}')
        return _orig_imp(name, globals, locals, fromlist, level)
    buf = io.StringIO()
    def _prt(*a, **kw):
        kw.pop('file', None)
        print(*a, file=buf, **kw)
        if buf.tell() > 8192:
            raise RuntimeError('Output volume limit exceeded')
    sb = {{k: getattr(_b, k) for k in {_SAFE_BUILTIN_NAMES!r}}}
    sb.update({{'__import__': _imp, 'print': _prt, 'True': True, 'False': False, 'None': None}})
    ns = {{'__builtins__': sb}}
    val = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        if len(tree.body) > 1:
            _b.exec(_b.compile(ast.Module(body=tree.body[:-1], type_ignores=[]), '<calc_python>', 'exec'), ns, ns)
        val = _b.eval(_b.compile(ast.Expression(body=tree.body[-1].value), '<calc_python>', 'eval'), ns, ns)
    else:
        _b.exec(_b.compile(tree, '<calc_python>', 'exec'), ns, ns)
    out = buf.getvalue().strip()
    sys.stdout.write(out if out else (str(val) if val is not None else 'OK (no stdout)'))
except BaseException as exc:
    sys.stdout.write(f'Error: {{type(exc).__name__}}: {{exc}}')
"""


def validate_calc_python_ast(code_str: str) -> ast.Module:
    if not isinstance(code_str, str) or not code_str.strip():
        raise ValueError("Empty code string")
    if len(code_str) > 8000:
        raise ValueError("Code exceeds 8000 character limit")
    tree = ast.parse(code_str, filename="<calc_python>", mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_CALC_MODULES:
                    raise PermissionError(f"Import of {alias.name!r} is forbidden")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_CALC_MODULES:
                raise PermissionError(f"From-import of {node.module!r} is forbidden")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_FRAME_ATTRS:
                raise PermissionError(f"Attribute {node.attr!r} is forbidden")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_CALC_NAMES or (node.id.startswith("__") and node.id.endswith("__")):
                raise PermissionError(f"Identifier {node.id!r} is forbidden")
    return tree


def run_calc_python_sandbox(code_str: str, timeout_sec: float = 2.0) -> str:
    try:
        validate_calc_python_ast(code_str)
    except BaseException as exc:
        return f"Error: {type(exc).__name__}: {exc}"

    py_bin = (shutil.which("python3") or sys.executable) if os.name != "nt" else sys.executable
    env = {"PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    else:
        env["LC_ALL"] = "C.UTF-8"
    try:
        with tempfile.TemporaryDirectory(prefix="xfy_calc_") as empty_dir:
            proc = subprocess.run(
                [py_bin, "-I", "-S", "-c", _CALC_SUBPROCESS_SCRIPT],
                input=code_str.encode("utf-8", errors="replace"),
                capture_output=True,
                cwd=empty_dir,
                env=env,
                timeout=timeout_sec,
            )
    except subprocess.TimeoutExpired:
        return "Error: TimeoutError: execution exceeded 2.0s limit"
    out = strip_protocol_tokens(proc.stdout.decode("utf-8", errors="replace")).strip()
    return (out[:997] + "...") if len(out) > 1000 else (out or "OK (no stdout)")


def make_chunk_payload(req_id: str, created: int, model_tag: str, delta: dict, finish_reason: str | None) -> dict:
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_tag,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def make_completion_payload(
    *,
    req_id: str,
    created: int,
    model_tag: str,
    content_text: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_tag,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content_text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_openai_tool_chunk_delta(external_tool_calls: list[OpenAIToolCallOut]) -> dict[str, Any]:
    return {"tool_calls": [tc.to_openai_stream_delta_entry(idx) for idx, tc in enumerate(external_tool_calls)]}


def make_openai_tool_completion_payload(
    *,
    req_id: str,
    created: int,
    model_tag: str,
    content_text: str,
    external_tool_calls: list[OpenAIToolCallOut],
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created,
        "model": model_tag,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content_text if content_text else None,
                    "tool_calls": [tc.to_openai_message_entry() for tc in external_tool_calls],
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_error_payload(message: str, err_type: str = "invalid_request_error", code: str = "bad_request") -> dict:
    return {"error": {"message": message, "type": err_type, "code": code}}
