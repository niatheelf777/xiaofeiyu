#!/usr/bin/env python3
"""Xiaofeiyu-12B-v4 HuggingFace ZeroGPU (48GB Blackwell PyTorch BF16) Split-Phase App.

Why in-process PyTorch BF16 on ZeroGPU:
  On HuggingFace ZeroGPU, GPU handles are dynamically injected inside `@spaces.GPU`
  forked worker contexts. External C++ `llama-server` subprocesses cannot access the
  forked GPU cleanly, whereas in-process PyTorch `bfloat16` gets the full `48 GB` VRAM
  slice.

This app:
  1. Loads `google/gemma-4-12B-it` in `bfloat16` (stripping unused vision/audio towers).
  2. Reads the exact `lora_a` / `lora_b` tensors directly from the 5 `24.3 MiB` `.gguf`
     cartridges in `niatheelf777/xiaofeiyu-12b-v4` and mounts them onto layers `36..47`
     (`o_proj`, `gate_proj`, `up_proj`, `down_proj`, `alpha/r = 2.0`).
  3. Runs strict Split-Phase decoding inside `@spaces.GPU(duration=20)`:
     - Phase A (`g_voice = 0.0`): `<|channel>thought`, `calc_python`, `send_code_or_doc`.
     - Phase B (`g_voice = 1.0`): 5-slider LoRA mixing exclusively inside
       `send_message{bubbles:[<|"|>...<|"|>]}`.
"""
from __future__ import annotations

import ast
import contextlib
import gc
import html
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterator

import gradio as gr
import numpy as np

try:
    import spaces
except ImportError:  # Allows local CPU/unit import checks without ZeroGPU runtime
    class _SpacesStub:
        @staticmethod
        def GPU(duration: int = 20):
            def _dec(fn):
                return fn
            return _dec
    spaces = _SpacesStub()

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

sys.dont_write_bytecode = True

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "google/gemma-4-12B-it")
ADAPTERS_REPO_ID = os.environ.get("ADAPTERS_REPO_ID", "niatheelf777/xiaofeiyu-12b-v4")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

FIVE_CARTRIDGE_FILES: tuple[str, ...] = (
    "s0_xiaofeiyu_backbone.gguf",
    "c1_neko_clingy.gguf",
    "c2_older_tease.gguf",
    "c3_brat_mesugaki.gguf",
    "c4_fragile_porcelain.gguf",
)

PERSONA_LABELS: tuple[tuple[str, str, str], ...] = (
    ("s0", "元气日常", "自然松弛的日常朋友搭子，短句接话不端架子"),
    ("c1", "甜系软萌", "软糯粘人、直球撒娇的高温度陪伴"),
    ("c2", "知性姐姐", "从容温柔、游刃有余的姐姐感与包纳感"),
    ("c3", "傲娇毒舌", "嘴硬心软、欢喜冤家式的机锋吐槽"),
    ("c4", "安静慢热", "轻声细语、细腻克制的慢热倾听"),
)

PRESET_WEIGHTS: dict[str, tuple[float, float, float, float, float]] = {
    "元气日常": (1.0, 0.0, 0.0, 0.0, 0.0),
    "甜系软萌": (1.0, 0.85, 0.0, 0.0, 0.0),
    "知性姐姐": (1.0, 0.0, 0.85, 0.0, 0.0),
    "傲娇毒舌": (1.0, 0.0, 0.0, 0.85, 0.0),
    "安静慢热": (1.0, 0.0, 0.0, 0.0, 0.85),
}

# Gemma 4 special token IDs & strings
EOS_ID = 1
BOS_ID = 2
TOOL_CALL_OPEN_ID = 48
TOOL_CALL_CLOSE_ID = 49
TOOL_RESP_OPEN_ID = 50
TOOL_RESP_CLOSE_ID = 51
QUOTE_DELIM_ID = 52
THINK_FLAG_ID = 98
CHANNEL_OPEN_ID = 100
CHANNEL_CLOSE_ID = 101
TURN_OPEN_ID = 105
TURN_END_ID = 106
NEWLINE_ID = 107
MULTIMODAL_IDS = (258880, 258881, 258882, 258883, 258884)

QUOTE_STR = '<|"|>'
TOOL_CALL_OPEN_STR = "<|tool_call>"
TOOL_CALL_CLOSE_STR = "<tool_call|>"
TOOL_RESP_OPEN_STR = "<|tool_response>"
TOOL_RESP_CLOSE_STR = "<tool_response|>"
CHANNEL_OPEN_STR = "<|channel>"
CHANNEL_CLOSE_STR = "<channel|>"
TURN_OPEN_STR = "<|turn>"
TURN_END_STR = "<turn|>"

SEND_MSG_STOP_VARIANTS = (
    'call:send_message{bubbles:[<|"|>',
    'call:send_message{bubbles: [<|"|>',
)
PHASE_A_STOPS = [*SEND_MSG_STOP_VARIANTS, TOOL_RESP_OPEN_STR, TURN_END_STR]
PHASE_B_STOPS = ["]}<tool_call|>", "] }<tool_call|>", TOOL_CALL_CLOSE_STR, TOOL_RESP_OPEN_STR, TURN_END_STR]

DEFAULT_PERSONA_HEADER = (
    "你是小肥鱼（昵称小肥鱼、鱼鱼），正在和朋友聊天。"
    "你自然真实、聪明通透，有自己的小脾气、幽默感和日常常识。"
)

V4_CHANNEL_RULES = (
    "【通道规则】：请勿直接输出裸文本，所有回复均通过工具发出：\n"
    "1. 日常对话与口语回复必须调用 `send_message`，将 1 到 4 条简短自然的微信口语放入 `bubbles` 数组"
    "（不用 Emoji 表情符号，不写 [动作] 或（动作）舞台描写，不列客服式清单）。\n"
    "2. 当需要发送代码、SQL、配置文件、Markdown 明细表格或长文时，"
    "调用 `send_code_or_doc(language, content)` 发送完整代码或格式化正文。需要同时发短句附言和代码块时，"
    "必须在同一轮紧接着连续输出 `send_message` 与 `send_code_or_doc` 两个 `<|tool_call>`。\n"
    "3. 遇到精确算术、账单平摊、日期推算或逻辑验算时，"
    "先调用 `calc_python(code)` 用 `print(...)` 算出精确结果，拿到返回后再在同一轮调用 `send_message` 回复。"
)

GEMMA4_V4_TOOL_DECLARATIONS = (
    "<|tool>declaration:send_message{"
    'description:<|"|>发送1到4条微信口语消息气泡（直接送达手机；若同轮需附带代码块，可在send_message后紧接输出send_code_or_doc）<|"|>,'
    "parameters:{properties:{bubbles:{"
    'description:<|"|>按顺序发送的1到4条简短口语气泡，不用Emoji，不写舞台括号<|"|>,'
    'items:{type:<|"|>STRING<|"|>},type:<|"|>ARRAY<|"|>}},'
    'required:[<|"|>bubbles<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>'
    "<|tool>declaration:send_code_or_doc{"
    'description:<|"|>发送代码块、SQL、Markdown表格、配置文件或格式化正文<|"|>,'
    "parameters:{properties:{"
    'language:{description:<|"|>语言标识，如python、sql、markdown、json、bash、text<|"|>,type:<|"|>STRING<|"|>},'
    'content:{description:<|"|>完整多行代码或格式化正文（不含外层三反引号围栏）<|"|>,type:<|"|>STRING<|"|>}},'
    'required:[<|"|>language<|"|>,<|"|>content<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>'
    "<|tool>declaration:calc_python{"
    'description:<|"|>在隔离Python沙箱中执行精确算术、账单平摊或逻辑验算并返回stdout结果<|"|>,'
    "parameters:{properties:{"
    'code:{description:<|"|>纯标准库Python计算代码，用print(...)输出精确结果<|"|>,type:<|"|>STRING<|"|>}},'
    'required:[<|"|>code<|"|>],type:<|"|>OBJECT<|"|>}}<tool|>'
)

SLIDER_TONE_SPECS: dict[int, tuple[str, str, str]] = {
    1: (
        "甜系软萌",
        "带一点软糯甜美的亲昵撒娇感，说话轻快贴心（绝不用Emoji或括号动作描写）",
        "此刻特别软萌粘人，习惯直球撒娇、求顺毛和贴心关心（可用“嘛、呀”等自然口语尾音，绝不用Emoji或括号动作描写）",
    ),
    2: (
        "知性姐姐",
        "带三分知性姐姐的从容与轻巧调侃感，稳稳接住对话（绝不用Emoji或括号动作描写）",
        "此刻带着知性姐姐的从容、包纳与温和逗弄感，一眼看穿对方心思，喜欢轻巧调侃两句再稳稳安抚（绝不用Emoji或括号动作描写）",
    ),
    3: (
        "傲娇毒舌",
        "带点嘴硬心软的机锋吐槽，先轻刺一句再好好接话（绝不用Emoji或括号动作描写）",
        "此刻嘴硬心软、爱欢喜冤家式斗嘴吐槽，表面略微嫌弃调侃，实则格外靠谱上心（绝不用Emoji或括号动作描写）",
    ),
    4: (
        "安静慢热",
        "语气轻柔安静、留白克制，细腻温和地倾听回应（绝不用Emoji或括号动作描写）",
        "此刻轻声细语、安静细腻且略带慢热腼腆，说话简短柔和、留白克制（绝不用Emoji或括号动作描写）",
    ),
}

PROTOCOL_TOKEN_RE = re.compile(
    r"<\|(?:tool|tool_call|tool_response|channel|turn|think|\"|\|)[^>\n]*>|"
    r"<(?:tool|tool_call|tool_response|channel|turn)\|>|<bos>|<eos>"
)

# ---------------------------------------------------------------------------
# 1. Deterministic GGUF LoRA Tensor Reader (`blk.{36..47}` -> L36..47 PyTorch)
# ---------------------------------------------------------------------------

GGUF_TENSOR_RE = re.compile(
    r"^blk\.(\d+)\.(attn_output|ffn_gate|ffn_up|ffn_down)\.weight\.(lora_a|lora_b)$"
)
GGUF_TO_HF_LEAF: dict[str, str] = {
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}
LAYER_NAME_RE = re.compile(r"^(?:model\.)?language_model\.layers\.(\d+)\.([a-z0-9_.]+)$")


def _read_gguf_tensors_pure_python(gguf_path: Path) -> tuple[float, dict[str, np.ndarray]]:
    """Fallback pure-Python GGUF v3 reader supporting F32 (0), F16 (1), BF16 (30)."""
    with gguf_path.open("rb") as f:
        magic, _ver, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic {magic!r} in {gguf_path}")

        def _read_str() -> str:
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln).decode("utf-8")

        def _read_val(vt: int) -> Any:
            if vt == 0: return struct.unpack("<B", f.read(1))[0]
            if vt == 1: return struct.unpack("<b", f.read(1))[0]
            if vt == 2: return struct.unpack("<H", f.read(2))[0]
            if vt == 3: return struct.unpack("<h", f.read(2))[0]
            if vt == 4: return struct.unpack("<I", f.read(4))[0]
            if vt == 5: return struct.unpack("<i", f.read(4))[0]
            if vt == 6: return struct.unpack("<f", f.read(4))[0]
            if vt == 7: return bool(struct.unpack("<B", f.read(1))[0])
            if vt == 8: return _read_str()
            if vt == 9:
                et, ln = struct.unpack("<IQ", f.read(12))
                return [_read_val(et) for _ in range(ln)]
            if vt == 10: return struct.unpack("<Q", f.read(8))[0]
            if vt == 11: return struct.unpack("<q", f.read(8))[0]
            if vt == 12: return struct.unpack("<d", f.read(8))[0]
            raise ValueError(f"Unsupported GGUF metadata type {vt}")

        alpha = 32.0
        align = 32
        for _ in range(n_kv):
            k = _read_str()
            vt = struct.unpack("<I", f.read(4))[0]
            v = _read_val(vt)
            if k == "adapter.lora.alpha":
                alpha = float(v)
            elif k == "general.alignment":
                align = int(v)

        specs: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            name = _read_str()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            ttype, offset = struct.unpack("<IQ", f.read(12))
            specs.append((name, dims, ttype, offset))

        pos = f.tell()
        data_start = ((pos + align - 1) // align) * align
        arrays: dict[str, np.ndarray] = {}
        for name, dims, ttype, offset in specs:
            np_shape = tuple(reversed(dims))
            numel = int(np.prod(np_shape))
            f.seek(data_start + offset)
            if ttype == 0:  # F32
                arr = np.frombuffer(f.read(numel * 4), dtype="<f4").reshape(np_shape).astype(np.float32)
            elif ttype == 1:  # F16
                arr = np.frombuffer(f.read(numel * 2), dtype="<f2").reshape(np_shape).astype(np.float32)
            elif ttype == 30:  # BF16
                u16 = np.frombuffer(f.read(numel * 2), dtype="<u2").reshape(np_shape)
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            else:
                raise ValueError(f"Unsupported GGUF tensor type {ttype} for {name}")
            arrays[name] = arr.copy()
        return alpha, arrays


def load_gguf_lora_cartridge(gguf_path: Path) -> tuple[float, int, dict[str, dict[str, np.ndarray]]]:
    """Read a GGUF LoRA cartridge into `{module_suffix: {'A': [r, in], 'B': [out, r]}}`."""
    gguf_path = Path(gguf_path)
    alpha = 32.0
    raw_arrays: dict[str, np.ndarray] = {}

    try:
        import gguf  # type: ignore[import-not-found]

        reader = gguf.GGUFReader(str(gguf_path))
        alpha_field = reader.fields.get("adapter.lora.alpha")
        if alpha_field is not None:
            alpha = float(alpha_field.contents())
        for tensor in reader.tensors:
            data = np.asarray(tensor.data)
            if str(tensor.tensor_type.name).upper() == "BF16" and data.dtype == np.uint16:
                data = (data.astype(np.uint32) << 16).view(np.float32)
            else:
                data = data.astype(np.float32)
            raw_arrays[str(tensor.name)] = data
    except ImportError:
        alpha, raw_arrays = _read_gguf_tensors_pure_python(gguf_path)

    grouped: dict[str, dict[str, np.ndarray]] = {}
    inferred_rank: int | None = None
    for t_name, arr in raw_arrays.items():
        m = GGUF_TENSOR_RE.match(t_name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        gguf_leaf = m.group(2)
        ab_key = "A" if m.group(3) == "lora_a" else "B"
        mod_suffix = f"layers.{layer_idx}.{GGUF_TO_HF_LEAF[gguf_leaf]}"
        grouped.setdefault(mod_suffix, {})[ab_key] = arr
        if ab_key == "A" and inferred_rank is None:
            inferred_rank = int(arr.shape[0])

    if not grouped or inferred_rank is None:
        raise ValueError(f"No valid L36..47 LoRA tensors found in {gguf_path}")
    for mod_suffix, pair in grouped.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"Incomplete A/B pair for {mod_suffix} in {gguf_path}")
        if pair["A"].shape[0] != inferred_rank or pair["B"].shape[1] != inferred_rank:
            raise ValueError(
                f"Rank mismatch in {gguf_path} for {mod_suffix}: A={pair['A'].shape}, B={pair['B'].shape}"
            )
    return alpha, inferred_rank, grouped


# ---------------------------------------------------------------------------
# 2. Thread-Local Split-Phase Gate & 5-Slider Linear Wrapper (`L36..47`)
# ---------------------------------------------------------------------------

class _GateContext(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.sliders: Any = None
        self.g_voice: float = 0.0


_GATE_CTX = _GateContext()


@contextlib.contextmanager
def split_phase_gate_scope(sliders: list[float] | tuple[float, ...], g_voice: float) -> Iterator[None]:
    prev_s, prev_gv = _GATE_CTX.sliders, _GATE_CTX.g_voice
    try:
        _GATE_CTX.sliders = tuple(float(x) for x in sliders)
        _GATE_CTX.g_voice = float(g_voice)
        yield
    finally:
        _GATE_CTX.sliders = prev_s
        _GATE_CTX.g_voice = prev_gv


if HAS_TORCH:
    class FrozenFiveSliderLinear(nn.Module):
        """Non-QKV `L36..47` linear hook holding stacked 5-slot GGUF LoRA adapters (`scaling = alpha / r`)."""

        def __init__(
            self,
            base: nn.Linear,
            a_stack: torch.Tensor,
            b_stack: torch.Tensor,
            scaling: float = 2.0,
        ) -> None:
            super().__init__()
            self.base = base
            self.base.requires_grad_(False)
            self.scaling = float(scaling)
            dev = base.weight.device
            dt = base.weight.dtype if base.weight.dtype.is_floating_point else torch.bfloat16
            self.register_buffer("a_stack", a_stack.to(device=dev, dtype=dt), persistent=False)
            self.register_buffer("b_stack", b_stack.to(device=dev, dtype=dt), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.base(x)
            gv = _GATE_CTX.g_voice
            if gv == 0.0 or _GATE_CTX.sliders is None:
                return out
            weights = _GATE_CTX.sliders
            if max(abs(w) for w in weights) < 1e-6:
                return out
            orig_dtype = x.dtype
            w_eff = torch.tensor(
                [w * self.scaling * gv for w in weights],
                device=x.device,
                dtype=self.a_stack.dtype,
            )
            xb = x.to(self.a_stack.dtype)
            if xb.dim() == 2:
                xb = xb.unsqueeze(0)
                r_mid = torch.einsum("bsi,nri->bsnr", xb, self.a_stack)
                delta = torch.einsum("bsnr,nor,n->bso", r_mid, self.b_stack, w_eff).squeeze(0).to(orig_dtype)
            else:
                r_mid = torch.einsum("bsi,nri->bsnr", xb, self.a_stack)
                delta = torch.einsum("bsnr,nor,n->bso", r_mid, self.b_stack, w_eff).to(orig_dtype)
            return out + delta


def _set_submodule(root: Any, dotted_name: str, new_mod: Any) -> None:
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def attach_five_gguf_loras(model: Any, gguf_paths: list[Path]) -> dict[str, Any]:
    """Load 5 `.gguf` LoRAs and mount `FrozenFiveSliderLinear` onto exact 48 modules (`L36..47`)."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to attach LoRA hooks")
    if len(gguf_paths) != 5:
        raise ValueError(f"Expected 5 GGUF cartridges, got {len(gguf_paths)}")

    parsed = [load_gguf_lora_cartridge(p) for p in gguf_paths]
    alphas = [p[0] for p in parsed]
    ranks = [p[1] for p in parsed]
    if len(set(ranks)) != 1 or len(set(alphas)) != 1:
        raise ValueError(f"Mismatched GGUF LoRA rank/alpha across slots: ranks={ranks}, alphas={alphas}")
    scaling = float(alphas[0]) / float(ranks[0])
    rank = int(ranks[0])

    model.requires_grad_(False)
    named_linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    wrapped_count = 0

    for full_name, lin in named_linears:
        rel = full_name[len("model.") :] if full_name.startswith("model.") else full_name
        m = LAYER_NAME_RE.match(rel)
        if not m:
            continue
        layer_idx = int(m.group(1))
        suffix = f"layers.{layer_idx}.{m.group(2)}"
        if suffix not in parsed[0][2]:
            continue

        a_list: list[torch.Tensor] = []
        b_list: list[torch.Tensor] = []
        for _, _, mod_map in parsed:
            pair = mod_map[suffix]
            a_np = pair["A"]
            b_np = pair["B"]
            if a_np.shape != (rank, lin.in_features):
                raise ValueError(
                    f"Shape mismatch for {full_name} A: got {a_np.shape}, expected {(rank, lin.in_features)}"
                )
            if b_np.shape != (lin.out_features, rank):
                raise ValueError(
                    f"Shape mismatch for {full_name} B: got {b_np.shape}, expected {(lin.out_features, rank)}"
                )
            a_list.append(torch.from_numpy(a_np))
            b_list.append(torch.from_numpy(b_np))

        wrapped = FrozenFiveSliderLinear(lin, torch.stack(a_list, dim=0), torch.stack(b_list, dim=0), scaling=scaling)
        _set_submodule(model, full_name, wrapped)
        wrapped_count += 1

    if wrapped_count != 48:
        raise RuntimeError(f"Expected 48 L36..47 non-QKV wrapped modules, got {wrapped_count}")
    return {"wrapped_modules": wrapped_count, "rank": rank, "alpha": alphas[0], "scaling": scaling}


# ---------------------------------------------------------------------------
# 3. Gemma 4 Prompt Renderer, Tool Parsers & Hardened `calc_python` Sandbox
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


def run_calc_python_sandbox(code_str: str, timeout_sec: float = 2.0) -> str:
    try:
        if not isinstance(code_str, str) or not code_str.strip():
            raise ValueError("Empty code string")
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
    except BaseException as exc:
        return f"Error: {type(exc).__name__}: {exc}"

    try:
        with tempfile.TemporaryDirectory(prefix="xfy_calc_") as empty_dir:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _CALC_SUBPROCESS_SCRIPT],
                input=code_str.encode("utf-8", errors="replace"),
                capture_output=True,
                cwd=empty_dir,
                timeout=timeout_sec,
            )
    except subprocess.TimeoutExpired:
        return "Error: TimeoutError: execution exceeded 2.0s limit"
    out = PROTOCOL_TOKEN_RE.sub("", proc.stdout.decode("utf-8", errors="replace")).strip()
    return (out[:997] + "...") if len(out) > 1000 else (out or "OK (no stdout)")


def build_slider_attitude_block(weights: tuple[float, float, float, float, float]) -> str:
    active_lines: list[str] = []
    for idx in (1, 2, 3, 4):
        w = float(weights[idx])
        if w <= 0.05:
            continue
        pct = int(round(w * 100))
        label, light_desc, strong_desc = SLIDER_TONE_SPECS[idx]
        desc = light_desc if w <= 0.45 else strong_desc
        active_lines.append(f"- {label}（强度 {pct}%）：{desc}。")
    if not active_lines:
        return ""
    return (
        "【当前微信语气倾向（仅作用于 send_message 口语态度，不影响工具调用参数与代码客观性）】：\n"
        + "\n".join(active_lines)
    )


def render_gemma4_prompt(
    history: list[dict[str, Any]],
    slider_weights: tuple[float, float, float, float, float],
) -> str:
    sys_hdr = f"{DEFAULT_PERSONA_HEADER}\n{V4_CHANNEL_RULES}"
    cue = build_slider_attitude_block(slider_weights)
    if cue:
        sys_hdr = f"{sys_hdr}\n{cue}"

    parts = [f"<bos>{TURN_OPEN_STR}system\n<|think|>\n{sys_hdr}{GEMMA4_V4_TOOL_DECLARATIONS}{TURN_END_STR}\n"]
    for item in history[-16:]:
        role = item.get("role")
        if role == "user":
            txt = PROTOCOL_TOKEN_RE.sub("", str(item.get("text") or "")).strip()
            if txt:
                parts.append(f"{TURN_OPEN_STR}user\n{txt}{TURN_END_STR}\n")
        elif role == "assistant":
            calls: list[str] = []
            resps: list[str] = []
            bubbles = [str(b).strip() for b in (item.get("bubbles") or []) if str(b).strip()]
            if bubbles:
                inner = ",".join(f"{QUOTE_STR}{b}{QUOTE_STR}" for b in bubbles)
                calls.append(f"{TOOL_CALL_OPEN_STR}call:send_message{{bubbles:[{inner}]}}{TOOL_CALL_CLOSE_STR}")
                resps.append(f"{TOOL_RESP_OPEN_STR}response:send_message{{delivered:true}}{TOOL_RESP_CLOSE_STR}")
            for card in item.get("code_cards") or []:
                lang = str(card.get("language") or "text")
                body = str(card.get("content") or "")
                calls.append(
                    f"{TOOL_CALL_OPEN_STR}call:send_code_or_doc{{language:{QUOTE_STR}{lang}{QUOTE_STR},"
                    f"content:{QUOTE_STR}{body}{QUOTE_STR}}}{TOOL_CALL_CLOSE_STR}"
                )
                resps.append(f"{TOOL_RESP_OPEN_STR}response:send_code_or_doc{{delivered:true}}{TOOL_RESP_CLOSE_STR}")
            if calls:
                parts.append(
                    f"{TURN_OPEN_STR}model\n{CHANNEL_OPEN_STR}thought\n{CHANNEL_CLOSE_STR}"
                    + "".join(calls)
                    + "".join(resps)
                    + f"{TURN_END_STR}\n"
                )
    parts.append(f"{TURN_OPEN_STR}model\n{CHANNEL_OPEN_STR}thought\n")
    return "".join(parts)


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

    def parse_gemma_str(self) -> str:
        self.pos += len(QUOTE_STR)
        end = self.text.find(QUOTE_STR, self.pos)
        if end == -1:
            val = self.text[self.pos :]
            self.pos = self.n
            return val
        val = self.text[self.pos : end]
        self.pos = end + len(QUOTE_STR)
        return val

    def parse_key(self) -> str:
        self._skip_ws()
        if self.text.startswith(QUOTE_STR, self.pos):
            return self.parse_gemma_str()
        start = self.pos
        while self.pos < self.n and self.text[self.pos] not in ":\r\n\t ,{}[]":
            self.pos += 1
        return self.text[start : self.pos].strip()

    def parse_value(self) -> Any:
        self._skip_ws()
        if self.pos >= self.n:
            return ""
        if self.text.startswith(QUOTE_STR, self.pos):
            return self.parse_gemma_str()
        ch = self._peek()
        if ch == "{":
            return self.parse_object()
        if ch == "[":
            self.pos += 1
            items: list[Any] = []
            while self.pos < self.n:
                self._skip_ws()
                if self._peek() == "]":
                    self.pos += 1
                    return items
                if self.text.startswith(TOOL_CALL_CLOSE_STR, self.pos):
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
        start = self.pos
        while self.pos < self.n and self.text[self.pos] not in ",}]":
            if self.text.startswith(TOOL_CALL_CLOSE_STR, self.pos):
                break
            self.pos += 1
        return self.text[start : self.pos].strip()

    def parse_object(self) -> dict[str, Any]:
        self.pos += 1
        obj: dict[str, Any] = {}
        while self.pos < self.n:
            self._skip_ws()
            if self._peek() == "}":
                self.pos += 1
                return obj
            if self.text.startswith(TOOL_CALL_CLOSE_STR, self.pos):
                return obj
            k = self.parse_key()
            self._skip_ws()
            if self._peek() != ":":
                break
            self.pos += 1
            v = self.parse_value()
            if k:
                obj[k] = v
            self._skip_ws()
            if self._peek() == ",":
                self.pos += 1
            elif self._peek() == "}":
                self.pos += 1
                return obj
            else:
                break
        return obj


def parse_gemma4_tool_calls(text: str) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    cursor = 0
    pat = re.compile(r"<\|tool_call>\s*call:([a-zA-Z0-9_]+)\s*\{")
    while cursor < len(text):
        m = pat.search(text, cursor)
        if not m:
            break
        t_name = m.group(1)
        parser = _GemmaArgsParser(text, m.end() - 1)
        args = parser.parse_object()
        end_pos = parser.pos
        while end_pos < len(text) and text[end_pos] in " \t\r\n":
            end_pos += 1
        if not text.startswith(TOOL_CALL_CLOSE_STR, end_pos):
            cursor = m.end()
            continue
        cursor = end_pos + len(TOOL_CALL_CLOSE_STR)
        calls.append((t_name, args))
    return calls


def extract_bubbles_from_phase_b(raw_phase_b: str) -> list[str]:
    """Parse `<|"|>...<|"|>` bubble sequence starting immediately after `bubbles:[<|"|>`."""
    cleaned = (
        raw_phase_b.replace(CHANNEL_OPEN_STR, "")
        .replace(CHANNEL_CLOSE_STR, "")
        .replace(TURN_OPEN_STR, "")
        .replace(TURN_END_STR, "")
        .replace("\r", "")
        .replace("\n", "")
    )
    for stop_suffix in ("]}<tool_call|>", "] }<tool_call|>", TOOL_CALL_CLOSE_STR):
        idx = cleaned.find(stop_suffix)
        if idx != -1:
            cleaned = cleaned[:idx]
            break

    bubbles: list[str] = []
    in_quote = True
    buf = cleaned
    while QUOTE_STR in buf:
        head, buf = buf.split(QUOTE_STR, 1)
        if in_quote:
            text_b = head.strip()
            if text_b:
                bubbles.append(text_b)
            in_quote = False
        else:
            in_quote = True
    if in_quote and buf.strip():
        bubbles.append(buf.strip())
    return bubbles


# ---------------------------------------------------------------------------
# 4. In-Process PyTorch Split-Phase Engine (Lazy ZeroGPU Placement)
# ---------------------------------------------------------------------------

class _ZeroGPUEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model: Any = None
        self.tokenizer: Any = None
        self.all_newline_ids: list[int] = [NEWLINE_ID]
        self.loaded_on_cuda = False
        self.lora_info: dict[str, Any] = {}

    def ensure_cpu_loaded(self) -> None:
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            from transformers import AutoModelForImageTextToText

            tok_path = hf_hub_download(repo_id=BASE_MODEL_ID, filename="tokenizer.json", token=HF_TOKEN)
            self.tokenizer = Tokenizer.from_file(tok_path)

            nl_set = {NEWLINE_ID}
            for _, tid in self.tokenizer.get_vocab().items():
                if 0 <= int(tid) < 1024:
                    s = self.tokenizer.decode([int(tid)], skip_special_tokens=False)
                    if "\n" in s or "\r" in s:
                        nl_set.add(int(tid))
            self.all_newline_ids = sorted(nl_set)

            gguf_paths = [
                Path(hf_hub_download(repo_id=ADAPTERS_REPO_ID, filename=fname, token=HF_TOKEN))
                for fname in FIVE_CARTRIDGE_FILES
            ]

            model = AutoModelForImageTextToText.from_pretrained(
                BASE_MODEL_ID,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                token=HF_TOKEN,
            )
            inner = getattr(model, "model", None)
            if inner is not None:
                for attr in ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector"):
                    if getattr(inner, attr, None) is not None:
                        setattr(inner, attr, None)
            gc.collect()

            self.lora_info = attach_five_gguf_loras(model, gguf_paths)
            model.eval()
            self.model = model

    def ensure_cuda_ready(self) -> torch.device:
        self.ensure_cpu_loaded()
        if not self.loaded_on_cuda:
            with self._lock:
                if not self.loaded_on_cuda:
                    self.model.to("cuda")
                    self.loaded_on_cuda = True
        return torch.device("cuda")


_ENGINE = _ZeroGPUEngine()


def _sample_next_token(
    logits_1d: torch.Tensor,
    *,
    banned_ids: list[int],
    temperature: float,
    top_p: float,
) -> int:
    logits = logits_1d.float()
    if banned_ids:
        logits[banned_ids] = -float("inf")
    if temperature <= 1e-4:
        return int(torch.argmax(logits).item())
    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probs[remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-12)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs, num_samples=1).item())


def _lm_step(
    model: Any,
    input_ids: torch.Tensor,
    past_kv: Any,
    *,
    sliders: tuple[float, float, float, float, float],
    g_voice: float,
) -> tuple[torch.Tensor, Any]:
    lm = model.model.language_model
    softcap = float(getattr(model.config.get_text_config(), "final_logit_softcapping", 30.0) or 30.0)
    with torch.inference_mode():
        with split_phase_gate_scope(sliders=sliders, g_voice=g_voice):
            out = lm(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            last_h = out.last_hidden_state[:, -1, :]
            raw_logits = F.linear(last_h.to(model.lm_head.weight.dtype), model.lm_head.weight).float()
            logits = softcap * torch.tanh(raw_logits / softcap)
    return logits.squeeze(0), out.past_key_values


def _generate_until_stop(
    model: Any,
    tokenizer: Any,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    stop_strings: list[str],
    banned_ids: list[int],
    temperature: float,
    top_p: float,
    sliders: tuple[float, float, float, float, float],
    g_voice: float,
    device: torch.device,
) -> tuple[str, str, str]:
    from transformers.cache_utils import DynamicCache

    ids_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    past_kv = DynamicCache(config=model.model.language_model.config)
    logits, past_kv = _lm_step(model, ids_tensor, past_kv, sliders=sliders, g_voice=g_voice)

    generated_ids: list[int] = []
    acc_text = ""
    matched_stop = ""
    stop_type = "limit"

    for _ in range(max_new_tokens):
        nxt = _sample_next_token(logits, banned_ids=banned_ids, temperature=temperature, top_p=top_p)
        generated_ids.append(nxt)
        acc_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        hit = ""
        best_idx = len(acc_text) + 1
        for sw in stop_strings:
            pos = acc_text.find(sw)
            if pos != -1 and pos < best_idx:
                best_idx = pos
                hit = sw
        if hit:
            acc_text = acc_text[:best_idx]
            matched_stop = hit
            stop_type = "word"
            break
        if nxt == EOS_ID:
            stop_type = "eos"
            break

        step_tensor = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits, past_kv = _lm_step(model, step_tensor, past_kv, sliders=sliders, g_voice=g_voice)

    return acc_text, matched_stop, stop_type


@spaces.GPU(duration=20)
def run_split_phase_pytorch(
    history: list[dict[str, Any]],
    weights: tuple[float, float, float, float, float],
    temperature: float = 0.85,
    top_p: float = 0.95,
) -> dict[str, Any]:
    """Full in-process PyTorch BF16 Split-Phase inference inside ZeroGPU 48GB slice."""
    t0 = time.perf_counter()
    device = _ENGINE.ensure_cuda_ready()
    model = _ENGINE.model
    tok = _ENGINE.tokenizer

    kickoff_ban = [CHANNEL_CLOSE_ID, TURN_END_ID, EOS_ID, TOOL_RESP_OPEN_ID, CHANNEL_OPEN_ID, *MULTIMODAL_IDS]
    thought_ban = [TURN_END_ID, EOS_ID, TOOL_RESP_OPEN_ID, CHANNEL_OPEN_ID, *MULTIMODAL_IDS]
    phase_a_ban = [CHANNEL_OPEN_ID, CHANNEL_CLOSE_ID, *MULTIMODAL_IDS]
    phase_b_ban = [
        CHANNEL_OPEN_ID, CHANNEL_CLOSE_ID, TOOL_CALL_OPEN_ID, TOOL_RESP_OPEN_ID,
        TOOL_RESP_CLOSE_ID, TURN_OPEN_ID, TURN_END_ID, EOS_ID,
        *_ENGINE.all_newline_ids, *MULTIMODAL_IDS,
    ]

    cur_prompt = render_gemma4_prompt(history, weights)
    bubbles_out: list[str] = []
    code_cards: list[dict[str, str]] = []
    thought_traces: list[str] = []

    for hop_idx in range(3):
        cur_ids = list(tok.encode(cur_prompt, add_special_tokens=False).ids)

        # Phase A: Kickoff (LoRA = 0.0)
        if not bubbles_out and not code_cards:
            kick_txt, _, _ = _generate_until_stop(
                model, tok, cur_ids,
                max_new_tokens=12,
                stop_strings=[],
                banned_ids=kickoff_ban,
                temperature=min(temperature, 0.65),
                top_p=top_p,
                sliders=weights,
                g_voice=0.0,
                device=device,
            )
            cur_prompt += kick_txt
            cur_ids = list(tok.encode(cur_prompt, add_special_tokens=False).ids)

        # Phase A: Thought (LoRA = 0.0)
        thought_txt, _, _ = _generate_until_stop(
            model, tok, cur_ids,
            max_new_tokens=640,
            stop_strings=[CHANNEL_CLOSE_STR],
            banned_ids=thought_ban,
            temperature=min(temperature, 0.65),
            top_p=top_p,
            sliders=weights,
            g_voice=0.0,
            device=device,
        )
        thought_traces.append(thought_txt.strip())
        cur_prompt += thought_txt + CHANNEL_CLOSE_STR + TOOL_CALL_OPEN_STR
        seg_prefix = TOOL_CALL_OPEN_STR

        called_calc = False
        resp_blocks: list[str] = []

        for _seg in range(4):
            cur_ids = list(tok.encode(cur_prompt, add_special_tokens=False).ids)
            a1_txt, stop_w, _ = _generate_until_stop(
                model, tok, cur_ids,
                max_new_tokens=1024,
                stop_strings=PHASE_A_STOPS,
                banned_ids=phase_a_ban,
                temperature=min(temperature, 0.60),
                top_p=top_p,
                sliders=weights,
                g_voice=0.0,
                device=device,
            )
            seg_full = seg_prefix + a1_txt
            seg_prefix = ""

            for t_name, t_args in parse_gemma4_tool_calls(seg_full):
                if t_name == "calc_python":
                    called_calc = True
                    calc_stdout = run_calc_python_sandbox(str(t_args.get("code") or ""))
                    resp_blocks.append(
                        f'{TOOL_RESP_OPEN_STR}response:calc_python{{stdout:{QUOTE_STR}{calc_stdout}{QUOTE_STR}}}{TOOL_RESP_CLOSE_STR}'
                    )
                elif t_name == "send_code_or_doc":
                    lang = str(t_args.get("language") or "text").strip()
                    body = str(t_args.get("content") or "").rstrip("\r\n")
                    if body:
                        code_cards.append({"language": lang, "content": body})
                    resp_blocks.append(f"{TOOL_RESP_OPEN_STR}response:send_code_or_doc{{delivered:true}}{TOOL_RESP_CLOSE_STR}")
                elif t_name == "send_message":
                    raw_b = t_args.get("bubbles") or []
                    for b in raw_b:
                        bs = str(b).strip()
                        if bs:
                            bubbles_out.append(bs)
                    resp_blocks.append(f"{TOOL_RESP_OPEN_STR}response:send_message{{delivered:true}}{TOOL_RESP_CLOSE_STR}")

            if stop_w in SEND_MSG_STOP_VARIANTS:
                cur_prompt += a1_txt + stop_w
                cur_ids = list(tok.encode(cur_prompt, add_special_tokens=False).ids)

                # Phase B: 5-Slider Voice Core ACTIVE (`g_voice = 1.0`) strictly inside quotes!
                b_txt, b_stop, _ = _generate_until_stop(
                    model, tok, cur_ids,
                    max_new_tokens=512,
                    stop_strings=PHASE_B_STOPS,
                    banned_ids=phase_b_ban,
                    temperature=temperature,
                    top_p=top_p,
                    sliders=weights,
                    g_voice=1.0,
                    device=device,
                )
                bubbles_out.extend(extract_bubbles_from_phase_b(b_txt))
                b_close = b_stop if b_stop.endswith(TOOL_CALL_CLOSE_STR) else "]}" + TOOL_CALL_CLOSE_STR
                cur_prompt += b_txt + b_close
                resp_blocks.append(f"{TOOL_RESP_OPEN_STR}response:send_message{{delivered:true}}{TOOL_RESP_CLOSE_STR}")
                continue

            cur_prompt += a1_txt
            break

        if called_calc and hop_idx < 2:
            cur_prompt += "".join(resp_blocks) + f"{CHANNEL_OPEN_STR}thought\n"
            continue
        break

    latency_ms = int(round((time.perf_counter() - t0) * 1000.0))
    return {
        "bubbles": bubbles_out or ["先等我缓口气，刚才走神啦"],
        "code_cards": code_cards,
        "thought": "\n---\n".join(t for t in thought_traces if t),
        "latency_ms": latency_ms,
        "lora_info": _ENGINE.lora_info,
    }


# ---------------------------------------------------------------------------
# 5. Warm Paper-Ink Gradio UI (`#F7F4EC` / `#1C2B22` / `#2C5E43`, Zero Emojis)
# ---------------------------------------------------------------------------

PAPER_INK_CSS = """
:root, .gradio-container {
  background-color: #F7F4EC !important;
  color: #1C2B22 !important;
  font-family: "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
  font-synthesis: none !important;
}
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 24px 20px !important;
}
.xfy-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  border-bottom: 1px solid #E2DDD0;
  padding-bottom: 16px;
  margin-bottom: 20px;
  gap: 16px;
}
.xfy-eyebrow {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: #2C5E43;
  margin-bottom: 6px;
}
.xfy-title {
  margin: 0;
  font-size: 28px;
  font-weight: 700;
  color: #1C2B22;
  letter-spacing: -0.01em;
}
.xfy-badge {
  background: #E2F1E8;
  border: 1px solid #B6D7C3;
  color: #215238;
  font-size: 12px;
  font-weight: 600;
  padding: 6px 14px;
  border-radius: 999px;
  white-space: nowrap;
}
.xfy-panel {
  background: #FCFBF7 !important;
  border: 1px solid #DDD6C7 !important;
  border-radius: 18px !important;
  padding: 16px !important;
}
.xfy-timeline {
  min-height: 440px;
  max-height: 540px;
  overflow-y: auto;
  padding: 18px;
  background: #F7F4EC;
  border: 1px solid #E2DDD0;
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.xfy-empty {
  margin: auto;
  text-align: center;
  color: #56665D;
  max-width: 420px;
  line-height: 1.6;
}
.xfy-turn-user {
  display: flex;
  justify-content: flex-end;
}
.xfy-bubble-user {
  background: #2C5E43;
  color: #FAF9F4;
  padding: 10px 15px;
  border-radius: 18px 18px 5px 18px;
  max-width: 78%;
  font-size: 14.5px;
  line-height: 1.55;
}
.xfy-turn-peer {
  display: flex;
  align-items: flex-end;
  gap: 10px;
}
.xfy-avatar {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: #DCE7DF;
  color: #2C5E43;
  font-size: 13px;
  font-weight: 700;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.xfy-burst {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  max-width: 78%;
}
.xfy-bubble-peer {
  background: #FFFFFF;
  color: #1C2B22;
  border: 1px solid #E2DDD0;
  padding: 10px 15px;
  border-radius: 18px 18px 18px 6px;
  font-size: 14.5px;
  line-height: 1.55;
}
.xfy-meta {
  font-size: 11px;
  color: #7F8D85;
  margin-top: 2px;
}
button.primary {
  background: #2C5E43 !important;
  border-color: #2C5E43 !important;
  color: #FAF9F4 !important;
}
"""


def render_chat_timeline_html(history: list[dict[str, Any]]) -> str:
    if not history:
        return (
            '<div class="xfy-timeline"><div class="xfy-empty">'
            '<div style="font-size:17px;font-weight:700;color:#1C2B22;margin-bottom:6px;">和小肥鱼打个招呼</div>'
            '<div>左侧调节五维人格滑条；思考与代码全程保持 LoRA = 0.0，仅在对话气泡内激活声音权重。</div>'
            "</div></div>"
        )
    blocks: list[str] = ['<div class="xfy-timeline">']
    for item in history:
        if item.get("role") == "user":
            safe = html.escape(str(item.get("text") or "")).replace("\n", "<br>")
            blocks.append(f'<div class="xfy-turn-user"><div class="xfy-bubble-user">{safe}</div></div>')
        else:
            bubbles_html = "".join(
                f'<div class="xfy-bubble-peer">{html.escape(str(b))}</div>'
                for b in (item.get("bubbles") or [])
            )
            meta = html.escape(str(item.get("meta") or ""))
            blocks.append(
                '<div class="xfy-turn-peer">'
                '<div class="xfy-avatar">鱼</div>'
                f'<div class="xfy-burst">{bubbles_html}<div class="xfy-meta">{meta}</div></div>'
                "</div>"
            )
    blocks.append("</div>")
    return "".join(blocks)


def _describe_mix(weights: tuple[float, float, float, float, float]) -> str:
    names = ("元气日常", "甜系软萌", "知性姐姐", "傲娇毒舌", "安静慢热")
    active = [f"{names[i]} {weights[i]:.2f}" for i in range(1, 5) if weights[i] >= 0.08]
    if not active:
        return "元气日常"
    return " / ".join(active)


def on_send_message(
    user_text: str,
    history: list[dict[str, Any]],
    w0: float,
    w1: float,
    w2: float,
    w3: float,
    w4: float,
):
    clean = (user_text or "").strip()
    if not clean:
        return "", history, render_chat_timeline_html(history), "", ""

    weights = (round(float(w0), 3), round(float(w1), 3), round(float(w2), 3), round(float(w3), 3), round(float(w4), 3))
    new_history = [*history, {"role": "user", "text": clean}]
    result = run_split_phase_pytorch(new_history, weights)
    mix_label = _describe_mix(weights)
    meta = f"{mix_label} · {result['latency_ms']} ms · BF16 Split-Phase"
    new_history.append(
        {
            "role": "assistant",
            "bubbles": result["bubbles"],
            "code_cards": result["code_cards"],
            "meta": meta,
        }
    )

    code_text = ""
    code_lang = "python"
    if result["code_cards"]:
        first_card = result["code_cards"][0]
        code_lang = first_card.get("language") or "python"
        if len(result["code_cards"]) == 1:
            code_text = first_card.get("content") or ""
        else:
            code_text = "\n\n".join(
                f"# --- [{c.get('language', 'text')}] ---\n{c.get('content', '')}"
                for c in result["code_cards"]
            )

    return (
        "",
        new_history,
        render_chat_timeline_html(new_history),
        gr.update(value=code_text, language=code_lang),
        result.get("thought") or "",
    )


def apply_preset(preset_name: str):
    w = PRESET_WEIGHTS.get(preset_name, (1.0, 0.0, 0.0, 0.0, 0.0))
    return w[0], w[1], w[2], w[3], w[4]


def clear_chat():
    return [], render_chat_timeline_html([]), gr.update(value=""), ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Xiaofeiyu-12B-v4 · ZeroGPU Split-Phase", css=PAPER_INK_CSS) as demo:
        gr.HTML(
            '<header class="xfy-header">'
            '<div><div class="xfy-eyebrow">XIAOFEIYU-12B-V4 · ZEROGPU PYTORCH BF16</div>'
            '<h1 class="xfy-title">小肥鱼 · 五维人格声带混音</h1></div>'
            '<div class="xfy-badge">Phase A: LoRA = 0.0 满血智商 · Phase B: 气泡声带激活</div>'
            "</header>"
        )

        chat_state = gr.State([])

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes=["xfy-panel"]):
                gr.Markdown("**人格快捷预设（Presets）**")
                with gr.Row():
                    btn_s0 = gr.Button("元气日常", size="sm")
                    btn_c1 = gr.Button("甜系软萌", size="sm")
                    btn_c2 = gr.Button("知性姐姐", size="sm")
                with gr.Row():
                    btn_c3 = gr.Button("傲娇毒舌", size="sm")
                    btn_c4 = gr.Button("安静慢热", size="sm")

                gr.Markdown("**五滑条权重混音（推荐风格和 ≤ 1.15）**")
                s_w0 = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="s0 · 元气日常 (Everyday Warmth)")
                s_w1 = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="c1 · 甜系软萌 (Sweet & Clingy)")
                s_w2 = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="c2 · 知性姐姐 (Poised Older Sister)")
                s_w3 = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="c3 · 傲娇毒舌 (Tsundere Wit)")
                s_w4 = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="c4 · 安静慢热 (Quiet & Gentle)")

                clear_btn = gr.Button("清空对话", variant="secondary")

            with gr.Column(scale=7):
                timeline_html = gr.HTML(render_chat_timeline_html([]))
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="和小肥鱼说点什么...（可试着让她算账、写段 Python 并吐槽你）",
                        show_label=False,
                        scale=5,
                        lines=2,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                code_card = gr.Code(
                    label="代码与文档卡（Phase A · LoRA = 0.0 原生智商输出）",
                    language="python",
                    interactive=False,
                    lines=8,
                )
                with gr.Accordion("查看 Phase A 思考链（LoRA = 0.0）", open=False):
                    thought_box = gr.Textbox(show_label=False, lines=4, interactive=False)

        for btn, pname in (
            (btn_s0, "元气日常"),
            (btn_c1, "甜系软萌"),
            (btn_c2, "知性姐姐"),
            (btn_c3, "傲娇毒舌"),
            (btn_c4, "安静慢热"),
        ):
            btn.click(lambda p=pname: apply_preset(p), outputs=[s_w0, s_w1, s_w2, s_w3, s_w4])

        send_btn.click(
            on_send_message,
            inputs=[user_input, chat_state, s_w0, s_w1, s_w2, s_w3, s_w4],
            outputs=[user_input, chat_state, timeline_html, code_card, thought_box],
        )
        user_input.submit(
            on_send_message,
            inputs=[user_input, chat_state, s_w0, s_w1, s_w2, s_w3, s_w4],
            outputs=[user_input, chat_state, timeline_html, code_card, thought_box],
        )
        clear_btn.click(clear_chat, outputs=[chat_state, timeline_html, code_card, thought_box])

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
