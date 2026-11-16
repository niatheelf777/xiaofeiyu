#!/usr/bin/env python3
"""1-Command Local Launcher for Xiaofeiyu (26B-A4B MoE / 12B Dense / E4B Voice Proxy).

Automatically pulls the base GGUF and 5 Xiaofeiyu GGUF LoRA cartridges (`s0`, `c1..c4`)
from HuggingFace (`niatheelf777/*`), starts `llama-server` with `--lora-init-without-apply`,
and starts the Split-Phase Gateway on `http://127.0.0.1:8081/v1`.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent

FIVE_CARTRIDGE_FILES: tuple[str, ...] = (
    "s0_xiaofeiyu_backbone.gguf",
    "c1_neko_clingy.gguf",
    "c2_older_tease.gguf",
    "c3_brat_mesugaki.gguf",
    "c4_fragile_porcelain.gguf",
)


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model_family: str
    base_hf_repo: str
    base_gguf_file: str
    adapters_hf_repo: str
    tokenizer_hf_repo: str
    default_ctx_size: int


PRESETS: dict[str, ModelPreset] = {
    "26b-a4b": ModelPreset(
        name="26b-a4b",
        model_family="26b-a4b",
        base_hf_repo="ggml-org/gemma-4-26B-A4B-it-GGUF",
        base_gguf_file="gemma-4-26B-A4B-it-Q4_K_M.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-26b-a4b",
        tokenizer_hf_repo="google/gemma-4-26B-A4B-it",
        default_ctx_size=16384,
    ),
    "12b": ModelPreset(
        name="12b",
        model_family="12b",
        base_hf_repo="ggml-org/gemma-4-12B-it-GGUF",
        base_gguf_file="gemma-4-12B-it-Q8_0.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-12b-v4",
        tokenizer_hf_repo="google/gemma-4-12B-it",
        default_ctx_size=16384,
    ),
    "e4b": ModelPreset(
        name="e4b",
        model_family="e4b",
        base_hf_repo="ggml-org/gemma-4-E4B-it-GGUF",
        base_gguf_file="gemma-4-E4B-it-Q8_0.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-translator-e4b",
        tokenizer_hf_repo="google/gemma-4-E4B-it",
        default_ctx_size=8192,
    ),
}


def download_hf_file(repo_id: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(filename).name
    if target.is_file() and target.stat().st_size > 0:
        return target

    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest_dir))).resolve()
        if downloaded != target.resolve() and downloaded.is_file():
            shutil.copy2(downloaded, target)
        return target
    except ImportError:
        import urllib.request

        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        tmp = target.with_suffix(target.suffix + ".part")
        sys.stderr.write(f"[launch_local] Downloading {url} -> {target}\n")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(target)
        return target


def resolve_llama_server(explicit_bin: Path | None) -> Path:
    if explicit_bin is not None:
        p = Path(explicit_bin)
        if not p.is_file():
            raise FileNotFoundError(f"llama-server binary not found at {p}")
        return p
    which_bin = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if which_bin:
        return Path(which_bin)
    raise FileNotFoundError("Could not find `llama-server` on PATH; pass --llama-server-bin <path>")


def build_launch_plan(
    *,
    preset_name: str,
    models_dir: Path,
    loras_dir: Path,
    llama_bin: str,
    host: str,
    port: int,
    llama_port: int,
    n_parallel: int,
    ctx_size: int | None,
    n_gpu_layers: int,
) -> dict[str, Any]:
    preset = PRESETS[preset_name]
    eff_ctx = int(ctx_size or preset.default_ctx_size)
    base_gguf_path = Path(models_dir) / preset.base_gguf_file
    lora_paths = [Path(loras_dir) / preset.name / fname for fname in FIVE_CARTRIDGE_FILES]
    tokenizer_path = Path(models_dir) / f"{preset.name}-tokenizer.json"

    llama_cmd = [
        llama_bin,
        "--host", "127.0.0.1",
        "--port", str(llama_port),
        "-m", str(base_gguf_path),
        "-ngl", str(n_gpu_layers),
        "-fa", "on",
        "-c", str(eff_ctx),
        "-np", str(n_parallel),
        *(["-cb"] if n_parallel > 1 else []),
        "--cache-ram", "0",
        "--special",
        "--no-webui",
        "--lora-init-without-apply",
        "--lora", ",".join(str(p) for p in lora_paths),
    ]
    gateway_cmd = [
        sys.executable,
        str(HERE / "pc2_split_phase_server.py"),
        "--host", host,
        "--port", str(port),
        "--llama-port", str(llama_port),
        "--model-family", preset.model_family,
        "--tokenizer-path", str(tokenizer_path),
        "--n-parallel", str(n_parallel),
        "--ctx-size", str(eff_ctx),
    ]
    return {
        "preset": preset.name,
        "base_hf_repo": preset.base_hf_repo,
        "base_gguf_file": preset.base_gguf_file,
        "adapters_hf_repo": preset.adapters_hf_repo,
        "tokenizer_hf_repo": preset.tokenizer_hf_repo,
        "lora_files": [str(p) for p in lora_paths],
        "llama_cmd": llama_cmd,
        "gateway_cmd": gateway_cmd,
    }


def wait_for_llama_health(port: int, timeout_sec: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            conn.close()
            if resp.status == 200 and '"ok"' in body:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server on port {port} did not become healthy within {timeout_sec}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="12b")
    ap.add_argument("--models-dir", type=Path, default=Path("./models"))
    ap.add_argument("--loras-dir", type=Path, default=Path("./loras"))
    ap.add_argument("--llama-server-bin", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--llama-port", type=int, default=8082)
    ap.add_argument("--n-parallel", type=int, default=1)
    ap.add_argument("--ctx-size", type=int, default=None)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--dry-run", action="store_true", help="Print resolved artifacts and commands without downloading or starting")
    args = ap.parse_args(argv)

    llama_bin_str = str(args.llama_server_bin) if args.llama_server_bin else (
        "llama-server" if args.dry_run else str(resolve_llama_server(None))
    )
    plan = build_launch_plan(
        preset_name=args.preset,
        models_dir=args.models_dir,
        loras_dir=args.loras_dir,
        llama_bin=llama_bin_str,
        host=args.host,
        port=args.port,
        llama_port=args.llama_port,
        n_parallel=args.n_parallel,
        ctx_size=args.ctx_size,
        n_gpu_layers=args.ngl,
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    preset = PRESETS[args.preset]
    download_hf_file(preset.base_hf_repo, preset.base_gguf_file, args.models_dir)
    tok_dl = download_hf_file(preset.tokenizer_hf_repo, "tokenizer.json", args.models_dir)
    target_tok = args.models_dir / f"{preset.name}-tokenizer.json"
    if tok_dl.resolve() != target_tok.resolve():
        shutil.copy2(tok_dl, target_tok)
    for fname in FIVE_CARTRIDGE_FILES:
        download_hf_file(preset.adapters_hf_repo, fname, args.loras_dir / preset.name)

    llama_proc = subprocess.Popen(plan["llama_cmd"])
    atexit.register(lambda: llama_proc.terminate() if llama_proc.poll() is None else None)
    wait_for_llama_health(args.llama_port)
    gw_proc = subprocess.Popen(plan["gateway_cmd"])
    atexit.register(lambda: gw_proc.terminate() if gw_proc.poll() is None else None)
    return gw_proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
