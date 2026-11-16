#!/usr/bin/env python3
"""1-Command Turnkey Local Launcher for Xiaofeiyu (12B / 12B-Q4 / 26B-A4B MoE / E4B).

Automatically pulls the base GGUF, tokenizer, and 5 Xiaofeiyu GGUF LoRA cartridges (`s0`, `c1..c4`)
from HuggingFace (`niatheelf777/*`), optionally fetches an official `llama-server` binary when
`--auto-download-llama` is passed, starts `llama-server` with `--lora-init-without-apply`, and
starts the Split-Phase Gateway serving BOTH:
  - Turnkey 2D Persona Orb Web UI:  http://127.0.0.1:8081/
  - OpenAI-compatible API (`/v1`):  http://127.0.0.1:8081/v1
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
import http.client
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any
import urllib.request
import webbrowser
import zipfile

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
LLAMA_RELEASES_LATEST_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

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
    lora_subdir: str | None = None


PRESETS: dict[str, ModelPreset] = {
    "26b-a4b": ModelPreset(
        name="26b-a4b",
        model_family="26b-a4b",
        base_hf_repo="niatheelf777/xiaofeiyu-26b-a4b",
        base_gguf_file="gemma-4-26B-A4B-it-UD-Q4_K_XL-xfy.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-26b-a4b",
        tokenizer_hf_repo="unsloth/gemma-4-26B-A4B-it",
        default_ctx_size=8192,
    ),
    "26b-a4b-iq4xs": ModelPreset(
        name="26b-a4b-iq4xs",
        model_family="26b-a4b",
        base_hf_repo="niatheelf777/xiaofeiyu-26b-a4b",
        base_gguf_file="gemma-4-26B-A4B-it-UD-IQ4_XS-xfy.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-26b-a4b",
        tokenizer_hf_repo="unsloth/gemma-4-26B-A4B-it",
        default_ctx_size=8192,
        lora_subdir="26b-a4b",
    ),
    "26b-a4b-12g": ModelPreset(
        name="26b-a4b-12g",
        model_family="26b-a4b",
        base_hf_repo="niatheelf777/xiaofeiyu-26b-a4b",
        base_gguf_file="gemma-4-26B-A4B-it-UD-IQ3_M-xfy.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-26b-a4b",
        tokenizer_hf_repo="unsloth/gemma-4-26B-A4B-it",
        default_ctx_size=8192,
        lora_subdir="26b-a4b",
    ),
    "26b-a4b-q8": ModelPreset(
        name="26b-a4b-q8",
        model_family="26b-a4b",
        base_hf_repo="ggml-org/gemma-4-26B-A4B-it-GGUF",
        base_gguf_file="gemma-4-26B-A4B-it-Q8_0.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-26b-a4b",
        tokenizer_hf_repo="unsloth/gemma-4-26B-A4B-it",
        default_ctx_size=16384,
        lora_subdir="26b-a4b",
    ),
    "12b": ModelPreset(
        name="12b",
        model_family="12b",
        base_hf_repo="ggml-org/gemma-4-12B-it-GGUF",
        base_gguf_file="gemma-4-12B-it-Q8_0.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-12b-v4",
        tokenizer_hf_repo="unsloth/gemma-4-12B-it",
        default_ctx_size=16384,
    ),
    "12b-q4": ModelPreset(
        name="12b-q4",
        model_family="12b",
        base_hf_repo="niatheelf777/xiaofeiyu-12b-v4",
        base_gguf_file="gemma-4-12B-it-UD-Q4_K_XL-xfy.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-12b-v4",
        tokenizer_hf_repo="unsloth/gemma-4-12B-it",
        default_ctx_size=8192,
        lora_subdir="12b",
    ),
    "e4b": ModelPreset(
        name="e4b",
        model_family="e4b",
        base_hf_repo="ggml-org/gemma-4-E4B-it-GGUF",
        base_gguf_file="gemma-4-E4B-it-Q8_0.gguf",
        adapters_hf_repo="niatheelf777/xiaofeiyu-translator-e4b",
        tokenizer_hf_repo="unsloth/gemma-4-E4B-it",
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
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        tmp = target.with_suffix(target.suffix + ".part")
        sys.stderr.write(f"[launch_local] Downloading {url} -> {target}\n")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(target)
        return target


def select_llama_release_asset(
    assets: list[dict[str, Any]],
    *,
    sys_platform: str | None = None,
    machine: str | None = None,
) -> dict[str, str] | None:
    """Choose a portable official `ggml-org/llama.cpp` release archive for the current OS/CPU."""
    plat = (sys_platform or sys.platform).lower()
    arch = (machine or platform.machine()).lower()
    is_arm = "arm" in arch or "aarch64" in arch

    candidates: list[tuple[int, dict[str, str]]] = []
    for item in assets:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        url = str(item.get("browser_download_url") or "").strip()
        lower = name.lower()
        if not name or not url or not (lower.endswith(".zip") or lower.endswith(".tar.gz")):
            continue
        if "cudart" in lower or "xcframework" in lower or "opencl" in lower:
            continue

        score = -1
        if plat.startswith("win"):
            if "bin-win-" in lower and ("x64" in lower or not is_arm):
                if "vulkan-x64" in lower:
                    score = 30
                elif "cpu-x64" in lower or "avx2-x64" in lower:
                    score = 20
        elif plat == "darwin":
            if "bin-macos-" in lower:
                if is_arm and "arm64" in lower:
                    score = 30
                elif not is_arm and "x64" in lower:
                    score = 30
        elif plat.startswith("linux"):
            if "bin-ubuntu-" in lower:
                if not is_arm and ("vulkan-x64" in lower or "x64" in lower):
                    score = 30 if "vulkan" in lower else 25

        if score > 0:
            candidates.append((score, {"name": name, "url": url}))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def _find_cached_llama_binary(bin_dir: Path, *, sys_platform: str | None = None) -> Path | None:
    if not bin_dir.is_dir():
        return None
    plat = (sys_platform or sys.platform).lower()
    exe_names = ("llama-server.exe", "llama-server") if plat.startswith("win") else ("llama-server",)
    for exe_name in exe_names:
        direct = bin_dir / exe_name
        if direct.is_file():
            return direct
        for found in sorted(bin_dir.rglob(exe_name)):
            if found.is_file():
                return found
    return None


def _safe_extract_archive(archive_path: Path, dest_dir: Path) -> None:
    root = dest_dir.resolve()
    name_lower = archive_path.name.lower()
    if name_lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                target = (root / member.filename).resolve()
                target.relative_to(root)
            zf.extractall(root)
        return
    if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                target = (root / member.name).resolve()
                target.relative_to(root)
            tf.extractall(root)
        return
    raise ValueError(f"Unsupported archive format: {archive_path.name}")


def auto_download_llama_server(bin_dir: Path, *, releases_api_url: str = LLAMA_RELEASES_LATEST_API) -> Path:
    cached = _find_cached_llama_binary(bin_dir)
    if cached is not None:
        return cached

    bin_dir.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        releases_api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "xiaofeiyu-launcher/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        release_data = json.loads(resp.read().decode("utf-8"))

    chosen = select_llama_release_asset(release_data.get("assets") or [])
    if chosen is None:
        raise FileNotFoundError(
            f"No prebuilt llama.cpp release archive matched ({sys.platform}, {platform.machine()}). "
            "Install llama-server manually and pass --llama-server-bin <path>."
        )

    archive_dest = bin_dir / chosen["name"]
    sys.stderr.write(f"[launch_local] Auto-downloading official llama.cpp binary ({chosen['name']})...\n")
    urllib.request.urlretrieve(chosen["url"], str(archive_dest))
    extract_dir = bin_dir / archive_dest.stem.replace(".tar", "")
    extract_dir.mkdir(parents=True, exist_ok=True)
    _safe_extract_archive(archive_dest, extract_dir)

    found = _find_cached_llama_binary(extract_dir)
    if found is None:
        raise FileNotFoundError(f"Extracted {chosen['name']}, but `llama-server` was not found inside {extract_dir}")
    if not sys.platform.lower().startswith("win"):
        found.chmod(found.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return found


def _format_missing_llama_message() -> str:
    if sys.platform.lower().startswith("win"):
        install_hint = "winget install ggml.llamacpp"
    elif sys.platform == "darwin":
        install_hint = "brew install llama.cpp"
    else:
        install_hint = "brew install llama.cpp  # or download from https://github.com/ggml-org/llama.cpp/releases"
    return (
        "Could not find `llama-server` on PATH.\n"
        "  Option 1 (automatic): re-run with `--auto-download-llama` to fetch the official ggml-org/llama.cpp binary.\n"
        f"  Option 2 (system pkg): `{install_hint}`\n"
        "  Option 3 (custom path): pass `--llama-server-bin /path/to/llama-server`"
    )


def resolve_llama_server(
    explicit_bin: Path | None,
    *,
    models_dir: Path = Path("./models"),
    auto_download: bool = False,
) -> Path:
    if explicit_bin is not None:
        p = Path(explicit_bin)
        if not p.is_file():
            raise FileNotFoundError(f"llama-server binary not found at {p}")
        return p
    which_bin = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if which_bin:
        return Path(which_bin)
    cached = _find_cached_llama_binary(Path(models_dir) / "bin")
    if cached is not None:
        return cached
    if auto_download:
        return auto_download_llama_server(Path(models_dir) / "bin")
    raise FileNotFoundError(_format_missing_llama_message())


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
    no_webui: bool = False,
) -> dict[str, Any]:
    preset = PRESETS[preset_name]
    eff_ctx = int(ctx_size or preset.default_ctx_size)
    lora_folder = preset.lora_subdir or preset.name
    base_gguf_path = Path(models_dir) / preset.base_gguf_file
    lora_paths = [Path(loras_dir) / lora_folder / fname for fname in FIVE_CARTRIDGE_FILES]
    tokenizer_path = Path(models_dir) / f"{lora_folder}-tokenizer.json"

    llama_cmd = [
        llama_bin,
        "--host",
        "127.0.0.1",
        "--port",
        str(llama_port),
        "-m",
        str(base_gguf_path),
        "-ngl",
        str(n_gpu_layers),
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-c",
        str(eff_ctx),
        "-np",
        str(n_parallel),
        *(["-cb"] if n_parallel > 1 else []),
        "--cache-ram",
        "0",
        "--special",
        "--no-webui",
        "--lora-init-without-apply",
        "--lora",
        ",".join(str(p) for p in lora_paths),
    ]
    gateway_cmd = [
        sys.executable,
        str(HERE / "pc2_split_phase_server.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--llama-port",
        str(llama_port),
        "--model-family",
        preset.model_family,
        "--tokenizer-path",
        str(tokenizer_path),
        "--n-parallel",
        str(n_parallel),
        "--ctx-size",
        str(eff_ctx),
        *(["--no-webui"] if no_webui else []),
    ]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return {
        "preset": preset.name,
        "model_family": preset.model_family,
        "base_hf_repo": preset.base_hf_repo,
        "base_gguf_file": preset.base_gguf_file,
        "adapters_hf_repo": preset.adapters_hf_repo,
        "tokenizer_hf_repo": preset.tokenizer_hf_repo,
        "tokenizer_path": str(tokenizer_path),
        "lora_files": [str(p) for p in lora_paths],
        "web_ui_url": None if no_webui else f"http://{display_host}:{port}/",
        "openai_api_url": f"http://{display_host}:{port}/v1",
        "llama_cmd": llama_cmd,
        "gateway_cmd": gateway_cmd,
    }


def wait_for_llama_health(port: int, timeout_sec: float = 180.0, host: str = "127.0.0.1") -> None:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(probe_host, port, timeout=2.0)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            conn.close()
            if resp.status == 200 and '"ok"' in body:
                return
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Service on {probe_host}:{port} did not become healthy within {timeout_sec}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="12b")
    ap.add_argument("--models-dir", type=Path, default=Path("./models"))
    ap.add_argument("--loras-dir", type=Path, default=Path("./loras"))
    ap.add_argument("--llama-server-bin", type=Path, default=None)
    ap.add_argument(
        "--auto-download-llama",
        action="store_true",
        help="Automatically fetch the official ggml-org/llama.cpp release binary if llama-server is not on PATH",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--llama-port", type=int, default=8082)
    ap.add_argument("--n-parallel", type=int, default=1)
    ap.add_argument("--ctx-size", type=int, default=None)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--no-webui", action="store_true", help="Disable serving the built-in 2D Persona Orb Web UI on /")
    ap.add_argument("--open", action="store_true", help="Open the local 2D Persona Orb Web UI in the default browser once ready")
    ap.add_argument("--dry-run", action="store_true", help="Print resolved artifacts and commands without downloading or starting")
    args = ap.parse_args(argv)

    try:
        llama_bin_str = (
            str(args.llama_server_bin)
            if args.llama_server_bin
            else (
                "llama-server"
                if args.dry_run
                else str(resolve_llama_server(None, models_dir=args.models_dir, auto_download=args.auto_download_llama))
            )
        )
    except FileNotFoundError as exc:
        sys.stderr.write(f"[launch_local] {exc}\n")
        return 2

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
        no_webui=args.no_webui,
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    preset = PRESETS[args.preset]
    lora_folder = preset.lora_subdir or preset.name
    download_hf_file(preset.base_hf_repo, preset.base_gguf_file, args.models_dir)
    tok_dl = download_hf_file(preset.tokenizer_hf_repo, "tokenizer.json", args.models_dir)
    target_tok = args.models_dir / f"{lora_folder}-tokenizer.json"
    if tok_dl.resolve() != target_tok.resolve():
        shutil.copy2(tok_dl, target_tok)
    for fname in FIVE_CARTRIDGE_FILES:
        download_hf_file(preset.adapters_hf_repo, fname, args.loras_dir / lora_folder)

    llama_proc = subprocess.Popen(plan["llama_cmd"])
    atexit.register(lambda: llama_proc.terminate() if llama_proc.poll() is None else None)
    wait_for_llama_health(args.llama_port)

    gw_proc = subprocess.Popen(plan["gateway_cmd"])
    atexit.register(lambda: gw_proc.terminate() if gw_proc.poll() is None else None)
    wait_for_llama_health(args.port, timeout_sec=30.0, host=args.host)

    sys.stderr.write(
        "\n============================================================\n"
        f"  Xiaofeiyu ({args.preset}) is ready!\n"
        + (f"  2D Persona Orb Web UI:  {plan['web_ui_url']}\n" if plan["web_ui_url"] else "")
        + f"  OpenAI-Compatible API:  {plan['openai_api_url']}\n"
        "============================================================\n\n"
    )
    if args.open and plan["web_ui_url"]:
        webbrowser.open(plan["web_ui_url"])

    return gw_proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
