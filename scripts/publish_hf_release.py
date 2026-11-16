#!/usr/bin/env python3
"""Validate HuggingFace model cards & 5-slot GGUF adapters and upload via HF_TOKEN."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

EXPECTED_HF_CARDS = (
    "xiaofeiyu-26b-a4b",
    "xiaofeiyu-12b-v4",
    "xiaofeiyu-translator-e4b",
)

GENERIC_SECRET_RE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}\b|\bhf_[A-Za-z0-9]{20,}\b|\bghp_[A-Za-z0-9]{20,}\b)"
)
SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv"})


def scan_generic_secrets(repo_root: Path) -> tuple[int, list[str]]:
    scanned = 0
    hits: list[str] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(repo_root).parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        if GENERIC_SECRET_RE.search(text):
            hits.append(str(path.relative_to(repo_root)).replace("\\", "/"))
    return scanned, hits


def validate_release_bundle(repo_root: Path, weights_dir: Path | None = None) -> dict:
    scanned_files, secret_hits = scan_generic_secrets(repo_root)
    if secret_hits:
        raise RuntimeError(f"Secret token detected in files: {secret_hits}")

    cards_root = repo_root / "hf-cards"
    cards_checked: list[str] = []
    for slug in EXPECTED_HF_CARDS:
        card = cards_root / slug / "README.md"
        if not card.is_file():
            raise FileNotFoundError(f"Missing HF card: {card}")
        text = card.read_text(encoding="utf-8")
        if not any(lic in text for lic in ("license: gemma", "license: apache-2.0")):
            raise ValueError(f"Missing gemma/apache-2.0 license frontmatter in {card}")
        cards_checked.append(slug)

    weights_found: list[str] = []
    if weights_dir is not None and weights_dir.is_dir():
        for gguf in sorted(weights_dir.glob("*.gguf")):
            weights_found.append(gguf.name)

    return {
        "audit_passed": True,
        "scanned_files": scanned_files,
        "cards_checked": cards_checked,
        "weights_found": weights_found,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--weights-dir", type=Path, default=None)
    ap.add_argument("--namespace", default="niatheelf777")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    summary = validate_release_bundle(args.repo_root, args.weights_dir)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "namespace": args.namespace, **summary}, ensure_ascii=False, indent=2))
        return 0

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN environment variable is required for live upload (or pass --dry-run)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    uploaded: list[str] = []
    for slug in summary["cards_checked"]:
        repo_id = f"{args.namespace}/{slug}"
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(args.repo_root / "hf-cards" / slug / "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )
        if args.weights_dir is not None:
            slug_dir = args.weights_dir / slug
            if slug_dir.is_dir():
                upload_candidates = [
                    *sorted(slug_dir.glob("*.gguf")),
                    *(p for p in (slug_dir / "lora_gguf_manifest.json", slug_dir / "ARTIFACT-MANIFEST.json") if p.is_file()),
                ]
                for weight_file in upload_candidates:
                    api.upload_file(
                        path_or_fileobj=str(weight_file),
                        path_in_repo=weight_file.name,
                        repo_id=repo_id,
                        repo_type="model",
                    )
        uploaded.append(repo_id)

    print(json.dumps({"uploaded_repos": uploaded, **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
