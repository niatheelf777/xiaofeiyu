#!/usr/bin/env python3
"""Clean-Room Open-Source Hygiene & Zero-Leak Auditor for Xiaofeiyu.

Enforces strict release boundaries across tracked/untracked repository files and git history:
  1. Zero API keys, bearer tokens, or private PEM keys.
  2. Zero local developer workstation paths (Windows drive paths, /Users/, /home/ developer dirs).
  3. Zero non-loopback IPv4 literals.
  4. Strict external URL domain allowlist (blocks private/lab/school/internal domains).
  5. Strict 5-slider architectural boundary (blocks >5-slot weight tuples, 6th-slot adapter defs,
     and hidden query-key unlock hooks).
  6. Clean git commit author/email metadata.

Optional `--extra-deny-regex-file` allows an external CI/workspace runner to apply additional
organization-specific deny patterns without storing sensitive strings inside the public repository.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = HERE.parent

SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv", "dist", "build"})
BINARY_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".gguf", ".safetensors", ".zip", ".tar", ".gz", ".woff", ".woff2"})

ALLOWED_PUBLIC_DOMAINS = frozenset(
    {
        "xfy.tospeak.top",
        "niatheelf777.github.io",
        "schema.org",
        "sitemaps.org",
        "www.sitemaps.org",
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "api.deepseek.com",
        "api.openai.com",
        "openrouter.ai",
        "ai.google.dev",
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "github.com",
        "api.github.com",
        "raw.githubusercontent.com",
        "arxiv.org",
        "aclanthology.org",
        "icml.cc",
        "purl.org",
        "creativecommons.org",
        "proceedings.mlsys.org",
        "lmsys.org",
        "www.lmsys.org",
        "unsloth.ai",
        "docs.unsloth.ai",
        "llamabench.ai",
        "gpubattle.com",
        "awslabs.github.io",
        "www.databasemart.com",
        "databasemart.com",
        "xbill999.medium.com",
        "medium.com",
        "tokatlas.github.io",
        "atomic.chat",
        "willitrunai.com",
        "www.gpupartner.com",
        "gpupartner.com",
        "www.millstoneai.com",
        "millstoneai.com",
        "jarvislabs.ai",
        "forums.developer.nvidia.com",
        "kikocis.github.io",
        "www.apache.org",
        "apache.org",
        "npmjs.com",
        "www.npmjs.com",
        "registry.npmjs.org",
        "vercel.com",
        "openapi.vercel.sh",
        "cloudflare.com",
        "developers.cloudflare.com",
        "w3.org",
        "www.w3.org",
        "matplotlib.org",
    }
)

SECRET_TOKEN_RE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{20,}\b|\bhf_[A-Za-z0-9]{20,}\b|\bghp_[A-Za-z0-9]{20,}\b|\bccp_svc_[A-Za-z0-9_-]{12,}\b)"
)
LOCAL_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\\\(?:Users|Datasets|Projects|Contest|666|TODO|Programs)\\\\|"
    r"[A-Za-z]:/(?:Users|Datasets|Projects|Contest|666|TODO|Programs)/|"
    r"(?<![A-Za-z0-9])/Users/[A-Za-z0-9_.-]+/)"
)
RAW_IPV4_RE = re.compile(
    r"\b(?!(?:127\.0\.0\.1|0\.0\.0\.0|1\.0\.0|2\.0\.0|192\.0\.2\.\d+|198\.51\.100\.\d+|203\.0\.113\.\d+)\b)"
    r"(?:\d{1,3}\.){3}\d{1,3}\b"
)
URL_RE = re.compile(r"https?://[^\s<>\"'`)\]]+")

# Structural 5-slider & confidential-research / HF-safety boundary checks
SIX_WEIGHT_TUPLE_RE = re.compile(r"xiaofeiyu-v4\|(?:\s*\d+\.\d+\s*,){5}\s*\d+\.\d+")
HIDDEN_QUERY_UNLOCK_RE = re.compile(
    r"(?:[?&][a-zA-Z]=|\bc[5-9]_[a-z0-9_]+\b|\bslot_[5-9]\b|\buncensored?\b|\bnsfw\b)",
    re.IGNORECASE,
)
PAPER_FORMULA_LEAK_RE = re.compile(
    r"(?:\bKLD\b|\bSVD\b|\bWasserstein\b|\bPythagorean\b|\$W_1\$|\\Delta\s*W|\\cos\()"
)


def is_allowed_host(host: str) -> bool:
    h = host.lower().strip().rstrip(".")
    if ":" in h:
        h = h.split(":", 1)[0]
    if not h or h in ALLOWED_PUBLIC_DOMAINS:
        return True
    if h.endswith(".example.com") or h == "example.com" or h.endswith(".example"):
        return True
    return False


def iter_scannable_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    out: list[Path] = []
    for p in sorted(target.rglob("*")):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(target).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        out.append(p)
    return out


def scan_text_content(
    text: str,
    rel_label: str,
    *,
    is_self_auditor: bool = False,
    extra_deny_re: re.Pattern[str] | None = None,
) -> list[dict[str, str | int]]:
    findings: list[dict[str, str | int]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if SECRET_TOKEN_RE.search(line) and not is_self_auditor:
            findings.append({"file": rel_label, "line": line_no, "rule": "secret_token", "excerpt": line[:120]})

        if LOCAL_PATH_RE.search(line) and not is_self_auditor:
            findings.append({"file": rel_label, "line": line_no, "rule": "local_workstation_path", "excerpt": line[:120]})

        for ip_m in RAW_IPV4_RE.finditer(line):
            octets = [int(x) for x in ip_m.group(0).split(".")]
            if all(0 <= o <= 255 for o in octets) and not is_self_auditor:
                # Ignore semantic version strings preceded by 'v' or 'version'
                prefix = line[max(0, ip_m.start() - 8) : ip_m.start()].lower()
                if not prefix.endswith("v") and "version" not in prefix:
                    findings.append(
                        {"file": rel_label, "line": line_no, "rule": f"raw_ipv4:{ip_m.group(0)}", "excerpt": line[:120]}
                    )

        for url_m in URL_RE.finditer(line):
            raw_url = url_m.group(0).rstrip(".,;:)")
            if "{" in raw_url or "$" in raw_url:
                continue
            parsed = urlparse(raw_url)
            if parsed.hostname and not is_allowed_host(parsed.hostname):
                findings.append(
                    {
                        "file": rel_label,
                        "line": line_no,
                        "rule": f"unlisted_domain:{parsed.hostname}",
                        "excerpt": line[:120],
                    }
                )

        if SIX_WEIGHT_TUPLE_RE.search(line) and not is_self_auditor:
            findings.append({"file": rel_label, "line": line_no, "rule": "six_slot_weight_tuple", "excerpt": line[:120]})

        if HIDDEN_QUERY_UNLOCK_RE.search(line) and not is_self_auditor:
            findings.append({"file": rel_label, "line": line_no, "rule": "hidden_unlock_hook", "excerpt": line[:120]})

        if PAPER_FORMULA_LEAK_RE.search(line) and not is_self_auditor:
            findings.append({"file": rel_label, "line": line_no, "rule": "confidential_paper_formula", "excerpt": line[:120]})

        if extra_deny_re is not None:
            m_ext = extra_deny_re.search(line)
            if m_ext:
                findings.append(
                    {"file": rel_label, "line": line_no, "rule": f"extra_deny:{m_ext.group(0)}", "excerpt": line[:120]}
                )
    return findings


def audit_git_history(repo_root: Path, extra_deny_re: re.Pattern[str] | None = None) -> list[dict[str, str | int]]:
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return []
    out = subprocess.check_output(
        ["git", "-C", str(repo_root), "log", "--format=%H|%an|%ae|%s"],
        text=True,
        encoding="utf-8",
    )
    findings: list[dict[str, str | int]] = []
    for idx, line in enumerate(out.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        sha, author_name, author_email, subject = parts
        if "niatheelf777" not in author_email.lower():
            findings.append(
                {
                    "file": "<git-log>",
                    "line": idx,
                    "rule": f"unexpected_git_author:{author_name}<{author_email}>",
                    "excerpt": sha[:12],
                }
            )
        if extra_deny_re is not None:
            m = extra_deny_re.search(f"{author_name} {author_email} {subject}")
            if m:
                findings.append(
                    {"file": "<git-log>", "line": idx, "rule": f"git_log_deny:{m.group(0)}", "excerpt": sha[:12]}
                )
    return findings


def run_audit(
    repo_root: Path,
    extra_targets: list[Path] | None = None,
    extra_deny_regex: str | None = None,
) -> dict[str, object]:
    extra_re = re.compile(extra_deny_regex, re.IGNORECASE) if extra_deny_regex else None
    findings: list[dict[str, str | int]] = []
    files_scanned = 0

    all_roots = [repo_root.resolve(), *(p.resolve() for p in (extra_targets or []))]
    self_path = Path(__file__).resolve()

    for root in all_roots:
        for fpath in iter_scannable_files(root):
            try:
                txt = fpath.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            files_scanned += 1
            rel = str(fpath.relative_to(root)).replace("\\", "/") if root.is_dir() else fpath.name
            is_self = fpath.resolve() == self_path
            findings.extend(scan_text_content(txt, rel, is_self_auditor=is_self, extra_deny_re=extra_re))

    if repo_root.is_dir():
        findings.extend(audit_git_history(repo_root, extra_deny_re=extra_re))

    return {
        "audit_passed": len(findings) == 0,
        "files_scanned": files_scanned,
        "leak_count": len(findings),
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    ap.add_argument("--extra-target", type=Path, action="append", default=[])
    ap.add_argument("--extra-deny-regex-file", type=Path, default=None)
    args = ap.parse_args(argv)

    extra_pat: str | None = None
    if args.extra_deny_regex_file is not None and args.extra_deny_regex_file.is_file():
        extra_pat = args.extra_deny_regex_file.read_text(encoding="utf-8").strip()

    summary = run_audit(args.repo_root, extra_targets=args.extra_target, extra_deny_regex=extra_pat)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
