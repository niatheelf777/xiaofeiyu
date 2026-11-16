#!/usr/bin/env python3
"""Unit tests for HuggingFace release bundle validation."""
from __future__ import annotations

from pathlib import Path
import pytest
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_zero_leak import run_audit  # noqa: E402
from publish_hf_release import validate_release_bundle  # noqa: E402


def test_repo_passes_hf_release_validation():
    summary = validate_release_bundle(REPO_ROOT)
    assert summary["audit_passed"] is True
    assert len(summary["cards_checked"]) == 3


def test_repo_passes_zero_leak_audit():
    summary = run_audit(REPO_ROOT)
    assert summary["audit_passed"] is True, f"Unexpected leaks: {summary['findings']}"
    assert summary["leak_count"] == 0


def test_zero_leak_auditor_catches_paths_domains_and_6slot(tmp_path: Path):
    bad = tmp_path / "bad.md"
    win_path = "C:/" + "Users/alice/secret"
    bad_url = "https://" + "internal.corp.invalid/v1"
    six_tup = "xiaofeiyu-v4|" + "1.0,0.0,0.0,0.0,0.0,1.0"
    bad.write_text(f"Path {win_path} and {bad_url} and {six_tup}\n", encoding="utf-8")
    res = run_audit(tmp_path)
    assert res["audit_passed"] is False
    rules = {str(f["rule"]).split(":")[0] for f in res["findings"]}
    assert "local_workstation_path" in rules
    assert "unlisted_domain" in rules
    assert "six_slot_weight_tuple" in rules


def test_secret_scanner_blocks_hardcoded_api_token(tmp_path: Path):
    for slug in ("xiaofeiyu-26b-a4b", "xiaofeiyu-12b-v4", "xiaofeiyu-translator-e4b"):
        d = tmp_path / "hf-cards" / slug
        d.mkdir(parents=True)
        (d / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Card\n", encoding="utf-8")
    (tmp_path / "leak.env").write_text("HF_TOKEN=" + "hf_" + "a" * 28 + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Secret token detected"):
        validate_release_bundle(tmp_path)
