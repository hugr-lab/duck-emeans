"""Tests of crap.py, the `make crap` orchestration: `uvx pytest tools/crap`. The llvm, cmake and lizard steps are not run."""

import pytest

import crap


def test_run_returns_stdout_and_exits_with_both_streams_on_failure():
    assert crap.run(["printf", "ok"]) == "ok"
    assert crap.run(["true"], quiet=False) is None
    with pytest.raises(SystemExit) as failure:
        crap.run(["sh", "-c", "echo out; echo err >&2; exit 3"])
    assert str(failure.value) == "command failed: sh -c echo out; echo err >&2; exit 3\nout\nerr\n"


def test_lizard_is_found_on_path_or_run_through_uvx(monkeypatch):
    monkeypatch.setattr(crap.shutil, "which", lambda name: "/opt/bin/lizard")
    assert crap.lizard() == ["/opt/bin/lizard"]
    monkeypatch.setattr(crap.shutil, "which", lambda name: None)
    assert crap.lizard() == ["uvx", "--quiet", "lizard"]


def test_tool_uses_xcrun_on_macos_only(monkeypatch):
    monkeypatch.setattr(crap.platform, "system", lambda: "Darwin")
    assert crap.tool("llvm-cov") == ["xcrun", "llvm-cov"]
    monkeypatch.setattr(crap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(crap.shutil, "which", lambda name: None)
    assert crap.tool("llvm-cov") == ["llvm-cov"]
