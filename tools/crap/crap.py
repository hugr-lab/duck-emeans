#!/usr/bin/env python3
"""CRAP score of every C function in src/ (make crap).

CRAP = CC^2 * (1 - coverage)^3 + CC, with CC from lizard and coverage from clang's source-based
coverage (llvm-cov) over every test that exercises the code:

  - test/c/test_core.c with -DEMEANS_FAULT_INJECTION (every out-of-memory path),
  - test/c/test_threads.c,
  - the SQL logic tests in test/sql on a coverage build of the extension.

Coverage is region coverage (the share of a function's code regions that ran), merged over all
three. Exits 2 when any function scores at or above --threshold (default 10).

Needs clang, llvm-profdata and llvm-cov (xcrun finds them on macOS), uvx (for lizard), cmake, and
configure/venv from `make configure`. The pure functions have tests in test_crap.py
(`uvx pytest tools/crap`).
"""

import argparse
import csv
import io
import json
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build", "crap")
CORE = ["src/core/recipe.c"]
COVERAGE_FLAGS = ["-fprofile-instr-generate", "-fcoverage-mapping", "-O0", "-g"]
# lizard reports the entry point under its macro name; the body it measured compiles to
# <name>_init_c_api_internal (<name>_init_c_api is the wrapper the macro generates around it).
ALIASES = {"DUCKDB_EXTENSION_ENTRYPOINT": "emeans_init_c_api_internal"}
# Functions below this score are left out of the report unless --all is given.
SHOW_FROM = 5.0


def tool(name):
    if platform.system() == "Darwin":
        return ["xcrun", name]
    return [shutil.which(name) or name]


def run(cmd, env=None, quiet=True):
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=quiet, text=True)
    if out.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{out.stdout if quiet else ''}{out.stderr if quiet else ''}")
    return out.stdout


def complexity():
    """(file, function) -> cyclomatic complexity, from lizard."""
    rows = csv.reader(io.StringIO(run(["uvx", "--quiet", "lizard", "src", "--csv"])))
    return {(r[6], ALIASES.get(r[7], r[7])): int(r[1]) for r in rows}


def profile_env(name):
    return dict(os.environ, LLVM_PROFILE_FILE=os.path.join(BUILD, f"{name}-%p.profraw"))


def core_tests():
    """Build and run the C tests with coverage; returns the binaries."""
    binaries = []
    for test, defines in (("test_core", ["-DEMEANS_FAULT_INJECTION"]), ("test_threads", [])):
        binary = os.path.join(BUILD, test)
        run(["clang", "-std=c11", "-pthread", *COVERAGE_FLAGS, *defines, "-Isrc/include", f"test/c/{test}.c", *CORE,
             "-lm", "-o", binary])
        run([binary], env=profile_env(test))
        binaries.append(binary)
    return binaries


def sql_tests():
    """Build the extension with coverage, run the SQL logic tests on it; returns the library."""
    cmake_dir = os.path.join(BUILD, "cmake")
    flags = " ".join(COVERAGE_FLAGS)
    run(["cmake", "-S", ".", "-B", cmake_dir, "-DEXTENSION_NAME=emeans", "-DTARGET_DUCKDB_VERSION_MAJOR=1",
         "-DTARGET_DUCKDB_VERSION_MINOR=5", "-DTARGET_DUCKDB_VERSION_PATCH=6", "-DCMAKE_BUILD_TYPE=Debug",
         "-DCMAKE_C_COMPILER=clang", f"-DCMAKE_C_FLAGS={flags}", "-DCMAKE_SHARED_LINKER_FLAGS=-fprofile-instr-generate"])
    run(["cmake", "--build", cmake_dir])
    library = os.path.join(cmake_dir, "libemeans.dylib" if platform.system() == "Darwin" else "libemeans.so")
    extension = os.path.join(BUILD, "emeans.duckdb_extension")
    python = os.path.join(ROOT, "configure", "venv", "bin", "python3")
    run([python, "extension-ci-tools/scripts/append_extension_metadata.py", "-l", library, "-o", extension, "-n",
         "emeans", "-dv", "v1.5.6", "-evf", "configure/extension_version.txt", "-pf", "configure/platform.txt"])
    run([python, "-m", "duckdb_sqllogictest", "--test-dir", "test/sql", "--external-extension", extension],
        env=profile_env("sql"))
    return library


def merged_profile():
    """Merges every .profraw under BUILD into one .profdata; returns its path."""
    raw = [os.path.join(BUILD, f) for f in os.listdir(BUILD) if f.endswith(".profraw")]
    profdata = os.path.join(BUILD, "all.profdata")
    run([*tool("llvm-profdata"), "merge", "-sparse", *raw, "-o", profdata])
    return profdata


def exported_functions(profdata, binary):
    """The llvm-cov function records of one binary."""
    export = json.loads(run([*tool("llvm-cov"), "export", "-format=text", f"-instr-profile={profdata}", binary]))
    return export["data"][0]["functions"]


def source_key(fn):
    """(file, function) of an llvm-cov record in src/, or None for other files."""
    path = os.path.relpath(fn["filenames"][0], ROOT)
    return (path, fn["name"].split(":")[-1]) if path.startswith("src/") else None


def region_flags(fn):
    """Executed or not, per code region (kind 0) in the function's own file (file id 0)."""
    return [r[4] > 0 for r in fn["regions"] if r[7] == 0 and r[5] == 0]


def merge_flags(old, flags):
    """Region-wise OR of two runs of the same function; a region-count mismatch keeps the first run."""
    if not old:
        return flags
    if len(old) != len(flags):
        return old
    return [a or b for a, b in zip(old, flags)]


def add_functions(covered, functions):
    """Merges the records of one binary into covered: (file, function) -> region flags."""
    for fn in functions:
        key = source_key(fn)
        if key:
            covered[key] = merge_flags(covered.get(key), region_flags(fn))


def fraction(flags):
    """Share of regions that ran; a function without regions counts as fully covered."""
    return sum(flags) / len(flags) if flags else 1.0


def coverage(binaries):
    """(file, function) -> region coverage in [0, 1], merged over all binaries."""
    profdata = merged_profile()
    covered = {}
    for binary in binaries:
        add_functions(covered, exported_functions(profdata, binary))
    return {key: fraction(flags) for key, flags in covered.items()}


def crap(cc, cov):
    return cc * cc * (1 - cov) ** 3 + cc


def score_rows(cc, cov):
    """(CRAP, function, file, CC, coverage or None) per function, worst first; unknown coverage scores as 0."""
    rows = []
    for (path, name), c in cc.items():
        k = cov.get((path, name))
        rows.append((crap(c, 0.0 if k is None else k), name, path, c, k))
    return sorted(rows, key=lambda r: (-r[0], r[1]))


def format_row(score, name, path, cc, cov):
    shown = "N/A" if cov is None else f"{100 * cov:5.1f}"
    return f"{name:30s} {path:36s} {cc:3d} {shown:>6s} {score:6.1f}"


def report(rows, show_all, threshold):
    """The report lines and the number of functions at or above the threshold."""
    lines = [f"{len(rows)} functions in src/, worst first (CRAP = CC^2 (1 - cov)^3 + CC, region coverage)\n",
             f"{'function':30s} {'file':36s} {'CC':>3s} {'cov%':>6s} {'CRAP':>6s}"]
    lines += [format_row(*row) for row in rows if show_all or row[0] >= SHOW_FROM]
    failing = sum(1 for row in rows if row[0] >= threshold)
    lines.append(f"\nworst CRAP {rows[0][0]:.1f}; {failing} function(s) at or above {threshold:g}")
    return lines, failing


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--all", action="store_true", help=f"list every function, not only those at/above {SHOW_FROM:g}")
    return parser.parse_args()


def main():
    args = parse_args()
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(BUILD)
    rows = score_rows(complexity(), coverage(core_tests() + [sql_tests()]))
    lines, failing = report(rows, args.all, args.threshold)
    print("\n".join(lines))
    return 2 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
