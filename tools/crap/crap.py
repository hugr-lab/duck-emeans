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
configure/venv from `make configure`.
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
# lizard reports the entry point under its macro name; the compiled function is <name>_init_c_api.
ALIASES = {"DUCKDB_EXTENSION_ENTRYPOINT": "emeans_init_c_api"}


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


def coverage(binaries):
    """(file, function) -> region coverage in [0, 1], merged over all binaries."""
    raw = [os.path.join(BUILD, f) for f in os.listdir(BUILD) if f.endswith(".profraw")]
    profdata = os.path.join(BUILD, "all.profdata")
    run([*tool("llvm-profdata"), "merge", "-sparse", *raw, "-o", profdata])
    covered = {}
    for binary in binaries:
        export = json.loads(run([*tool("llvm-cov"), "export", "-format=text", f"-instr-profile={profdata}", binary]))
        for fn in export["data"][0]["functions"]:
            path = os.path.relpath(fn["filenames"][0], ROOT)
            if not path.startswith("src/"):
                continue
            # code regions (kind 0) in the function's own file (file id 0): executed or not
            flags = [r[4] > 0 for r in fn["regions"] if r[7] == 0 and r[5] == 0]
            key = (path, fn["name"].split(":")[-1])
            old = covered.get(key)
            covered[key] = [a or b for a, b in zip(old, flags)] if old and len(old) == len(flags) else old or flags
    return {key: (sum(f) / len(f) if f else 1.0) for key, f in covered.items()}


def crap(cc, cov):
    return cc * cc * (1 - cov) ** 3 + cc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--all", action="store_true", help="list every function, not only those at/above 5")
    args = parser.parse_args()

    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(BUILD)
    cc = complexity()
    cov = coverage(core_tests() + [sql_tests()])

    rows = []
    for (path, name), c in cc.items():
        k = cov.get((path, name))
        rows.append((crap(c, k if k is not None else 0.0), name, path, c, k))
    rows.sort(key=lambda r: (-r[0], r[1]))

    print(f"{len(rows)} functions in src/, worst first (CRAP = CC^2 (1 - cov)^3 + CC, region coverage)\n")
    print(f"{'function':30s} {'file':36s} {'CC':>3s} {'cov%':>6s} {'CRAP':>6s}")
    for score, name, path, c, k in rows:
        if args.all or score >= 5:
            shown = "N/A" if k is None else f"{100 * k:5.1f}"
            print(f"{name:30s} {path:36s} {c:3d} {shown:>6s} {score:6.1f}")
    worst = rows[0][0]
    failing = [r for r in rows if r[0] >= args.threshold]
    print(f"\nworst CRAP {worst:.1f}; {len(failing)} function(s) at or above {args.threshold:g}")
    return 2 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
