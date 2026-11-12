#!/usr/bin/env python3
"""CRAP gate for the C in src/ (make crap).

Builds every test that exercises the code with clang's source-based coverage, runs them, writes one
LCOV report, and scores it with crap_cpp.py (lizard's complexity per function, line coverage from the
report, CRAP = CC^2 * (1 - coverage)^3 + CC):

  - test/c/test_core.c with -DEMEANS_FAULT_INJECTION (every out-of-memory path),
  - test/c/test_threads.c,
  - the SQL logic tests in test/sql on a coverage build of the extension.

Exits 2 when any function scores at or above --threshold (default 10).

Needs clang, llvm-profdata and llvm-cov (xcrun finds them on macOS), lizard (on PATH, or through uvx),
cmake, and configure/venv from `make configure`. `python3 tools/crap/crap_cpp.py --selftest` checks the
scoring; `uvx pytest tools/crap` runs that and this file's own tests.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUILD = os.path.join(ROOT, "build", "crap")
# Absolute, like cmake's paths, so llvm-cov sees one source file across all the binaries.
CORE = [os.path.join(ROOT, f) for f in ("src/core/recipe.c", "src/core/rng.c", "src/core/assign.c",
                                       "src/core/parallel.c", "src/core/emeans.c", "src/core/geo.c")]
# As in the Makefile and CMake: on macOS the kernel uses Accelerate.
PLATFORM_FLAGS = ["-DEMEANS_USE_ACCELERATE", "-framework", "Accelerate"] if platform.system() == "Darwin" else []
COVERAGE_FLAGS = ["-fprofile-instr-generate", "-fcoverage-mapping", "-O0", "-g"]
SCORER = os.path.join(ROOT, "tools", "crap", "crap_cpp.py")


def tool(name):
    if platform.system() == "Darwin":
        return ["xcrun", name]
    return [shutil.which(name) or name]


def run(cmd, env=None, quiet=True):
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=quiet, text=True)
    if out.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{out.stdout if quiet else ''}{out.stderr if quiet else ''}")
    return out.stdout


def lizard():
    """lizard from PATH, or through uvx."""
    return [shutil.which("lizard")] if shutil.which("lizard") else ["uvx", "--quiet", "lizard"]


def profile_env(name):
    return dict(os.environ, LLVM_PROFILE_FILE=os.path.join(BUILD, f"{name}-%p.profraw"))


def core_tests():
    """Build and run the C tests with coverage; returns the binaries."""
    binaries = []
    for test, defines in (("test_core", ["-DEMEANS_FAULT_INJECTION"]), ("test_threads", [])):
        binary = os.path.join(BUILD, test)
        run(["clang", "-std=c11", "-pthread", *COVERAGE_FLAGS, *defines, "-Isrc/include",
             os.path.join(ROOT, f"test/c/{test}.c"), *CORE, *PLATFORM_FLAGS, "-lm", "-o", binary])
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


def lcov_report(binaries):
    """One LCOV report over every binary, from the merged profiles; returns its path."""
    raw = [os.path.join(BUILD, f) for f in os.listdir(BUILD) if f.endswith(".profraw")]
    profdata = os.path.join(BUILD, "all.profdata")
    run([*tool("llvm-profdata"), "merge", "-sparse", *raw, "-o", profdata])
    objects = [arg for binary in binaries[1:] for arg in ("-object", binary)]
    report = os.path.join(BUILD, "coverage.lcov")
    with open(report, "w") as out:
        out.write(run([*tool("llvm-cov"), "export", "-format=lcov", f"-instr-profile={profdata}", binaries[0], *objects]))
    return report


def lizard_csv():
    """lizard's csv over src/; returns its path."""
    path = os.path.join(BUILD, "lizard.csv")
    with open(path, "w") as out:
        out.write(run([*lizard(), "--csv", "-l", "cpp", "src"]))
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--all", action="store_true", help="list every function, not only the worst 30")
    return parser.parse_args()


def main():
    args = parse_args()
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(BUILD)
    report = lcov_report(core_tests() + [sql_tests()])
    scorer = [sys.executable, SCORER, "--lizard-csv", lizard_csv(), "--lcov", report, "--threshold", str(args.threshold),
              "--limit", "0" if args.all else "30", "src"]
    return subprocess.run(scorer, cwd=ROOT).returncode


if __name__ == "__main__":
    sys.exit(main())
