#!/usr/bin/env python3
"""CRAP gate for the C in src/ (make crap).

Builds every test that exercises the code with clang's source-based coverage, runs them, writes one
LCOV report, and scores it with crap_cpp.py (lizard's complexity per function, line coverage from the
report, CRAP = CC^2 * (1 - coverage)^3 + CC):

  - test/c/test_core.c with -DEMEANS_FAULT_INJECTION (every out-of-memory path),
  - test/c/test_threads.c,
  - the SQL logic tests in test/sql, in DuckDB's unittest with the C++ extension linked in and only the
    extension's sources instrumented.

Exits 2 when any function scores at or above --threshold (default 10).

Needs clang, llvm-profdata and llvm-cov (xcrun finds them on macOS), lizard (on PATH, or through uvx),
cmake and ninja (the first run builds DuckDB, as `make release` does). `python3 tools/crap/crap_cpp.py --selftest` checks the
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
    """Build DuckDB and this extension, only the extension instrumented (EMEANS_COVERAGE), and run the SQL
    logic tests in DuckDB's unittest. unittest loads the extension from the build's local repository, as
    `make test_release` does, so the coverage is in the loadable binary; returns it."""
    cmake_dir = os.path.join(BUILD, "duckdb")
    run(["cmake", "-G", "Ninja", "-S", "duckdb", "-B", cmake_dir, "-DCMAKE_BUILD_TYPE=Release",
         f"-DDUCKDB_EXTENSION_CONFIGS={os.path.join(ROOT, 'extension_config.cmake')}", "-DEMEANS_COVERAGE=ON",
         "-DOVERRIDE_GIT_DESCRIBE=v2.0.0", "-DENABLE_UNITTEST_CPP_TESTS=FALSE",
         f"-DUNITTEST_ROOT_DIRECTORY={ROOT}/", "-DCMAKE_C_COMPILER=clang",
         "-DCMAKE_CXX_COMPILER=clang++",
         "-DCMAKE_EXE_LINKER_FLAGS=-fprofile-instr-generate", "-DCMAKE_SHARED_LINKER_FLAGS=-fprofile-instr-generate"])
    run(["cmake", "--build", cmake_dir])
    out = run([os.path.join(cmake_dir, "test", "unittest"), "test/sql/*"], env=profile_env("sql"))
    # A missing extension skips every file and still passes.
    if "require emeans" in out:
        sys.exit(f"the SQL tests skipped on 'require emeans':\n{out}")
    return os.path.join(cmake_dir, "extension", "emeans", "emeans.duckdb_extension")


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
