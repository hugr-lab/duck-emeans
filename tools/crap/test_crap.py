"""Tests of the pure functions in crap.py: `uvx pytest tools/crap`. The llvm and cmake steps are not run."""

import os

import pytest

import crap


def test_run_returns_stdout_and_exits_with_both_streams_on_failure():
    assert crap.run(["printf", "ok"]) == "ok"
    assert crap.run(["true"], quiet=False) is None
    with pytest.raises(SystemExit) as failure:
        crap.run(["sh", "-c", "echo out; echo err >&2; exit 3"])
    assert str(failure.value) == "command failed: sh -c echo out; echo err >&2; exit 3\nout\nerr\n"


def record(path, name, regions):
    """An llvm-cov function record with the fields crap.py reads: filenames, name and regions."""
    return {"filenames": [os.path.join(crap.ROOT, path)], "name": name, "regions": regions}


# A region is [line_start, col_start, line_end, col_end, execution_count, file_id, expanded_file_id, kind].
def region(count, file_id=0, kind=0):
    return [1, 1, 2, 1, count, file_id, 0, kind]


def test_source_key_keeps_src_and_strips_the_file_prefix_of_static_names():
    assert crap.source_key(record("src/core/recipe.c", "recipe.c:law_batch", [])) == ("src/core/recipe.c", "law_batch")
    assert crap.source_key(record("src/x.c", "emeans_round_pow2", [])) == ("src/x.c", "emeans_round_pow2")
    assert crap.source_key(record("test/c/test_core.c", "main", [])) is None
    assert crap.source_key(record("duckdb_capi/duckdb.h", "f", [])) is None


def test_region_flags_count_code_regions_of_the_own_file_only():
    regions = [region(3), region(0), region(0, file_id=1), region(0, kind=2), region(7)]
    assert crap.region_flags(record("src/a.c", "f", regions)) == [True, False, True]


def test_merge_flags():
    assert crap.merge_flags(None, [True, False]) == [True, False]
    assert crap.merge_flags([], [True, False]) == [True, False]
    assert crap.merge_flags([True, False], [False, True]) == [True, True]
    # A different region count means a different build of the function: the first run is kept.
    assert crap.merge_flags([True, False, False], [True, True]) == [True, False, False]


def test_add_functions_merges_runs_and_ignores_other_files():
    covered = {}
    crap.add_functions(covered, [record("src/a.c", "f", [region(1), region(0)]), record("test/t.c", "main", [region(1)])])
    crap.add_functions(covered, [record("src/a.c", "f", [region(0), region(2)])])
    assert covered == {("src/a.c", "f"): [True, True]}


def test_fraction():
    assert crap.fraction([True, False, False, False]) == 0.25
    assert crap.fraction([]) == 1.0


def test_crap_formula():
    # CC 4 at 50%: 16 * 0.125 + 4 = 6; CC 3 untested: 9 + 3 = 12; CC 5 fully covered: 5.
    assert crap.crap(4, 0.5) == 6.0
    assert crap.crap(3, 0.0) == 12.0
    assert crap.crap(5, 1.0) == 5.0


def test_score_rows_sorts_worst_first_then_by_name_and_scores_unknown_coverage_as_zero():
    cc = {("src/a.c", "b"): 3, ("src/a.c", "a"): 3, ("src/a.c", "covered"): 4, ("src/a.c", "unknown"): 2}
    cov = {("src/a.c", "b"): 0.0, ("src/a.c", "a"): 0.0, ("src/a.c", "covered"): 1.0}
    rows = crap.score_rows(cc, cov)
    assert [(r[1], r[0]) for r in rows] == [("a", 12.0), ("b", 12.0), ("unknown", 6.0), ("covered", 4.0)]
    assert rows[2][4] is None


def test_report_lists_scores_from_five_unless_all_and_counts_failures():
    rows = crap.score_rows({("src/a.c", "big"): 12, ("src/a.c", "ok"): 2, ("src/a.c", "fine"): 3},
                           {("src/a.c", "big"): 0.5, ("src/a.c", "ok"): 1.0, ("src/a.c", "fine"): 1.0})
    lines, failing = crap.report(rows, show_all=False, threshold=10.0)
    # big: 144 * 0.125 + 12 = 30.0; the two covered ones score 2 and 3, below the display cut-off.
    assert lines[0].startswith("3 functions in src/")
    assert [line.split()[0] for line in lines[2:-1]] == ["big"]
    assert "  50.0" in lines[2] and lines[2].rstrip().endswith("30.0")
    assert lines[-1] == "\nworst CRAP 30.0; 1 function(s) at or above 10"
    assert failing == 1
    lines, failing = crap.report(rows, show_all=True, threshold=40.0)
    assert [line.split()[0] for line in lines[2:-1]] == ["big", "fine", "ok"]
    assert failing == 0


def test_format_row_shows_na_for_unknown_coverage():
    assert crap.format_row(6.0, "f", "src/a.c", 2, None).split() == ["f", "src/a.c", "2", "N/A", "6.0"]
