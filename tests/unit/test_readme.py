"""The README's results table must be exactly what the stored test run produces."""

from pathlib import Path

from retrieval_labs.evaluation.report_tables import TEST_RUN, load_rows, markdown_table

ROOT = Path(__file__).resolve().parents[2]


def test_readme_results_table_matches_the_stored_test_run() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = markdown_table(load_rows(TEST_RUN, ROOT / "reports" / "benchmark"))
    assert table in readme


def test_readme_links_point_to_existing_files() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in [
        "LICENSE",
        "docs/EVALUATION.md",
        "docs/ENGINEERING_LOG.md",
        "docs/DECISIONS.md",
        "reports/benchmark/",
        "demo/streamlit_app.py",
        "scripts/prepare_cuad.py",
    ]:
        assert target in readme, target
        assert (ROOT / target).exists(), target
