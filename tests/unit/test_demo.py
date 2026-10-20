"""The Streamlit demo: what it may import, and that it renders with and without the dataset.

The AppTest runs use the BM25 retriever, so no model is loaded; the demo builds every method
through the same factory as the benchmark (tests/unit/test_factory.py).
"""

import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "demo" / "streamlit_app.py"
PROCESSED = ROOT / "data" / "processed" / "cuad"


def imported_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def forbidden_imports(path: Path) -> list[str]:
    forbidden = ("retrieval_labs.cuad.gold", "retrieval_labs.extraction", "anthropic")
    names = imported_names(path)
    return sorted(n for n in names if n.startswith(forbidden) or n.endswith(".load_gold"))


def test_demo_never_touches_gold_or_llm_extraction() -> None:
    assert forbidden_imports(APP) == []


def test_import_check_detects_violations(tmp_path: Path) -> None:
    for source in [
        "from retrieval_labs.cuad.store import load_contracts, load_gold",
        "from retrieval_labs.cuad import gold",
        "import retrieval_labs.extraction.anthropic_extractor",
        "from anthropic import Anthropic",
    ]:
        path = tmp_path / "app.py"
        path.write_text(source, encoding="utf-8")
        assert forbidden_imports(path), source


def run_app(monkeypatch: pytest.MonkeyPatch, processed_dir: Path) -> AppTest:
    monkeypatch.setenv("RETRIEVAL_LABS_DEMO_RETRIEVER", "bm25")
    monkeypatch.setenv("RETRIEVAL_LABS_PROCESSED_DIR", str(processed_dir))
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    return app


def test_demo_without_dataset_explains_setup_and_still_shows_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = run_app(monkeypatch, tmp_path)
    assert len(app.error) == 1
    assert "prepare_cuad.py --download" in app.error[0].value
    subheaders = [s.value for s in app.subheader]
    assert subheaders == [
        "Test split — evaluated once, all methods frozen",
        "Dev split — used to compare methods before the test run",
    ]
    assert any("| hybrid+rerank |" in m.value for m in app.markdown)


@pytest.mark.data
def test_demo_shows_evidence_for_every_clause_type(monkeypatch: pytest.MonkeyPatch) -> None:
    if not (PROCESSED / "manifest.json").is_file():
        pytest.skip("prepared CUAD dataset not found")
    app = run_app(monkeypatch, PROCESSED)
    assert not app.error
    assert len(app.selectbox[0].options) == 102
    thresholds = [c.value for c in app.caption if c.value.startswith("Top-1 score")]
    assert len(thresholds) == 5
    app.selectbox[0].select_index(1).run()
    assert not app.exception
    assert len([c for c in app.caption if c.value.startswith("Top-1 score")]) == 5
