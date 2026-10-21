"""Guards against gold-annotation leakage into inference code (brief Section 4.5)."""

import ast
import json
from pathlib import Path

import retrieval_labs
from retrieval_labs.cuad.prepare import prepare_dataset
from retrieval_labs.cuad.store import CONTRACTS_FILE
from retrieval_labs.documents import Contract

GOLD_MODULE = "retrieval_labs.cuad.gold"
# Only dataset preparation and (future) evaluation code may touch gold annotations.
ALLOWED_GOLD_IMPORTERS = ("retrieval_labs.cuad", "retrieval_labs.evaluation")
CONTRACT_FIELDS = {"contract_id", "title", "text", "text_sha256"}

PACKAGE_DIR = Path(retrieval_labs.__file__).parent


def module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported_modules(source: str, module: str) -> set[str]:
    """Absolute names of everything a module imports, including ``from x import y`` as x.y."""
    package = module.rsplit(".", 1)[0]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, base] if base else anchor)
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def imports_gold(source: str, module: str) -> bool:
    return GOLD_MODULE in imported_modules(source, module)


def is_allowed_importer(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in ALLOWED_GOLD_IMPORTERS)


def test_only_allowed_packages_import_gold() -> None:
    violations = [
        module_name(path)
        for path in PACKAGE_DIR.rglob("*.py")
        if imports_gold(path.read_text(encoding="utf-8"), module_name(path))
        and not is_allowed_importer(module_name(path))
    ]
    assert violations == []


def test_boundary_check_detects_violations() -> None:
    """The checker itself must catch every import form, or the test above proves nothing."""
    module = "retrieval_labs.retrieval.bm25"
    for source in [
        "import retrieval_labs.cuad.gold",
        "from retrieval_labs.cuad.gold import GoldAnnotation",
        "from retrieval_labs.cuad import gold",
        "from ..cuad.gold import GoldSpan",
        "from ..cuad import gold",
    ]:
        assert imports_gold(source, module), source
        assert not is_allowed_importer(module)
    assert not imports_gold("from retrieval_labs.documents import Contract", module)


def test_documents_module_does_not_depend_on_cuad() -> None:
    source = (PACKAGE_DIR / "documents.py").read_text(encoding="utf-8")
    imports = imported_modules(source, "retrieval_labs.documents")
    assert not any(name.startswith("retrieval_labs.cuad") for name in imports)


# Inference code works on any Contract; it must not depend on the dataset package at all.
INFERENCE_MODULES = (
    "retrieval_labs.clauses",
    "retrieval_labs.documents",
    "retrieval_labs.ingestion",
    "retrieval_labs.retrieval",
    "retrieval_labs.extraction",
    "retrieval_labs.pipeline",
    "retrieval_labs.db",
    "retrieval_labs.worker",
)


def is_inference_module(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in INFERENCE_MODULES)


def test_inference_modules_do_not_import_the_dataset_package() -> None:
    checked, violations = [], []
    for path in PACKAGE_DIR.rglob("*.py"):
        module = module_name(path)
        if not is_inference_module(module):
            continue
        checked.append(module)
        imports = imported_modules(path.read_text(encoding="utf-8"), module)
        violations += [(module, n) for n in imports if n.startswith("retrieval_labs.cuad")]
    assert "retrieval_labs.pipeline" in checked
    assert violations == []


def importers_of(package: str) -> list[str]:
    return sorted(
        module_name(path)
        for path in PACKAGE_DIR.rglob("*.py")
        if any(
            n == package or n.startswith(package + ".")
            for n in imported_modules(path.read_text(encoding="utf-8"), module_name(path))
        )
    )


def test_vendor_sdk_is_isolated_in_one_module() -> None:
    assert importers_of("anthropic") == ["retrieval_labs.extraction.anthropic_extractor"]


def test_model_libraries_are_isolated_in_adapter_modules() -> None:
    allowed = {"retrieval_labs.retrieval.embeddings", "retrieval_labs.retrieval.reranker"}
    assert set(importers_of("sentence_transformers")) <= allowed
    assert importers_of("torch") == []
    assert importers_of("transformers") == []


def test_contract_has_no_annotation_fields() -> None:
    assert set(Contract.model_fields) == CONTRACT_FIELDS


def test_contracts_file_contains_only_contract_fields(cuad_archive: Path, tmp_path: Path) -> None:
    prepare_dataset(cuad_archive, tmp_path, dev_size=1, seed=42)
    lines = (tmp_path / CONTRACTS_FILE).read_text(encoding="utf-8").splitlines()
    assert lines
    assert all(set(json.loads(line)) == CONTRACT_FIELDS for line in lines)
