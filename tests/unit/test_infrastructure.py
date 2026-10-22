"""Static guards on the Docker setup: local-only ports, free extractor, pinned models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from retrieval_labs.retrieval.embeddings import BGE_SMALL
from retrieval_labs.retrieval.reranker import MINILM_RERANKER

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))


def test_only_the_api_publishes_a_port_and_only_on_loopback() -> None:
    services = load_yaml("compose.yaml")["services"]
    published = {name: s["ports"] for name, s in services.items() if "ports" in s}
    assert published == {"api": ["127.0.0.1:8000:8000"]}


def test_the_dev_override_publishes_only_on_loopback() -> None:
    for service in load_yaml("compose.dev.yaml")["services"].values():
        assert all(port.startswith("127.0.0.1:") for port in service["ports"])


def test_containers_never_get_the_paid_provider_or_its_key() -> None:
    compose = load_yaml("compose.yaml")
    for name, service in compose["services"].items():
        environment = service.get("environment") or {}
        assert "ANTHROPIC_API_KEY" not in environment, name
        if "LLM_PROVIDER" in environment:
            assert environment["LLM_PROVIDER"] == "fake", name


def test_the_ci_workflow_parses_and_checks_the_cpu_only_build() -> None:
    # An invalid workflow makes GitHub start a run with no jobs at all (LOG-019).
    jobs = load_yaml(".github/workflows/ci.yml")["jobs"]
    assert set(jobs) == {"checks", "integration", "image"}
    for name, job in jobs.items():
        for step in job["steps"]:
            assert isinstance(step.get("run", ""), str), name
    image_commands = [step.get("run", "") for step in jobs["image"]["steps"]]
    assert any('grep -q "CPU-only: torch" build.log' in run for run in image_commands)


def test_only_the_models_step_goes_online() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=1" in dockerfile
    online = {
        name
        for name, service in load_yaml("compose.yaml")["services"].items()
        if (service.get("environment") or {}).get("HF_HUB_OFFLINE") == "0"
    }
    assert online == {"models"}


def load_download_script():
    spec = importlib.util.spec_from_file_location(
        "download_models", ROOT / "scripts" / "download_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHfApi:
    def __init__(self, sha_override: str | None = None, size_delta: int = 0) -> None:
        self.sha_override = sha_override
        self.size_delta = size_delta

    def model_info(self, repo: str, revision: str, files_metadata: bool):
        expected = next(f for spec, f in self.expected.items() if spec.repo == repo)
        siblings = [
            SimpleNamespace(rfilename=name, size=size + self.size_delta)
            for name, size in expected.items()
        ]
        return SimpleNamespace(sha=self.sha_override or revision, siblings=siblings)


@pytest.fixture
def script():
    module = load_download_script()
    FakeHfApi.expected = module.EXPECTED
    return module


def test_download_list_pins_the_measured_models(script) -> None:
    assert set(script.EXPECTED) == {BGE_SMALL, MINILM_RERANKER}
    assert sum(sum(files.values()) for files in script.EXPECTED.values()) == 226_321_698


def test_matching_metadata_passes_the_check(script) -> None:
    assert script.check(FakeHfApi()) == []


def test_a_size_change_blocks_the_download(script) -> None:
    problems = script.check(FakeHfApi(size_delta=1))
    assert len(problems) == 17


def test_a_moved_revision_blocks_the_download(script) -> None:
    problems = script.check(FakeHfApi(sha_override="0" * 40))
    assert len(problems) == 2
