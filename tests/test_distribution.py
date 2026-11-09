"""Installed-product boundaries; no model download or native training."""
import hashlib
from importlib.metadata import distribution
from importlib.resources import files
import json
from pathlib import Path

from conceptmod.textsliders import h3_cli, minimax_h3_backend, pika_h3_flex


def test_installed_distribution_contains_exact_runtime_and_data():
    dist = distribution("minimax-h3-particle-sliders")
    assert dist.version == "0.1.0.dev0"
    resources = files("conceptmod.textsliders")
    manifest = json.loads(resources.joinpath("data/h3-extraction.json").read_text())
    assert manifest["source_commit"] == "fae34ef80f8ad0fbe02e5d9abc3c0e770798d5f7"
    for record in manifest["sources"]:
        name = record["destination_path"]
        if not name.startswith("conceptmod/"):
            continue
        installed = Path(dist.locate_file(name))
        assert installed.is_file(), name
        assert hashlib.sha256(installed.read_bytes()).hexdigest() == record.get("current_sha256", record["sha256"]), name
    python_files = {str(p) for p in dist.files if str(p).endswith(".py")}
    expected = {r["destination_path"] for r in manifest["sources"] if r["destination_path"].endswith(".py")
                and r["destination_path"].startswith("conceptmod/")}
    assert python_files == expected | {"conceptmod/textsliders/h3_cli.py"}
    assert not any(str(p).startswith(("particle_sliders/", "hypergan/", "hndl/")) for p in dist.files)
    assert set(resources.joinpath("data").iterdir()).issuperset(resources.joinpath("data", name) for name in
        ("pika-h3-flex-slider.hndl", "pika-h3-flex-critic.hndl", "pika-h3-flex-release.json", "pika-h3-flex-session.json"))
    assert Path(minimax_h3_backend.__file__).resolve() == Path(dist.locate_file("conceptmod/textsliders/minimax_h3_backend.py")).resolve()
    assert pika_h3_flex.RELEASE["model_id"] == "Pika-Labs/Pika-H3-Flex"


def test_installed_entrypoints_and_legacy_wrappers(monkeypatch):
    dist = distribution("minimax-h3-particle-sliders")
    scripts = {p.name: p for p in dist.entry_points if p.group == "console_scripts"}
    assert set(scripts) == {"h3-flex", "h3-legacy-uni", "h3-legacy-diag"}
    assert not any(p.group == "particle_sliders.response_backends" for p in dist.entry_points)
    from conceptmod.textsliders import train_lora_minimax_h3, diag_minimax_h3_uni
    calls = []
    monkeypatch.setattr(train_lora_minimax_h3, "main", lambda: calls.append("uni") or {"result": "fixture"})
    monkeypatch.setattr(diag_minimax_h3_uni, "main", lambda: calls.append("diag") or {"result": "fixture"})
    assert scripts["h3-legacy-uni"].load()() is None
    assert scripts["h3-legacy-diag"].load()() is None
    assert calls == ["uni", "diag"]
    assert scripts["h3-flex"].load() is pika_h3_flex.main
