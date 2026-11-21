"""Bounded H3 consumer checks for the deliberate current shared-source cohort."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest
import torch

pytest.importorskip("particle_sliders", reason="current consumer checks require installed shared core")
pytest.importorskip("diffusers", reason="current consumer checks require Pika-native Diffusers")
from conceptmod.textsliders import pika_h3_flex_session as h3
from particle_sliders.lifecycle import SessionConfig, CheckpointError
from particle_sliders.engine import _hash


@pytest.fixture
def original_fixture():
    # Preserve the original factory bytes and their source identity. Do not copy
    # the generic lifecycle or rewrite packet metadata to accept another cohort.
    spec = importlib.util.spec_from_file_location("h3_current_fixture", Path(__file__).with_name("test_pika_h3_flex_session.py"))
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    return fixture


@pytest.mark.parametrize("tamper", ["changed", "missing", "extra"])
def test_complete_core_closure_rejects_tampering(tmp_path, monkeypatch, tamper):
    import particle_sliders
    lock = h3.require_shared_runtime()
    assert lock["core_revision"] == "ec1161f5d74481be3540cb0dfa63d0e8f8b54e90"
    assert len(lock["core_sources"]) == 59
    source = Path(particle_sliders.__file__).parent
    for name in lock["core_sources"]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    monkeypatch.setattr(particle_sliders, "__file__", str(tmp_path / "__init__.py"))
    assert h3.require_shared_runtime() == lock
    if tamper == "changed":
        (tmp_path / "inference.py").write_text("# changed source\n")
    elif tamper == "missing":
        (tmp_path / "inference.py").unlink()
    else:
        (tmp_path / "unexpected.py").write_text("# unreviewed source\n")
    with pytest.raises(RuntimeError, match="source closure differs"):
        h3.require_shared_runtime()


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_h3_current_exact_resume_raw_and_off(original_fixture, tmp_path, preset, checkpointing):
    fixture = original_fixture
    config = SessionConfig(fixture.FIXTURE.name, "h3-current-cpu-consumer-v1", seed=7,
                           steps=4, batch=2, ema_decay=None, preset=preset, checkpointing=checkpointing)
    with fixture.open_fixture(config) as whole:
        whole.train()
        expected = _hash(whole.game.state_dict())
    with fixture.open_fixture(config) as first:
        baseline = first.game.rows[0].neutral.clone()
        torch.testing.assert_close(first.evaluate(0, strength=0), baseline, rtol=0, atol=0)
        first.train(stop_after=2)
        first.checkpoint(tmp_path / "resume.pt")
    with fixture.open_fixture(config, resume=tmp_path / "resume.pt") as resumed:
        resumed.train()
        assert _hash(resumed.game.state_dict()) == expected
        torch.testing.assert_close(resumed.evaluate(0, strength=0), baseline, rtol=0, atol=0)
        runtime = resumed.game.backend.runtime
        runtime._check_fixed_pika()
        artifact = tmp_path / "raw.pt"
        resumed.export(artifact, selection="raw")
        expected_weights = deepcopy(resumed.game.network.state_dict())
        with fixture.open_fixture(config, artifact=artifact, operation="response") as inference:
            assert _hash(inference.game.network.state_dict()) == _hash(expected_weights)
            torch.testing.assert_close(inference.evaluate(0, strength=0), baseline, rtol=0, atol=0)
        with pytest.raises(ValueError, match="raw|RAW"):
            resumed.export(selection="ema")


def test_h3_update_error_keeps_primary_and_closes_independent_forwards(original_fixture, monkeypatch):
    fixture = original_fixture
    config = SessionConfig(fixture.FIXTURE.name, "h3-current-cleanup-v1", seed=7, steps=2, batch=2, checkpointing=True)
    session = fixture.open_fixture(config)
    adapter = session.game.backend
    runtime = adapter.runtime
    original_close = adapter.close
    marker = ValueError("H3 update primary")
    def fail_step():
        raise marker
    def fail_close():
        raise RuntimeError("H3 checkpoint close secondary")
    monkeypatch.setattr(session.game, "step", fail_step)
    monkeypatch.setattr(adapter, "close", fail_close)
    try:
        with pytest.raises(ValueError) as caught:
            session.step()
        assert caught.value is marker
        assert any("H3 checkpoint close secondary" in note for note in marker.__notes__)
        assert runtime.network.loras == [] and adapter._handles
        assert not session.closed
    finally:
        monkeypatch.setattr(adapter, "close", original_close)
        session.close()
    assert session.closed and adapter._handles == []
