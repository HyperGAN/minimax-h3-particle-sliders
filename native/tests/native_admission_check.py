"""Allocation-free checks for the installed native migration entry points."""
from importlib import import_module

import pytest

from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation


@pytest.mark.parametrize("variant", ["native", "phase"])
def test_fresh_training_refuses_before_constructing_native_runtime(variant, tmp_path, monkeypatch):
    package = "minimax_h3_" + variant
    contracts = import_module(package + ".contracts")
    lifecycle = import_module(package + ".lifecycle")
    cache = tmp_path / "saved-cache"
    cache.touch()
    options = {
        "snapshot": str(tmp_path), "device": "cuda:0", "ema_decay": None,
        "cache": str(cache), "cache_sha256": "0" * 64,
        "pg_version": "0.11.0", "pg_source_profile": "release-0.11.0",
        "penalty_input_domain": "continuous", "source_profile": contracts.SOURCE_PROFILE,
        "initialization_policy": contracts.INITIALIZATION_POLICY,
        "checkpointing": True,
    }
    if variant == "phase":
        options.update(conditioning=str(cache), conditioning_sha256="0" * 64)
    config = SessionConfig.from_dict(
        {"backend": contracts.BACKEND_NAME, "cohort": "source-test", "options": options}
    )

    def unexpected_loader(*args, **kwargs):
        pytest.fail("Fresh BF16 training reached the native loader")

    monkeypatch.setattr(lifecycle, "PikaH3FlexBackend", unexpected_loader)
    with pytest.raises(UnsupportedOperation, match="FP4"):
        lifecycle.open_training(config)
    with pytest.raises(ValueError, match="CUDA"):
        contracts.validate_options(dict(options, device="cpu"))


@pytest.mark.parametrize("variant", ["native", "phase"])
def test_fresh_cli_refuses_before_creating_output(variant, tmp_path, capsys):
    main = import_module("minimax_h3_" + variant + ".cli").main
    output = tmp_path / "new-run"
    with pytest.raises(SystemExit) as excluded:
        main(["train", "--config", str(tmp_path / "unused"), "--output", str(output)])
    assert excluded.value.code == 2
    assert "FP4" in capsys.readouterr().err
    assert not output.exists()
