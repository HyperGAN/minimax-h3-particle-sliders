"""Cross-install compatibility for random miniature H3 fixtures, never weights.

Run produce with the original reviewed modules, then verify with installed
product modules (and vice versa), in the same exact shared runtime cohort.
Fixture payloads go outside the repository and must never be published.
"""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("produce", "verify"))
    parser.add_argument("--fixture-module", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--expected-product-root", type=Path, required=True)
    args = parser.parse_args()
    from conceptmod.textsliders import pika_h3_flex_session as h3
    assert Path(h3.__file__).resolve().is_relative_to(args.expected_product_root.resolve())
    spec = importlib.util.spec_from_file_location("h3_compat_fixture", args.fixture_module)
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    from particle_sliders.lifecycle import SessionConfig
    from particle_sliders.pg09 import _hash
    for checkpointing in (False, True):
        folder = args.artifact_dir / str(checkpointing).lower()
        config = SessionConfig(fixture.FIXTURE.name, "h3-extraction-compat-v1", seed=7,
                               steps=4, batch=2, ema_decay=0.5, checkpointing=checkpointing)
        if args.mode == "produce":
            folder.mkdir(parents=True, exist_ok=True)
            with fixture.open_fixture(config) as session:
                session.train(stop_after=2)
                session.checkpoint(folder / "resume.pt")
                session.train()
                expected = {"network": deepcopy(session.game.network.state_dict()),
                            "ema_state": deepcopy(session.ema.state_dict()), "signature": _hash(session.game._signature),
                            "responses": {}}
                for selection in ("raw", "ema"):
                    session.export(folder / (selection + ".pt"), selection=selection)
                    if selection == "ema":
                        session.game.network.load_state_dict(session.ema.values)
                    expected["responses"][selection] = session.evaluate(0, seed=19)
                torch.save(expected, folder / "expected.pt")
        else:
            expected = torch.load(folder / "expected.pt", weights_only=True)
            with fixture.open_fixture(config, resume=folder / "resume.pt") as session:
                session.train()
                assert _hash(session.game.network.state_dict()) == _hash(expected["network"])
                assert _hash(session.ema.state_dict()) == _hash(expected["ema_state"])
                assert _hash(session.game._signature) == expected["signature"]
            for selection in ("raw", "ema"):
                with fixture.open_fixture(config, artifact=folder / (selection + ".pt"), operation="response") as session:
                    torch.testing.assert_close(session.evaluate(0, seed=19), expected["responses"][selection], rtol=0, atol=0)
        print(args.mode, "checkpointing=" + str(checkpointing), "raw/EMA/resume: exact random-fixture compatibility")


if __name__ == "__main__":
    main()
