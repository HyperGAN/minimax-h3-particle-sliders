"""Product-owned, source-pinned HyperGAN/HNDL builder for the H3 LoRA graph.

The general source-closure checks are adapted from particle-sliders'
``fibo_hndl.verified_builder`` (MIT). No former product network, core-package
payload, ParticleGAN version, training formula or native qualification is implied.
This module imports only stdlib until checked_builder() is called.
"""
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).parent
PROFILE = "h3-native95-product-hndl010-builder-v1"
HYPERGAN_COMMIT = "5f368a3770cf5bdae0448995c9f76e29af8049f7"
HNDL_VERSION = "0.10.0"
LORA_SHA256 = "e32094eb55b9178e8c42d7bf483a5915593d5fe6ae3cbca7478559cf80f8dc4c"
BUILDER_SOURCES_SHA256 = "1db7a69f7733b063d8f56dd2312fdf5c3c9dceaeaab875f16b2bb497e47d59a6"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_builder():
    """Return the exact external build_network and its checked H3 identity.

    Admission binds every HNDL Python source, the reviewed HyperGAN builder
    closure, its clean commit provenance and this product's two-line graph.
    It establishes source identity only; it does not construct a network or
    certify installed models, training updates, continuation or rendering.
    """
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("H3 HNDL requires Python>=3.11,<3.15")
    manifest = ROOT / "builder-sources.json"
    if _sha(manifest) != BUILDER_SOURCES_SHA256:
        raise ValueError("H3 requires the exact reviewed builder source manifest")
    expected = json.loads(manifest.read_text())
    if (expected.get("schema"), expected.get("profile"),
            expected.get("hndl_version"), expected.get("hypergan_commit"),
            expected.get("lora_sha256"), expected.get("source_policy")) != (
            "h3-native95-product-builder-source-closure-v1", PROFILE,
            HNDL_VERSION, HYPERGAN_COMMIT, LORA_SHA256,
            "all-hndl-python-and-declared-hypergan-builder-files-v1"):
        raise ValueError("H3 builder source manifest contradicts the binding")
    if _sha(ROOT / "data/pika-h3-flex-slider.hndl") != LORA_SHA256:
        raise ValueError("H3 native LoRA graph differs from the original two-line graph")
    try:
        import hndl
        from hypergan.hndl_networks import build_network
    except ModuleNotFoundError as exc:
        raise ImportError(
            "H3 requires explicit hndl==0.10.0 and the pinned HyperGAN train dependency"
        ) from exc
    if version("hndl") != HNDL_VERSION:
        raise ValueError("H3 requires hndl==0.10.0")
    hndl_root = Path(hndl.__file__).parent
    actual = {str(p.relative_to(hndl_root)): _sha(p)
              for p in hndl_root.rglob("*.py")}
    if actual != expected["hndl"]:
        raise ValueError("H3 HNDL Python source closure differs from the reviewed source")
    source = inspect.getsourcefile(build_network)
    if source is None:
        raise ValueError("H3 requires inspectable HyperGAN build_network source")
    hypergan_root = Path(source).parent
    provenance = hypergan_root / "_build_provenance.json"
    if not provenance.is_file() or json.loads(provenance.read_text()) != {
            "schema_version": 1, "commit": HYPERGAN_COMMIT, "dirty": False}:
        raise ValueError("H3 requires exact clean HyperGAN commit build provenance")
    if any(_sha(hypergan_root / name) != digest
           for name, digest in expected["hypergan"].items()):
        raise ValueError("H3 HyperGAN builder source closure differs from the reviewed source")
    identity = dict(
        profile=PROFILE,
        hndl_version=HNDL_VERSION,
        hypergan_commit=HYPERGAN_COMMIT,
        hypergan_build_provenance_sha256=_sha(provenance),
        builder_sources_sha256=_sha(manifest),
        builder_source_sha256=_sha(__file__),
        source_policy=expected["source_policy"],
        h3_lora_sha256=LORA_SHA256,
    )
    return build_network, identity
