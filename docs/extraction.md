# Standalone H3 extraction

This is a clean export from
`HyperGAN/particle-sliders@fae34ef80f8ad0fbe02e5d9abc3c0e770798d5f7`
for destination `HyperGAN/minimax-h3-particle-sliders`. This note records the
historical local extraction; source publication is documented in PUBLICATION.md.
It does not retire old copies, merge umbrella PR149 or authorize model execution.

## Exact content boundary

The packaged `data/h3-extraction.json` lists 25 allowlisted original files,
their source/destination paths, original Git blob SHA-1, SHA-256 and sizes.
At phase1, the six H3 runtime modules, two namespace initializers, four YAML configs and
prompts, four Pika manifests/graphs and LICENSE are byte-identical to that head.
Original license and model notices are retained. No umbrella history is grafted;
original commit/path/blob identities provide traceability.

The source distribution also carries the three H3 test modules and five H3
guide/qualification files. Adaptations are limited to:

- Tests read installed H3 code/configs rather than paths relative to the umbrella.
- The generic scaling hazard test imports the installed shared helper.
- One unrelated Music3-defaults test is omitted; no Music3 source is copied.
- The original shared-session test file remains byte-identical: its fixture
  factory source hash participates in cohort identity. The separate compatibility
  driver adds cross-install raw/EMA/resume coverage with recomputation off/on.
- Three historical cross-model documentation links point to immutable original
  source URLs, rather than broken local files.

Generated packaging files add an explicit package/resource allowlist, three
console entry points, installed-distribution tests and a cross-install fixture
compatibility driver. The legacy command wrappers discard historical result
dictionaries so console scripts return successful exit codes. They do not alter
the underlying recipes. No new shared-GAN command or backend registration is added.

## Cohort preservation

Retain `conceptmod.textsliders`, `pika-h3-flex`, the protocol names, adapter keys,
HNDL bytes, model/geometry metadata, exact dependency source lock and original
runtime source bytes. These enter architecture/checkpoint signatures. Namespace
or runtime-source changes would require a distinct cohort and reviewed migration;
this extraction does neither. Compatibility is tested across original-source and
installed-product processes in the same exact CPU/software runtime.

The historical UNI/MSE recipe and modern adapter-GAN candidate remain distinct.
The modern path imports the installed shared lifecycle, official PG objective,
checkpoint utilities, optimizer policy and verified HyperGAN/HNDL builder. The
H3-specific projection hooks and graph definitions remain product code. No generic
implementation or dependency source tree is copied into this distribution.

## Build and installed checks

```bash
python -m build --no-isolation
python -m pip install --no-deps dist/minimax_h3_particle_sliders-0.1.0.dev0-py3-none-any.whl
python -m pytest --import-mode=importlib tests
```

The local validation builds wheel and sdist offline using existing build tools,
installs each in a fresh target outside all product checkouts, and runs tests
with no product checkout on the import path. Wheel tests use copied tests/docs
only; sdist tests use a wheel rebuilt from its extracted archive. Imported H3
module locations are asserted to belong to the installed target. Package data,
exact original runtime hashes, entry points, fixed Pika isolation, native geometry,
official refusal, shared CPU updates, checkpoint/resume, raw/EMA export and atomic
mismatch refusals are exercised. Neither test lane downloads or loads weights.

`tools/verify_original_cohort.py` produces and verifies **random miniature**
fixtures between original and installed processes, including raw/EMA responses
and exact resumed parameters/signatures with checkpointing off/on. Generated
fixture payloads stay outside this repository and are excluded from all artifacts.

The original extraction used core0.12 at `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`.
That baseline remains in `extraction/phase1-tested` and `extraction/pre-a794-reviewed`.
The subsequent deliberate adoption below supersedes only the modern candidate pin.

Native wide-vector admission is still refused before loading. Full H3 memory
fit, pretrained loading, AV quality, quantized compatibility/training, six-step
control and native-only export remain unqualified. Prior JSON receipts describe
their historical source stages, not a deployment or model release.

## Deliberate shared cohort update

The coordinator authorized one local modern-candidate adoption of exact shared
core `a794e3b4a8813b012f2c4faa3da82b78d4a90214` after its source review and
bounded synthetic core CUDA checks. This worker performs CPU consumer checks
only. Neither that evidence nor this adoption qualifies native H3 or H3 CUDA.

The optional dependency URL pins that commit. The lock now includes all 50
package files, verified against the exact reviewed core Git tree and wheel
SHA256 `22e47ced4af8cb874d818db2f4006738f5f978cfcff324693931b6ab23f21c8a`.
Changed, missing or extra files fail admission; normal bytecode caches are ignored.
PG0.9.4, HyperGAN5f368a, HNDL0.10, model/runtime pins, loss, bank, native
geometry, fixed Pika behavior and quantization status are unchanged.

This intentionally changes the session-guard source and shared dependency/cohort
identity. Old raw, EMA and resume packets must refuse; no metadata rebinding,
profile alias or old-artifact acceptance path is introduced. The original tested
branches and all original source/path/blob provenance remain intact. The provenance
manifest adds current hashes and an explicit update ledger for the two affected
runtime/resource files, rather than overwriting their historical hashes.

The installed wheel/sdist checks repeat lifecycle, both presets, exact resume,
raw/EMA export, slider OFF, atomic EMA conversion failure, cleanup errors, native
refusal and full-source admission. The preserved fixture factory is byte-identical;
new consumer regressions live in a separate test file. Private fixture packets,
logs, wheels and environment trees stay outside the product repository.

Historical UNI and default standalone-native behavior remain byte-identical.
The native wide-vector gate still refuses before model loading. Repository
source publication is separate from native admission; no GPU, weights, model
release or deployment is included.
