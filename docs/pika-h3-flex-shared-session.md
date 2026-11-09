# H3 Flex shared response session — source draft

This adds an H3-owned consumer of the real `particle-sliders-core` response
lifecycle. Both the slider's down/up projections and the conditional critic
execute HNDL graphs built by **HyperGAN develop**. It does not copy a GAN
objective, regularizer, optimizer, checkpoint engine or training loop.

**Native H3 training is still refused.** Official ParticleGAN0.9.4 release/auto
does not admit the genuine wide continuous response. No driver, permissions,
GPU jobs, pretrained weights, native output or model publication are involved
in the software qualification below.

## Exact source cohort

| Component | Reviewed identity |
| --- | --- |
| Preserved H3 native-boundary commit | `3e252a9b18197cbd79a943b47db37b3fea0a385d` |
| Shared core candidate | `a794e3b4a8813b012f2c4faa3da82b78d4a90214`, core0.12.0 |
| Shared candidate wheel receipt | `22e47ced4af8cb874d818db2f4006738f5f978cfcff324693931b6ab23f21c8a` |
| Official ParticleGAN | `0.9.4`, `release-0.9.4/auto` |
| HyperGAN develop | `5f368a3770cf5bdae0448995c9f76e29af8049f7` |
| HNDL | `0.10.0` |
| Native Pika model/runtime | Original exact pins in [the native-boundary guide](pika-h3-flex.md) |

The candidate is a separate, parent-owned dependency/publication. This H3
change does not merge, vendor or modify it. The source-checkout consumer
requires the separately installed candidate and verifies its critical source
hashes of the complete 50-file package closure from `data/pika-h3-flex-session.json`. It does not change `sys.path` or
install dependencies at runtime. The common verified builder gate currently
lives in `particle_sliders.fibo_hndl.verified_builder`; H3 calls that gate,
which verifies HyperGAN's clean build provenance and HNDL's complete source
closure. The executed graph sources are H3-specific, not Fibo's graphs.

## Actual execution and ownership

`HNDLH3FlexBackend` inherits pinned local loading, frozen Pika-at-1 behavior,
native packing and the guarded lifecycle from the original H3 boundary. It
overrides slider construction with `pika-h3-flex-slider.hndl`: two bias-free
linear projections with rank8/alpha8, the existing adapter-only LoRA
architecture. They run in float32, casting the resulting delta to the frozen
native host dtype. Scale zero calls the complete original PEFT forward.
No particle bank is fabricated; the declared arm is `adapter_gan`.

`pika-h3-flex-critic.hndl` describes the conditional raw-logit critic. Both
graphs are resolved and built through the actual verified HyperGAN/HNDL
builder. Graph source hashes and runtime provenance enter authoritative
architecture/checkpoint identity. A unit test compares HNDL projection values
and input derivatives with the matching native linear expression, including
a bfloat16 host with float32 trainable leaves.

`H3ResponseAdapter` exposes the complete packed video velocity followed by
audio velocity as a genuine continuous vector. Positive and neutral rows
retain their own native conditioning/layout and share video/audio noise and
times. Teacher responses use frozen base plus Pika, with slider zero. Student
responses use neutral conditioning at slider+1. Row identity is an explicit
one-hot critic condition; no response projection or image reshape is used.
This draft binds the **first native t2va evaluation point**, prepared by the
existing native boundary. It does not qualify control over all six sampling
steps or audiovisual quality.

`Binding` transfers the actual `ConditionalPG09` game, observation hook and
cleanup callback to the shared `ResponseSession`. The shared package owns
updates, EMA, observation RNG, checkpoint/resume and v1 numerical response
export/reload. Optional recomputation uses its reversible
`checkpoint_with_adapter_scale` handles; native internal checkpointing must
be off. Cleanup attempts checkpoint-handle removal and HNDL-forward removal
independently, including when one fails. Constructor failures also unwind owned
forwards, guards and native trainability/mode flags. Secondary cleanup failures
are attached to the original error; `retry_cleanup` retains pending owners for
repair and retry without replacing unrelated forwards.

The custom slider context remains necessary. Its trainable wrappers are not
named PEFT adapters; Pika is the only PEFT name and remains fixed at1. The
shared PEFT-name selection helper cannot scale these custom graph leaves.
Existing fixed-companion and signed/fractional-scale tests remain intact.

## Native preflight and API

The following uses the public session configuration and delegates preparation,
packet validation and restore to the exact shared lifecycle implementation:

```python
from particle_sliders.lifecycle import SessionConfig
from conceptmod.textsliders.pika_h3_flex_session import open_session

config = SessionConfig("pika-h3-flex", "fresh-h3-response-cohort",
    snapshot="/path/to/pinned-pika-snapshot",
    rows=[{"neutral": "A fox in a forest.", "positive": "A red fox in a forest."},
          {"neutral": "A quiet room.", "positive": "A brightly lit quiet room."}],
    duration=5, resolution="480p", steps=4, batch=2, seed=7)
open_session(config)  # currently refuses before native loading
```

The declaration fixes genuine geometry: 124 prepared frames at832×480 produce
`(1,1398528)`. Source tests execute the official gate and verify that the native
loader is never called:

```text
ValueError: released PG09 cannot admit flat continuous vectors with width >=170
```

Geometry uses the pinned release temporal contract and canvas sizes. Both
480p (832×480) and 768p (1344×768) packing are tested; 768p has response shape
`(1,3593664)` at the same duration. Preparation emits a warning when native
alignment changes duration. `model_identity` and adapter `metadata()` retain
requested/prepared frame counts, actual duration and packed geometry, so a
five-second request remains visibly 120 requested versus 124 prepared frames.

Public options reject fixture substitution, response width/shape overrides,
source-profile overrides and penalty-domain overrides. The in-progress
upstream continuous-domain proposal is not consumed. Removing this refusal
requires a separately reviewed source/profile adoption. Full-width critic
allocation, upstream optimizer memory behavior, hardware fit, pretrained
loading, native-only v2 export, standard-LoRA interchange, six-step media
sampling and quantized training remain unqualified. The standalone GGUF is
still an unqualified inference candidate, not a supported session input.

## Historical software checks

The original source-stage shared lane used candidate4abc0eb/core0.12, officialPG0.9.4, actual
HyperGAN5f368a/HNDL0.10, Python3.12.15, Torch2.14.1+cpu, PEFT0.21.2 and
Transformers5.22.1 with the pinned Diffusers source. Selected existing requests
dependencies are reused read-only in a task-local dependency view. This is not
a pristine full Pika environment or pretrained qualification.

The six-coordinate **random miniature native-H3 architecture fixture** has
its own explicit name/identity and is not selectable through public H3
options. It exercises actual HNDL projection/critic execution, shared PG
updates, scale-captured recomputation, unchanged frozen base/Pika weights,
observation isolation, exact four-step versus2+2 resume, shared export/reload
and atomic changed-cohort, model-identity and ParticleGAN-profile refusal.
Rejection preserves game/checkpoint state, RNG, gradient buffers, parameter
identities and ParticleGAN thermal state. Its artificial geometry never licenses the native
1.4M-wide model. No actual H3 training run is performed.

Run from this repository with the separately installed shared candidate:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  -o pythonpath=. tests/test_pika_h3_flex_session.py tests/test_pika_h3_flex.py
```

`-o pythonpath=.` prevents the repository's older core0.11 source path from
shadowing the installed candidate. The source hash gate rejects a mixed cohort.
The original Pika/legacy H3 suite is also retained in its original environment.
The [review correction receipt](pika-h3-flex-review-corrections.json) records
the final 71 shared-lane and 95 legacy-lane passes after Claude's source
review. These suites overlap. The original qualification receipt is retained
as historical evidence rather than relabeled with later results.

The shared core declares Python >=3.11. Standalone cleanup uses optional,
deduplicated exception notes so it does not depend on `add_note`; this
compatibility branch is simulated in tests, not a Python 3.10 runtime claim.
Both qualified CPU runs above have zero skips. The shared test module explicitly
skips if `particle_sliders` is absent, distribution metadata is absent, the
installed core version differs from 0.12.0, Pika-native Diffusers or PEFT is
absent, or Pika pipeline support is missing. Without the shared installation,
the observed reason was `shared H3 fixture requires particle_sliders package`
(one module skipped, pytest exit 5 because no tests were collected). Native
architecture tests skip absent Diffusers/PEFT or missing Pika pipeline support;
a source-hash mismatch in an available runtime fails rather than skips.

The final regression set covers direct `bind()` adapter-construction refusal,
idempotent projection cleanup, aggregated cleanup failures, removal of closed
module registrations, partial legacy-network construction, and frame-metadata
restore/preparation mismatches. Active legacy behavior remains unchanged.
`geometry_provenance` in the release JSON names the pinned functions/defaults
and published transformer config behind every added geometry constant.

## Publication overlap

Historical draft [PR140](https://github.com/HyperGAN/particle-sliders/pull/140)
is a divergent stacked LTX/H3 audit. Its exact overlapping backend and guide
were compared from base `ed3e4a9a724aaaf9acc657be63f6d4556ebc45be` to head
`0650d06c8d5fdeba314dc180041fec3d27fd86aa`. That backend adds
`require_h3_cpu_fixture` guards to construction, forward, sampling and adapter
I/O; its replacement guide closes the historical native/UNI path. Those are
intentional policy changes beyond this H3 Flex implementation.

This patch preserves main's historical UNI formulation and native behavior,
including the original `named_modules()` plain-Linear selection order and RNG
initialization, verified by an exact state-dict parity test. Complete PEFT host
support remains additive and excludes LoRA child modules. The Pika loader is
separate. PR140's closures, historical shared core, LTX payload and qualification
claims are not merged or cherry-picked. Parent retains the choice of any later
policy reconciliation and controls actual-diff review, dependency assembly,
CI/release-impact checks and merge.

The standalone review branch subsequently adopts exact core `a794e3b4a8813b012f2c4faa3da82b78d4a90214`.
This changes shared source/cohort identity deliberately; old packets are refused.
See the [adoption note](extraction.md#deliberate-shared-cohort-update). Earlier
counts and JSON receipts above remain historical, not relabeled new-core results.
