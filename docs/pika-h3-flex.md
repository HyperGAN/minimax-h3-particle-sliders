# Pika H3 Flex native boundary

The later [shared response-session draft](pika-h3-flex-shared-session.md)
adds real shared lifecycle and HyperGAN/HNDL execution on a separate core0.12
software cohort. Native wide-domain execution remains refused; the historical
UNI trainer is not silently migrated.

The existing H3 product code lives in `HyperGAN/particle-sliders`, under
`conceptmod/textsliders/minimax_h3_backend.py`. There is no separate H3 or
MiniMax video slider repository among the 21 public HyperGAN repositories
inspected on 2026-10-31. `music3-particle-sliders` is a different audio model.
The workstation also has an older, locally modified project at
`historical embedded source checkout`, with untracked `train_lora_h3.py`, prompt
files and `H3.md`. Those files and its environment are preserved.

This change adds an opt-in native loader, real packing/sampler qualification
and a shared-core admission check. It **does not complete a GAN migration**.
The existing H3 UNI objective stays unchanged. Its two-row mock pack is not
pretrained-model qualification. No existing H3 slider is relabeled as Pika.

## Release identity

| Component | Immutable revision |
| --- | --- |
| Pika weights / manifests | `Pika-Labs/Pika-H3-Flex@3ff166447d5edb1406107ab8370c42000ce173e4` |
| Pika inference repository observed head | `Pika-Labs/Pika-H3-Flex@4bf8ea508891a432106e70f675d287c27f02306d` |
| MiniMax base from Pika manifests | `MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe` |
| Pika's documented Diffusers source | `huggingface/diffusers@d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f` |
| Historical standalone admission reference | `HyperGAN/particle-sliders@3c25f785d14073e6f627c0fa9d2297a041c79913`, core 0.11.0 / official ParticleGAN 0.9.3 |
| Optional shared-session cohort | Core 0.12.0 / official ParticleGAN 0.9.4 / HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` / HNDL 0.10.0; [exact source locks and limits](pika-h3-flex-shared-session.md) |

Pika is a post-trained MiniMax-H3 derivative. The release supplies rank/alpha
128/128 adapters: `fl2va/adapter.safetensors` for `transformer` and
`ref2va/adapter.safetensors` for `transformer_ref`. This boundary supports
**t2va with the FL2VA adapter only**. Reference and keyframe requests remain
unqualified. The exact manifests and inspected contract-source checksums are
in `conceptmod/textsliders/data/pika-h3-flex-release.json`.

The sampler uses knots `[1000, 750, 500, 250, 100, 40, 0]`, video shift 12
and audio shift 3. Both scheduler configs must set
`shift_provided_sigmas=true` and `stochastic_sampling=true`. A generic six-step
Euler sampler is not equivalent. The native pipeline owns these operations.
VAEs remain float32; the other base components load at their native mixed
precision under the requested bfloat16 policy.

Sources: [research release](https://pika.art/blog/pika-h3-flex-research),
[pinned model card](https://huggingface.co/Pika-Labs/Pika-H3-Flex/tree/3ff166447d5edb1406107ab8370c42000ce173e4),
[official Diffusers contract](https://huggingface.co/docs/diffusers/main/en/api/pipelines/pika_h3_flex),
[Pika's pinned environment notes](https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/3ff166447d5edb1406107ab8370c42000ce173e4/examples/README.md).
The model and derivative adapters retain the MiniMax H3 Community License
Agreement, including its territorial restrictions and redistribution notices.
The code licenses do not relicense those weights. Hosted upscaling and audio
stem services are outside this local boundary.

## Quantization finding

A full-architecture inference candidate is:

| Field | Value |
| --- | --- |
| Hub repository | `Abiray/MiniMax-H3-GGUF` |
| Revision | `9fc3454d3ebe1be1bade862cd4a5011f325a22cb` |
| File | `unet/MiniMax-H3-FL2VA-Q4_K_M.gguf` |
| Bytes | `19864208217` |
| SHA-256 | `d66125a46c9bee7de3586420843263d9c5185a8d5516949f13b21cd85516e37b` |

Use the native Comfy-layout Pika adapter, not the Diffusers-layout FP32
adapter, when evaluating this candidate in a compatible Comfy GGUF loader.
Pika documents the native conversion's fused QKV/rank handling in its
[adapter guide](https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/3ff166447d5edb1406107ab8370c42000ce173e4/comfy/README.md).
The quantizer's [model card](https://huggingface.co/Abiray/MiniMax-H3-GGUF)
establishes base-model provenance and file layout. It does not establish
end-to-end Pika adapter compatibility or gradients. This candidate is **not
loaded by the new Diffusers backend** and has not been rendered here.
Pruned H3 GGUFs are not interchangeable with the full release adapter.

The existing `historical `smoke_test_h3.py` helper` instead uses TorchAO int8
weight-only base inference. Its old environment has Diffusers 0.40.0.dev0 at
`dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d`; it is not the Pika runtime pin.
Broken optional TorchAO extension imports were observed during CPU tests.
Neither this old int8 result nor GGUF availability qualifies quantized Pika
training. No weights were downloaded during this task.

## Native API

Stage the specified release locally in a separate, reviewed backend
environment. The loader performs no downloads, overrides component sources
to the supplied local path, checks the release adapter's SHA-256, validates
the sampler and verifies hashes of the inspected Diffusers contract files.
It also requires published hashes for all **30 local base/configuration files**
needed by t2va, including every transformer, text-encoder and VAE shard.
The provenance JSON records the official pinned Hugging Face tree endpoint,
file sizes, LFS SHA-256 hashes and ordinary-file Git blob SHA-1 hashes.
Missing provenance, missing files, mismatches and unlisted component files
(including alternative checkpoints or tokenizer sidecars) fail closed before
native loading. This streams approximately 144 GB of local files when the full
release is staged; no network call or automatic download occurs.
Keep the staged base/configuration files immutable throughout loading; their
path-based component loaders run after verification. The adapter has a stronger
byte-level guarantee: one captured byte payload is hashed and decoded into
safetensors, and those exact tensors are passed to `load_lora_weights`. A later
change to the on-disk adapter cannot replace the captured release payload.

These are the published hashes of the **Pika-packaged** base at its pinned
revision. They do not independently establish tensor equivalence to MiniMax's
differently packaged original checkpoint. Only the audio-VAE weight file has
the same same-path LFS hash in both official trees; the other weight packages
differ. No pretrained shards were downloaded or hashed during this task.

```python
from conceptmod.textsliders.pika_h3_flex import PikaH3FlexBackend, gradient_probe

backend = PikaH3FlexBackend('/path/to/pinned-pika-snapshot', device='cuda:0')
packed = backend.prepare_t2va('A red fox walking through a snowy pine forest.')
receipt = gradient_probe(backend, packed)  # no optimizer update
result = backend.render('A red fox walking through a snowy pine forest.', strength=0)
print(result['frame_metadata'])
```

Preparation exposes `packed.frame_metadata`; `gradient_probe` propagates it
into its receipt, and rendering adds `frame_metadata` to the result. These
include requested/prepared frame counts, requested duration, actual duration
derived from the prepared count at 24 fps, and `alignment_changed`. A five-second
request reports 120 requested frames, 124 prepared frames and approximately
5.166667 seconds. Whenever alignment changes duration, the helper emits a
`UserWarning` identifying both counts and durations; render warns before
generation and validates the returned frame count afterward. This is frame-grid
metadata, not independent validation of an encoded media container's timeline.

The example requires enough memory for the resident model; it is not a
consumer-GPU memory recipe. Native pretrained loading, conditioning, media
decoding and GPU performance remain unqualified in this task. The CLI exposes
`provenance`, `admit-gan` and `probe`, and has no training command.

`prepare_t2va` uses the native text encoder, row layout, initial noise and
timestep-plan blocks. `velocity` calls the native transformer directly because
ModularPipeline execution is no-grad. Scale zero leaves the Pika PEFT adapter
active at 1.0; a separate custom attention slider owns the float32 trainable
leaves. The scale context must include backward when checkpointing is enabled.
`gradient_probe` checks finite nonzero slider gradients without an optimizer
or touching parameter `.grad` buffers. It does not implement a loss recipe.

After slider attachment, pipeline/model fuse, unload, disable, additional
adapter loading and deletion operations are rejected before mutation. A
`set_adapters` roundtrip may select only `pika_release` at unit weight, and
re-freezes all base components. Each H3 operation checks that the transformer
and PEFT layer identities are intact, Pika remains enabled/unmerged at 1.0,
and all base parameters remain frozen. Direct native-internal mutation is
unsupported and causes the next operation to fail closed.

The current shared `lora_scale_context` scales every PEFT name it finds,
including Pika. A test executes that unmodified helper and confirms H3 refuses
to run while its companion adapter is zeroed. H3's custom context scales only
its custom slider leaves and preserves Pika at zero, fractional and signed
slider strengths. It is not registered with the generic shared adapter binder;
future selective shared scaling is separate coordinated work.

The new shared selective helper operates on **PEFT adapter names**. H3's
trainable slider is an `AttnLoRANetwork` of custom `_AttnLoRA` forward wrappers
with its own float32 down/up leaves; it is not a named PEFT adapter. The only
PEFT adapter here is the fixed Pika release. Selecting that name would scale
the companion instead of the slider, while selecting a hypothetical slider
PEFT name would not address these custom leaves. H3 therefore keeps its
custom context, which changes only wrapper multipliers and spans backward.
Migrating the representation would require separate adapter-format and
lifecycle qualification; adopting a selective API alone cannot perform it.

The pinned native source uses `vae.config.clip_length=17` and
`vae.tokens_chunk_size=5`: prepared frames must satisfy **17*n+5**, not 4*k+1.
The API accepts integer durations 5–15 seconds. Valid native requests of 120,
240 and 360 frames align upward to 124, 243 and 362 frames respectively.
Invalid durations are rejected, and prepared outputs must match the exact
native alignment for the request. A pre-alignment request of 120 frames is
valid; a prepared output of 120 frames is not.

## GAN admission in the two software lanes

`admit_shared_gan((1, native_width))` delegates unchanged native vector geometry
to `particle_sliders.pg09.validate_pg09_domain`. The standalone boundary was
checked against core 0.11.0 / official ParticleGAN 0.9.3. The optional shared
session is separately source-locked and tested against core 0.12.0 / official
ParticleGAN 0.9.4. Both reject the real wide packed response:

```text
ValueError: released PG09 cannot admit flat continuous vectors with width >=170
```

Native 832×480, five-second preparation aligns to 124 frames and produces
`[14430, 96]` video rows and `[414, 32]` audio rows: a combined response width
of **1,398,528**. This exact prepared geometry was checked against the installed
official gate and rejected before a model forward.

Do not flatten into a pretend image, reduce to a tiny response, use a
categorical fallback or change the source-profile alias. Do not rebind old
checkpoints to another ParticleGAN profile. The standalone probe implements no
training objective; the [optional shared session](pika-h3-flex-shared-session.md)
connects the real shared GAN lifecycle and HyperGAN/HNDL graphs, with native
execution still refused. Tiny CPU fixtures qualify that software wiring only.
No shared-core or HyperGAN upstream code is modified in this H3 patch.

## Executed checks and review

All tests are tiny/synthetic CPU or metadata fixtures. Architecture fixtures
run against the actual pinned Diffusers MiniMaxH3Transformer and
PEFT 0.20.0, with small random parameters, not pretrained weights. They verify
fixed-Pika scale-zero identity, independent signed slider scale, finite
nonzero slider gradients, unchanged frozen base/PEFT parameters, genuine
native row packing and different shifted audio/video times. A native
denoising block executes exactly six transformer evaluations on the packed
fixture. This is architecture evidence, not generated-media quality.
The expanded checks compare a stock random base-plus-PEFT model with slider
scale zero before/after pipeline/model `set_adapters` roundtrips, check the
exact attention host set excluding `lora_A`/`lora_B` children, reject destructive
lifecycle changes and reject the unmodified shared generic-scaling hazard.
None of these checks qualifies pretrained loading, media output or GGUF.

Run from the repository root in the isolated Pika environment:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_pika_h3_flex.py tests/test_minimax_h3_slider.py
```

Native architecture tests skip when Pika Diffusers/PEFT are absent. The
release metadata, local-only loader wiring and admission forwarding tests
still run. Source hash mismatches fail rather than masquerading as a pass.
On this host `nvidia-smi` could not communicate with the driver. Storage had
456 GiB free on a 94%-used volume before source staging; GPU ownership could
not be established, so no GPU or pretrained-model smoke was attempted.

Claude's design feedback, relayed by the parent task, supported preserving
the UNI/GAN boundary, native preparation, fixed Pika scale, independent
slider gradients and treating GGUF as unqualified inference-only. These
points and the subsequent bounded hardening findings are implemented where
supported by the official source. Final hardened-diff review and a
pretrained/quantized smoke remain pending. The GGUF remains an unqualified
candidate, not a supported input to this loader.
