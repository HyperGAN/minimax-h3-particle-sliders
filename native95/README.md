# Experimental native H3 · ParticleGAN 0.9.5

This additive subproject exposes the captured full MiniMax/Pika H3 runtime and
an explicit runner API. It preserves the existing repository package and its
0.9.4 defaults. This source proposal has not been installed or executed as a
new public package. H3 has no completed matched finished-output LR comparison
in the retained audit; there is no measured winning rate to publish.

The model is Pika-H3-Flex revision
`3ff166447d5edb1406107ab8370c42000ce173e4`, with the original fixed Pika release
adapter at strength1, original six-step schedules and full video/audio output
coordinates. The captured phase profile uses all50 main blocks, both refiners,
rank8/alpha8 HNDL branches, full native attention, group size at most5, FF512,
BF16 base/fixed Pika and the original FP32 islands/game. No CPU model arithmetic
or model offload is supported by the runner API. CPU control, file loading,
saved conditioning and output transfers do not change that rule.

## Install the opt-in subproject

From the reviewed repository checkout containing this directory:

```sh
python -m pip install ./native95
```

Use a separate environment. The dependency metadata pins ParticleGAN0.9.5,
HNDL0.10.0, HyperGAN`5f368a3`, Diffusers`d0b9b1b`, Transformers5.22.1 and the
compatible core source commit`f7001a821c89df6c2e8473da4f6914db45516aeb`.
That core commit changes packaging metadata from41f; the12 files checked by
both unchanged H3 core guards match byte-for-byte. Core0.14.1 remains required;
a newer version number alone is insufficient. Original native source/data,
builder provenance and dependency guards remain active. This subproject does
not download or include model weights. Installed-wheel or full-model fit is
not established by a static source match.

The original `minimax_h3_native95` and `minimax_h3_phase95` namespaces are copied
exactly. New owner helpers live only in sibling `minimax_h3_runner95`; the
original runtime hashes and manifests are not rewritten. The registered
training/inference backend is `minimax_h3_phase_owned_av95`. `h3-phase95 check`
reports its declared geometry without running a model. The older CLI's cache
and render-pair flows can encode text on CPU; those flows are not a supported
substitute for this GPU-only runner boundary.

## The runner owns admission and lifetime

The captured runtime fixes these physical devices, in this order:

```text
GPU-72c1b506-891d-b8bc-b353-e020585e1c47
GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69
```

This is an experimental hardware-specific profile, not a portable device map.
The external process runner selects that exact CUDA mask before entry. It
supplies a `RunnerAdmission` with four real physical shared `HeldLease`s
(research and ordinary lease per GPU), one real role-exclusive lease and a
`check_live(caps, admission)` callback. The library independently validates
FD/path/inode identity and actual kernel FLOCK READ/WRITE modes. The callback
must verify actual policy admission, correct lease-to-GPU mappings, SH/EX modes,
parent/PDEATH/lifetime, all other CURRENT forecasts and live caps/floors. It
must return `True` only after those checks and continuously monitor long native
calls. A constant callback, admission flag or open FD is not that evidence.

Captured limits are host/RSS72GiB, NVML37/42GiB, allocators35/40GiB, entry/live
free5GiB per GPU and host free16GiB, with every other CURRENT forecast retained.
There is no elapsed work deadline. The external runner must enforce allocator
limits before model work and monitor its process family throughout; the library
does not reserve capacity, implement a scheduler, stop other jobs or claim fit.
The current private executor's observed47GiB host-entry forecast is its scoped
policy decision, not a public permission to ignore the72GiB cap or other jobs.

Enter `Runner(admission)` in a fresh worker before framework imports. It scopes
the original admission marker only after real owner checks. No Root identity,
Root receipt, private grant path or manual flag-forging is required. One process
thread and one DiT/game owner at a time are supported. Keep the runner's leases
until all owners close. Failed or foreign-replaced cleanup remains pending;
the marker is removed last and `close()` can retry only pending owners. Repair
foreign state through its owner; never overwrite it to manufacture a clean exit.

## Training uses an already completed matching cache

Provide the complete original model snapshot, completed full text conditioning
and a completed paired native teacher cache with their real hashes. Conditions
must retain verbatim prompts, full64-layer encoder/hidden50 provenance, tags,
all native packed fields and original geometry. The cache must contain every
positive/neutral teacher return. Missing or ambiguous work is not a completed
cache. Opening training verifies and consumes the saved data; it does not call
the original teacher collector or re-encode prompts. New preparation would be
separate explicit work; this proposal does not authorize replaying saved calls
or provide a GPU text-encoding replacement for the original CPU encoder.

The following callable is an integration example. `admission` comes from the
real external runner described above, and `config_dict` names genuine local
completed artifacts. No executable dummy admission is supplied.

```python
def train_saved_cache(admission, config_dict, full_path=None):
    from minimax_h3_runner95 import Runner
    with Runner(admission) as runner:
        from particle_sliders.lifecycle import SessionConfig
        config = SessionConfig.from_dict(config_dict)
        with runner.open_training(config, resume=full_path) as session:
            def save(metrics):
                step = metrics['step']
                session.checkpoint(f'FULL-{step:06d}.pt')
                session.export_inference(f'RAW-{step:06d}.pt', selection='raw')
            session.train(on_step=save)
```

Choose a new cohort for genuinely fresh work and an unused output directory.
The backend/profile/initialization strings in `session.example.json` are exact;
replace only its local artifact paths/hashes and explicit requested recipe.
The example G0.012 and D multiplier1.5 are the retained starting recipe,
**not a measured recommendation**. They imply nominal D0.018000000000000002.
There is no particle bank and no prior LR. Batch2, h128, seed7 and full five-second
480p geometry are shown. Original alignment produces124 video frames and
1,398,528 packed response coordinates; never substitute a smaller response.

Fresh construction applies the existing scoped ParticleGAN initialization
before optimizer creation: same-Parameter down QR at Kaiming-sqrt5 host variance
`1/(3*fan_in)`, zero UP retained, critic host law KEEP, pretrained/fixed weights
untouched. No bank law is invented. Model-weight EMA is disabled explicitly.
Official ParticleGAN optimizer scalar statistics remain unchanged.

## FULL restore and RAW inference

Pass a non-None original typed FULL packet/path to `runner.open_training` for
resume. The original exact config/cohort, full manifest, game/optimizers/scalar
statistics, cursors/RNG and initialization receipt must match. Resume restores
that whole state and invokes no scoped fresh initializer. Do not replace
scheduled optimizer groups with nominal starting rates, change h128, retag a
factory/profile, reset a cohort or replay a committed update. A mismatch is a
refusal to reconcile explicitly, not permission for a lossy migration.

RAW inference constructs only the original native inference owner, restores
raw adapter weights and never constructs the training game/cache/critic or
reruns initialization. Use `runner.open_inference(raw_packet, options=options,
strength=alpha, output_type='native_av_latents', native_request={'prompt':prompt})`;
options contain the genuine snapshot, `device='cuda:0'`, BF16 and the completed
matching conditioning path/hash. Its session `render(prompt=prompt,
strength=alpha, seed=seed, output_type='native_av_latents')` returns **internal
AV latents**, not finished media. Original strengths0..1 and scheduler/RNG
semantics remain. The caller owns a genuine once-only request ledger and must
not replay completed or UNKNOWN native calls.

Persist both native latent streams and their actual provenance before decoding.
Close the complete inference/DiT owner. `runner.decode_latents(snapshot, latents,
device='cuda:0')` then delegates original sequential GPU FP32 video/audio VAE
phases; original `minimax_h3_phase95.render.write_av(result, new_directory)`
writes all frames and native32kHz stereo float32 WAV. The output is a full PNG
sequence plus WAV, not a claimed muxed video. The original `decode_saved` API
can continue its own genuine completed latent receipts without TE/DiT/RAW
replay; do not synthesize receipts for another workflow.

## Captured execution helpers and results

`minimax_h3_runner95/captured` includes byte-exact standalone Source assets for
HNDL row512, combined original PEFT/Pika+HNDL row512, synchronous diagnostics,
and24/26 and25/25 placement/cache-trim owners. `helpers-manifest.json` identifies
them. They are not automatically activated by this new runner API. FF512 and
grouped checkpoints already belong to the unchanged phase runtime.

The retained private training profile composes those exact helpers, completed
conditioning bookkeeping and strict FULL0 comparisons under its own execution
owner. Its saved factory/session identity remains24/26 even when a separately
witnessed execution override uses25/25. Archives do not turn an ordinary fresh
public session into that captured cohort. Exact saved-state compatibility must
be established by the original manifest/restore checks; no paths, source hashes
or profile fields are normalized. Placement changes do not establish bitwise
equivalence, whole-model fit or a new LR result.

Software observations and bounded numerical cases retain their original scopes;
they do not establish finished AV usefulness. There is no final H3 LR row yet.
When genuine matched finished outputs are available, rank one intended-effect
ON-minus-OFF score averaged equally across matched cases. Preserve completed
comparisons and ties. Teacher SSE, drift and visual beauty do not select or
break ties. A qualified catastrophic failure stops the affected candidate;
weak effect is separate and uncertain checks remain unmeasured. This source
proposal creates no new qualification, confirmation or calibration gate.
