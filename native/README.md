# Experimental native H3 source migration

The standalone native packages pin Particle Sliders core 0.15.2 at
`ec1161f5d74481be3540cb0dfa63d0e8f8b54e90` and ParticleGAN 0.11.1.
The product preserves full native video and audio coordinates, the official
optimizer, scoped initialization, and RAW state. CPU verification covers
installed source admission and ownership. Native FP4 training and rendering
on these pins remain pending. The fresh training and cache commands refuse
before model allocation. Fresh native work requires a separately qualified
FP4 or smaller H3 derivative with an explicit stock EMPF host.

Install this candidate in a separate environment.

    python -m pip install ./native

The captured BF16 work uses the previous exact code, optimizer and runtime.
The [archived runner profile](https://github.com/HyperGAN/minimax-h3-particle-sliders/blob/043f8c89ca6425f2f6eec05d24138572e3e19bc1/native95/src/minimax_h3_runner95/profile.json) records
that cohort. The following runner notes describe the captured placement and
resource contracts. Fresh work requires the separate FP4 or smaller H3
derivative and new provenance. Existing FULL and RAW files retain their
original dependency and model identities.

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

## Exact continuation uses an already completed matching cache

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
def resume_saved_cache(admission, config_dict, full_path):
    from minimax_h3_runner import Runner
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

For exact continuation, pass the original session config and FULL packet from
the matching source cohort. The backend, profile, and initialization strings
in `session.example.json` document the BF16 layout for inspection.
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
phases; original `minimax_h3_phase.render.write_av(result, new_directory)`
writes all frames and native32kHz stereo float32 WAV. The output is a full PNG
sequence plus WAV, not a claimed muxed video. The original `decode_saved` API
can continue its own genuine completed latent receipts without TE/DiT/RAW
replay; do not synthesize receipts for another workflow.

## Captured execution helpers and results

`minimax_h3_runner/captured` includes byte-exact standalone Source assets for
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

## Installed runner software checks

All four new cases in `tests/runner_owner_check.py` passed on 2026-11-11 against the genuine offline-built and installed wheel. They cover actual temporary-file flock modes and identity, policy refusal, foreign-marker/pending cleanup retry, and exact-once delegation to the original training methods. The run used a neutral directory, exact installed Source/RECORD checks and no source `PYTHONPATH`. See the [software validation record](validation/installed-owner-api-20261111.json). No GPU, pretrained model or optimizer work was performed. Full-native execution through the public wheel and the H3 LR comparison remain unverified.

## Scheduled audiovisual denoising on an owned FP4 session

For the separate original FP4 runner, use the `minimax_h3_scheduling`
extension with an already-opened strict RAW session. Call
`render_latents(session, prompt=prompt, seed=seed, step_gains=gains)` with
six native gains. The coupled video and audio evaluations share the current
gain. Return the resulting latent pair to the same FP4 runner's native
decoders. This extension can be imported from this checkout beside the
frozen FP4 environment. Effective strengths must be in `[0,1]`.
Pretrained scheduled audiovisual effects are pending.

### Compare doses on both native clocks

Set `record_native_grids=True` on a fixed `render_latents` call to record
`video_sigmas` and `audio_sigmas` alongside the existing timesteps and
`effective_strengths`. This verifies both scheduler cursors and grids during
the fixed trajectory. Gated traces already carry both grids.

To verify a matched fixed and gated pair offline, use the shared core's
[`interval_dose`](https://github.com/HyperGAN/particle-sliders/blob/07dbd331c3a548709949032c01ce19e7c64a2520/packages/particle-sliders-core/src/particle_sliders/strength_schedule.py)
from revision `07dbd331c3a548709949032c01ce19e7c64a2520`. The verifier
requires identical video and audio timesteps and sigma grids across the two
traces before integrating their resolved coefficients on each clock.

```python
from minimax_h3_scheduling import native_dose_residuals

residuals = native_dose_residuals(
    fixed_trace, gated_trace, coefficient_format=measured_scalar_format
)
assert all(residual == 0 for residual in residuals.values())
```

Pass the coefficient format confirmed for the exact native runtime. The
captured pruned RAW2 runtime was verified at the float32 scalar boundary.
For that runtime, use `coefficient_format="binary32"` with strengths that
are already resolved at that boundary. The shared verifier rejects a
coefficient that would still round to binary32. The signed residuals are
fixed minus gated and use exact rational arithmetic. Verify zero separately
for video and audio. This verifies scalar dose equality. Finished audiovisual
effects require the matched native renders and observer.

### Calibrated activation gating

An `AVGatePlan` selects one common RAW strength for both modalities before
that interval's native transformer call. The original text-to-AV path is
single-conditional. The gate reads the packed `hidden_states` and
`audio_hidden_states` before the edit, averages their rows in FP32, and
concatenates the video channel means followed by the audio channel means.
It checks both native solver cursors and each modality's packed row times.

Build the shared core wheel at revision
[`b1f780de`](https://github.com/HyperGAN/particle-sliders/commit/b1f780decc516808cfd3019d5e5b674457161f86).
Give its path to the gate plan alongside the original FP4 environment. The
facade verifies and loads only the standalone steering module from the wheel.
The captured runtime, adapter and checkpoint source pins keep their original
identities. Use a fitted unit direction and the threshold or bias from a
calibration record for this joint readout, exact FP4 cohort and slider.

```python
from minimax_h3_scheduling import render_latents
from minimax_h3_scheduling.activation_gate import AVGatePlan

plan = AVGatePlan(
    core_wheel=reviewed_core_wheel,
    method="cfp",
    direction=fitted_joint_direction,
    parameter=calibrated_threshold,
    calibration_sha256=calibration_record_sha256,
)
latents, trace = render_latents(
    raw_session, prompt=prompt, seed=seed, strength=qualified_strength,
    gate_plan=plan,
)
```

For CLABS, supply `method="clabs"`, the calibrated bias as `parameter`,
`clabs` with the measured projection statistics and explicit controller fields,
and a gate `seed`. For the matched random control, provide `method="random"`,
its matched firing rate as `parameter` and a gate `seed`. This control takes
no direction. The gate state resets for each render. The trace records both
native sigma grids, the decisions, the realized firing rate, the applied
strengths and the sigma-integrated dose for each modality. With the same
prompt, RAW checkpoint, nominal strength and native seed, replay with
`step_gains=[value / qualified_strength for value in trace["effective_strengths"]]`.

Fit the direction from paired native readouts in matching geometry and on the
same FP4 model. Calibrate the firing rate using separate active rollouts.
Compare decoded, complete audiovisual outputs on a disjoint held-out set with
native OFF and constant-strength RAW, and record the applied gains separately
from the intended-effect and integrity checks. The latent return is internal
to the existing native decoders. There is no qualified H3 AV direction, gate
calibration, or measured adaptive gain for the pruned Warm Sunset cohort.
The captured pruned FP4 timing panel did not reach its required RAW8 checkpoint.
See [the shared steering and calibration protocol](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering).
