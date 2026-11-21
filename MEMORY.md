# Memory requirements

| Operation / profile | GPU count | Calculated VRAM per GPU (GiB) | Calculated host RAM (GiB) | Input limits / precision |
| --- | ---: | --- | ---: | --- |
| Public original BF16 AV training | 2 | GPU0: 96, GPU1: 96 | 80 | B2;5s480p/124frames; text≤256; saved24/26; group5/FF512/row512 |
| Public original BF16 AV sampling | 2 | GPU0: 48, GPU1: 52 | 80 | B1;5s480p/124frames; text≤256; saved24/26; native6steps |
| Native sampling, largest release geometry | 2 | GPU0: 96, GPU1: 96 | 80 | B1;15s768p/362frames; text≤256; BF16; saved24/26 |
| Separate local FP4 AV training cohort | 2 | GPU0: 72, GPU1: 72 | 48 | B2;5s480p; text≤256; balanced25/25; full architecture/Pika |
| BF16 text conditioning phase | 1 | 80 | 128 | One prompt;≤4096tokens; full Qwen3-VL32B; no KV |
| Separate local FP4 text conditioning phase | 1 | 32 | 64 | One prompt;≤4096tokens; full Qwen FP4/BF16; no KV |
| Separate native AV VAE decode | 1 | 48 | 64 | B1;5s480p; frozen video/audioVAEs; FP32; DiTclosed first |


These are **calculated planning budgets, not measured peaks or hard runtime ceilings**.
They apply only to the bounded profiles in the table, for one process/slider at a
time. GiB means 2^30 bytes. Add concurrent jobs' resident allocations; do not sum
mutually exclusive loading, conditioning, training and decoding phases as if all
their models were resident together. Allow the stated host RAM independently of
VRAM. Larger batches, longer inputs, extra references, FP32 bases, different
attention kernels or allocator fragmentation can require more. There is no finite
universal maximum for arbitrary inputs. These calculations start no model jobs.

## Which model / execution profile

Public native95 is the original Pika AV profile, not the new H3 image model.
It uses `Pika-Labs/Pika-H3-Flex@3ff166447d5edb1406107ab8370c42000ce173e4`,
the full original MiniMax transformer plus a frozen rank128 Pika release adapter.
The [public runner profile](https://github.com/HyperGAN/minimax-h3-particle-sliders/blob/043f8c89ca6425f2f6eec05d24138572e3e19bc1/native95/src/minimax_h3_runner95/profile.json)
pins saved24/26 main blocks, refiner1/1, GPU0 game, group5 recomputation,
FF512 and row512. It is explicitly hardware-specific, not a portable automatic map.
See [native contracts](https://github.com/HyperGAN/minimax-h3-particle-sliders/blob/043f8c89ca6425f2f6eec05d24138572e3e19bc1/native95/src/minimax_h3_phase95/contracts.py),
[release geometry/asset metadata](https://github.com/HyperGAN/minimax-h3-particle-sliders/blob/043f8c89ca6425f2f6eec05d24138572e3e19bc1/native95/src/minimax_h3_phase95/data/pika-h3-flex-release.json)
and [model-file identity](https://github.com/HyperGAN/minimax-h3-particle-sliders/blob/043f8c89ca6425f2f6eec05d24138572e3e19bc1/native95/src/minimax_h3_phase95/model-files.json).

The separately authored November13 FP4 cohort is **not shipped in this public
profile**; it uses the same complete architecture/fixed Pika with a new quantized
identity, never an old FULL/teacher cache retag. The local FP4 training allowance
above remains a calculated budget, not a claim that its full update fit was proven.

## Calculation

* Original DiT tensor bytes 61.728 GiB + frozen Pika serialized upper
  2.577 GiB. Exact retained balanced25/25 weight-placement metadata
  gives33,922,910,720 and33,741,115,392 bytes (31.593/31.424 GiB).
  Public24/26 moves one main block and its Pika branches fromGPU0 toGPU1.
  A block is `2*(4*5376*7168+3*5376*14336+2*5376+256+
  2688*18*5376+18*5376)` bytes, with fixed rank128 Q/K/V/out/FF adapter
  `2*128*(4*(5376+7168)+(5376+28672)+(14336+5376))` bytes.
  Result: 30.366/32.651 GiB; this is placement arithmetic,
  not a claim the public saved profile changed to25/25.
* PG adapter is ordinary rank8 HNDL on208 attention hosts, 20,873,216 scalars,
  **no particle bank**. The default full1398528-coordinate critic's
  source [bounded critic](https://github.com/HyperGAN/minimax-h3-particle-sliders/blob/043f8c89ca6425f2f6eec05d24138572e3e19bc1/native95/src/minimax_h3_phase95/bounded_critic.py)
  uses `(1398528+2)→16→1`, giving exactly 22,376,513 parameters.
  FP32 training/update envelope is 3.222 GiB onGPU0. Frozen Pika
  receives neither gradients nor optimizer state. Its2.577 GiB source adapter
  is FP32 on disk; native BF16 casting halves its tensor residency to~1.289 GiB.
* Default5s aligns to124 native frames;37 video latent frames ×15×26 patches
  + stereo audio rows gives~14.9k packed rows including short text. Group5
  checkpointing retains boundaries and recomputes five blocks together: a
  conservative B2 active-group envelope `2*B*T*5*(8*5376+3*7168+14336)`
  bytes is about22 GiB; reserve42 GiB including native feed-forward, FP32 HNDL,
  retained game/penalty graphs and recomputation temporaries. Add4 boundaries,
  GPU0 trainable/update allowance and8 runtime GiB, round both devices to96.
* At15s768p,362 alignedframes produce107 video latent frames ×24×42 patches,
  plus1206 stereo audio rows and text. This is~7.3× as many tokens. For sampling,
  scale the7 GiB live-forward envelope to52, add frozen weights and10 workspace,
  and round to96 GiB/card. The fixed training critic explicitly accepts only
  response1398528/condition2: longer/larger training is NOT implemented by that
  bounded profile, even though the release geometry planner accepts it.
* Local FP4 frozen upper bound: `(DiT_tensor_bytes/2)*0.5625 +1GiB` for all
  packed matrices/FP32 islands/scales, plus frozen Pika. This is
  20.938 GiB total, approximately half per balanced card.
  It reduces weight residency; it does not remove the full native graphs or
  critic/update allocation. Retain the same42+4=46 GiB graph/boundary allowance
  as BF16; quantization alone does not shrink it.72 GiB/card is the calculated
  default training budget with the same row/group controls, not an old BF16
  checkpoint promotion.
* BF16 Qwen3-VL weights alone are62.133 GiB. For4096 tokens add2.54 GiB
  returned hidden states and10 workspace, round to80. FP4 conditioning uses
  17.475 GiB all-parameter upper packing +6 islands +2.54 hidden
  +5 workspace, round to32. `use_cache=False` avoids an autoregressive KV cache.
* VAEs add10.264 GiB serialized upper residency; allow another copy
  for FP32 conversion,12 live decode and8 workspace, round to48. Decode follows
  DiT closure; phase maxima are separate. Larger768p/15s decode is not covered
  by the5s480p decode row and can exceed it.
* Host80 for original native loading covers the61.73 GiB full CPU DiT before
  explicit placement +9.2 GiB largest shard +1.29 GiB Pika and checkpoint/cache
  copies/runtime. Conservative BF16 text loading needs128 host GiB (62.13 GiB
  full weights +46.49 GiB largest shard/load-copy + hidden-state/runtime margin),
  or64 for GPU-streamed FP4 with a46.49 GiB
  largest source shard. Separate FP4 training needs48 host GiB for streamed
  quantized loading, the9.2 GiB largest DiT source shard, snapshots and runtime.

All attention estimates require fused SDPA, native geometry, no offload, no EMA
and the stated checkpoint controls. The table is deliberately a budget, not a new
qualification gate for historical results or a requirement to replay old work.


## Optimizer and estimate policy

Training figures freeze pretrained weights, use RAW current adapters and explicitly
disable weight EMA. They include no full-base gradients or EMA shadow model.
The captured `bispf_optimized` recipe uses zero-momentum DualNorm, not Adam's two
full-size moment tensors. Its direction blend still retains protected gradients,
clones originals/displacements, and converts up to two protected normals to
FP64. For a conservative planning allowance, the tables reserve **80 bytes per
trainable scalar in total**: 8 for FP32 weights+gradients, 16 for two FP64 normals,
and 56 for protected FP32 gradients, FP32/FP64 direction/projection/polar,
rollback and serialization temporaries. This is an explicit allowance, not a claim
that every temporary is simultaneously allocated or that it is an exact peak.
Small scalar optimizer statistics remain enabled. Recipe changes need a new ledger.
See the original [ParticleGAN recipes](https://github.com/255BITS/ParticleGAN/blob/1b8355b184b21bb7d81104ee590813c262f6df0a/particlegan/recipes.py),
[DualNorm](https://github.com/255BITS/ParticleGAN/blob/1b8355b184b21bb7d81104ee590813c262f6df0a/particlegan/optim/dualnorm.py)
and [direction blend](https://github.com/255BITS/ParticleGAN/blob/1b8355b184b21bb7d81104ee590813c262f6df0a/particlegan/optim/direction_blend.py).

Activation entries below are architecture-based envelopes. Fused/memory-efficient
SDPA is assumed for the pretrained model; a math attention fallback that materializes
all score matrices can exceed them. No model benchmark or runtime measurement was
performed for this documentation. Machine-readable inputs, arithmetic and all table
rows are in [docs/memory/inputs.json](docs/memory/inputs.json).

## Explicit rounding ledger

| Operation | Calculated GPU subtotal (GiB) → planning budget | Host subtotal (GiB) → planning budget |
| --- | --- | --- |
| Public original BF16 AV training | GPU0: 87.588 → 96; GPU1: 86.651 → 96 | 79.207 → 80 |
| Public original BF16 AV sampling | GPU0: 45.366 → 48; GPU1: 47.651 → 52 | 79.207 → 80 |
| Native sampling, largest release geometry | GPU0: 92.366 → 96; GPU1: 94.651 → 96 | 79.207 → 80 |
| Separate local FP4 AV training cohort | GPU0: 67.692 → 72; GPU1: 64.469 → 72 | 44.190 → 48 |
| BF16 text conditioning phase | 74.673 → 80 | 119.167 → 128 |
| Separate local FP4 text conditioning phase | 31.015 → 32 | 60.034 → 64 |
| Separate native AV VAE decode | 40.528 → 48 | 49.770 → 64 |

The detailed host terms and per-device arithmetic are included in the machine-readable inputs.
