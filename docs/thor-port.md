# Jetson AGX Thor port: fork change record

This document records the changes made on 2026-09-24 to run the existing
Qwen3.8 Flash Next deployment on one Jetson AGX Thor. It is the maintainer-facing
companion to the [Thor runbook](thor.md), which covers starting, stopping,
cleanup and test commands.

The settings and limitations below describe the original bring-up. The
2026-09-26 follow-up enables full decode CUDA graphs and MTP in the Thor
wrapper. It adds `files/patch_thor_runner.py` for dummy inputs and early
multimodal preparation, `tests/test_thor_runner.py` for startup regressions, and
`bench/thor-decode.py` for measured throughput and draft acceptance. See the
runbook for current settings, measurements and the startup-workaround limits.

The port adds an opt-in Thor profile and two GPU kernel compatibility fixes.
It reuses the upstream model image, checkpoint, PLE offload implementation,
download procedure and shutdown script. No container rebuild was required.

## Upstream baseline and tested environment

The working changes were made against upstream commit:

```text
6b5086458023474a7809ea30e1bcf42f03dcd75f
2026-09-19 — docs: how to run the NVIDIA checkpoint, and what it costs (#52)
```

The baseline identifies the code before the Thor changes; it is not a commit
containing the port. This record does not assert that a GitHub fork or release
has already been published.

| Component | Tested value |
| --- | --- |
| GPU | NVIDIA Jetson AGX Thor, compute capability 11.0 / SM110 |
| Host | Ubuntu 24.04, Jetson Linux R38.2.2, aarch64 |
| Driver / CUDA | 580.00 / CUDA 13.0 |
| Usable shared RAM | Approximately 122.8 GiB |
| Image tag | `vllm/vllm-openai:qwen38-flash-next` |
| PyTorch in image | `2.13.0+cu130` |
| vLLM in image | `0.1.dev20073+g8e685d198` |
| Checkpoint | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |
| Checkpoint snapshot | `925d7be6c14c6c9442ef83e8f05b5a3c39304f69` |

Docker reported this repository digest for the installed image:

```text
vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
```

Its local image ID was:

```text
sha256:d464f3b466fa9c45ddbff8a812e80564503b6879a9fd95c1a47514f3f0df5a4a
```

The tag is mutable. These identities are recorded for reproduction, not as a
claim that every future image bearing that tag will work. The launcher permits
an `IMAGE` environment override, including the digest reference above.

## Changes and reasons

| File | Change | Reason |
| --- | --- | --- |
| `start-thor.sh` (new) | Exports a conservative Thor profile, then invokes `start.sh`. | Keeps Thor defaults in one place while retaining the existing serving pipeline. |
| `start.sh` | Detects Thor using `/etc/nv_tegra_release` and the GPU name; selects the NVIDIA Docker runtime; adds `DOCKER_GPU_RUNTIME` and `MOE_BACKEND` overrides; runs Thor GPU preflight; reports the platform/backend. | Makes runtime selection explicit on Jetson and allows selection of a working MoE implementation. Environment overrides survive `.env` loading. |
| `start.sh` | Suppresses the Spark-specific VM tuning recommendation on Thor. | The Spark sysctl values were not validated on this host. No privileged tuning is applied automatically. |
| `start.sh` | Replaces archive-pruning `ls`/pipeline logic with a Python glob, retaining the newest 20 archive sets. | On a first launch, no matching archive files caused `ls` to fail and `set -euo pipefail` to abort startup. An empty archive now succeeds. |
| `scripts/check_thor.py` (new) | Checks SM110, CUDA 13+, model files, BF16 GEMM, Triton JIT, persistent top-k and reported Marlin NVFP4 support. | Catches missing GPU/model support before loading approximately 99 GiB of checkpoint files. This is a small preflight, not a full inference certification. |
| `files/patch_qsa_fp8_kv.py` | Adds SM110 to the architectures excluded from cooperative top-k dispatch. | The image's cooperative kernel failed on Thor with a cluster-configuration error. The existing persistent top-k path works. This fix applies to BF16 and FP8 KV alike. |
| `scripts/smoke-test.sh` | Adds validated `MIN_DECODE_TPS` configuration, retaining the default of 15; zero explicitly disables the speed gate and reports a warning. Generalizes the nondeterminism warning. | The conservative Thor profile runs at about 6 tokens/s, below the Spark baseline. Functional correctness and an explicitly chosen performance target need separate results. |
| `.env.sample` | Documents the Thor wrapper and runtime/backend knobs. | Makes the entry point discoverable without replacing the Spark sample profile. |
| `tests/test_thor_profile.py` (new) | Checks wrapper defaults, explicit overrides, argument forwarding, empty archives and archive retention. | Guards the new launch behavior without Docker or model-sized allocations. |
| `tests/test_thor_moe.py` (new) | Small Marlin NVFP4 MoE calculation compared with the same dequantized weights in FP32. | Provides a numerical check of the chosen MoE fallback independently of model loading. |
| `tests/test_thor_qsa.py` (new) | Exercises patched QSA dispatch and index expansion with fixed logits and different visible lengths. | Checks the persistent top-k path selects valid blocks and expands the correct token indices. Scoring is mocked to isolate selection/expansion. |
| `docs/thor.md` (new) | Operator runbook with commands, measured results and limitations. | Gives Thor users a tested launch/stop procedure. |
| `docs/thor-port.md` (new) | This baseline, rationale and maintenance record. | Makes the fork's divergence reviewable and reproducible. |
| `README.md`, `CHANGELOG.md` | Link the Thor documentation and summarize the port. | Separates Thor evidence from historical Spark results. |
| `.gitignore` | Allowlists the new source, test and documentation files. | This repository ignores everything by default; new files otherwise disappear from a normal fork commit. |

### Kernel failures that motivated the changes

**NVFP4 MoE:** Automatic backend selection chose `VLLM_CUTLASS`. The full
checkpoint loaded, but warmup failed in `cutlass_fp4_moe_mm`, with TMA/CUDA
errors. The Thor profile selects `--moe-backend marlin`, which uses weight-only
FP4 compression with higher-precision activations. This is a compatibility
fallback; it does not establish that Thor hardware lacks native FP4 or that
other software versions cannot use it.

**QSA top-k:** After fixing MoE and memory sizing, a later decode warmup failed:

```text
cooperative_topk launch failed:
a kernel launch error has occurred due to cluster misconfiguration
```

The original predicate allowed cooperative top-k for capability >= 9.0 while
excluding the SM12x family. The patch also excludes SM110, directing Thor to
`persistent_topk`. Existing SM12x dispatch and other architectures keep their
previous predicates. The edit lives in the patch generator, not only in the
ignored generated Python file, so subsequent launches reproduce it.

### Initial Thor defaults and their intent

Explicit nonempty environment overrides take precedence over these wrapper
defaults. The wrapper exports them before calling `start.sh`, whose environment
snapshot restores them after sourcing `.env`.

| Setting | Thor default | Why |
| --- | --- | --- |
| `DOCKER_GPU_RUNTIME` | `nvidia` | Explicit Jetson GPU runtime. |
| `MOE_BACKEND` | `marlin` | Avoid the failing auto-selected CUTLASS MoE path. |
| `GDN_DECODE_KERNEL` | `triton` | Conservative bring-up choice; alternative GDN decode kernels were not compared on Thor. |
| `MAX_MODEL_LEN` | `262144` | Restore native context after initial 32K bring-up. |
| `YARN` | `0` | Use native rope settings during bring-up. |
| `MAX_NUM_SEQS` | `1` | Validate single-request serving before concurrency tuning. |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | Limit prefill working memory. |
| `MTP_NUM_SPECULATIVE_TOKENS` | `0` | Validate baseline decoding before speculative decoding. |
| `OVERHEAD_GIB` | `14` | Replace the insufficient Spark estimate of 5.6 GiB. |
| `KV_TARGET_GIB` | `8` | Cover about 7.2 GiB BF16 KV plus budget rounding headroom. |
| `KV_CACHE_DTYPE` | `auto` | BF16 KV for this checkpoint, avoiding FP8 KV quality tradeoffs during bring-up. |
| `CUDAGRAPH_MODE` | `NONE` | Validate eager execution before graph capture. |
| `COMPILATION_MODE` | `0` | Disable torch.compile during bring-up; Triton/FlashInfer kernel JIT still occurs. |

Marlin completed the initial profiling pass, but the original overhead estimate
produced **-2.04 GiB available KV**. With 14 GiB overhead, the launcher derived a
GPU budget of **89.78 GiB** (`gpu_memory_utilization=0.731`) and a **94 GiB**
container cap using this host's existing configuration. vLLM allocated
**4.89 GiB of KV**, reporting 169,738 cache tokens. That initial run served
32,768 tokens with a 4 GiB KV target. The wrapper now requests 262,144 tokens
and an 8 GiB KV target; full-context Thor validation is still pending.
Configured concurrency remains one.

## Validation and its limits

Testing on 2026-09-24 established:

- Approximately 70.88 GiB of GPU model weights loaded; the patched server
  reached readiness after approximately 432 seconds.
- Health, 32,768-token model metadata, arithmetic generation (`17 × 23 = 391`),
  parsed tool calls, image recognition of the supplied fixture and metrics worked.
- The final functional smoke run returned **7 passed, 0 failed, 1 warning**.
  It used `MIN_DECODE_TPS=0`; the warning explicitly reports the disabled speed
  floor. The original 15 tokens/s gate failed during that bring-up.
- Two 400-token requests measured **5.8–6.1 tokens/s**, including request overhead.
- Temperature-zero responses differed in one run and matched in another.
  Determinism is not guaranteed.
- Marlin MoE's isolated numerical test produced relative error **0.006344**.
  The isolated QSA dispatch/expansion test and expanded GPU preflight passed.
- Three CPU regressions, shell/Python syntax checks and `git diff --check` passed.

A standalone Marlin probe stalled when run concurrently with the serving
container. It was stopped; subsequent API generation succeeded. Run standalone
GPU regressions with serving stopped. The isolated passing Marlin result is
not a claim that the concurrent probe passed.

The initial port did not validate video requests, larger contexts, MTP, CUDA graphs,
multiple concurrent requests, sustained load/soak operation, other checkpoints,
or supervised systemd restarts on Thor. Spark behavior was preserved in the
relevant dispatch/defaults, but the changes were not rerun on a physical Spark.
See the [runbook](thor.md#reproduce-the-current-checks) for reproduction commands.

## Host cleanup and fork contents

The user-provided `~/clean.sh` was run with sudo between failed startup attempts.
It sets `vm.nr_hugepages` to zero and drops filesystem caches. Available memory
rose from about 44 GiB to 119 GiB without rebooting. The earlier suspicion of
unrecoverable driver-held memory was not established. This external script is
not a repository dependency or an automatic startup step.

Include the source, tests and documentation in the table above in the fork.
Keep `.env` (which may contain credentials), `.last_launch.sh`, logs, checkpoint
files, packed PLE data, extracted `.orig` files and generated patched modules
out of the source commit. Their existing ignore behavior is retained. The
original `LICENSE` and attribution headers remain in place.

`stop.sh` and `download.sh` were not modified. Start with `./start-thor.sh` and
stop with `./stop.sh`. Existing systemd units still call `start.sh`; transfer the
Thor wrapper's settings into `.env` before using the supervisor. Automatic Thor
runtime detection alone does not apply the full Thor profile.

## Maintaining the fork

When rebasing, review the small upstream-facing changes in `start.sh`, the QSA
patch generator and the smoke test separately from the new Thor-only files.
Keep Spark benchmark claims associated with their original hardware.

Before updating the vLLM image, use a fresh checkout for validation or explicitly
refresh the extracted upstream sources. The current extraction helper reuses
existing `.orig` files, so merely changing `IMAGE` can leave patches based on
an older image. The generator requires unique source anchors and fails when
those anchors change; do not bypass that check without reviewing the new code.

Recheck internal package paths and kernel interfaces, run the isolated GPU and
CPU regressions, then test a complete launch and API requests. Record the new
image digest and actual smoke results. Remove a compatibility workaround only
after its replacement has passed on Thor. Treat performance tuning as a
separate change so the working baseline remains reproducible.
