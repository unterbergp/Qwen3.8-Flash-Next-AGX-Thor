# Jetson AGX Thor

For the upstream baseline, file-by-file rationale and maintenance notes, see the
[fork change record](thor-port.md).

The model runs on this Thor with the model-specific image from `.env.sample`.
Use the Thor wrapper:

```bash
./start-thor.sh
```

On a fresh clone, first copy `.env.sample` to `.env` and run `./download.sh`.
This machine already has the checkpoint and packed PLE table. Stop the server
with `./stop.sh`. The API is at `http://localhost:8888/v1` with the default port.

## Thor profile

The wrapper uses a 262,144-token context, one concurrent sequence, 1,024-token
prefill chunks, an 8 GiB KV target, 16 GiB estimated runtime overhead with MTP
(14 GiB without it), BF16 KV,
Marlin MoE and Triton GDN decode. Full decode CUDA graphs and three-token MTP
are enabled; torch.compile remains disabled. The V2 model runner is explicitly
pinned for both target and draft, and graph capture sizes follow MTP depth and
configured concurrency (`CUDAGRAPH_CAPTURE_SIZES=auto`). At the default MTP 3,
capture sizes are `1,4`: one token for draft decode, four for target verification
and draft prefill. Input and output share
the context budget. Native context
needs about 7.2 GiB BF16 KV; the former 4 GiB target was insufficient.
The existing PLE offload, host reserves, container limits and memory watchdog
remain enabled because the CPU and GPU share physical memory.

The wrapper overrides those Spark settings in `.env`. Explicit environment
values override the wrapper, for example `MAX_MODEL_LEN=65536 ./start-thor.sh`.
The image, model, credentials, cache, port, binding and host reserves still come
from `.env`.

The original eager baseline remains available through explicit overrides:

```bash
MTP_NUM_SPECULATIVE_TOKENS=0 CUDAGRAPH_MODE=NONE ./start-thor.sh
```

For graphs without speculation, use `MTP_NUM_SPECULATIVE_TOKENS=0 ./start-thor.sh`.
Keep the reduced draft vocabulary configured in `.env`; use the Spanish
variant for Spanish traffic. Do not export all of `.env` before launching the
wrapper: exported Spark settings override its Thor defaults.

`./start-thor.sh --no-launch` prepares patches and writes `.last_launch.sh`.
It may pull the image or build the packed table; it does not start serving or
run the GPU preflight.

The optional systemd supervisor invokes `start.sh`, not the wrapper. Copy the
Thor profile values from `start-thor.sh` into `.env` before enabling supervised
restarts. Service installation is covered in the main README.

## Compatibility changes

- `start.sh` detects Thor and uses `--runtime=nvidia --gpus all` even if Docker
  defaults to `runc`. `DOCKER_GPU_RUNTIME=auto|nvidia|default` overrides this.
- The wrapper selects `--moe-backend marlin`. Automatic selection chose the
  image's CUTLASS NVFP4 MoE kernel, which failed during warmup on this Thor.
- The QSA patch excludes SM110 from cooperative top-k selection. That kernel
  failed with a cluster-configuration error; Thor now uses persistent top-k.
  Spark's SM12x path and other architectures retain their existing dispatch.
- Thor estimates 16 GiB runtime overhead with MTP, or 14 GiB without it.
  The Spark estimate is insufficient here. These estimates feed the existing
  budget calculation; the host reserve still caps the GPU allocation.
- A Thor-only MRV2 patch uses synthetic model inputs for dummy runs, avoiding
  gathers from uninitialized request history. It prepares multimodal profile
  inputs before allocating model/runner tensors; actual vision execution remains
  in the memory profiler. Startup synchronization is outside the token-generation
  loop. Patch anchors fail closed if the image source changes.
- GPU preflight checks CUDA 13+, BF16 matrix multiplication, Triton compilation,
  persistent top-k, model files and reported Marlin support before weight loading.
- Spark-specific sysctl recommendations are suppressed on Thor. No kernel
  tuning is applied automatically. An unrelated first-launch failure with an
  empty log archive is also fixed.

The startup ordering addresses an observed Thor failure: startup sometimes
changed draft weights or PLE indexing constants, causing zero draft acceptance
or a CUDA indexing assertion. Instrumentation could mask the problem, so an
instrumented benchmark alone is insufficient validation. The underlying writer
has not been identified; a [related upstream Thor investigation](https://github.com/vllm-project/vllm/issues/54906#issuecomment-5596317838)
also reports persistent CUDA state changing around multimodal setup. This is a
local startup workaround, not a general driver or upstream vLLM fix.

Keep the model-specific image: this repository depends on its Qwen3.8 Flash
Next implementation and Python layout. A generic Jetson vLLM image is not a
substitute. The local image contains PyTorch 2.13.0+cu130 and vLLM
`0.1.dev20073+g8e685d198`; mutable tags may change later.

## Memory cleanup

If the server is stopped but too little memory is available, this host's
`~/clean.sh` disables reserved huge pages and drops filesystem caches:

```bash
sudo bash ~/clean.sh
./start-thor.sh
```

This is a host-provided script, not part of the repository. It restored available
memory from about 44 GiB to 119 GiB during testing; no reboot was needed.
The launcher does not invoke privileged cleanup automatically.

## Initial bring-up validation (2026-09-24)

Hardware/software: Thor SM110, Jetson Linux R38.2.2, driver 580.00, CUDA 13.0,
122.8 GiB usable shared RAM. The stock Mia NVFP4 checkpoint loaded successfully;
startup took approximately 7 minutes. Text generation correctly answered
17 × 23, parsed tool calling worked, and image input identified the supplied
vision fixture. Health, model metadata and metrics endpoints passed.

The conservative profile measured **5.8–6.1 tokens/s** for 400-token responses,
including request overhead. This is below the original Spark smoke test's
15 tokens/s floor. Temperature-zero outputs differed between two runs;
deterministic generation is not guaranteed. The final smoke run reported
7 passes, 0 failures and one warning for the explicitly disabled speed floor.

The historical eager baseline used the following command to report speed
without a performance floor (explicitly reported as a warning):

```bash
MIN_DECODE_TPS=0 EXPECT_LEN=262144 ./scripts/smoke-test.sh
```

## MTP and graph validation (2026-09-26)

Measured on the same Thor/image/checkpoint listed above, with an idle server,
one sequence, native context, BF16 KV and 1,024-token prefill chunks. Each
figure is the median of three 400-token requests at temperature zero with
thinking disabled; wall time includes prefill and HTTP overhead.

| Profile | Prose tokens/s | Code tokens/s |
| --- | ---: | ---: |
| CUDA graphs, MTP off | 28.35 | 28.15 |
| CUDA graphs, MTP 3, clean startup 1 | 36.27 | 59.02 |
| CUDA graphs, MTP 3, clean startup 2 | 36.72 | 59.47 |

MTP improves this code workload by about 2.1× over graphs alone. Acceptance
varies by content: the clean run accepted about 1.0–1.1 draft tokens per step
for prose and 2.4–2.6 for code, out of three proposed. The older ~6 tokens/s
result is historical, not the matched control in this table. Upstream Spark
figures use their own configuration/workload and are not a head-to-head test.

The MTP measurements above came from `./start-thor.sh` without audit hooks,
CUDA launch blocking, sanitizer instrumentation or a mounted diagnostic Triton
cache. Target verification, draft prefill and draft decode graphs captured;
a separate live-request debugger check observed `cudaGraphLaunch` from
`at::cuda::CUDAGraph::replay()`. Debugger-assisted requests are excluded from
the timing results.

Both fresh, uninstrumented startups passed all six acceptance measurements
and all eight smoke checks at the normal 15 tokens/s floor, including text,
parsed tools and vision. The second benchmark ran after its vision check. A 7,189-token prompt returned its
embedded passphrase; a 7,224-token follow-up did the same with 5,824 cached
prompt tokens. A one-token thinking-budget request forced the reasoning end
marker and returned the correct answer. These checks exercise chunked prefill,
MTP rollback and a cached conversation. A full 262,144-token prompt, concurrent
GPU workloads and long-duration serving have not been validated.

Local evidence is retained in `logs/thor-graphs-k0-isolated-20260926.jsonl`,
`logs/thor-mtp3-clean1.jsonl`, `logs/thor-mtp3-clean2.jsonl`,
`logs/thor-mtp3-clean1-smoke.log`, `logs/thor-mtp3-clean2-smoke.log`,
`logs/thor-mtp3-clean1-context.log` and `logs/thor-cudagraph-replay-proof.log`.
Runtime logs are intentionally untracked.

## Reproduce the current checks

Use the default 15 tokens/s gate for the graph/MTP profile:

```bash
EXPECT_LEN=262144 ./scripts/smoke-test.sh
```

For an isolated comparison, run the same benchmark against each profile, with
no other requests in flight. The subshell exports API credentials only for the
benchmark; do not export `.env` into the shell that launches the Thor wrapper.

```bash
(
  set -a; source .env; set +a
  python3 bench/thor-decode.py --tag thor-mtp3 --require-speculation --out logs/thor-mtp3.jsonl
)
```

The benchmark sends three 400-token prose requests and three code requests at
temperature zero with thinking disabled. It records full responses, wall-clock
throughput including prefill/HTTP overhead, and changes in speculative-decoding
counters. Check accepted tokens as well as throughput: MTP can be enabled and
still make generation slower when proposals are rejected.

CPU regressions and syntax checks:

```bash
python3 tests/test_thor_profile.py
python3 tests/test_thor_runner.py
bash -n start.sh start-thor.sh scripts/smoke-test.sh
git diff --check
```

GPU regressions, independent of the checkpoint: run these with the serving
container stopped. A standalone Marlin probe stalled when run alongside the
server; its isolated run passed. Concurrent GPU workloads are not validated.

```bash
docker run --rm --runtime=nvidia --gpus all -e OMP_NUM_THREADS=1 \
  -v "$PWD/tests/test_thor_moe.py:/test_thor_moe.py:ro" \
  --entrypoint python3 vllm/vllm-openai:qwen38-flash-next /test_thor_moe.py

# Requires generated files/qsa_ops_patched.py from a launch/preparation:
docker run --rm --runtime=nvidia --gpus all -e OMP_NUM_THREADS=1 \
  -v "$PWD/tests/test_thor_qsa.py:/test_thor_qsa.py:ro" \
  -v "$PWD/files/qsa_ops_patched.py:/qsa_thor.py:ro" \
  --entrypoint python3 vllm/vllm-openai:qwen38-flash-next /test_thor_qsa.py
```

Marlin NVFP4 MoE matched the dequantized FP32 reference with relative error
0.006344. The QSA regression passed dispatch, visible-block selection and token
expansion checks during the initial port validation. The current CPU suite
passes 11 tests covering profile defaults and overrides, graph sizing, dummy
input dispatch, startup ordering, argument forwarding and archive retention.
