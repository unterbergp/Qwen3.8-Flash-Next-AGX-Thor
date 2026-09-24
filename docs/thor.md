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
prefill chunks, an 8 GiB KV target, 14 GiB estimated runtime overhead, BF16 KV,
Marlin MoE and Triton GDN decode. MTP, CUDA graphs and torch.compile are disabled.
The initial 32,768-token bring-up profile was validated; the restored native
context still needs end-to-end validation on Thor. This profile is not
performance-tuned. Input and output share the context budget. Native context
needs about 7.2 GiB BF16 KV; the former 4 GiB target was insufficient.
The existing PLE offload, host reserves, container limits and memory watchdog
remain enabled because the CPU and GPU share physical memory.

The wrapper overrides those Spark settings in `.env`. Explicit environment
values override the wrapper, for example `MAX_MODEL_LEN=65536 ./start-thor.sh`.
Larger contexts, MTP and graphs have not been validated on Thor in this session.
The image, model, credentials, cache, port, binding and host reserves still come
from `.env`.

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
- Thor's runtime-overhead estimate is 14 GiB. Spark's 5.6 GiB estimate produced
  a negative KV budget; the revised profile granted 4.89 GiB of KV on this host.
- GPU preflight checks CUDA 13+, BF16 matrix multiplication, Triton compilation,
  persistent top-k, model files and reported Marlin support before weight loading.
- Spark-specific sysctl recommendations are suppressed on Thor. No kernel
  tuning is applied automatically. An unrelated first-launch failure with an
  empty log archive is also fixed.

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

## Validation on this machine

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

Run the functional smoke checks while reporting speed without a performance
floor (the disabled floor is explicitly reported as a warning):

```bash
MIN_DECODE_TPS=0 EXPECT_LEN=262144 ./scripts/smoke-test.sh
```

`MIN_DECODE_TPS` otherwise defaults to 15. Set a nonzero value to enforce your
own performance requirement; the Thor profile has not been speed-tuned.

CPU regressions and syntax checks:

```bash
python3 tests/test_thor_profile.py
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
expansion checks. Three CPU regressions passed, covering profile defaults,
overrides, argument forwarding, first-launch pruning and archive retention.
