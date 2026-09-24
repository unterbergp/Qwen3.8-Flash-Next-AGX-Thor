"""Small image/GPU compatibility check; does not load model weights.

Run inside the serving image with --runtime=nvidia --gpus all.
This does not prove that every model kernel or a full serving workload works.
"""
from pathlib import Path
import torch
import triton
import triton.language as tl
import vllm


@triton.jit
def add_one(x, y, N: tl.constexpr):
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    tl.store(y + i, tl.load(x + i, i < N, other=0) + 1, i < N)


def main():
    capability = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} SM{capability[0]}{capability[1]}", flush=True)
    if capability != (11, 0):
        raise RuntimeError("This profile requires Jetson Thor (SM110)")
    if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) < 13:
        raise RuntimeError("Thor requires a CUDA 13+ image")
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, vLLM {vllm.__version__}", flush=True)
    package = Path(vllm.__file__).parent
    if not (package / "models/qwen3_8_flash_next/nvidia/model.py").is_file():
        raise RuntimeError("Image lacks this repository's Qwen3.8 Flash Next implementation")
    x = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    add_one[(2,)](x, y, 256)
    torch.testing.assert_close(y, x * 2)
    a = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(a @ a, a * 128)
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import is_fp4_marlin_supported
    if not is_fp4_marlin_supported():
        raise RuntimeError("Image has no Marlin NVFP4 support for this GPU")
    # Exercise the top-k fallback without loading the checkpoint.
    from vllm import _custom_ops  # Register the native operators.
    logits = torch.arange(1024, device="cuda", dtype=torch.float32).reshape(1, -1)
    visible = torch.tensor([1024], device="cuda", dtype=torch.int32)
    selected = torch.empty((1, 512), device="cuda", dtype=torch.int32)
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
    torch.ops._C.persistent_topk(logits, visible, selected, workspace, 512, 1024)
    torch.testing.assert_close(
        selected[0].sort().values,
        torch.arange(512, 1024, device="cuda", dtype=torch.int32),
    )
    torch.cuda.synchronize()
    print("PASS: CUDA, BF16 GEMM, Triton JIT, persistent top-k, model files, Marlin NVFP4 capability check", flush=True)


if __name__ == "__main__":
    main()
