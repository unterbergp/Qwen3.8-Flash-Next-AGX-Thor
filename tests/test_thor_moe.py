"""Small GPU regression for Thor's Marlin NVFP4 MoE fallback.

Run in the model image with --runtime=nvidia --gpus all; no checkpoint needed.
Compares two routed experts against the same dequantized weights in FP32.
"""
import torch
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_nvfp4_like,
)
from vllm.scalar_type import scalar_types


def main():
    torch.manual_seed(7)
    experts, hidden, intermediate, tokens = 2, 256, 256, 4

    def weights(rows, columns):
        return [
            rand_marlin_weight_nvfp4_like(
                torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16) * 0.02,
                16,
            )
            for _ in range(experts)
        ]

    w1 = weights(2 * intermediate, hidden)
    w2 = weights(hidden, intermediate)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[0], [1], [0], [1]], device="cuda", dtype=torch.int32)
    output = fused_marlin_moe(
        x,
        torch.stack([v[1] for v in w1]),
        torch.stack([v[1] for v in w2]),
        None, None,
        torch.stack([v[2] for v in w1]),
        torch.stack([v[2] for v in w2]),
        torch.ones(tokens, 1, device="cuda"),
        ids,
        scalar_types.float4_e2m1f.id,
        global_scale1=torch.stack([v[3] for v in w1]),
        global_scale2=torch.stack([v[3] for v in w2]),
    )
    reference = []
    for i in range(tokens):
        expert = i % experts
        gate, up = (x[i:i + 1].float() @ w1[expert][0].float()).chunk(2, dim=-1)
        reference.append((torch.nn.functional.silu(gate) * up) @ w2[expert][0].float())
    reference = torch.cat(reference)
    relative_error = (output.float() - reference).norm() / reference.norm()
    assert torch.isfinite(output).all() and relative_error < 0.03, relative_error
    print(f"PASS: Marlin NVFP4 MoE relative error {relative_error.item():.6f}")


if __name__ == "__main__":
    main()
