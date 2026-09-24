"""GPU regression for Thor's patched QSA top-k dispatch and index expansion.

Mount files/qsa_ops_patched.py as /qsa_thor.py alongside this test in the
serving image. Fixed logits isolate selection/expansion from QSA scoring.
"""
from unittest.mock import patch

import torch
import vllm._custom_ops  # Register the CUDA operators.
import qsa_thor


def main():
    rows, columns, block_topk, compress_ratio = 3, 4096, 512, 4
    device = "cuda"
    logits = torch.arange(columns, device=device, dtype=torch.float32).repeat(rows, 1)
    visible = torch.tensor([512, 2048, 4096], device=device, dtype=torch.int32)
    lengths = visible * compress_ratio
    positions = lengths - 1
    requests = torch.arange(rows, device=device, dtype=torch.int32)
    q = torch.zeros(rows, 1, 16, device=device, dtype=torch.bfloat16)
    cache = torch.empty(64, 64, 1, 16, device=device, dtype=torch.bfloat16)
    pages = torch.arange(64, device=device, dtype=torch.int32).repeat(rows, 1)
    with patch.object(qsa_thor, "qsa_mqa_paged", return_value=(logits, visible)):
        selected = qsa_thor.qsa_select_paged_tokens(
            q, cache, pages, requests, positions, lengths,
            block_topk * compress_ratio, compress_ratio,
        )
    for i, blocks in enumerate([512, 2048, 4096]):
        valid = selected[i][selected[i] >= 0].sort().values
        expected = torch.arange(
            (blocks - block_topk) * compress_ratio,
            blocks * compress_ratio,
            device=device, dtype=torch.int32,
        )
        torch.testing.assert_close(valid, expected)
    print("PASS: Thor QSA dispatch, visible-block selection, and token expansion")


if __name__ == "__main__":
    main()
