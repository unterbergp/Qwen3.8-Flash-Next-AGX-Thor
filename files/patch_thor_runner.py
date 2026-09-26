#!/usr/bin/env python3
"""Prepare MRV2 initialization and dummy inputs on Thor.

The model's dedicated dummy helper supplies EOS-filled PLE context without
gathering live request history. Real requests retain prepare_inputs unchanged.
MTP loading is synchronized before runner-state allocation and after startup
initialization; there is no per-token synchronization. Multimodal profile
inputs are prepared before model allocation, while encoder execution stays in
the profiler. This startup ordering is a Thor workaround; the original stray
writer has not been identified. Applied only by the Thor launcher; fail closed
when the image source changes.
"""
from pathlib import Path

OLD = "            **self.model_state.prepare_inputs(input_batch, self.req_states),"
NEW = """            **(
                self.model_state.prepare_dummy_inputs(
                    input_batch.num_reqs_after_padding,
                    input_batch.num_tokens_after_padding,
                )
                if dummy_run
                else self.model_state.prepare_inputs(input_batch, self.req_states)
            ),"""


def patch(source: str) -> str:
    if source.count(OLD) != 1:
        raise ValueError("Thor runner patch: expected exactly one input preparation anchor")
    source = source.replace(OLD, NEW)
    return source


def patch_startup(source: str) -> str:
    edits = [
        ("        # Initialize the components that require the model.",
         "        # Finish draft weight work before allocating persistent runner state.\n"
         "        if self.speculator is not None:\n"
         "            torch.cuda.synchronize(self.device)\n\n"
         "        # Initialize the components that require the model."),
        ("        get_offloader().post_init()",
         "        get_offloader().post_init()\n"
         "        if self.speculator is not None:\n"
         "            torch.cuda.synchronize(self.device)"),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("Thor runner patch: startup anchor missing or ambiguous")
        source = source.replace(old, new)
    return source


MM_PREP = """                mm_budget = MultiModalBudget(
                    self.vllm_config,
                    self.mm_registry,
                    enable_cache=False,
                )
                dummy_mm_inputs = get_dummy_encoder_profile_inputs(
                    self.mm_registry,
                    mm_budget,
                )
"""
MM_INIT_ANCHOR = "        # Speculative decoding."
MM_EARLY = """        # Prepare CPU multimodal inputs before allocating model/runner tensors.
        # Keep actual encoder execution in profile_run for memory accounting.
        self._thor_mm_profile = None
        if self.supports_mm_inputs and self.is_first_pp_rank:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and not mm_config.skip_mm_profiling:
                budget = MultiModalBudget(
                    self.vllm_config, self.mm_registry, enable_cache=False
                )
                dummy = get_dummy_encoder_profile_inputs(self.mm_registry, budget)
                self._thor_mm_profile = (budget, dummy)
                logger.info("Thor: prepared multimodal profile inputs before model allocation")

"""


def patch_early_mm(source: str) -> str:
    edits = [
        (MM_INIT_ANCHOR, MM_EARLY + MM_INIT_ANCHOR),
        (MM_PREP,
         "                mm_budget, dummy_mm_inputs = self._thor_mm_profile\n"
         "                self._thor_mm_profile = None\n"),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("Thor runner patch: multimodal anchor missing or ambiguous")
        source = source.replace(old, new)
    return source


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    source = (here / "thor_model_runner.py.orig").read_text()
    patched = patch_early_mm(patch_startup(patch(source)))
    (here / "thor_model_runner.py").write_text(patched)
    print("patched Thor MRV2 startup and dummy inputs")
