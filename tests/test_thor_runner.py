"""Exercise dummy versus real input dispatch without CUDA/model weights."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("patch_thor_runner", ROOT / "files/patch_thor_runner.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class RunnerPatchTests(unittest.TestCase):
    def prepare(self, dummy, state):
        source = "def run(self, input_batch, dummy_run):\n    return {\n" + patcher.OLD + "\n    }\n"
        scope = {}
        exec(patcher.patch(source), scope)
        batch = SimpleNamespace(num_reqs_after_padding=1, num_tokens_after_padding=1024)
        runner = SimpleNamespace(model_state=state, req_states=object())
        return scope["run"](runner, batch, dummy)

    def test_dummy_never_reads_live_request_history(self):
        class State:
            def prepare_inputs(self, *args):
                raise AssertionError("Dummy request must not gather live request history")

            def prepare_dummy_inputs(self, requests, tokens):
                return {"shape": (requests, tokens), "ngram_context": "EOS"}

        self.assertEqual(self.prepare(True, State()), {"shape": (1, 1024), "ngram_context": "EOS"})

    def test_real_requests_keep_live_input_path(self):
        class State:
            def prepare_inputs(self, batch, requests):
                return {"ngram_context": "real history"}

            def prepare_dummy_inputs(self, *args):
                raise AssertionError("Real request must not use dummy history")

        self.assertEqual(self.prepare(False, State()), {"ngram_context": "real history"})

    def test_startup_barriers_surround_state_initialization(self):
        source = """def initialize(self):
        # Initialize the components that require the model.
        initialize_state()
        get_offloader().post_init()
"""
        for speculative in (False, True):
            events = []
            scope = {
                "torch": SimpleNamespace(cuda=SimpleNamespace(
                    synchronize=lambda device: events.append("sync"))),
                "initialize_state": lambda: events.append("state"),
                "get_offloader": lambda: SimpleNamespace(
                    post_init=lambda: events.append("post_init")),
            }
            exec(patcher.patch_startup(source), scope)
            scope["initialize"](SimpleNamespace(
                speculator=object() if speculative else None, device="cuda"))
            self.assertEqual(events, ["sync", "state", "post_init", "sync"]
                             if speculative else ["state", "post_init"])

    def test_multimodal_setup_precedes_allocations_but_encoder_is_profiled(self):
        source = """class Runner:
    def __init__(self, enabled=True, skip=False):
        self.supports_mm_inputs = enabled
        self.is_first_pp_rank = True
        self.model_config = SimpleNamespace(multimodal_config=SimpleNamespace(skip_mm_profiling=skip))
        self.vllm_config = object()
        self.mm_registry = object()
""" + patcher.MM_INIT_ANCHOR + """
        events.append("allocate")

    def profile_run(self):
        if self.supports_mm_inputs and self.is_first_pp_rank:
            mm_config = self.model_config.multimodal_config
            if mm_config is not None and not mm_config.skip_mm_profiling:
""" + patcher.MM_PREP + """                events.append(("encoder", mm_budget, dummy_mm_inputs))
"""
        for enabled, skip in ((True, False), (False, False), (True, True)):
            events = []
            def budget(*args, **kwargs):
                events.append("budget")
                return "budget-object"
            def dummy(*args):
                events.append("dummy")
                return "dummy-inputs"
            scope = dict(SimpleNamespace=SimpleNamespace, events=events,
                         MultiModalBudget=budget, get_dummy_encoder_profile_inputs=dummy,
                         logger=SimpleNamespace(info=lambda *args: None))
            exec(patcher.patch_early_mm(source), scope)
            runner = scope["Runner"](enabled, skip)
            runner.profile_run()
            expected = ["budget", "dummy", "allocate",
                        ("encoder", "budget-object", "dummy-inputs")] if enabled and not skip else ["allocate"]
            self.assertEqual(events, expected)
            self.assertIsNone(runner._thor_mm_profile)

    def test_source_drift_is_rejected(self):
        for source in ("", patcher.OLD * 2):
            with self.assertRaises(ValueError):
                patcher.patch(source)
        for source in ("", patcher.MM_INIT_ANCHOR + patcher.MM_PREP * 2):
            with self.assertRaises(ValueError):
                patcher.patch_early_mm(source)


if __name__ == "__main__":
    unittest.main()
