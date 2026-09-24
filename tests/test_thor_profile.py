"""Check wrapper precedence and argument forwarding without Docker or weights."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ThorProfileTests(unittest.TestCase):
    def run_profile(self, overrides=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(ROOT / "start-thor.sh", root)
            (root / "start.sh").write_text(
                "python3 -c 'import json,os,sys; "
                "print(json.dumps([dict(os.environ),sys.argv[1:]]))' \"$@\"\n"
            )
            env = {"PATH": os.environ["PATH"], **(overrides or {})}
            result = subprocess.run(
                ["bash", str(root / "start-thor.sh"), "--no-launch"],
                env=env, capture_output=True, text=True, check=True,
            )
            return json.loads(result.stdout)

    def test_defaults_and_arguments(self):
        env, args = self.run_profile()
        self.assertEqual(args, ["--no-launch"])
        self.assertEqual(env["DOCKER_GPU_RUNTIME"], "nvidia")
        self.assertEqual(env["MOE_BACKEND"], "marlin")
        self.assertEqual(env["MAX_MODEL_LEN"], "262144")
        self.assertEqual(env["KV_TARGET_GIB"], "8")
        self.assertEqual(env["MTP_NUM_SPECULATIVE_TOKENS"], "0")
        self.assertEqual(env["CUDAGRAPH_MODE"], "NONE")
        self.assertEqual(env["KV_CACHE_DTYPE"], "auto")
        self.assertEqual(env["OVERHEAD_GIB"], "14")
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("IMAGE", env)

    def test_explicit_environment_wins(self):
        overrides = {"MAX_MODEL_LEN": "65536", "MTP_NUM_SPECULATIVE_TOKENS": "3",
                     "MOE_BACKEND": "auto", "IMAGE": "local/test", "PORT": "9000"}
        env, _ = self.run_profile(overrides)
        for key, value in overrides.items():
            self.assertEqual(env[key], value)


class ArchivePruningTests(unittest.TestCase):
    def test_first_launch_and_retention(self):
        source = (ROOT / "start.sh").read_text()
        code = source.split("<<'PRUNE_PY'\n", 1)[1].split("\nPRUNE_PY", 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["python3", "-c", code, tmp], check=True)
            root = Path(tmp)
            for i in range(23):
                for suffix in ("container", "memwatch", "timeout"):
                    path = root / f"run-{i:02}-{suffix}.log"
                    path.touch()
                    os.utime(path, (i + 1, i + 1))
            subprocess.run(["python3", "-c", code, tmp], check=True)
            self.assertEqual(len(list(root.glob("*-container.log"))), 20)
            self.assertFalse((root / "run-02-memwatch.log").exists())
            self.assertTrue((root / "run-03-memwatch.log").exists())


if __name__ == "__main__":
    unittest.main()
