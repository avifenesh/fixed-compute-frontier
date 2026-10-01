#!/usr/bin/env python3
"""CPU tests of serve_arm.sh and the shell scripts around it: no GPU, no SGLang, no network.

serve_arm.sh runs for real (bash, curl, setsid, arm_kind.py with torch reading small trainable files). Three things are
fake, all in this directory: fake_launch.py stands in for `python -m sglang.launch_server` (health, a chat answer, and the
`negeig: active` lines a real plugin logs), fake_negeig_cli.py for sglang/negeig_sglang.py (merge-lora with a receipt,
export-gates, install-plugin, static-check), and a stub nvidia-smi. The fake launch is found by the same pgrep pattern the
script uses for a real server, and stop_sglang is pointed at a pattern that only matches the fakes of one test (a real
SGLang server on the rig is never signalled; when one is running the end-to-end classes skip).

    python -m unittest test_serve_arm -v          (run_tests.sh does this with the CPU caps)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional

import testlib
from testlib import HERE, ScratchCase

SCRIPT = HERE / "serve_arm.sh"
BOX_DIR = HERE.parent / "box"
HAVE_TORCH = importlib.util.find_spec("torch") is not None
REAL_GUARD = r"[s]glang\.launch_server|^[s]glang::"   # the pattern serve_arm.sh itself uses before a launch
MODEL_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def sglang_running() -> bool:
    return subprocess.run(["pgrep", "-f", REAL_GUARD], capture_output=True).returncode == 0


def alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


def make_trainable(path: Path, kind: str, salt: float = 0.0) -> Path:
    """A tiny file in the shape train27.py saves. kind: wide (LoRA plus gate), ctrl (LoRA only), gate_only, junk."""
    import torch

    t = {}
    if kind in ("wide", "ctrl"):
        t["base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.default.weight"] = torch.full((2, 4), salt)
        t["base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.default.weight"] = torch.full((4, 2), salt)
    if kind in ("wide", "gate_only"):
        t["layers.0.linear_attn.negeig_w"] = torch.full((8, 4), salt)
    if kind == "junk":
        t["something.else"] = torch.zeros(1)
    torch.save(t, path)
    return path


class ServeCase(ScratchCase):
    """A scratch box: model dir, fake CUDA home, fake venv python, stub nvidia-smi, and the env that points
    serve_arm.sh at all of it."""

    def setUp(self) -> None:
        super().setUp()
        t = self.tmp
        self.model = t / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}")
        (self.model / ".negeig_revision").write_text(MODEL_REV + "\n")
        self.cuda = t / "cuda"
        (self.cuda / "bin").mkdir(parents=True)
        (self.cuda / "bin" / "nvcc").write_text("#!/usr/bin/env bash\necho stub\n")
        (self.cuda / "bin" / "nvcc").chmod(0o755)
        self.vpy_dir = t / "vpy" / "bin"
        self.vpy_dir.mkdir(parents=True)
        self.vpy = self.vpy_dir / "python3"
        # `-m sglang.launch_server` is the fake server (the label makes its command line match the script's guard),
        # `import sglang` answers a version, everything else is this interpreter
        self.vpy.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "-m" ] && [ "$2" = "sglang.launch_server" ]; then shift 2; '
            f'exec "{sys.executable}" "{HERE}/fake_launch.py" --label sglang.launch_server "$@"; fi\n'
            'if [ "$1" = "-c" ] && [[ "$2" == *"import sglang"* ]]; then echo 0.5.20; exit 0; fi\n'
            f'exec "{sys.executable}" "$@"\n')
        self.vpy.chmod(0o755)
        smi = t / "bin" / "nvidia-smi"
        smi.parent.mkdir()
        smi.write_text(
            "#!/usr/bin/env bash\n"
            'n=${STUB_GPUS:-8}\n'
            'case "$*" in\n'
            '  *memory.used*) for i in $(seq 1 $n); do if [ "$i" = "$n" ]; then echo " ${STUB_GPU_USED_MIB:-0}"; else echo " 0"; fi; done ;;\n'
            '  *name*) for i in $(seq 1 $n); do echo " NVIDIA B300 SXM6 AC"; done ;;\n'
            '  *) exit 9 ;;\n'
            'esac\n')
        smi.chmod(0o755)
        self.eval_root = t / "eval"
        self.merged_root = t / "merged"
        self.plugin_dir = t / "plugin"
        self.port = testlib.free_port()
        self.record = t / "launch.jsonl"
        self.cli_log = t / "cli.jsonl"
        self.cli = HERE / "fake_negeig_cli.py"
        self.addCleanup(self.kill_fakes)

    # -- running the script --------------------------------------------------------------------------------------
    def env(self, **extra: str) -> Dict[str, str]:
        e = {"SERVE_VPY": str(self.vpy), "SERVE_MODEL_DIR": str(self.model), "NEGEIG_CLI": str(self.cli),
             "NVIDIA_SMI": str(self.tmp / "bin" / "nvidia-smi"), "EVAL_ROOT": str(self.eval_root),
             "MERGED_ROOT": str(self.merged_root), "PLUGIN_DIR": str(self.plugin_dir), "ALLOW_NO_REV": "1",
             "NEGEIG_CUDA_HOME": str(self.cuda), "NEGEIG_W": str(self.tmp / "w"), "SGLANG_POLL_S": "1",
             "PORT": str(self.port), "FAKE_RECORD": str(self.record), "FAKE_CLI_LOG": str(self.cli_log),
             "SGLANG_PROC_PATTERN": f"[f]ake_launch[.]py.*{re.escape(str(self.tmp))}"}
        for k in list(os.environ):
            if k.startswith(("FAKE_", "STUB_")) or k in ("DP", "TP", "CTX", "RADIX", "READY_S", "MEMFRAC", "EXTRA",
                                                         "ATTN_BACKEND", "MAMBA_RATIO", "ARM_DIR", "BOXDIR", "E27",
                                                         "SGLANG_PLUGINS", "NEGEIG_GATES"):
                e[k] = ""
        e.update(extra)
        return e

    def sh(self, *args: str, timeout: float = 240.0, **env: str) -> subprocess.CompletedProcess:
        return testlib.run(["bash", SCRIPT, *args], env=self.env(**env), timeout=timeout)

    def trainable(self, kind: str, name: Optional[str] = None, salt: float = 0.0) -> Path:
        return make_trainable(self.tmp / (name or f"trainable_{kind}.pt"), kind, salt)

    def up(self, kind: str, arm: str = "a", trainable: Optional[Path] = None, *extra: str, **env: str):
        args = ["up", "--arm", arm, "--kind", kind, "--dp", "2", "--port", str(self.port), "--ready-s", "60"]
        if kind != "base":
            args += ["--trainable", str(trainable or self.trainable(kind))]
        return self.sh(*args, *extra, **env)

    # -- reading what it left ------------------------------------------------------------------------------------
    def arm_dir(self, arm: str = "a") -> Path:
        return self.eval_root / arm

    def j(self, *parts: str):
        return json.loads(Path(self.eval_root, *parts).read_text())

    def launches(self) -> List[dict]:
        if not self.record.is_file():
            return []
        return [json.loads(l) for l in self.record.read_text().splitlines() if l.strip()]

    def cli_calls(self) -> List[List[str]]:
        if not self.cli_log.is_file():
            return []
        return [json.loads(l) for l in self.cli_log.read_text().splitlines() if l.strip()]

    def live_fakes(self) -> List[int]:
        return [r["pid"] for r in self.launches() if alive(r["pid"])]

    def kill_fakes(self) -> None:
        for r in self.launches():
            try:
                os.killpg(r["pid"], signal.SIGKILL)
            except OSError:
                try:
                    os.kill(r["pid"], signal.SIGKILL)
                except OSError:
                    pass
        subprocess.run(["pkill", "-9", "-f", f"fake_launch[.]py.*{self.tmp}"], capture_output=True)

    def assertRc(self, p: subprocess.CompletedProcess, rc: int) -> None:
        self.assertEqual(p.returncode, rc, f"exit {p.returncode}, wanted {rc}\nstdout:\n{p.stdout[-2500:]}\nstderr:\n{p.stderr[-2500:]}")

    def assertStopped(self, why: str) -> None:
        """A failed `up` leaves no server running and no serving.json."""
        self.assertEqual(self.live_fakes(), [], why)
        self.assertFalse((self.eval_root / "serving.json").exists(), why)


# ----------------------------------------------------------------------------------------------------------
class TestPrintCmd(ServeCase):
    """--print-cmd validates and prints the commands and touches nothing."""

    def pc(self, *args: str, **env: str) -> dict:
        p = self.sh("up", *args, "--print-cmd", **env)
        self.assertRc(p, 0)
        out: Dict[str, str] = {}
        for line in p.stdout.splitlines():
            k, _, v = line.partition(": ")
            out[k if _ else line.split("=")[0]] = v if _ else line
        out["_head"] = p.stdout.splitlines()[0]
        out["_n"] = str(len(p.stdout.splitlines()))
        return out

    BASE_FLAGS = ("--served-model-name qwen3.8-27b --tp-size 1 --dp-size 8 --host 127.0.0.1 --port {port} --dtype bfloat16 "
                  "--mem-fraction-static 0.85 --context-length 131072 --mamba-full-memory-ratio 3 "
                  "--linear-attn-backend triton --linear-attn-prefill-backend triton --linear-attn-decode-backend triton "
                  "--trust-remote-code --disable-radix-cache")

    @unittest.skipUnless(HAVE_TORCH, "torch is needed to write a trainable file")
    def test_base_ctrl_wide_lines(self):
        flags = self.BASE_FLAGS.format(port=self.port)
        b = self.pc("--arm", "b", "--kind", "base")
        self.assertEqual(b["launch"], f"{self.vpy} -m sglang.launch_server --model-path {self.model} {flags}")
        self.assertEqual(b["env"], "plugin not loaded (SGLANG_PLUGINS unset)")
        self.assertEqual(b["_head"], f"arm=b kind=base ranks=8 arm_dir={self.eval_root}/b")
        self.assertNotIn("merge", b)
        self.assertNotIn("gates", b)
        self.assertEqual(b["_n"], "3", "base prints the header, the env line and the launch line, nothing else")

        t = self.trainable("ctrl")
        merged = f"{self.merged_root}/c"
        c = self.pc("--arm", "c", "--kind", "ctrl", "--trainable", str(t))
        self.assertEqual(c["merge"], f"{self.vpy} {self.cli} merge-lora --base {self.model} --trainable {t} --out {merged} "
                                     "--scale 2.0 --rounding stochastic --seed 20261001 --min-delta-kept 0.95")
        self.assertEqual(c["launch"], f"{self.vpy} -m sglang.launch_server --model-path {merged} {flags}")
        self.assertEqual(c["env"], "plugin not loaded (SGLANG_PLUGINS unset)")
        self.assertNotIn("gates", c)

        w = self.trainable("wide")
        wd = self.pc("--arm", "w", "--kind", "wide", "--trainable", str(w))
        gates = f"{self.eval_root}/w/gates.safetensors"
        self.assertEqual(wd["gates"], f"{self.vpy} {self.cli} export-gates --trainable {w} --out {gates} "
                                      f"--config {self.model}/config.json")
        self.assertEqual(wd["env"], f"PYTHONPATH={self.plugin_dir} SGLANG_PLUGINS=negeig NEGEIG_GATES={gates}")
        self.assertIn(f"--model-path {self.tmp}/merged/w ", wd["launch"])
        self.assertTrue(wd["merge"].endswith("--min-delta-kept 0.95"))

    def test_the_three_triton_backends_and_no_speculative_flags_on_every_kind(self):
        launch = self.pc("--arm", "b", "--kind", "base")["launch"]
        for flag in ("--linear-attn-backend triton", "--linear-attn-prefill-backend triton",
                     "--linear-attn-decode-backend triton", "--dtype bfloat16", "--trust-remote-code"):
            self.assertIn(flag, launch)
        for flag in ("speculative", "replayssm", "tf32", "--enable-torch-compile", "--quantization", "--kv-cache-dtype"):
            self.assertNotIn(flag, launch)

    def test_options_reach_the_command(self):
        o = self.pc("--arm", "b", "--kind", "base", "--dp", "4", "--tp", "2", "--ctx", "65536", "--mem-fraction", "0.7",
                    "--radix", "on", "--extra", "--chunked-prefill-size 8192 --foo")
        self.assertEqual(o["_head"], f"arm=b kind=base ranks=8 arm_dir={self.eval_root}/b")
        for part in ("--tp-size 2", "--dp-size 4", "--context-length 65536", "--mem-fraction-static 0.7"):
            self.assertIn(part, o["launch"])
        self.assertNotIn("--disable-radix-cache", o["launch"])
        self.assertTrue(o["launch"].endswith("--trust-remote-code --chunked-prefill-size 8192 --foo"), o["launch"])

    def test_env_defaults_and_option_precedence(self):
        o = self.pc("--arm", "b", "--kind", "base", DP="3", CTX="4096", ATTN_BACKEND="flashinfer", MAMBA_RATIO="5")
        self.assertIn("--dp-size 3", o["launch"])
        self.assertIn("--context-length 4096", o["launch"])
        self.assertIn("--attention-backend flashinfer", o["launch"])
        self.assertIn("--mamba-full-memory-ratio 5", o["launch"])
        o = self.pc("--arm", "b", "--kind", "base", "--dp", "6", DP="3")
        self.assertIn("--dp-size 6", o["launch"])

    def test_nothing_is_created(self):
        self.pc("--arm", "b", "--kind", "base")
        self.assertFalse(self.eval_root.exists())
        self.assertFalse(self.merged_root.exists())
        self.assertFalse(self.plugin_dir.exists())
        self.assertEqual(self.launches(), [])

    def test_no_cuda_home_is_not_needed_to_print(self):
        p = self.sh("up", "--arm", "b", "--kind", "base", "--print-cmd", NEGEIG_CUDA_HOME=str(self.tmp / "nope"))
        self.assertRc(p, 0)


# ----------------------------------------------------------------------------------------------------------
class TestValidation(ServeCase):
    """Refusals before anything starts. rc 2 is a usage error, rc 1 a refusal."""

    def refuse(self, args: List[str], text: str, rc: int = 1, **env: str) -> None:
        p = self.sh(*args, **env)
        self.assertRc(p, rc)
        self.assertIn(text, p.stderr)
        self.assertEqual(self.launches(), [])
        self.assertFalse((self.eval_root / "serving.json").exists())

    def base(self, *extra: str) -> List[str]:
        return ["up", "--arm", "a", "--kind", "base", *extra]

    def test_usage_errors_exit_2(self):
        for args in ([], ["bogus"], ["up", "--bogus"], ["--help"], ["up", "-h"], ["down", "--bogus"]):
            p = self.sh(*args)
            self.assertRc(p, 2)
            self.assertIn("serve_arm.sh up", p.stderr, args)

    def test_a_value_option_as_the_last_argument_exits_2(self):
        for opt in ("--arm", "--kind", "--trainable", "--dp", "--tp", "--port", "--ctx", "--radix", "--ready-s",
                    "--mem-fraction", "--extra"):
            p = self.sh("up", opt)
            self.assertRc(p, 2)
            self.assertIn(f"{opt} needs a value", p.stderr)

    def test_arm_and_kind(self):
        self.refuse(["up", "--kind", "base"], "--arm NAME is required")
        for bad in ("a/b", "a b", "../x", "ä", ""):
            self.refuse(["up", "--arm", bad, "--kind", "base"], "must be [A-Za-z0-9_.-]+" if bad else "--arm NAME is required")
        self.refuse(["up", "--arm", "a"], "--kind must be base, ctrl or wide")
        self.refuse(["up", "--arm", "a", "--kind", "narrow"], "--kind must be base, ctrl or wide")

    def test_numbers(self):
        for opt, val in (("--dp", "0"), ("--dp", "x"), ("--dp", "-1"), ("--tp", "0"), ("--tp", "1.5"), ("--port", "0"),
                         ("--ctx", "0"), ("--ctx", "1e5"), ("--ready-s", "0"), ("--ready-s", "")):
            self.refuse(self.base(opt, val), "is not a positive integer")
        for val in ("0", "1.5", "-0.5", "abc", "0.", ""):
            self.refuse(self.base("--mem-fraction", val), "is not a fraction in (0, 1]")
        for tp in ("5", "7", "49", "64"):
            self.refuse(self.base("--tp", tp), "does not divide 48")
        self.refuse(self.base("--radix", "maybe"), "--radix must be on or off")
        for tp in ("1", "2", "3", "4", "6", "8"):
            self.assertRc(self.sh(*self.base("--tp", tp, "--print-cmd")), 0)
        for frac in ("0.85", ".9", "1", "1.0", "0.5"):
            self.assertRc(self.sh(*self.base("--mem-fraction", frac, "--print-cmd")), 0)

    @unittest.skipUnless(HAVE_TORCH, "torch is needed to write a trainable file")
    def test_trainable_rules(self):
        t = self.trainable("ctrl")
        self.refuse(self.base("--trainable", str(t)), "kind base takes no --trainable")
        self.refuse(["up", "--arm", "a", "--kind", "ctrl"], "needs --trainable FILE")
        self.refuse(["up", "--arm", "a", "--kind", "wide"], "needs --trainable FILE")
        self.refuse(["up", "--arm", "a", "--kind", "ctrl", "--trainable", str(self.tmp / "nope.pt")], "not found")

    def test_model_dir_and_revision(self):
        self.refuse(self.base(), "model dir", SERVE_MODEL_DIR=str(self.tmp / "nomodel"))
        (self.model / ".negeig_revision").unlink()
        self.refuse(self.base(), "missing: the model was not staged by bootstrap.sh", ALLOW_NO_REV="")
        (self.model / ".negeig_revision").write_text("0" * 40 + "\n")
        self.refuse(self.base(), f"is not revision {MODEL_REV}", ALLOW_NO_REV="")
        (self.model / ".negeig_revision").write_text(f"revision {MODEL_REV}\n")
        self.assertRc(self.sh(*self.base("--print-cmd"), ALLOW_NO_REV=""), 0)

    @unittest.skipUnless(HAVE_TORCH, "torch is needed to write a trainable file")
    def test_tools_must_exist(self):
        self.refuse(self.base(), "not found or not executable", SERVE_VPY=str(self.tmp / "nopy"))
        t = self.trainable("ctrl")
        self.refuse(["up", "--arm", "a", "--kind", "ctrl", "--trainable", str(t)], "missing", NEGEIG_CLI=str(self.tmp / "nocli.py"))
        # a base arm never runs the CLI, so a missing one does not matter for it
        p = self.sh(*self.base("--print-cmd"), NEGEIG_CLI=str(self.tmp / "nocli.py"))
        self.assertRc(p, 0)

    @unittest.skipIf(any((Path(c) / "bin" / "nvcc").exists() for c in ("/usr/local/cuda-13.2", "/usr/local/cuda-13.0", "/usr/local/cuda")),
                     "this machine has a CUDA toolkit, the refusal cannot be reached")
    def test_no_cuda_toolkit_refuses(self):
        self.refuse(self.base(), "no CUDA toolkit with nvcc", NEGEIG_CUDA_HOME=str(self.tmp / "none"))

    def test_status_and_down_take_no_arm(self):
        p = self.sh("status")
        self.assertRc(p, 1)
        self.assertIn("no server recorded", p.stdout)


# ----------------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_TORCH, "torch is needed to write a trainable file")
class TestUp(ServeCase):
    """The whole `up` flow against the fakes: what is written, what is launched, what is refused."""

    @classmethod
    def setUpClass(cls) -> None:
        if sglang_running():
            raise unittest.SkipTest("a real SGLang server is running on this machine")

    def test_base(self):
        # the plugin variables of the calling shell must not reach a base server
        p = self.up("base", SGLANG_PLUGINS="negeig", NEGEIG_GATES="/nowhere/gates.safetensors")
        self.assertRc(p, 0)
        ad = self.arm_dir()
        serve = self.j("a", "serve.json")
        self.assertEqual({k: serve[k] for k in ("context_length", "radix", "tp_size", "dp_size", "port", "dtype", "kind",
                                                "linear_attn_backend", "linear_prefill_backend", "linear_decode_backend",
                                                "attn_backend", "sglang_version", "model_rev", "gpu_name", "gpu_count",
                                                "extra_args", "mem_fraction", "gates_sha256")},
                         {"context_length": 131072, "radix": "off", "tp_size": 1, "dp_size": 2, "port": self.port,
                          "dtype": "bfloat16", "kind": "base", "linear_attn_backend": "triton",
                          "linear_prefill_backend": "triton", "linear_decode_backend": "triton", "attn_backend": "default",
                          "sglang_version": "0.5.20", "model_rev": MODEL_REV, "gpu_name": "NVIDIA B300 SXM6 AC",
                          "gpu_count": 8, "extra_args": "", "mem_fraction": 0.85, "gates_sha256": None})
        self.assertEqual(serve["model_dir"], str(self.model))
        self.assertIn("--disable-radix-cache", serve["cmdline"])
        eng = self.j("a", "engagement.json")
        self.assertTrue(eng["ok"])
        self.assertEqual((eng["negeig_lines"], eng["active_lines"], eng["ranks_expected"]), (0, 0, 2))
        arm = self.j("a", "arm.json")
        self.assertEqual((arm["arm"], arm["kind"], arm["engagement"]["ok"]), ("a", "base", True))
        self.assertNotIn("merge_receipt", arm)
        self.assertEqual(arm["serve"], serve)
        sv = self.j("serving.json")
        self.assertEqual((sv["arm"], sv["kind"], sv["port"], sv["dir"]), ("a", "base", self.port, str(ad)))
        [launch] = self.launches()
        self.assertEqual(sv["pid"], launch["pid"])
        self.assertTrue(alive(sv["pid"]))
        self.assertIsNone(launch["plugins"], "SGLANG_PLUGINS leaked into a base server")
        self.assertIsNone(launch["gates"], "NEGEIG_GATES leaked into a base server")
        self.assertEqual(launch["cuda_home"], str(self.cuda))
        self.assertEqual(launch["jit_deepgemm"], "0")
        self.assertEqual(launch["path0"], str(self.vpy_dir))
        self.assertEqual(launch["argv"][launch["argv"].index("--model-path") + 1], str(self.model))
        self.assertEqual(launch["argv"][launch["argv"].index("--dp-size") + 1], "2")
        self.assertIn("--disable-radix-cache", launch["argv"])
        self.assertEqual(self.cli_calls(), [], "a base arm never runs the negeig CLI")
        self.assertFalse(self.merged_root.joinpath("a").exists())
        self.assertEqual((ad / "launch.cmd").read_text().strip(), "launch: " + serve["cmdline"])
        # the liveness probe: think-off, temperature 0, the served name
        chat = [json.loads(l) for l in Path(str(self.record) + ".chat").read_text().splitlines()]
        self.assertEqual(len(chat), 1)
        self.assertEqual(chat[0]["model"], "qwen3.8-27b")
        self.assertEqual(chat[0]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(chat[0]["temperature"], 0)
        self.assertIn("17 times 23", chat[0]["messages"][0]["content"])
        # status sees it, down stops it
        s = self.sh("status")
        self.assertRc(s, 0)
        self.assertIn(f"health on :{self.port} = 200", s.stdout)
        d = self.sh("down")
        self.assertRc(d, 0)
        self.assertIn("stopped, GPUs free (max used 0 MiB)", d.stdout)
        self.assertEqual(self.live_fakes(), [])
        self.assertFalse((self.eval_root / "serving.json").exists())
        self.assertRc(self.sh("status"), 1)

    def test_wide(self):
        t = self.trainable("wide")
        p = self.up("wide", "w", t)
        self.assertRc(p, 0)
        gates = self.arm_dir("w") / "gates.safetensors"
        sha = hashlib.sha256(gates.read_bytes()).hexdigest()
        eng = self.j("w", "engagement.json")
        self.assertTrue(eng["ok"], eng)
        self.assertEqual((eng["active_lines"], eng["active_bare"], eng["active_logged"], eng["installed_lines"], eng["layers_per_rank"]),
                         (2, 2, 2, 2, [48]))
        self.assertEqual(eng["max_w"], ["0.5"])
        [launch] = self.launches()
        self.assertEqual(launch["plugins"], "negeig")
        self.assertEqual(launch["gates"], str(gates))
        self.assertTrue(launch["pythonpath"].startswith(str(self.plugin_dir)), launch["pythonpath"])
        self.assertEqual(launch["argv"][launch["argv"].index("--model-path") + 1], str(self.merged_root / "w"))
        serve = self.j("w", "serve.json")
        self.assertEqual(serve["gates_sha256"], sha)
        self.assertEqual(serve["kind"], "wide")
        self.assertEqual(serve["model_dir"], str(self.merged_root / "w"))
        self.assertEqual(serve["plugin_sha256"], hashlib.sha256(self.cli.read_bytes()).hexdigest())
        arm = self.j("w", "arm.json")
        self.assertEqual((arm["kind"], arm["engagement"]["ok"]), ("wide", True))
        self.assertEqual(arm["merge_receipt"]["delta_kept"], 0.99)
        self.assertEqual(arm["merge_receipt"]["trainable_sha256"], hashlib.sha256(t.read_bytes()).hexdigest())
        self.assertTrue((self.arm_dir("w") / "MERGE-RECEIPT.json").is_file())
        self.assertEqual(self.j("w", "trainable.kind.json")["kind"], "wide")
        self.assertTrue(self.plugin_dir.joinpath("negeig_sglang.py").is_file())
        calls = self.cli_calls()
        self.assertEqual([c[0] for c in calls], ["merge-lora", "export-gates", "install-plugin", "static-check"])
        ml = calls[0]
        for flag, val in (("--base", str(self.model)), ("--trainable", str(t)), ("--out", str(self.merged_root / "w")),
                          ("--scale", "2.0"), ("--rounding", "stochastic"), ("--seed", "20261001"),
                          ("--min-delta-kept", "0.95")):
            self.assertEqual(ml[ml.index(flag) + 1], val, flag)
        eg = calls[1]
        self.assertEqual(eg[eg.index("--config") + 1], str(self.model / "config.json"))
        self.assertEqual(self.j("serving.json")["kind"], "wide")
        self.assertRc(self.sh("down"), 0)

    def test_wide_under_tensor_parallel_expects_one_line_per_rank_with_fewer_layers(self):
        p = self.up("wide", "w", None, "--tp", "2")
        self.assertRc(p, 0)
        eng = self.j("w", "engagement.json")
        self.assertEqual((eng["active_lines"], eng["ranks_expected"], eng["layers_per_rank"]), (4, 4, [24]))
        self.assertEqual(self.j("w", "serve.json")["tp_size"], 2)

    def test_ctrl(self):
        p = self.up("ctrl", "c")
        self.assertRc(p, 0)
        self.assertIn("no negeig_w tensors", (self.arm_dir("c") / "ctrl.export.err").read_text())
        self.assertFalse((self.arm_dir("c") / "gates.safetensors").exists())
        self.assertFalse((self.arm_dir("c") / "ctrl_gates_must_not_exist.safetensors").exists())
        eng = self.j("c", "engagement.json")
        self.assertTrue(eng["ok"])
        self.assertEqual(eng["negeig_lines"], 0)
        [launch] = self.launches()
        self.assertIsNone(launch["plugins"])
        self.assertIsNone(launch["gates"])
        self.assertEqual(launch["argv"][launch["argv"].index("--model-path") + 1], str(self.merged_root / "c"))
        arm = self.j("c", "arm.json")
        self.assertEqual((arm["kind"], arm["merge_receipt"]["scale"]), ("ctrl", 2.0))
        self.assertEqual([c[0] for c in self.cli_calls()], ["merge-lora", "export-gates"])
        self.assertIsNone(self.j("c", "serve.json")["gates_sha256"])

    def test_base_and_arm_serve_records_agree_on_every_compared_key(self):
        """What compare_general.py checks between two arms is what serve_arm.sh wrote: nothing differs for one launch recipe."""
        import compare_general as cg

        self.assertRc(self.up("base", "b"), 0)
        self.assertRc(self.sh("down"), 0)
        self.assertRc(self.up("wide", "w"), 0)
        b, w = self.j("b", "serve.json"), self.j("w", "serve.json")
        for key in cg.SERVE_KEYS:
            self.assertIn(key, b, key)
            self.assertEqual(b[key], w[key], key)
        info = cg.arm_validity(self.j("w", "arm.json"))
        self.assertEqual((info["kind"], info["engagement_ok"], info["delta_kept"]), ("wide", True, 0.99))
        # a different extra argument on one arm is a protocol difference compare_general refuses
        self.assertRc(self.sh("down"), 0)
        self.assertRc(self.up("wide", "w2", None, "--extra", "--chunked-prefill-size 4096"), 0)
        self.assertNotEqual(self.j("w2", "serve.json")["extra_args"], b["extra_args"])
        self.assertEqual(self.launches()[-1]["argv"][-2:], ["--chunked-prefill-size", "4096"])
        self.assertIn("extra_args", [k for k in cg.SERVE_KEYS if b[k] != self.j("w2", "serve.json")[k]])

    def test_the_declared_kind_has_to_match_the_file(self):
        p = self.up("wide", "a", self.trainable("ctrl"))
        self.assertRc(p, 1)
        self.assertIn("declared kind wide but the trainable is ctrl", p.stderr)
        p = self.up("ctrl", "a", self.trainable("wide"))
        self.assertRc(p, 1)
        self.assertIn("declared kind ctrl but the trainable is wide", p.stderr)
        p = self.up("wide", "a", self.trainable("gate_only"))
        self.assertRc(p, 1)
        self.assertIn("neither a wide nor a ctrl arm", p.stderr)
        p = self.up("ctrl", "a", self.trainable("junk"))
        self.assertRc(p, 1)
        self.assertEqual(self.launches(), [])
        self.assertEqual(self.cli_calls(), [])

    def test_the_trainable_may_be_a_full_checkpoint(self):
        import torch

        t = self.tmp / "ckpt_000600.pt"
        torch.save({"step": 600, "trainable": torch.load(self.trainable("ctrl"), weights_only=True), "opt": {}, "sched": {}}, t)
        self.assertRc(self.up("ctrl", "c", t), 0)

    # -- the refusals after the launch --------------------------------------------------------------------------
    def red_wide(self, text: str, **env: str) -> dict:
        p = self.up("wide", "w", None, **env)
        self.assertRc(p, 1)
        self.assertIn("engagement check failed for w (wide)", p.stderr)
        self.assertIn(text, p.stderr)
        self.assertStopped(text)
        arm = self.j("w", "arm.json")  # the record of the failure stays, with the reason
        self.assertFalse(arm["engagement"]["ok"])
        self.assertTrue(arm["engagement"]["problems"])
        self.assertEqual(len(self.launches()), 1)
        return arm

    def test_red_wide_one_rank_silent(self):
        self.red_wide("1 'negeig: active' lines, expected 2", FAKE_ACTIVE_RANKS="1")

    def test_red_wide_no_engagement_line(self):
        self.red_wide("0 'negeig: active' lines, expected 2", FAKE_NO_ENGAGE="1")

    def test_red_wide_other_gates_loaded(self):
        self.red_wide("not the exported file", FAKE_SHA="ab" * 32)

    def test_red_wide_zero_gate(self):
        self.red_wide("the gate is zero or not finite", FAKE_MAXW="0.0")

    def test_red_wide_nan_gate(self):
        self.red_wide("the gate is zero or not finite", FAKE_MAXW="nan")

    def test_red_wide_wrong_layer_count(self):
        self.red_wide("expected 48", FAKE_LAYERS="47")

    def test_red_wide_an_extra_negeig_line(self):
        self.red_wide("other 'negeig:' line", FAKE_EXTRA_LINE="1")

    def test_red_wide_installed_line_of_another_patch_version(self):
        self.red_wide("patch version", FAKE_INSTALLED_VERSION="2")

    def test_wide_bare_only_and_logged_only_logs_are_both_accepted(self):
        for form in ("bare", "logged"):
            with self.subTest(form=form):
                p = self.up("wide", "w", None, FAKE_LOG_FORM=form)
                self.assertRc(p, 0)
                eng = self.j("w", "engagement.json")
                self.assertTrue(eng["ok"], eng)
                self.assertEqual(eng["active_lines"], 2)
                self.assertRc(self.sh("down"), 0)

    def test_red_base_that_carries_the_plugin_banner(self):
        p = self.up("base", "a", None, FAKE_FORCE_ENGAGE="1")
        self.assertRc(p, 1)
        self.assertIn("the plugin must not be loaded", p.stderr)
        self.assertStopped("base with a banner")
        self.assertFalse(self.j("a", "arm.json")["engagement"]["ok"])

    def test_red_ctrl_that_carries_the_plugin_banner(self):
        p = self.up("ctrl", "c", None, FAKE_FORCE_ENGAGE="1")
        self.assertRc(p, 1)
        self.assertIn("ctrl: the log has 6 'negeig:' line(s) but the plugin must not be loaded", p.stderr)
        self.assertStopped("ctrl with a banner")

    def test_red_probe_wrong_answer(self):
        p = self.up("base", "a", None, FAKE_ANSWER="42")
        self.assertRc(p, 1)
        self.assertIn("the probe did not answer 391 (got: 42)", p.stderr)
        self.assertStopped("wrong probe answer")
        self.assertFalse((self.arm_dir() / "arm.json").exists(), "no arm.json for a server that never answered")

    def test_red_probe_http_error(self):
        p = self.up("base", "a", None, FAKE_CHAT_STATUS="500")
        self.assertRc(p, 1)
        self.assertIn("the probe did not answer 391", p.stderr)
        self.assertStopped("probe 500")

    def test_red_server_crash_is_seen_at_once(self):
        p = self.up("base", "a", None, FAKE_CRASH="1")
        self.assertRc(p, 1)
        self.assertIn("sglang exited early", p.stdout)
        self.assertIn("did not become healthy within 60 s", p.stderr)
        self.assertIn("planned crash", p.stdout, "the log tail is printed")
        self.assertStopped("crash")

    def test_red_ready_deadline(self):
        p = self.up("base", "a", None, "--ready-s", "3", FAKE_DELAY="60")
        self.assertRc(p, 1)
        self.assertIn("sglang not healthy after 3 s", p.stdout)
        self.assertIn("did not become healthy within 3 s", p.stderr)
        self.assertStopped("deadline")

    # -- the merge and the gates ------------------------------------------------------------------------------
    def test_red_merge_fails(self):
        p = self.up("ctrl", "c", None, FAKE_MERGE_FAIL="1")
        self.assertRc(p, 1)
        self.assertIn("merge-lora failed", p.stderr)
        self.assertIn("planned failure", p.stderr)
        self.assertEqual(self.launches(), [])

    def test_red_merge_kept_too_little(self):
        p = self.up("ctrl", "c", None, FAKE_DELTA_KEPT="0.5")
        self.assertRc(p, 1)
        self.assertIn("merge receipt check failed", p.stderr)
        self.assertIn("is below 0.95", p.stderr)
        self.assertEqual(self.launches(), [])
        self.assertFalse((self.eval_root / "serving.json").exists())

    def test_red_ctrl_file_that_export_gates_accepts(self):
        p = self.up("ctrl", "c", None, FAKE_EXPORT_ACCEPTS_CTRL="1")
        self.assertRc(p, 1)
        self.assertIn("it carries gate tensors", p.stderr)
        self.assertEqual(self.launches(), [])
        self.assertFalse((self.arm_dir("c") / "ctrl_gates_must_not_exist.safetensors").exists())

    def test_red_ctrl_export_gates_fails_for_another_reason(self):
        p = self.up("ctrl", "c", None, FAKE_EXPORT_OTHER_FAIL="1")
        self.assertRc(p, 1)
        self.assertIn("failed for another reason than 'no negeig_w tensors'", p.stderr)
        self.assertEqual(self.launches(), [])

    def test_red_static_check_fails(self):
        p = self.up("wide", "w", None, FAKE_STATIC_RC="1")
        self.assertRc(p, 1)
        self.assertIn("static-check failed", p.stderr)
        self.assertEqual(self.launches(), [])

    def test_red_export_gates_fails_on_a_wide_arm(self):
        p = self.up("wide", "w", None, FAKE_EXPORT_OTHER_FAIL="1")
        self.assertRc(p, 1)
        self.assertIn("export-gates failed", p.stderr)
        self.assertEqual(self.launches(), [])

    def test_merge_is_reused_for_the_same_file_and_rebuilt_for_another(self):
        t = self.trainable("ctrl", salt=0.0)
        self.assertRc(self.up("ctrl", "c", t), 0)
        self.assertRc(self.sh("down"), 0)
        self.assertEqual([c[0] for c in self.cli_calls()].count("merge-lora"), 1)
        p = self.up("ctrl", "c", t)
        self.assertRc(p, 0)
        self.assertIn("reusing the finished merge", p.stdout)
        self.assertEqual([c[0] for c in self.cli_calls()].count("merge-lora"), 1, "the same file is not merged twice")
        self.assertRc(self.sh("down"), 0)
        t2 = self.trainable("ctrl", name="trainable_other.pt", salt=1.0)
        p = self.up("ctrl", "c", t2)
        self.assertRc(p, 0)
        self.assertIn("exists without a valid receipt for this trainable: removing it", p.stdout)
        self.assertEqual([c[0] for c in self.cli_calls()].count("merge-lora"), 2)
        receipt = json.loads((self.merged_root / "c" / "MERGE-RECEIPT.json").read_text())
        self.assertEqual(receipt["trainable_sha256"], hashlib.sha256(t2.read_bytes()).hexdigest())
        # a merge directory with a receipt that kept too little is not reused either
        self.assertRc(self.sh("down"), 0)
        rp = self.merged_root / "c" / "MERGE-RECEIPT.json"
        doc = json.loads(rp.read_text())
        doc["delta_kept"] = 0.4
        rp.write_text(json.dumps(doc))
        self.assertRc(self.up("ctrl", "c", t2), 0)
        self.assertEqual([c[0] for c in self.cli_calls()].count("merge-lora"), 3)

    def test_down_purge_removes_the_merged_model_and_only_with_purge(self):
        self.assertRc(self.up("ctrl", "c"), 0)
        self.assertRc(self.sh("down"), 0)
        self.assertTrue((self.merged_root / "c").is_dir())
        self.assertRc(self.up("ctrl", "c"), 0)
        d = self.sh("down", "--purge")
        self.assertRc(d, 0)
        self.assertFalse((self.merged_root / "c").exists())
        self.assertTrue((self.arm_dir("c") / "arm.json").is_file(), "the arm record stays")

    # -- one server at a time ----------------------------------------------------------------------------------
    def test_a_second_server_is_refused_and_replace_swaps_it(self):
        self.assertRc(self.up("base", "one"), 0)
        first = self.j("serving.json")["pid"]
        p = self.up("base", "two")
        self.assertRc(p, 1)
        self.assertIn("an SGLang server is already running", p.stderr)
        self.assertTrue(alive(first))
        self.assertEqual(self.j("serving.json")["arm"], "one", "the refusal leaves the running server's record alone")
        p = self.up("base", "two", None, "--replace")
        self.assertRc(p, 0)
        self.assertIn("--replace: stopping the running server", p.stdout)
        self.assertFalse(alive(first))
        self.assertEqual(self.j("serving.json")["arm"], "two")
        self.assertEqual(len(self.live_fakes()), 1)

    def test_a_busy_port_is_refused(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", self.port))
            s.listen(1)
            p = self.up("base")
            self.assertRc(p, 1)
            self.assertIn(f"port {self.port} is already in use", p.stderr)
        self.assertEqual(self.launches(), [])

    def test_a_rerun_clears_the_earlier_records_first(self):
        self.assertRc(self.up("base", "a"), 0)
        self.assertRc(self.sh("down"), 0)
        p = self.up("base", "a", None, FAKE_ANSWER="42")
        self.assertRc(p, 1)
        for name in ("arm.json", "serve.json", "engagement.json"):
            self.assertFalse((self.arm_dir() / name).exists(), f"{name} of the earlier run survived a failed one")

    # -- down and status ---------------------------------------------------------------------------------------
    def test_down_with_nothing_running(self):
        d = self.sh("down")
        self.assertRc(d, 0)
        self.assertIn("stopped, GPUs free", d.stdout)

    def test_down_refuses_while_a_gpu_is_held_by_something_else(self):
        self.assertRc(self.up("base"), 0)
        d = self.sh("down", STUB_GPU_USED_MIB="5000")
        self.assertRc(d, 1)
        self.assertIn("a GPU still holds 5000 MiB after stopping SGLang", d.stderr)
        self.assertEqual(self.live_fakes(), [], "the server itself was stopped")
        self.assertTrue((self.eval_root / "serving.json").exists(), "the record stays until the GPUs are free")
        d = self.sh("down")
        self.assertRc(d, 0)
        self.assertFalse((self.eval_root / "serving.json").exists())

    def test_down_without_nvidia_smi_does_not_claim_free_gpus(self):
        d = self.sh("down", NVIDIA_SMI=str(self.tmp / "no-smi"))
        self.assertRc(d, 0)
        self.assertIn("stopped, GPUs free", d.stdout)
        self.assertNotIn("max used", d.stdout)

    def test_status_after_a_killed_server(self):
        self.assertRc(self.up("base"), 0)
        os.killpg(self.j("serving.json")["pid"], signal.SIGKILL)
        for _ in range(50):
            if not self.live_fakes():
                break
            subprocess.run(["sleep", "0.1"])
        s = self.sh("status")
        self.assertRc(s, 1)
        self.assertIn(f"health on :{self.port} = 000", s.stdout)


# ----------------------------------------------------------------------------------------------------------
class TestShellScripts(ScratchCase):
    """bash -n on every shell script of the runner, and the CLI shape of prepare_box.sh."""

    def scripts(self) -> List[Path]:
        return sorted(HERE.glob("*.sh"))

    def test_the_scripts_are_there(self):
        names = {p.name for p in self.scripts()}
        for n in ("serve_arm.sh", "run_general.sh", "prepare_box.sh"):
            self.assertIn(n, names)

    def test_bash_n(self):
        for p in self.scripts():
            r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{p.name}: {r.stderr}")

    def test_the_scripts_are_executable_and_start_with_a_shebang(self):
        for p in self.scripts():
            self.assertTrue(os.access(p, os.X_OK), f"{p.name} is not executable")
            self.assertEqual(p.read_text().splitlines()[0], "#!/usr/bin/env bash", p.name)

    def test_shellcheck_when_it_is_installed(self):
        sc = subprocess.run(["sh", "-c", "command -v shellcheck"], capture_output=True, text=True).stdout.strip()
        if not sc:
            self.skipTest("shellcheck is not installed")
        for p in self.scripts():
            r = subprocess.run([sc, "-S", "warning", "-x", str(p)], capture_output=True, text=True, cwd=str(HERE))
            self.assertEqual(r.returncode, 0, f"{p.name}:\n{r.stdout}")

    def test_prepare_box_usage_and_unknown_mode_exit_2(self):
        for args in ([], ["--help"], ["bogus"]):
            r = testlib.run(["bash", HERE / "prepare_box.sh", *args], env={"NEGEIG_W": str(self.tmp / "w")})
            self.assertEqual(r.returncode, 2, args)
        r = testlib.run(["bash", HERE / "prepare_box.sh", "--help"], env={"NEGEIG_W": str(self.tmp / "w")})
        self.assertIn("prepare_box.sh all", r.stdout)

    def test_prepare_box_check_on_an_empty_workspace_names_every_gap(self):
        r = testlib.run(["bash", HERE / "prepare_box.sh", "check"],
                        env={"NEGEIG_W": str(self.tmp / "w"), "PREP_OFFLINE": "1"})
        self.assertEqual(r.returncode, 1)
        for gap in ("evals venv", "BFCL venv", "mmlupro_test.jsonl"):
            self.assertIn(gap, r.stderr)
        self.assertFalse((self.tmp / "w").exists() and any((self.tmp / "w").iterdir()), "check changes nothing")


if __name__ == "__main__":
    unittest.main()
