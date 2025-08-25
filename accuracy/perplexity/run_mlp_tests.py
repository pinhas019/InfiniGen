"""Run MLP eviction-policy experiments (automation for `table2.sh` mlp runs).

This script follows the commands in `accuracy/perplexity/table2.sh` but focuses on testing the
`mlp` eviction policy for OPT and LLaMA experiments.

Usage (PowerShell):
  # dry-run (print commands only)
  python run_mlp_tests.py --dry-run

  # quick smoke tests (fast mode) - short seq and 1 sample per run
  python run_mlp_tests.py --fast

  # run full experiments (heavy compute and requires models and setup per README)
  python run_mlp_tests.py

Notes / prerequisites (from accuracy/README.md):
 - Install transformers (v4.35 suggested) and dependencies. Example:
     git clone -b v4.35-release https://github.com/huggingface/transformers.git
     cd transformers
     pip install -e .
 - Place llama-2 models and set environment variable LLAMA_PATH to the folder containing
   llama-2-7b / llama-2-13b directories. Or pass --llama-path to override.
 - Ensure the repo's `setup` folder contains generated partial weights and skewing matrix
   (follow README.md -> setup steps).

This script runs the python entry points `opt.py` and `llama.py` which are expected to live
in the same directory as this script (accuracy/perplexity).
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path
import datetime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root (InfiniGen/accuracy)

# Defaults mirroring table2.sh
PARTIAL = 0.2
SEQLEN = 2048
OPT_SIZES = ["6.7b", "13b", "30b"]
LLAMA_SIZES = ["7b", "13b"]
DATASETS = ["wikitext2", "ptb"]

OPT_ALPHA = 4
OPT_BUDGET = 0.2
LLAMA_ALPHA = 5
LLAMA_BUDGET = 0.2

LOG_DIR = HERE / "mlp_test_logs"
LOG_DIR.mkdir(exist_ok=True)


def check_prereqs(llama_path):
    # Check python import availability for quick feedback
    try:
        import transformers  # noqa: F401
    except Exception:
        print("Warning: `transformers` not importable. Follow accuracy/README.md to install it (suggested v4.35).")

    # Check that opt.py and llama.py exist
    opt_py = HERE / "opt.py"
    llama_py = HERE / "llama.py"
    if not opt_py.exists():
        print(f"Missing: {opt_py}. `opt.py` must be present in {HERE}")
    if not llama_py.exists():
        print(f"Missing: {llama_py}. `llama.py` must be present in {HERE}")

    # Check LLAMA_PATH
    if llama_path is None:
        llama_path = os.environ.get("LLAMA_PATH")
    if llama_path is None:
        print("Warning: LLAMA_PATH not set. LLaMA experiments will fail unless you set --llama-path or LLAMA_PATH env var.")
    else:
        p = Path(llama_path)
        if not p.exists():
            print(f"Warning: LLAMA_PATH={llama_path} does not exist on disk")

    # Check setup generated weights/skewing matrix (best-effort)
    setup_weights_dir = ROOT / "setup" / "weights"
    if not setup_weights_dir.exists():
        print(f"Note: expected setup weights at {setup_weights_dir} not found — follow README setup steps if running full experiments.")


def run_cmd(cmd, log_path, dry_run=False):
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] CMD: {' '.join(cmd)}")
    if dry_run:
        print(f"Dry-run: would write logs to {log_path}")
        return 0

    with open(log_path, "wb") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            try:
                sys.stdout.buffer.write(line)
            except Exception:
                # fallback if stdout.buffer not available
                try:
                    sys.stdout.write(line.decode(errors='ignore'))
                except Exception:
                    pass
        proc.wait()
        return proc.returncode


def build_and_run(commands, dry_run=False):
    results = []
    for cmd, logname in commands:
        logpath = LOG_DIR / logname
        # ensure parent exists
        logpath.parent.mkdir(parents=True, exist_ok=True)
        rc = run_cmd(cmd, str(logpath), dry_run=dry_run)
        results.append((cmd, rc, str(logpath)))
        if rc != 0 and not dry_run:
            print(f"Command failed (rc={rc}): {' '.join(cmd)}. See {logpath}")
    return results


def make_opt_commands(models_root, fast=False):
    cmds = []
    seqlen = 128 if fast else SEQLEN
    eval_samples = 1 if fast else 0

    for size in OPT_SIZES:
        model_dir = Path(models_root) / f"opt-{size}"
        model_arg = str(model_dir)
        for dataset in DATASETS:
            # Only mlp eviction policy runs (capacity 0.8) as requested
            logname = f"opt-{size}_{dataset}_80pct_mlp.log"
            cmd = [sys.executable, str(HERE / "opt.py"),
                   "--model", model_arg,
                   "--eval_dataset", dataset,
                   "--seq_len", str(seqlen),
                   "--eval_samples", str(eval_samples),
                   "--model_name", f"opt-{size}",
                   "--infinigen",
                   "--partial_weight_ratio", str(PARTIAL),
                   "--partial_weight_path", str(ROOT / "setup" / "weights" / f"opt-{size}_{PARTIAL}"),
                   "--alpha", str(OPT_ALPHA),
                   "--budget", str(OPT_BUDGET),
                   "--capacity", "0.8",
                   "--eviction_policy", "mlp"]
            cmds.append((cmd, logname))
    return cmds


def make_llama_commands(llama_path, fast=False):
    cmds = []
    seqlen = 128 if fast else SEQLEN
    eval_samples = 1 if fast else 0

    for size in LLAMA_SIZES:
        # model is expected at {LLAMA_PATH}/llama-2-{size}
        model_arg = str(Path(llama_path) / f"llama-2-{size}")
        for dataset in DATASETS:
            logname = f"llama-{size}_{dataset}_80pct_mlp.log"
            cmd = [sys.executable, str(HERE / "llama.py"),
                   "--model", model_arg,
                   "--eval_dataset", dataset,
                   "--seq_len", str(seqlen),
                   "--eval_samples", str(eval_samples),
                   "--model_name", f"llama-{size}",
                   "--infinigen",
                   "--partial_weight_ratio", str(PARTIAL),
                   "--partial_weight_path", str(ROOT / "setup" / "weights" / f"llama-2-{size}_{PARTIAL}"),
                   "--skewing_matrix_path", str(ROOT / "setup" / "skewing_matrix" / f"llama-2-{size}.pt"),
                   "--alpha", str(LLAMA_ALPHA),
                   "--budget", str(LLAMA_BUDGET),
                   "--capacity", "0.8",
                   "--eviction_policy", "mlp"]
            cmds.append((cmd, logname))
    return cmds


def main():
    parser = argparse.ArgumentParser(description="Run MLP-eviction experiments (automation for table2.sh mlp runs)")
    parser.add_argument("--models-root", type=str, default=str(ROOT / "setup" / "opt-model"),
                        help="Root directory containing opt-* model directories (default: setup/opt-model)")
    parser.add_argument("--llama-path", type=str, default=os.environ.get("LLAMA_PATH"),
                        help="Path to LLAMA models if not set in LLAMA_PATH env var")
    parser.add_argument("--fast", action="store_true", help="Fast mode: small seq_len and 1 eval sample (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't run them")
    parser.add_argument("--only-opt", action="store_true", help="Run only OPT experiments")
    parser.add_argument("--only-llama", action="store_true", help="Run only LLaMA experiments")

    args = parser.parse_args()

    check_prereqs(args.llama_path)

    commands = []
    if not args.only_llama:
        commands += make_opt_commands(args.models_root, fast=args.fast)
    if not args.only_opt:
        if args.llama_path is None:
            print("Skipping LLaMA runs because LLAMA_PATH not provided (use --llama-path or set LLAMA_PATH env var)")
        else:
            commands += make_llama_commands(args.llama_path, fast=args.fast)

    if not commands:
        print("No commands to run. Exiting.")
        return

    print(f"Built {len(commands)} commands. Logs will be written to {LOG_DIR}")
    results = build_and_run(commands, dry_run=args.dry_run)

    # Summary
    print('\n=== SUMMARY ===')
    for cmd, rc, log in results:
        status = 'OK' if rc == 0 else f'FAIL (rc={rc})'
        print(f"{status}: {' '.join(cmd)} -> {log}")

    print('\nFinished. Review logs in', LOG_DIR)


if __name__ == '__main__':
    main()
