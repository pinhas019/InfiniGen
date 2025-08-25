"""Run a single model/dataset with mlp eviction policy and capture logs.

Usage examples (PowerShell):
  # fast smoke run for OPT 6.7b on wikitext2
  python .\accuracy\perplexity\run_single_mlp.py --model-type opt --size 6.7b --dataset wikitext2 --fast

  # if running LLaMA specify --llama-path or set LLAMA_PATH env var
  python .\accuracy\perplexity\run_single_mlp.py --model-type llama --size 7b --dataset ptb --llama-path C:\path\to\llama

This script writes logs to `accuracy/perplexity/mlp_single_logs/`.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import datetime

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LOG_DIR = HERE / "mlp_single_logs"
LOG_DIR.mkdir(exist_ok=True)

PARTIAL = 0.2

OPT_ALPHA = 4
OPT_BUDGET = 0.2
LLAMA_ALPHA = 5
LLAMA_BUDGET = 0.2


def run_cmd(cmd, log_path, dry_run=False):
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] CMD: {' '.join(cmd)}")
    if dry_run:
        print(f"Dry-run: would write logs to {log_path}")
        return 0

    with open(log_path, 'wb') as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            try:
                sys.stdout.buffer.write(line)
            except Exception:
                try:
                    sys.stdout.write(line.decode(errors='ignore'))
                except Exception:
                    pass
        proc.wait()
        return proc.returncode


def build_command(args):
    if args.model_type == 'opt':
        model_arg = str(Path(args.models_root) / f"opt-{args.size}")
        seqlen = 128 if args.fast else 2048
        eval_samples = 1 if args.fast else 0
        cmd = [sys.executable, str(HERE / 'opt.py'),
               '--model', model_arg,
               '--eval_dataset', args.dataset,
               '--seq_len', str(seqlen),
               '--eval_samples', str(eval_samples),
               '--model_name', f'opt-{args.size}',
               '--infinigen',
               '--partial_weight_ratio', str(PARTIAL),
               '--partial_weight_path', str(ROOT / 'setup' / 'weights' / f"opt-{args.size}_{PARTIAL}"),
               '--alpha', str(OPT_ALPHA),
               '--budget', str(OPT_BUDGET),
               '--capacity', '0.8',
               '--eviction_policy', 'mlp']
        return cmd, f"opt-{args.size}_{args.dataset}_mlp.log"

    elif args.model_type == 'llama':
        llama_path = args.llama_path or os.environ.get('LLAMA_PATH')
        if llama_path is None:
            raise SystemExit('LLAMA_PATH not provided (use --llama-path or set LLAMA_PATH env var)')
        model_arg = str(Path(llama_path) / f"llama-2-{args.size}")
        seqlen = 128 if args.fast else 2048
        eval_samples = 1 if args.fast else 0
        cmd = [sys.executable, str(HERE / 'llama.py'),
               '--model', model_arg,
               '--eval_dataset', args.dataset,
               '--seq_len', str(seqlen),
               '--eval_samples', str(eval_samples),
               '--model_name', f'llama-{args.size}',
               '--infinigen',
               '--partial_weight_ratio', str(PARTIAL),
               '--partial_weight_path', str(ROOT / 'setup' / 'weights' / f"llama-2-{args.size}_{PARTIAL}"),
               '--skewing_matrix_path', str(ROOT / 'setup' / 'skewing_matrix' / f"llama-2-{args.size}.pt"),
               '--alpha', str(LLAMA_ALPHA),
               '--budget', str(LLAMA_BUDGET),
               '--capacity', '0.8',
               '--eviction_policy', 'mlp']
        return cmd, f"llama-{args.size}_{args.dataset}_mlp.log"

    else:
        raise SystemExit('Unknown model type; choose "opt" or "llama"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['opt', 'llama'], required=True)
    parser.add_argument('--size', required=True, help='model size string, e.g. 6.7b or 7b')
    parser.add_argument('--dataset', required=True, help='wikitext2 or ptb')
    parser.add_argument('--models-root', default=str(ROOT / 'setup' / 'opt-model'))
    parser.add_argument('--llama-path', default=os.environ.get('LLAMA_PATH'))
    parser.add_argument('--fast', action='store_true', help='fast smoke-run')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    try:
        cmd, logname = build_command(args)
    except SystemExit as e:
        print(e)
        return

    logpath = LOG_DIR / logname
    rc = run_cmd(cmd, str(logpath), dry_run=args.dry_run)
    print('\nDone. Log:', logpath)
    if rc != 0:
        print('Run exited with non-zero return code:', rc)

if __name__ == '__main__':
    main()
