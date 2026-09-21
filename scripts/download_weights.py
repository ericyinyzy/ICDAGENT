"""Fetch the ICDAGENT checkpoints from the Hugging Face Hub.

    python scripts/download_weights.py                  # everything (~35 GB)
    python scripts/download_weights.py --benchmark mimic4-icd9   # ~25 GB
    python scripts/download_weights.py --list

Everything lands under `weights/` in the layout `icdagent/config.py` expects, so
the stage scripts work with no further configuration. Downloads resume, so
rerunning after an interruption is cheap.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icdagent.config import WEIGHTS_ROOT

REPO = 'ericyinyzy/ICDAGENT'

# subdirectory in the Hub repo -> (local name, approximate size, what it is)
COMPONENTS = {
    'base/huatuogpt-o1-8b': ('base/huatuogpt-o1-8b', '16 GB',
                             'shared R_code base (mirror of FreedomIntelligence/HuatuoGPT-o1-8B)'),
    'rcode-icd9-lora': ('rcode-icd9-lora', '0.9 GB', 'R_code LoRA, ICD-9'),
    'rcode-icd10-lora': ('rcode-icd10-lora', '1.2 GB', 'R_code LoRA, ICD-10'),
    'rcrit-icd9': ('rcrit-icd9', '8.3 GB', 'R_crit, ICD-9 (full Qwen3-4B)'),
    'rcrit-icd10': ('rcrit-icd10', '8.3 GB', 'R_crit, ICD-10 (full Qwen3-4B)'),
}

NEEDED = {
    'mimic4-icd9': ['base/huatuogpt-o1-8b', 'rcode-icd9-lora', 'rcrit-icd9'],
    'mimic3-icd9': ['base/huatuogpt-o1-8b', 'rcode-icd9-lora', 'rcrit-icd9'],
    'mimic4-icd10': ['base/huatuogpt-o1-8b', 'rcode-icd10-lora', 'rcrit-icd10'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', choices=sorted(NEEDED),
                    help='only fetch what this benchmark needs (default: everything)')
    ap.add_argument('--output', default=WEIGHTS_ROOT)
    ap.add_argument('--list', action='store_true', help='show the components and exit')
    args = ap.parse_args()

    if args.list:
        for remote, (_, size, what) in COMPONENTS.items():
            print('%-24s %8s  %s' % (remote, size, what))
        return

    from huggingface_hub import snapshot_download

    wanted = NEEDED[args.benchmark] if args.benchmark else list(COMPONENTS)
    for remote in wanted:
        local, size, what = COMPONENTS[remote]
        target = os.path.join(args.output, local)
        print('\n==> %s (%s) -> %s' % (what, size, target))
        snapshot_download(repo_id=REPO, allow_patterns=remote + '/*',
                          local_dir=args.output, resume_download=True)
    print('\nWeights are in %s/ -- the stage scripts will find them automatically.'
          % args.output)


if __name__ == '__main__':
    main()
