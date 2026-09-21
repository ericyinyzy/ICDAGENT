"""Stage 2 - the critical agent verifies each (code, rationale) pair.

    python scripts/stage2_critic.py \
        --coding coding_round1.json --benchmark mimic4-icd9 \
        --output critic_round1.json --rejected rejected_round1.json

Accepted codes and their refined explanations go to `--output`; every rejected
code is written to `--rejected` with the feedback stage 3 needs. Loads R_crit
only.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icdagent.agents import CriticalAgent
from icdagent.cli import add_model_overrides, resolve
from icdagent.utils import dump_json, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coding', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--rejected', required=True)
    add_model_overrides(ap)
    args = ap.parse_args()

    cfg = resolve(args)
    coded = load_json(args.coding)
    descriptions = load_json(cfg.code_descriptions)
    verdicts, rejected = CriticalAgent(cfg, descriptions).verify(coded)
    dump_json(verdicts, args.output)
    dump_json(rejected, args.rejected)
    accepted = sum(len(v['extracted_codes']) for v in verdicts)
    print('stage2: %d accepted -> %s, %d rejected -> %s'
          % (accepted, args.output, len(rejected), args.rejected))


if __name__ == '__main__':
    main()
