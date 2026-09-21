"""Stage 3 - the coding agent re-picks from the codes R_crit rejected.

    python scripts/stage3_reselect.py \
        --rejected rejected_round1.json --benchmark mimic4-icd9 \
        --output coding_round2.json

Everything R_crit rejected for one admission becomes a candidate set that
R_code chooses from again, writing fresh rationales for what it keeps. The
output has the same shape as stage 1, so feed it straight back into stage 2 for
the next verification round. Loads R_code only.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icdagent.agents import CodingAgent
from icdagent.cli import add_model_overrides, resolve
from icdagent.utils import dump_json, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rejected', required=True)
    ap.add_argument('--output', required=True)
    add_model_overrides(ap)
    args = ap.parse_args()

    cfg = resolve(args)
    rejected = load_json(args.rejected)
    if not rejected:
        dump_json([], args.output)
        print('stage3: nothing to reconsider')
        return
    results = CodingAgent(cfg).reselect(rejected)
    kept = sum(len(set(r['extracted_labels'])) for r in results)
    dump_json(results, args.output)
    print('stage3: %d rejected codes over %d admissions -> %d codes kept -> %s'
          % (len(rejected), len(results), kept, args.output))


if __name__ == '__main__':
    main()
