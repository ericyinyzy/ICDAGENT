"""Stage 1 - the coding agent extracts codes and preliminary rationales.

    python scripts/stage1_coding.py \
        --notes notes.json --benchmark mimic4-icd9 --output coding_round1.json

`--notes` is a JSON list with `subject_id`, `hadm_id`, `TEXT` and `LABELS`
(semicolon-separated gold codes). Loads R_code only.
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
    ap.add_argument('--notes', required=True)
    ap.add_argument('--output', required=True)
    add_model_overrides(ap)
    args = ap.parse_args()

    cfg = resolve(args)
    notes = load_json(args.notes)
    results = CodingAgent(cfg).extract(notes)
    dump_json(results, args.output)
    codes = sum(len(set(r['extracted_labels'])) for r in results)
    print('stage1: %d notes -> %d candidate codes -> %s' % (len(results), codes, args.output))


if __name__ == '__main__':
    main()
