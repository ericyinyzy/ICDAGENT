"""Combine the per-round critic outputs into the final prediction file.

    python scripts/merge_rounds.py \
        --rounds critic_round1.json critic_round2.json --output predictions.json

Rounds are accumulated: a code accepted in any round is kept, so later rounds can
only add codes. Round 2 holds one record per regenerated code, so records are
grouped by admission first.
"""
import argparse
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icdagent.utils import dump_json, load_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', nargs='+', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    merged = OrderedDict()
    for round_idx, path in enumerate(args.rounds):
        for rec in load_json(path):
            key = (str(rec['subject_id']), str(rec['hadm_id']))
            if key not in merged:
                merged[key] = {
                    'subject_id': key[0],
                    'hadm_id': key[1],
                    'label': rec.get('label', []),
                    'extracted_codes': [],
                    'ans': {},
                    'ori_code': rec.get('ori_code', []),
                }
            entry = merged[key]
            for code in rec['extracted_codes']:
                if code not in entry['extracted_codes']:
                    entry['extracted_codes'].append(code)
            if isinstance(rec.get('ans'), dict):
                entry['ans'].update(rec['ans'])
            if round_idx == 0:
                entry['ori_ans'] = rec.get('ori_ans', '')

    records = list(merged.values())
    dump_json(records, args.output)
    total = sum(len(r['extracted_codes']) for r in records)
    print('merged %d rounds -> %d admissions, %d codes -> %s'
          % (len(args.rounds), len(records), total, args.output))


if __name__ == '__main__':
    main()
