"""Score an ICDAGENT prediction file against the MIMIC gold labels.

    python scripts/evaluate.py --predictions out.json --benchmark mimic4-icd9

The prediction file is a list of records holding `subject_id`, `hadm_id`,
`extracted_codes` (the codes R_crit accepted) and `label` (the gold codes).
Predicted codes outside the benchmark label space are kept as false positives
rather than silently dropped.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icdagent.config import BENCHMARKS
from icdagent.metrics import all_metrics, build_matrices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--benchmark', required=True, choices=sorted(BENCHMARKS))
    ap.add_argument('--pred-key', default='extracted_codes')
    ap.add_argument('--gold-key', default='label')
    ap.add_argument('--label-space', default=None,
                    help='overrides the benchmark default')
    args = ap.parse_args()

    cfg = BENCHMARKS[args.benchmark]
    space_path = args.label_space or cfg.label_space
    with open(space_path) as f:
        label_space = json.load(f)
    with open(args.predictions) as f:
        records = json.load(f)

    yhat, y = build_matrices(records, label_space, args.pred_key, args.gold_key,
                             cfg.strip_dot_for_eval)
    m = all_metrics(yhat, y)
    print('%s  (n=%d, |L|=%d)' % (args.benchmark, len(records), len(label_space)))
    print('             Macro    Micro')
    print('Precision  %7.2f  %7.2f' % (m['prec_macro'] * 100, m['prec_micro'] * 100))
    print('Recall     %7.2f  %7.2f' % (m['rec_macro'] * 100, m['rec_micro'] * 100))
    print('F1         %7.2f  %7.2f' % (m['f1_macro'] * 100, m['f1_micro'] * 100))


if __name__ == '__main__':
    main()
