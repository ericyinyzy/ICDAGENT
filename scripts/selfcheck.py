"""Verify the install before spending GPU hours on a full run.

    python scripts/selfcheck.py --benchmark mimic4-icd9

Checks the pinned dependencies, the visible GPUs, the downloaded weights, the
bundled data files, and the metric implementation. Needs no MIMIC data.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from icdagent.config import BENCHMARKS
from icdagent.metrics import all_metrics, build_matrices
from icdagent.utils import load_json

EXPECTED = {'vllm': '0.8.3', 'transformers': '4.51.3', 'cachetools': '5.5.2'}

results = []


def check(label, ok, detail=''):
    results.append(ok)
    print('[%s] %-34s %s' % ('ok' if ok else 'FAIL', label, detail))


def check_versions():
    import importlib
    for pkg, want in EXPECTED.items():
        try:
            got = importlib.import_module(pkg).__version__
        except Exception as exc:
            check(pkg, False, 'not importable: %s' % exc)
            continue
        check(pkg, got == want, 'found %s, expected %s' % (got, want))
    try:
        import torch
        n = torch.cuda.device_count()
        check('CUDA devices', n >= 1, '%d visible (2x A100-80GB used in the paper)' % n)
    except Exception as exc:
        check('CUDA devices', False, str(exc))


def check_weights(cfg):
    for label, path, must_have in [
            ('R_code base', cfg.coding_base, 'config.json'),
            ('R_code LoRA', cfg.coding_lora, 'adapter_config.json'),
            ('R_crit', cfg.critic_model, 'config.json')]:
        present = os.path.isfile(os.path.join(path, must_have))
        check(label, present, path if present else '%s missing (run download_weights.py)' % path)

    cfg_path = os.path.join(cfg.coding_lora, 'adapter_config.json')
    if os.path.isfile(cfg_path):
        adapter = load_json(cfg_path)
        check('LoRA rank', adapter.get('r') == 32 and adapter.get('lora_alpha') == 64,
              'r=%s alpha=%s' % (adapter.get('r'), adapter.get('lora_alpha')))


def check_data(cfg):
    for label, path, expected in [
            ('label space', cfg.label_space, None),
            ('code descriptions', cfg.code_descriptions, None)]:
        try:
            n = len(load_json(path))
            check(label, n > 0, '%d entries' % n)
        except Exception as exc:
            check(label, False, '%s: %s' % (path, exc))
    ids = 'data/test_ids/%s_test_1000.json' % cfg.label_space.split('/')[-1][:-5]
    try:
        n = len(load_json(ids))
        check('test ids', n == 1000, '%d admissions' % n)
    except Exception as exc:
        check('test ids', False, '%s: %s' % (ids, exc))


def check_metrics():
    """One admission, one hit, one miss, one false positive."""
    space = ['|A|', '|B|', '|C|']
    records = [{'extracted_codes': ['|A|', '|C|'], 'label': ['|A|', '|B|']}]
    yhat, y = build_matrices(records, space)
    m = all_metrics(yhat, y)
    ok = (abs(m['prec_micro'] - 0.5) < 1e-9 and abs(m['rec_micro'] - 0.5) < 1e-9
          and abs(m['f1_micro'] - 0.5) < 1e-9)
    check('metrics', ok, 'micro P/R/F1 = %.2f/%.2f/%.2f, expected 0.50 each'
          % (m['prec_micro'], m['rec_micro'], m['f1_micro']))

    # a code predicted outside the label space must still count against precision
    oov = [{'extracted_codes': ['|A|', '|ZZZ|'], 'label': ['|A|']}]
    m2 = all_metrics(*build_matrices(oov, space))
    check('out-of-space predictions', abs(m2['prec_micro'] - 0.5) < 1e-9,
          'micro precision %.2f, expected 0.50' % m2['prec_micro'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benchmark', default='mimic4-icd9', choices=sorted(BENCHMARKS))
    args = ap.parse_args()
    cfg = BENCHMARKS[args.benchmark]

    print('ICDAGENT selfcheck -- %s\n' % args.benchmark)
    check_versions()
    check_weights(cfg)
    check_data(cfg)
    check_metrics()

    failed = results.count(False)
    print('\n%d checks, %d failed' % (len(results), failed))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
