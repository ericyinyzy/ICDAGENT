"""Shared argument handling for the stage scripts."""
import copy

from .config import BENCHMARKS


def add_model_overrides(parser):
    parser.add_argument('--benchmark', required=True, choices=sorted(BENCHMARKS))
    parser.add_argument('--coding-base', default=None)
    parser.add_argument('--coding-lora', default=None)
    parser.add_argument('--critic-model', default=None)
    parser.add_argument('--code-descriptions', default=None)


def resolve(args):
    """Benchmark defaults with any command-line overrides applied."""
    cfg = copy.deepcopy(BENCHMARKS[args.benchmark])
    for field in ('coding_base', 'coding_lora', 'critic_model', 'code_descriptions'):
        value = getattr(args, field, None)
        if value:
            setattr(cfg, field, value)
    return cfg
