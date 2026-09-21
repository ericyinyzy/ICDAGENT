"""Per-benchmark configuration for the released ICDAGENT checkpoints.

Defaults match the layout `scripts/download_weights.py` creates. R_code is one
shared HuatuoGPT-o1-8B base with a per-benchmark LoRA adapter (r=32, alpha=64)
attached at run time; R_crit is a fully fine-tuned Qwen3-4B loaded directly.
Override any of them from the command line if your weights live elsewhere.
"""
import os
from dataclasses import dataclass, field

WEIGHTS_ROOT = os.environ.get('ICDAGENT_WEIGHTS', 'weights')


def _w(*parts):
    return os.path.join(WEIGHTS_ROOT, *parts)


@dataclass
class Config:
    version: str                # '9' or '10'
    coding_base: str
    coding_lora: str
    critic_model: str
    code_descriptions: str
    label_space: str
    strip_dot_for_eval: bool    # ICD-10 label space is dot-free, predictions are not
    coding_sampling: dict = field(default_factory=lambda: dict(
        temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=8192))
    critic_sampling: dict = field(default_factory=lambda: dict(
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048))


ICD9 = Config(
    version='9',
    coding_base=_w('base', 'huatuogpt-o1-8b'),
    coding_lora=_w('rcode-icd9-lora'),
    critic_model=_w('rcrit-icd9'),
    code_descriptions='data/code_descriptions_icd9.json',
    label_space='data/label_space/mimic4_icd9.json',
    strip_dot_for_eval=False,
)

ICD10 = Config(
    version='10',
    coding_base=_w('base', 'huatuogpt-o1-8b'),
    coding_lora=_w('rcode-icd10-lora'),
    critic_model=_w('rcrit-icd10'),
    code_descriptions='data/code_descriptions_icd10.json',
    label_space='data/label_space/mimic4_icd10.json',
    strip_dot_for_eval=True,
)

# MIMIC-III is evaluated zero-shot with the ICD-9 checkpoints; only the label
# space changes.
MIMIC3_ICD9 = Config(
    **{**ICD9.__dict__, 'label_space': 'data/label_space/mimic3_icd9.json'})

BENCHMARKS = {
    'mimic3-icd9': MIMIC3_ICD9,
    'mimic4-icd9': ICD9,
    'mimic4-icd10': ICD10,
}
