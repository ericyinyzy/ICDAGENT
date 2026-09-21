"""Code extraction and rationale segmentation shared by both agents."""
import json
import re

# ICD-9 codes carry a decimal point (250.00, V58.66, E878.1); ICD-10 codes in the
# MIMIC-IV label space are dot-free and start with a digit or a letter (E119, 0UT74ZZ).
CODE_PATTERNS = {
    '9': r'\|([VE]?\d{1,3}(?:\.\d{1,2})?)\|',
    '10': r'\|([0A-Z][^|]*)\|',
}
STAR_PATTERN = r'\*\*([VE]?\d{1,3}(?:\.\d{1,2})?)\*\*'


def extract_icd_codes(text, version):
    return re.findall(CODE_PATTERNS[version], text)


def extract_icd_codes_star(text):
    return re.findall(STAR_PATTERN, text)


def parse_generation(text, version):
    """Codes emitted by R_code, falling back to the bold-markdown form."""
    codes = extract_icd_codes(text, version)
    if not codes:
        codes = extract_icd_codes_star(text)
    return codes


def split_rationales(generation, version):
    """Map each code to the paragraph(s) of `generation` that justify it.

    R_code emits '## Thinking <paragraphs> ## Final Response <summary>'; only the
    thinking half holds the per-code rationales, so the summary is dropped.
    """
    thinking = generation.split('## Final Response')[0]
    by_code = {}
    for para in thinking.split('\n\n'):
        for code in set(extract_icd_codes(para, version)):
            by_code.setdefault('|' + code + '|', []).append(para)
    return {code: '\n\n'.join(paras) for code, paras in by_code.items()}


def normalize(code, strip_dot=False):
    code = str(code).strip('|')
    if strip_dot:
        code = code.replace('.', '')
    return '|' + code + '|'


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def dump_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)
