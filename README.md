# ICDAGENT

Inference and evaluation code for **ICDAGENT: Empowering Agentic Large Language
Models for Explainable Medical Coding** (ACL 2026).

ICDAGENT assigns ICD codes to a clinical note and explains every assignment. A
coding agent `R_code` extracts codes together with preliminary rationales, and a
critical agent `R_crit` audits each rationale with chain-of-thought reasoning,
keeping the codes it can ground in the note and rejecting the rest. The codes
rejected for an admission are then handed back to `R_code` as a candidate set to
reconsider, so a valid code lost to a weak rationale can still be recovered. The
reported results use `K=2` rounds.

## What is released

| Item | Released |
| --- | --- |
| `R_code` LoRA adapters (ICD-9, ICD-10) | yes |
| `R_crit` checkpoints (ICD-9, ICD-10) | yes |
| Inference pipeline and evaluation code | yes |
| Test-set identifiers and label spaces | yes |
| Official ICD code descriptions | yes |
| Clinical notes, gold labels, training data, training code | no |

MIMIC-III and MIMIC-IV are governed by the PhysioNet data use agreement, so no
note text or label file is redistributed here. Everything needed to rebuild the
exact evaluation inputs from your own credentialed copy is included.

## Models

All weights live in one Hub repo, [`ericyinyzy/ICDAGENT`](https://huggingface.co/ericyinyzy/ICDAGENT),
under one subdirectory each.

| Subdirectory | Agent | Size | What it is |
| --- | --- | --- | --- |
| `base/huatuogpt-o1-8b` | `R_code` | 16 GB | shared base, mirror of [HuatuoGPT-o1-8B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-8B) (Apache-2.0) |
| `rcode-icd9-lora` | `R_code` | 0.9 GB | LoRA adapter, ICD-9 |
| `rcode-icd10-lora` | `R_code` | 1.2 GB | LoRA adapter, ICD-10 |
| `rcrit-icd9` | `R_crit` | 8.3 GB | fully fine-tuned Qwen3-4B, ICD-9 |
| `rcrit-icd10` | `R_crit` | 8.3 GB | fully fine-tuned Qwen3-4B, ICD-10 |

Both `R_code` adapters (r=32, alpha=64) share the same base and are attached at
run time rather than merged, which is how the reported results were produced.
`R_crit` is fully fine-tuned, so it is loaded directly with no adapter.

MIMIC-III-ICD-9 is evaluated zero-shot with the ICD-9 pair; only the label space
changes.

## Setup

The pinned versions are the ones the reported results were produced with:

```bash
conda create -y -n icdagent python=3.10
conda activate icdagent
pip install -r requirements.txt
```

`cachetools` must stay at 5.5.2. vLLM 0.8.x reaches into internals that
cachetools 6 and later removed, and with a newer one LoRA loading fails with
`'LoRALRUCache' object has no attribute '_LRUCache__update'`.

Inference needs two A100-80GB GPUs, matching the setup reported in the paper;
`tensor_parallel_size` follows the visible device count. Evaluation alone needs
only NumPy.

## Download the weights

```bash
python scripts/download_weights.py                          # all five, ~35 GB
python scripts/download_weights.py --benchmark mimic4-icd9  # just what ICD-9 needs
```

They land in `weights/` in the layout `icdagent/config.py` expects. Set
`ICDAGENT_WEIGHTS` or pass `--coding-base` / `--coding-lora` / `--critic-model`
to any stage script to use a different location.

## Prepare the evaluation input

`data/test_ids/` holds the 1,000 sampled admissions per benchmark, in the exact
order they were evaluated. Join them against your own credentialed MIMIC tables
to produce a JSON list of records:

```json
[{"subject_id": "13394703", "hadm_id": "23079758",
  "TEXT": "<discharge summary text>", "LABELS": "250.00;401.9;V58.66"}]
```

`LABELS` is the semicolon-separated gold code list. The splits follow
MIMIC-IV-ICD (Nguyen et al., 2023) and the MIMIC-III split of Mullenbach et al.
(2018).

## Run

One command for the full `K=2` pipeline and scoring:

```bash
bash scripts/run_all.sh mimic4_icd9_test.json mimic4-icd9 runs/icd9
```

It chains the stages below. They are separate processes on purpose: each agent
claims 90% of GPU memory, so the two cannot be co-resident.

```bash
# 1. R_code: notes -> candidate codes with rationales
python scripts/stage1_coding.py --notes mimic4_icd9_test.json \
    --benchmark mimic4-icd9 --output runs/icd9/coding_round1.json

# 2. R_crit: judge every (code, rationale) pair
python scripts/stage2_critic.py --coding runs/icd9/coding_round1.json \
    --benchmark mimic4-icd9 --output runs/icd9/critic_round1.json \
    --rejected runs/icd9/rejected_round1.json

# 3. R_code: reconsider the rejected codes as a candidate set
python scripts/stage3_reselect.py --rejected runs/icd9/rejected_round1.json \
    --benchmark mimic4-icd9 --output runs/icd9/coding_round2.json

# 4. R_crit again on what came back
python scripts/stage2_critic.py --coding runs/icd9/coding_round2.json \
    --benchmark mimic4-icd9 --output runs/icd9/critic_round2.json \
    --rejected runs/icd9/rejected_round2.json

# 5. accumulate the rounds, then score
python scripts/merge_rounds.py \
    --rounds runs/icd9/critic_round1.json runs/icd9/critic_round2.json \
    --output runs/icd9/predictions.json
python scripts/evaluate.py --predictions runs/icd9/predictions.json \
    --benchmark mimic4-icd9
```

Rounds accumulate: a code accepted in round 1 is kept, and round 2 can only add
to it. `predictions.json` carries the accepted codes per admission along with
the explanation `R_crit` settled on for each.

Swap `--benchmark` for `mimic4-icd10` or `mimic3-icd9` to run the other two; the
prompts, label space and code descriptions follow from it.

## Reported results

Table 1 of the paper, reproduced by `scripts/evaluate.py`:

| Benchmark | P-macro | P-micro | R-macro | R-micro | F1-macro | F1-micro |
| --- | --- | --- | --- | --- | --- | --- |
| MIMIC-III-ICD-9 | 29.62 | 59.35 | 25.58 | 46.87 | 27.45 | 52.37 |
| MIMIC-IV-ICD-9 | 31.92 | 60.19 | 28.90 | 49.15 | 30.34 | 54.11 |
| MIMIC-IV-ICD-10 | 26.03 | 54.73 | 22.73 | 40.77 | 24.27 | 46.73 |

Both agents decode with sampling rather than greedily (`R_code` at temperature
0.7 / top-p 0.8, `R_crit` at temperature 0.6 / top-p 0.95 / top-k 20), so a
rerun lands near these numbers rather than exactly on them. Re-running this code
end to end on a freshly built environment over a 20-admission subset of
MIMIC-IV-ICD-9 gave 45.95 / 70.56 / 45.62 / 49.51 / 45.78 / **58.19**, against
46.02 / 69.86 / 46.22 / 50.16 / 46.12 / **58.40** for the original run on the
same subset: every metric within 0.7 points, with per-admission code sets
agreeing at Jaccard 0.62.

Two evaluation details matter for matching the table. Macro metrics average only
over codes that appear at least once in the gold labels of the split. And
because the MIMIC-IV ICD-10 label space is dot-free while the model emits dotted
codes, ICD-10 predictions are stripped of dots before matching; `evaluate.py`
handles this from the benchmark config. Predicted codes that fall outside the
label space are counted as false positives rather than dropped.

## Layout

```
icdagent/
  config.py      per-benchmark model paths, sampling and label spaces
  prompts.py     the six prompts: coding, critic and reselection x ICD-9/10
  agents.py      CodingAgent and CriticalAgent, served with vLLM
  metrics.py     macro / micro precision, recall and F1
  utils.py       code extraction and rationale segmentation
  cli.py         shared argument handling for the stage scripts
scripts/
  download_weights.py  fetch the checkpoints from the Hub
  stage1_coding.py     R_code, initial extraction
  stage2_critic.py     R_crit, verification
  stage3_reselect.py   R_code, reconsider the rejected codes
  merge_rounds.py      accumulate rounds into final predictions
  evaluate.py          scoring
  run_all.sh           all of the above, end to end
data/
  test_ids/            the 1,000 evaluated admissions per benchmark
  label_space/         the full ICD code space of each benchmark
  code_descriptions_icd9.json, code_descriptions_icd10.json
```

## Citation

```bibtex
@inproceedings{yin-etal-2026-icdagent,
  title     = "{ICDAGENT}: Empowering Agentic Large Language Models for Explainable Medical Coding",
  author    = "Yin, Ziyi and Cao, Yuanpu and Wang, Ting and Chen, Jinghui and Ma, Fenglong",
  booktitle = "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year      = "2026"
}
```

## License

Code in this repository is released under the MIT License. The `R_code` base is
a mirror of HuatuoGPT-o1-8B (Apache-2.0); `R_crit` derives from Qwen3-4B
(Apache-2.0). Use of MIMIC-III and MIMIC-IV remains subject to the PhysioNet
data use agreement.
