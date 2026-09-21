"""The two ICDAGENT agents, each served through vLLM.

They are deliberately kept in separate processes: both claim 90% of GPU memory,
so a single process cannot hold them at once. `scripts/stage*.py` load one agent
each and exit, which is also how the original experiments were run.

R_code is HuatuoGPT-o1-8B plus a LoRA adapter and does two jobs: the initial
extraction pass, and rationale regeneration for codes R_crit rejected.
R_crit is a fully fine-tuned Qwen3-4B judging one (code, rationale) pair per
prompt.
"""
from collections import OrderedDict

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from . import prompts
from .utils import parse_generation, split_rationales

BATCH_SIZE = 12


def _build_llm(model, max_lora_rank=None):
    kwargs = dict(
        model=model,
        dtype='bfloat16',
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=32768,
        enforce_eager=False,
    )
    if max_lora_rank is not None:
        kwargs.update(max_lora_rank=max_lora_rank, enable_lora=True)
    return LLM(**kwargs)


def _chat(tokenizer, system, user):
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        tokenize=False, add_generation_prompt=True)


class CodingAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.coding_base)
        self.llm = _build_llm(cfg.coding_base, max_lora_rank=32)
        self.lora = LoRARequest('lora_adapter', 1, cfg.coding_lora)
        self.params = SamplingParams(**cfg.coding_sampling)

    def _generate(self, prompt_batch):
        outs = self.llm.generate(prompt_batch, self.params, lora_request=self.lora)
        return [o.outputs[0].text for o in outs]

    def extract(self, notes):
        """Round 1: clinical note -> codes with preliminary rationales."""
        v = self.cfg.version
        results = []
        for i in range(0, len(notes), BATCH_SIZE):
            batch = notes[i:i + BATCH_SIZE]
            prompt_batch = [
                _chat(self.tokenizer, prompts.coding_system(v),
                      prompts.coding_user(v, e['TEXT'])) for e in batch]
            for entry, text in zip(batch, self._generate(prompt_batch)):
                codes = parse_generation(text, v)
                results.append({
                    'subject_id': str(entry['subject_id']),
                    'hadm_id': str(entry['hadm_id']),
                    'TEXT': entry['TEXT'],
                    'ans': text,
                    'label': ['|' + c.strip('|') + '|' for c in entry['LABELS'].split(';')],
                    'extracted_labels': codes,
                    'rationales': split_rationales(text, v) if codes else {},
                })
        return results

    def reselect(self, rejected):
        """Later rounds: give the codes R_crit rejected a second chance.

        All codes rejected for one admission are pooled into a single candidate
        set and R_code re-picks from it, writing a fresh rationale for whatever
        it keeps. Dropping a code here is how the round discards it.
        """
        v = self.cfg.version
        grouped = OrderedDict()
        for entry in rejected:
            key = (str(entry['subject_id']), str(entry['hadm_id']))
            if key not in grouped:
                grouped[key] = {
                    'subject_id': key[0],
                    'hadm_id': key[1],
                    'TEXT': entry['clnote'],
                    'label': entry.get('label', []),
                    'candidates': [],
                }
            grouped[key]['candidates'].append(entry['code'])

        notes = list(grouped.values())
        results = []
        for i in range(0, len(notes), BATCH_SIZE):
            batch = notes[i:i + BATCH_SIZE]
            prompt_batch = [
                _chat(self.tokenizer, prompts.reselection_system(v),
                      prompts.reselection_user(v, e['TEXT'], '\n\n'.join(e['candidates'])))
                for e in batch]
            for entry, text in zip(batch, self._generate(prompt_batch)):
                codes = parse_generation(text, v)
                results.append({
                    'subject_id': entry['subject_id'],
                    'hadm_id': entry['hadm_id'],
                    'TEXT': entry['TEXT'],
                    'ans': text,
                    'label': entry['label'],
                    'extracted_labels': codes,
                    'rationales': split_rationales(text, v) if codes else {},
                })
        return results


class CriticalAgent:
    def __init__(self, cfg, code_descriptions):
        self.cfg = cfg
        self.desc = code_descriptions
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.critic_model)
        self.llm = _build_llm(cfg.critic_model)
        self.params = SamplingParams(**cfg.critic_sampling)

    def _describe(self, code, parenthesised=False):
        if code not in self.desc:
            return code
        return f'{code}({self.desc[code]})' if parenthesised else f'{code}: {self.desc[code]}'

    def verify(self, coded):
        """Judge every (code, rationale) pair R_code produced.

        Returns the per-note verdicts and a flat list of rejected codes, each
        with the feedback R_code needs to retry it in the next round. Every
        prompt is independent, so pairs are batched across notes -- in later
        rounds a note often carries just one code.
        """
        v = self.cfg.version
        pairs = [(entry, code, rationale)
                 for entry in coded
                 for code, rationale in self._rationales(entry).items()]

        accepted = {}
        explanations = {}
        rejected = []
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i:i + BATCH_SIZE]
            prompt_batch = [
                _chat(self.tokenizer, prompts.critic_system(v),
                      prompts.critic_user(v, entry['TEXT'], self._describe(code),
                                          rationale, self._describe(code, True)))
                for entry, code, rationale in batch]
            outs = self.llm.generate(prompt_batch, self.params)
            for (entry, code, rationale), out in zip(batch, outs):
                text = out.outputs[0].text
                verdict = text.split('## Final Response')[-1]
                key = id(entry)
                if 'is a correct' in text:
                    accepted.setdefault(key, []).append(code)
                    explanations.setdefault(key, {})[code] = verdict
                else:
                    rejected.append({
                        'subject_id': entry['subject_id'],
                        'hadm_id': entry['hadm_id'],
                        'clnote': entry['TEXT'],
                        'label': entry['label'],
                        'code': self._describe(code),
                        'feedback': verdict,
                        'ori_rational': rationale,
                    })

        verdicts = []
        for entry in coded:
            key = id(entry)
            judged = bool(self._rationales(entry))
            verdicts.append({
                'subject_id': entry['subject_id'],
                'hadm_id': entry['hadm_id'],
                'ans': explanations.get(key, {}) if judged
                       else 'No codes need to be judged! We skip this!',
                'label': entry['label'],
                'extracted_codes': accepted.get(key, []),
                'ori_ans': entry['ans'],
                'ori_code': ['|' + c.strip('|') + '|' for c in set(entry['extracted_labels'])],
            })
        return verdicts, rejected

    def _rationales(self, entry):
        """Code -> rationale for a record, whatever stage produced it."""
        if 'rationales' in entry:
            return entry['rationales']
        if not set(entry['extracted_labels']):
            return {}
        return split_rationales(entry['ans'], self.cfg.version)
