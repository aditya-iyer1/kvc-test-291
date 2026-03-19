#!/usr/bin/env python3
import json
from pathlib import Path

from transformers import AutoTokenizer

DATA_PATH = Path('/home/jovyan/kvc-test-291/KVCache-Factory/data/LongBench/triviaqa.jsonl')
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
TEMPLATE = "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}"
BUDGET = 3150
SINK = 32
RECENT = 3118
FIXED_PREFIX = 3118

prefix, rest = TEMPLATE.split('{context}')
mid, suffix = rest.split('{input}')


def tok_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])


def char_to_tok(tokenizer, text, char_idx):
    return tok_len(tokenizer, text[:char_idx])


def last_question_pos(prompt, tokenizer):
    idx = prompt.rfind('Question:')
    needle = 'Question:'
    if idx == -1:
        idx = prompt.rfind('Q:')
        needle = 'Q:'
    if idx == -1:
        return None, None
    return char_to_tok(tokenizer, prompt, idx), needle


def overlap_len(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    rows = []
    with DATA_PATH.open() as f:
        for line in f:
            ex = json.loads(line)
            prompt = TEMPLATE.format(**ex)
            total_len = tok_len(tokenizer, prompt)
            context_start = tok_len(tokenizer, prefix)
            input_start = tok_len(tokenizer, prefix + ex['context'] + mid)
            exemplar_start = context_start
            exemplar_end = input_start
            exemplar_len = exemplar_end - exemplar_start
            q_pos, q_marker = last_question_pos(prompt, tokenizer)
            rows.append({
                'length_field': ex['length'],
                'total_len': total_len,
                'context_start': context_start,
                'input_start': input_start,
                'exemplar_start': exemplar_start,
                'exemplar_end': exemplar_end,
                'exemplar_len': exemplar_len,
                'exemplar_mid': (exemplar_start + exemplar_end) / 2.0,
                'question_pos': q_pos,
                'question_marker': q_marker,
            })

    long_rows = [r for r in rows if r['length_field'] > BUDGET]

    evict_stats = []
    fully_evicted = 0
    almost_fully_evicted = 0
    fixed_prefix_any_overlap = 0
    fixed_prefix_retained_fracs = []
    for r in long_rows:
        total_len = r['total_len']
        evict_start = SINK
        evict_end = max(SINK, total_len - RECENT)
        evicted = overlap_len(r['exemplar_start'], r['exemplar_end'], evict_start, evict_end)
        frac_evicted = evicted / r['exemplar_len'] if r['exemplar_len'] > 0 else 0.0
        evict_stats.append(frac_evicted)
        if evicted == r['exemplar_len']:
            fully_evicted += 1
        if frac_evicted >= 0.95:
            almost_fully_evicted += 1

        fixed_keep = overlap_len(r['exemplar_start'], r['exemplar_end'], 0, FIXED_PREFIX)
        fixed_prefix_retained_fracs.append(fixed_keep / r['exemplar_len'] if r['exemplar_len'] > 0 else 0.0)
        if fixed_keep > 0:
            fixed_prefix_any_overlap += 1

    q_positions = [r['question_pos'] for r in rows if r['question_pos'] is not None]
    exemplar_mids = [r['exemplar_mid'] for r in rows]
    exemplar_starts = [r['exemplar_start'] for r in rows]
    exemplar_ends = [r['exemplar_end'] for r in rows]

    print('TriviaQA prompt structure diagnostic\n')
    print(f'Total examples: {len(rows)}')
    print(f'Examples with length > {BUDGET}: {len(long_rows)}')
    print()
    print('Prompt token positions (means over all examples):')
    print(f'  Mean few-shot exemplar start: {sum(exemplar_starts)/len(exemplar_starts):.1f}')
    print(f'  Mean few-shot exemplar end / actual input start: {sum(exemplar_ends)/len(exemplar_ends):.1f}')
    print(f'  Mean few-shot exemplar midpoint: {sum(exemplar_mids)/len(exemplar_mids):.1f}')
    print(f'  Mean question position: {sum(q_positions)/len(q_positions):.1f}')
    print()
    print(f'10% sliding-window analysis (retain first {SINK} + last {RECENT} tokens):')
    print(f'  Mean fraction of exemplar tokens evicted on long examples: {sum(evict_stats)/len(evict_stats):.3f}')
    print(f'  Fraction of long examples where exemplars are fully evicted: {fully_evicted/len(long_rows):.3f}')
    print(f'  Fraction of long examples where >=95% of exemplar tokens are evicted: {almost_fully_evicted/len(long_rows):.3f}')
    print(f'  Mean fraction of exemplar tokens retained by fixed-prefix: {sum(fixed_prefix_retained_fracs)/len(fixed_prefix_retained_fracs):.3f}')
    print(f'  Fraction of long examples with any fixed-prefix exemplar retention: {fixed_prefix_any_overlap/len(long_rows):.3f}')
    print()
    print('Interpretation:')
    print('  StreamingLLMSliding keeps only the first 32 sink tokens from the front of the prompt.')
    print('  TriviaQA few-shot exemplars live near the front, but extend far beyond token 32.')
    print('  The original fixed-prefix StreamingLLM retains that front-loaded exemplar block; the sliding variant largely evicts it.')

if __name__ == '__main__':
    main()
