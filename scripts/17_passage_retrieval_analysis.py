#!/usr/bin/env python3
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

DATA_PATH = Path('/home/jovyan/kvc-test-291/KVCache-Factory/data/LongBench/passage_retrieval_en.jsonl')
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
TEMPLATE = (
    'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph '
    'the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\n'
    'Please enter the number of the paragraph that the abstract is from. The answer format must be '
    'like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
)
BUDGETS = [3150, 6300, 15750]
SINK = 32


def tok_len(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)['input_ids'])


def char_to_tok(tokenizer, text, char_idx):
    return tok_len(tokenizer, text[:char_idx])


def find_target_span(prompt, answer_number):
    label = f'Paragraph {answer_number}:'
    start = prompt.find(label)
    if start == -1:
        return None, None, label
    next_match = re.search(r'\n\nParagraph \d+:', prompt[start + len(label):])
    if next_match:
        end = start + len(label) + next_match.start()
    else:
        marker = '\n\nThe following is an abstract.'
        end = prompt.find(marker, start)
        if end == -1:
            end = len(prompt)
    return start, end, label


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    rows = []

    with DATA_PATH.open() as f:
        for line in f:
            ex = json.loads(line)
            prompt = TEMPLATE.format(**ex)
            total_len = tok_len(tokenizer, prompt)
            match = re.search(r'Paragraph\s+(\d+)', ex['answers'][0])
            if not match:
                continue
            ans_num = match.group(1)
            start_char, end_char, label = find_target_span(prompt, ans_num)
            if start_char is None:
                continue
            start_tok = char_to_tok(tokenizer, prompt, start_char)
            end_tok = char_to_tok(tokenizer, prompt, end_char)
            rows.append({
                'total_len': total_len,
                'target_start': start_tok,
                'target_end': end_tok,
                'target_mid': (start_tok + end_tok) / 2.0,
                'target_label': label,
            })

    print('Passage retrieval prompt-position diagnostic\n')
    print(f'Total examples analyzed: {len(rows)}')
    print(f'Mean target paragraph start token: {sum(r["target_start"] for r in rows)/len(rows):.1f}')
    print(f'Mean target paragraph midpoint token: {sum(r["target_mid"] for r in rows)/len(rows):.1f}')
    print(f'Mean prompt length tokenized with Mistral: {sum(r["total_len"] for r in rows)/len(rows):.1f}')
    print()

    for budget in BUDGETS:
        prefix_end = budget - SINK
        sliding_tail_start = None
        sliding_evicted = 0
        fixed_prefix = 0
        sliding_tail = 0
        strict_last32 = 0
        for r in rows:
            total_len = r['total_len']
            target_start = r['target_start']
            target_mid = r['target_mid']
            tail_start = max(SINK, total_len - (budget - SINK))
            evict_start = SINK
            evict_end = tail_start

            if evict_start <= target_mid < evict_end:
                sliding_evicted += 1
            if 0 <= target_start < prefix_end:
                fixed_prefix += 1
            if tail_start <= target_mid < total_len:
                sliding_tail += 1
            if max(0, total_len - 32) <= target_mid < total_len:
                strict_last32 += 1

        n = len(rows)
        print(f'[budget={budget}]')
        print(f'  Fraction with target in sliding evicted zone [{SINK}:{"total_len-(budget-32)"}]: {sliding_evicted/n:.3f}')
        print(f'  Fraction with target retained by fixed-prefix [0:{prefix_end}]: {fixed_prefix/n:.3f}')
        print(f'  Fraction with target in actual sliding retained tail [total_len-{budget-SINK}:total_len]: {sliding_tail/n:.3f}')
        print(f'  Fraction with target in strict last-32 tail [total_len-32:total_len]: {strict_last32/n:.3f}')
        print()

    print('Interpretation:')
    print('  The target paragraph is distributed across the 30-paragraph context rather than concentrated at the front.')
    print('  Fixed-prefix and sliding-window retain different structural regions, but neither guarantees the target paragraph.')
    print('  That is why passage_retrieval_en can end up similarly hard for both methods, even though they fail for different structural reasons.')

if __name__ == '__main__':
    main()
