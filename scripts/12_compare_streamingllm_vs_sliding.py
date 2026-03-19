#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

SCRIPT_DIR = Path('/home/jovyan/kvc-test-291/scripts')
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / 'results'
config_path = SCRIPT_DIR / 'config.env'
if config_path.exists():
    for line in config_path.read_text().splitlines():
        if line.startswith('RESULTS_DIR='):
            RESULTS_DIR = Path(line.split('=', 1)[1].strip().strip('"\''))
            break

SUMMARY_PATH = RESULTS_DIR / 'scores' / 'summary_run8.json'
KVC_DIR = PROJECT_DIR / 'KVCache-Factory'
METRICS_PATH = KVC_DIR / 'metrics.py'

spec = importlib.util.spec_from_file_location('metrics_mod', METRICS_PATH)
metrics_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_mod)

DATASETS = [
    'narrativeqa', 'hotpotqa', '2wikimqa', 'gov_report', 'qmsum',
    'multi_news', 'triviaqa', 'passage_retrieval_en', 'lcc', 'repobench-p'
]
BUDGETS = {
    '10pct': ('budget_10pct', 3150),
    '20pct': ('budget_20pct', 6300),
    '50pct': ('budget_50pct', 15750),
}
DATASET2METRIC = {
    'narrativeqa': metrics_mod.qa_f1_score,
    'hotpotqa': metrics_mod.qa_f1_score,
    '2wikimqa': metrics_mod.qa_f1_score,
    'triviaqa': metrics_mod.qa_f1_score,
    'gov_report': metrics_mod.rouge_score,
    'qmsum': metrics_mod.rouge_score,
    'multi_news': metrics_mod.rouge_score,
    'passage_retrieval_en': metrics_mod.retrieval_score,
    'lcc': metrics_mod.code_sim_score,
    'repobench-p': metrics_mod.code_sim_score,
}

summary = json.loads(SUMMARY_PATH.read_text())


def score_file(path: Path, dataset: str):
    if not path.exists():
        return None
    metric = DATASET2METRIC[dataset]
    preds = 0
    total = 0.0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            pred = data['pred']
            if dataset in ['trec', 'triviaqa', 'samsum', 'lsht']:
                pred = pred.lstrip('\n').split('\n')[0]
            score = 0.0
            for gt in data['answers']:
                score = max(score, metric(pred, gt, all_classes=data.get('all_classes')))
            total += score
            preds += 1
    return round(100.0 * total / preds, 2) if preds else None


def fmt(v):
    return 'MISSING' if v is None else f'{v:.2f}'


def fmt_delta(orig, new):
    if orig is None or new is None:
        return 'MISSING'
    delta = round(new - orig, 2)
    return f'{delta:+.2f}'


print('StreamingLLM vs StreamingLLMSliding\n')
sliding_overalls = {}
for budget, (budget_dir, cap) in BUDGETS.items():
    print(f'[{budget}]')
    print(f"{'Dataset':24s} {'StreamingLLM':>14s} {'Sliding':>14s} {'Delta':>10s}")
    print('-' * 66)
    sliding_scores = []
    orig_scores = []
    passage_verdict = None
    for dataset in DATASETS:
        orig = summary.get('StreamingLLM', {}).get(budget, {}).get('datasets', {}).get(dataset, {}).get('score')
        raw_path = RESULTS_DIR / budget_dir / f'mistral-7b-instruct-v0.2_{cap}' / dataset / 'StreamingLLMSliding.json'
        sliding = score_file(raw_path, dataset)
        label = dataset + ('  <==' if dataset == 'passage_retrieval_en' else '')
        print(f'{label:24s} {fmt(orig):>14s} {fmt(sliding):>14s} {fmt_delta(orig, sliding):>10s}')
        if orig is not None:
            orig_scores.append(orig)
        if sliding is not None:
            sliding_scores.append(sliding)
        if dataset == 'passage_retrieval_en':
            if orig is None or sliding is None:
                passage_verdict = 'not measurable (missing raw Sliding results)'
            elif sliding > orig:
                passage_verdict = 'improved'
            elif sliding < orig:
                passage_verdict = 'worsened'
            else:
                passage_verdict = 'no effect'
    orig_overall = round(sum(orig_scores) / len(orig_scores), 2) if orig_scores else None
    sliding_overall = round(sum(sliding_scores) / len(sliding_scores), 2) if sliding_scores else None
    sliding_overalls[budget] = (orig_overall, sliding_overall, passage_verdict)
    print('-' * 66)
    print(f"{'Overall mean':24s} {fmt(orig_overall):>14s} {fmt(sliding_overall):>14s} {fmt_delta(orig_overall, sliding_overall):>10s}")
    print(f'Verdict passage_retrieval_en: {passage_verdict}')
    print()

print('Summary verdicts:')
for budget in ['10pct', '20pct', '50pct']:
    orig, sliding, verdict = sliding_overalls[budget]
    print(f'  {budget}: passage_retrieval_en {verdict}; overall {fmt(orig)} -> {fmt(sliding)}')
