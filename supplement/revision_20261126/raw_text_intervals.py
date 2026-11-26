"""Paired intervals for the raw-text answering control (saved outputs only).

Raw-text search vs the same writer's Qwen-extracted graph and vs Qwen dense, Gemma 3 27B,
initial stage, k = 10. Protocol as in the paper: 100,000 whole-cluster bootstrap draws
(pointwise 95% percentile intervals), 200,000 cluster sign flips (two-sided, plus-one),
Holm across the four contrasts. Seed 20261126.
"""
import csv, json
from pathlib import Path
import numpy as np

SUPP = Path(__file__).resolve().parents[1]                     # supplement root
SEED, DRAWS, FLIPS, BATCH = 20261126, 100_000, 200_000, 500
z = np.load(SUPP / 'statistics/validated_inputs.npz', allow_pickle=False)
control = {}
for r in csv.DictReader((SUPP / 'review_diagnostics/control_correctness.csv').open()):
    control.setdefault((r['dataset'], r['condition']), {})[r['question_id']] = int(r['correct'])

def column(dataset, method, extractor):
    cols = [json.loads(str(c)) for c in z[dataset + '_columns']]
    want = ('gemma3_27b_it', method, extractor, 'parent', 10)
    idx = [i for i, c in enumerate(cols) if (c['model'], c['method'], c['extractor'], c['stage'], c['k']) == want]
    assert len(idx) == 1, want
    return z[dataset + '_values'][:, idx[0]].astype(float)

rows, pvals = [], []
for d, (dataset, writer) in enumerate([('mbe', 'qwen38_kg'), ('housing', 'terra_kg')]):
    ids = [str(i) for i in z[dataset + '_ids']]
    raw = np.array([control[(dataset, 'raw_passage_cues')][i] for i in ids], dtype=float)
    groups, inv = np.unique(z[dataset + '_groups'], return_inverse=True)
    G = len(groups)
    counts = np.bincount(inv, minlength=G).astype(float)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, d]))
    for name, comp in [('graph_same_writer_qwen_extraction', column(dataset, writer, 'qwen')),
                       ('qwen_dense', column(dataset, 'qwen3_dense', 'not_applicable'))]:
        diff = raw - comp
        num = np.bincount(inv, weights=diff, minlength=G)
        boot = np.empty(DRAWS)
        for s in range(0, DRAWS, BATCH):
            w = rng.multinomial(G, np.full(G, 1 / G), size=BATCH).astype(float)
            boot[s:s + BATCH] = (w @ num) / (w @ counts)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        obs = abs(num.sum()); extreme = 0
        for _ in range(FLIPS // BATCH):
            sign = rng.integers(0, 2, size=(BATCH, G)) * 2 - 1
            extreme += int((np.abs(sign @ num) >= obs - 1e-12).sum())
        p = (extreme + 1) / (FLIPS + 1)
        rows.append({'dataset': dataset, 'raw_text_writer': writer, 'comparator': name,
                     'raw_text_accuracy': 100 * raw.mean(), 'comparator_accuracy': 100 * comp.mean(),
                     'delta_pp': 100 * diff.mean(), 'ci95_pp': [100 * lo, 100 * hi], 'p_signflip': p,
                     'raw_only_correct': int(((raw == 1) & (comp == 0)).sum()),
                     'comparator_only_correct': int(((raw == 0) & (comp == 1)).sum())})
        pvals.append(p)
order = np.argsort(pvals); running = 0.0
for rank, i in enumerate(order):
    running = min(1.0, max(running, (len(pvals) - rank) * pvals[i])); rows[i]['p_holm_4'] = running
out = {'seed': SEED, 'bootstrap_draws': DRAWS, 'sign_flips': FLIPS, 'answer_model': 'gemma3_27b_it',
       'stage': 'initial', 'k': 10, 'contrasts': rows,
       'note': 'HousingQA graph cell is the run with a different generation seed and GPU memory setting (dagger in the appendix).'}


def _close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    return a == b

assert _close(out, json.loads(Path(__file__).with_name('raw_text_intervals.json').read_text())), 'raw-text intervals differ'
print(json.dumps({'status': 'PASS', 'replayed': 'raw_text_intervals', 'contrasts': len(rows)}))
for r in []:
    print(f"{r['dataset']:8s} raw({r['raw_text_writer']}) vs {r['comparator']:34s} {r['raw_text_accuracy']:.2f} vs {r['comparator_accuracy']:.2f}  "
          f"d={r['delta_pp']:+.2f} [{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}] p={r['p_signflip']:.4f} holm={r['p_holm_4']:.4f}")
