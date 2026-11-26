"""HousingQA pooled writer contrasts with and without the 384 questions shared with the prompt pilot.
Saved outputs only; ordinary accuracy; same protocol as the paper's pooled summaries
(100,000 cluster bootstrap draws, 200,000 cluster sign flips, Holm over the 8 HousingQA contrasts)."""
import json
from pathlib import Path
import numpy as np

SUPP = Path(__file__).resolve().parents[1]                     # supplement root
split = json.loads((SUPP / 'revision_20261126/housing_pilot_question_ids.json').read_text())
z = np.load(SUPP / 'statistics/validated_inputs.npz', allow_pickle=False)
ids = [str(i) for i in z['housing_ids']]
pilot = {str(i) for part in ('dev', 'test') for i in split[part]}
overlap = {part: sum(str(i) in set(ids) for i in split[part]) for part in ('dev', 'test')}
assert overlap == {'dev': 116, 'test': 268}, overlap
keep = np.array([i not in pilot for i in ids])
assert keep.sum() == 4584 - 384

cols = [json.loads(str(c)) for c in z['housing_columns']]
lookup = {(c['model'], c['method'], c['extractor'], c['stage'], c['k']): i for i, c in enumerate(cols)}
models = sorted({c['model'] for c in cols})
values = z['housing_values'].astype(float)
SEED, DRAWS, FLIPS, BATCH = 20261126, 100_000, 200_000, 500

def per_question(writer, baseline):
    diffs = [values[:, lookup[(m, writer, e, s, k)]] - values[:, lookup[(m, baseline, 'not_applicable', s, k)]]
             for e in ('qwen', 'llama') for m in models for s in ('parent', 'reranked') for k in (1, 5, 10, 20)]
    assert len(diffs) == 112
    return np.column_stack(diffs).mean(axis=1)

def analyse(mask, stream):
    groups, inv = np.unique(z['housing_groups'][mask], return_inverse=True)
    G = len(groups)
    counts = np.bincount(inv, minlength=G).astype(float)
    labels, cols_ = [], []
    for w in ('qwen38_kg', 'gemma4_kg', 'mistral_kg', 'terra_kg'):
        for b in ('qwen3_dense', 'bm25'):
            labels.append((w, b)); cols_.append(per_question(w, b)[mask])
    pq = np.column_stack(cols_)
    num = np.zeros((G, pq.shape[1])); np.add.at(num, inv, pq)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, stream]))
    boot = np.empty((DRAWS, pq.shape[1]))
    for s in range(0, DRAWS, BATCH):
        w = rng.multinomial(G, np.full(G, 1 / G), size=BATCH).astype(float)
        boot[s:s + BATCH] = (w @ num) / (w @ counts)[:, None]
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    obs = np.abs(num.sum(axis=0)); ext = np.zeros(pq.shape[1])
    for _ in range(FLIPS // BATCH):
        sign = (rng.integers(0, 2, size=(BATCH, G)) * 2 - 1).astype(float)
        ext += (np.abs(sign @ num) >= obs - 1e-12).sum(axis=0)
    p = (ext + 1) / (FLIPS + 1)
    order = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(order):
        run = min(1.0, max(run, (len(p) - r) * p[i])); adj[i] = run
    return [{'writer': w, 'baseline': b, 'delta_pp': 100 * pq[:, j].mean(), 'ci95_pp': [100 * lo[j], 100 * hi[j]],
             'p_signflip': float(p[j]), 'p_holm_8': float(adj[j])} for j, (w, b) in enumerate(labels)], int(mask.sum()), G

result = {'pilot_overlap': overlap}
for name, mask, stream in [('all_4584', np.ones(len(ids), bool), 1), ('without_pilot_4200', keep, 2)]:
    rows, n, G = analyse(mask, stream)
    result[name] = {'questions': n, 'clusters': G, 'contrasts': rows}


def _close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    return a == b

assert _close(result, json.loads(Path(__file__).with_name('housing_without_pilot.json').read_text())), 'pilot exclusion differs'
print(json.dumps({'status': 'PASS', 'replayed': 'housing_without_pilot', 'questions_kept': int(keep.sum())}))
for a, b in []:
    print(f"{a['writer']:10s} vs {a['baseline']:11s} all {a['delta_pp']:+.2f} [{a['ci95_pp'][0]:+.2f},{a['ci95_pp'][1]:+.2f}]   "
          f"no-pilot {b['delta_pp']:+.2f} [{b['ci95_pp'][0]:+.2f},{b['ci95_pp'][1]:+.2f}] holm {b['p_holm_8']:.4f}")
