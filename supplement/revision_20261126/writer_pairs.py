"""Pooled writer-vs-writer contrasts on identical graphs (both extractors, 7 answer models, 4 depths,
2 stages = 112 matched settings per pair). Saved outputs only; ordinary accuracy; same protocol as the
pooled summaries; Holm over the 12 pairs (6 per dataset)."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np

SUPP = Path(__file__).resolve().parents[1]                     # supplement root
z = np.load(SUPP / 'statistics/validated_inputs.npz', allow_pickle=False)
WRITERS = ['terra_kg', 'gemma4_kg', 'qwen38_kg', 'mistral_kg']
SEED, DRAWS, FLIPS, BATCH = 20261126, 100_000, 200_000, 500
rows = []
for stream, dataset in enumerate(['mbe', 'housing']):
    cols = [json.loads(str(c)) for c in z[dataset + '_columns']]
    lookup = {(c['model'], c['method'], c['extractor'], c['stage'], c['k']): i for i, c in enumerate(cols)}
    models = sorted({c['model'] for c in cols})
    values = z[dataset + '_values'].astype(float)
    groups, inv = np.unique(z[dataset + '_groups'], return_inverse=True)
    G = len(groups); counts = np.bincount(inv, minlength=G).astype(float)
    pairs, pq = list(combinations(WRITERS, 2)), []
    for a, b in pairs:
        d = [values[:, lookup[(m, a, e, s, k)]] - values[:, lookup[(m, b, e, s, k)]]
             for e in ('qwen', 'llama') for m in models for s in ('parent', 'reranked') for k in (1, 5, 10, 20)]
        pq.append(np.column_stack(d).mean(axis=1))
    pq = np.column_stack(pq)
    num = np.zeros((G, pq.shape[1])); np.add.at(num, inv, pq)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 10 + stream]))
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
    for j, (a, b) in enumerate(pairs):
        rows.append({'dataset': dataset, 'writer_a': a, 'writer_b': b, 'delta_pp': 100 * pq[:, j].mean(),
                     'ci95_pp': [100 * lo[j], 100 * hi[j]], 'p_signflip': float(p[j])})
order = np.argsort([r['p_signflip'] for r in rows]); run = 0.0
for r_, i in enumerate(order):
    run = min(1.0, max(run, (len(rows) - r_) * rows[i]['p_signflip'])); rows[i]['p_holm_12'] = run


def _close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    return a == b

assert _close(rows, json.loads(Path(__file__).with_name('writer_pairs.json').read_text())), 'writer pairs differ'
print(json.dumps({'status': 'PASS', 'replayed': 'writer_pairs', 'pairs': len(rows)}))
for r in []:
    print(f"{r['dataset']:8s} {r['writer_a']:10s} - {r['writer_b']:10s} {r['delta_pp']:+.2f} [{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}] holm {r['p_holm_12']:.4f}")
