"""Pooled writer contrasts, option-phrase strata and the neither-found breakdown (25 Nov 2026 revision).

Saved outputs only: no model calls, no retrieval, no network. HousingQA is analysed
in ordinary accuracy only (no class-specific quantities are read or reported).

Protocol (mirrors reproducibility/.../statistics/reproduce_statistics.py):
  * whole-cluster bootstrap, 100,000 draws, batches of 500, question-weighted ratio
    recomputed in each draw, pointwise 95% percentile intervals;
  * 200,000 cluster-level sign flips (one shared sign per cluster), two-sided,
    plus-one correction;
  * clusters: 844 MBE fact-pattern groups, 225 HousingQA question-text groups.

Default run (public inputs only): python pooled_statistics.py
  recomputes the 16 pooled contrasts (bootstrap and sign-flip replay, about 15 s) and the
  neither-found breakdown, and compares them with final_statistics.json.
Option-phrase strata need the restricted originals:
  python pooled_statistics.py --restricted-queries PATH --benchmark PATH
"""
import csv
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

REL = Path(__file__).resolve().parents[1]                     # supplement root
NPZ = REL / 'statistics/validated_inputs.npz'
TRANSITIONS = REL / 'review_diagnostics/mbe_requested_reference_transitions.csv'
MEMBERSHIP_K10 = REL / 'review_diagnostics/mbe_reranked_k10_reference_membership_by_question.csv'
PUBLIC_QUERIES = REL / 'queries/mbe_terra_kg.jsonl'
TERRA_QUERIES = None   # restricted: original Terra MBE programs (set with --restricted-queries)
BENCHMARK = None       # restricted: Bar Exam QA items with choices and answers (set with --benchmark)
OUT = Path(__file__).resolve().parent

SEED = 20261125
STREAMS = {'mbe_pooled': 1, 'housing_pooled': 2, 'mbe_strata': 3}
DRAWS, FLIPS, BATCH = 100_000, 200_000, 500
WRITERS = [('qwen38_kg', 'Qwen 3.8'), ('gemma4_kg', 'Gemma 4'), ('mistral_kg', 'Mistral'), ('terra_kg', 'Terra')]
BASELINES = [('qwen3_dense', 'Qwen dense'), ('bm25', 'BM25')]
EXTRACTORS = ['qwen', 'llama']
STAGES = ['parent', 'reranked']
DEPTHS = [1, 5, 10, 20]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise(text):
    return ' '.join(text.casefold().split())


def load(dataset, z):
    cols = [json.loads(str(c)) for c in z[dataset + '_columns']]
    lookup = {(c['model'], c['method'], c['extractor'], c['stage'], c['k']): i for i, c in enumerate(cols)}
    models = sorted({c['model'] for c in cols})
    values = z[dataset + '_values'].astype(np.float64)
    groups, inverse = np.unique(z[dataset + '_groups'], return_inverse=True)
    return lookup, models, values, groups, inverse


def matched_cells(lookup, models, writer, baseline):
    cells = []
    for extractor in EXTRACTORS:
        for model in models:
            for stage in STAGES:
                for k in DEPTHS:
                    g = lookup[(model, writer, extractor, stage, k)]
                    b = lookup[(model, baseline, 'not_applicable', stage, k)]
                    cells.append(((model, extractor, stage, k), g, b))
    assert len(cells) == 112, len(cells)
    return cells


def cluster_sums(per_question, inverse, n_groups):
    per_question = np.atleast_2d(per_question.T).T if per_question.ndim == 1 else per_question
    totals = np.zeros((n_groups, per_question.shape[1]))
    np.add.at(totals, inverse, per_question)
    return totals


def resample(rng, numerators, denominators, den_index):
    """Bootstrap ratios. numerators: G x C, denominators: G x D, den_index: C -> column of D."""
    G = numerators.shape[0]
    boot = np.empty((DRAWS, numerators.shape[1]))
    for first in range(0, DRAWS, BATCH):
        w = rng.multinomial(G, np.full(G, 1 / G), size=BATCH).astype(np.float64)
        boot[first:first + BATCH] = (w @ numerators) / (w @ denominators)[:, den_index]
    return boot


def sign_flip(rng, numerators):
    G = numerators.shape[0]
    observed = np.abs(numerators.sum(axis=0))
    extremes = np.zeros(numerators.shape[1], dtype=np.int64)
    for _ in range(FLIPS // BATCH):
        s = rng.integers(0, 2, size=(BATCH, G), dtype=np.int8).astype(np.float64) * 2 - 1
        extremes += (np.abs(s @ numerators) >= observed[None, :] - 1e-12).sum(axis=0)
    return (extremes + 1) / (FLIPS + 1)


def holm(pvalues):
    order = np.argsort(pvalues, kind='stable')
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = min(1.0, max(running, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def pooled(dataset, z):
    lookup, models, values, groups, inverse = load(dataset, z)
    G = len(groups)
    counts = np.bincount(inverse, minlength=G).astype(np.float64)[:, None]
    labels, per_question, per_cell = [], [], []
    for writer, wname in WRITERS:
        for baseline, bname in BASELINES:
            cells = matched_cells(lookup, models, writer, baseline)
            diffs = np.column_stack([values[:, g] - values[:, b] for _, g, b in cells])
            labels.append((writer, wname, baseline, bname))
            per_question.append(diffs.mean(axis=1))
            per_cell.append(diffs.mean(axis=0))
    pq = np.column_stack(per_question)
    numerators = cluster_sums(pq, inverse, G)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, STREAMS[dataset + '_pooled']]))
    boot = resample(rng, numerators, counts, np.zeros(pq.shape[1], dtype=int))
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    p = sign_flip(rng, numerators)
    rows = []
    for i, (writer, wname, baseline, bname) in enumerate(labels):
        cell = per_cell[i]
        rows.append({'dataset': dataset, 'writer': writer, 'writer_label': wname, 'baseline': baseline,
                     'baseline_label': bname, 'cells': 112,
                     'delta_pp': 100 * pq[:, i].mean(), 'ci95_pp': [100 * low[i], 100 * high[i]],
                     'p_signflip': float(p[i]),
                     'cells_higher': int((cell > 1e-12).sum()), 'cells_tied': int((np.abs(cell) <= 1e-12).sum()),
                     'cells_lower': int((cell < -1e-12).sum())})
    # four-writer mean, descriptive (interval from the same draws; no test)
    four = {}
    for b, (baseline, bname) in enumerate(BASELINES):
        idx = [i for i, lab in enumerate(labels) if lab[2] == baseline]
        mean_boot = boot[:, idx].mean(axis=1)
        four[baseline] = {'baseline_label': bname, 'delta_pp': 100 * pq[:, idx].mean(axis=1).mean(),
                          'interval95_pp_descriptive': [100 * x for x in np.quantile(mean_boot, [0.025, 0.975])]}
    return rows, four, {'clusters': G, 'questions': int(values.shape[0])}


def option_strata(z):
    benchmark = {}
    with BENCHMARK.open(newline='') as stream:
        for row in csv.DictReader(stream):
            benchmark[row['idx']] = row
    queries = json.loads(TERRA_QUERIES.read_text())
    public = {r['question_id']: r['original_query_sha256'] for r in map(json.loads, PUBLIC_QUERIES.read_text().splitlines())}
    assert set(benchmark) == set(public) == {q['idx'] for q in queries} and len(queries) == 1071
    phrase_pattern = re.compile(r"\bCONTAINS\s+'((?:\\.|''|[^'\\])*)'", re.I)
    stratum, bins, sole_correct = {}, Counter(), 0
    for query in queries:
        item = benchmark[query['idx']]
        program = query['parent_cypher_query']
        assert hashlib.sha256(program.encode()).hexdigest() == public[query['idx']]
        source_text = normalise(' '.join([item['prompt'], item['question']]))
        choices = {c: normalise(item['choice_' + c]) for c in 'abcd' if item['choice_' + c].strip()}
        raw = phrase_pattern.findall(program)
        assert len(raw) == len(re.findall(r'\bCONTAINS\b', program, re.I))
        options = set()
        for phrase in {normalise(s.replace("\\'", "'").replace("''", "'")) for s in raw}:
            if len(re.findall(r'\w+', phrase)) < 2:
                continue
            pattern = re.compile(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)')
            if pattern.search(source_text):
                continue
            hits = [c for c, value in choices.items() if pattern.search(value)]
            if len(hits) == 1:
                options.update(hits)
        bins[len(options)] += 1
        if not options:
            label = 'no_option_phrase'
        elif len(options) == 1:
            correct = item['answer'].strip().lower() in options
            sole_correct += int(correct)
            label = 'sole_correct' if correct else 'sole_distractor'
        else:
            label = 'multiple_options'
        stratum[query['idx']] = label
    counts = Counter(stratum.values())

    lookup, models, values, groups, inverse = load('mbe', z)
    G = len(groups)
    ids = [str(i) for i in z['mbe_ids']]
    assert set(ids) == set(stratum)
    labels_q = np.array([stratum[i] for i in ids])
    strata = ['no_option_phrase', 'sole_correct', 'sole_distractor', 'multiple_options']
    masks = np.column_stack([(labels_q == s).astype(np.float64) for s in strata])
    denominators = cluster_sums(masks, inverse, G)
    columns, den_index, per_cell_means = [], [], {}
    numer_cols = []
    for baseline, bname in BASELINES:
        cells = matched_cells(lookup, models, 'terra_kg', baseline)
        diffs = np.column_stack([values[:, g] - values[:, b] for _, g, b in cells])
        dq = diffs.mean(axis=1)
        for si, s in enumerate(strata):
            numer_cols.append(dq * masks[:, si])
            columns.append((baseline, s))
            den_index.append(si)
            m = masks[:, si].astype(bool)
            per_cell_means[(baseline, s)] = diffs[m].mean(axis=0)
    numerators = cluster_sums(np.column_stack(numer_cols), inverse, G)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, STREAMS['mbe_strata']]))
    boot = resample(rng, numerators, denominators, np.array(den_index))
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    p = sign_flip(rng, numerators)
    rows, interactions = [], {}
    for c, (baseline, s) in enumerate(columns):
        si = strata.index(s)
        cell = per_cell_means[(baseline, s)]
        rows.append({'baseline': baseline, 'stratum': s, 'questions': int(masks[:, si].sum()),
                     'delta_pp': 100 * numerators[:, c].sum() / denominators[:, si].sum(),
                     'ci95_pp': [100 * low[c], 100 * high[c]], 'p_signflip_vs_zero': float(p[c]),
                     'cells_positive': int((cell > 1e-12).sum()), 'cells_tied': int((np.abs(cell) <= 1e-12).sum()),
                     'cells': 112})
    for baseline, _ in BASELINES:
        ci_ = columns.index((baseline, 'sole_correct'))
        di_ = columns.index((baseline, 'sole_distractor'))
        diff = boot[:, ci_] - boot[:, di_]
        est = (numerators[:, ci_].sum() / denominators[:, 1].sum()) - (numerators[:, di_].sum() / denominators[:, 2].sum())
        p_boot = min(1.0, 2 * min(((diff <= 0).sum() + 1) / (DRAWS + 1), ((diff >= 0).sum() + 1) / (DRAWS + 1)))
        interactions[baseline] = {'correct_minus_distractor_pp': 100 * est,
                                  'ci95_pp': [100 * x for x in np.quantile(diff, [0.025, 0.975])],
                                  'p_bootstrap_two_sided': p_boot,
                                  'p_method': 'two-sided bootstrap-inversion p from the same 100,000 cluster draws (plus-one); not a sign-flip test, because the contrast compares two disjoint question strata'}
    audit = {'questions_by_number_of_options_matched': {str(k): bins[k] for k in range(5)},
             'stratum_counts': dict(counts), 'sole_option_correct': sole_correct,
             'sole_option_distractor': bins[1] - sole_correct,
             'sole_option_correct_share': sole_correct / bins[1]}
    return rows, interactions, audit


def neither_found(z):
    lookup, models, values, _, _ = load('mbe', z)
    with TRANSITIONS.open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['prior_group'] == 'all']
    cells = {}
    for r in rows:
        key = (r['model'], r['extractor'], r['stage'], int(r['k']))
        cells.setdefault(key, {})[r['reference_hit_group']] = r
    assert len(cells) == 112, len(cells)
    totals = Counter()
    question_totals = Counter()
    checked = 0
    for (model, extractor, stage, k), groups in cells.items():
        assert set(groups) == {'both_hit', 'KG_only_hit', 'dense_only_hit', 'neither_hit'}
        assert sum(int(g['questions']) for g in groups.values()) == 1071
        kg = int(values[:, lookup[(model, 'terra_kg', extractor, stage, k)]].sum())
        dense = int(values[:, lookup[(model, 'qwen3_dense', 'not_applicable', stage, k)]].sum())
        assert sum(int(g['KG_correct']) for g in groups.values()) == kg, (model, extractor, stage, k)
        assert sum(int(g['dense_correct']) for g in groups.values()) == dense, (model, extractor, stage, k)
        for name, g in groups.items():
            assert int(g['net_correct']) == int(g['KG_only_correct']) - int(g['dense_only_correct'])
            assert int(g['net_correct']) == int(g['KG_correct']) - int(g['dense_correct'])
            totals[name] += int(g['net_correct'])
            question_totals[name] += int(g['questions'])
        checked += 1
    net = sum(totals.values())
    # cross-check the single cell reported in the manuscript (Gemma 3 27B, reranked, k = 10, Qwen extraction)
    with MEMBERSHIP_K10.open() as stream:
        membership = list(csv.DictReader(stream))
    neither_k10 = sum(1 for m in membership if m['terra_qwen_requested_reference'] == '0' and m['qwen_dense_requested_reference'] == '0')
    paper_cell = cells[('gemma3_27b_it', 'qwen', 'reranked', 10)]
    return {'cells_verified_against_saved_correctness': checked,
            'net_additional_correct_by_reference_group': dict(totals),
            'question_cell_pairs_by_reference_group': dict(question_totals),
            'net_additional_correct_total': net,
            'share_from_neither': totals['neither_hit'] / net,
            'membership_definition': 'annotated passage present in the requested top-k list of each system at the cell stage and depth (before any trimming to the answerer input budget)',
            'cross_check_gemma3_27b_reranked_k10_qwen': {
                'neither_questions_from_membership_file': neither_k10,
                'neither_questions_in_transitions': int(paper_cell['neither_hit']['questions']),
                'neither_net': int(paper_cell['neither_hit']['net_correct']),
                'total_net': sum(int(g['net_correct']) for g in paper_cell.values())}}


def close(a, b, tol=1e-9):
    if isinstance(a, dict):
        return set(a) == set(b) and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= tol
    return a == b


def main():
    import argparse
    global TERRA_QUERIES, BENCHMARK
    parser = argparse.ArgumentParser()
    parser.add_argument('--restricted-queries', type=Path)
    parser.add_argument('--benchmark', type=Path)
    args = parser.parse_args()
    saved = json.loads((OUT / 'final_statistics.json').read_text())
    assert saved['inputs_sha256']['validated_inputs.npz'] == sha256(NPZ)
    z = np.load(NPZ, allow_pickle=False)
    pooled_rows, four = [], {}
    for dataset in ['mbe', 'housing']:
        rows, four_d, _ = pooled(dataset, z)
        pooled_rows += rows
        four[dataset] = four_d
    adjusted = holm(np.array([r['p_signflip'] for r in pooled_rows]))
    for r, a in zip(pooled_rows, adjusted):
        r['p_holm_16'] = float(a)
    assert close(pooled_rows, saved['pooled_writer_contrasts']), 'pooled contrasts differ'
    assert close(four, saved['four_writer_mean_descriptive']), 'four-writer means differ'
    assert close(neither_found(z), saved['neither_found_breakdown']), 'neither-found breakdown differs'
    checked = ['pooled_writer_contrasts', 'four_writer_mean_descriptive', 'neither_found_breakdown']
    if args.restricted_queries and args.benchmark:
        TERRA_QUERIES, BENCHMARK = args.restricted_queries, args.benchmark
        assert saved['inputs_sha256'][TERRA_QUERIES.name] == sha256(TERRA_QUERIES)
        assert saved['inputs_sha256'][BENCHMARK.name] == sha256(BENCHMARK)
        strata_rows, interactions, audit = option_strata(z)
        assert close({'rows': strata_rows, 'correct_minus_distractor': interactions, 'audit': audit},
                     saved['option_phrase_strata']), 'option strata differ'
        checked.append('option_phrase_strata')
    print(json.dumps({'status': 'PASS', 'replayed': checked, 'bootstrap_draws': DRAWS, 'sign_flips': FLIPS,
                      'seed': SEED}))


if __name__ == '__main__':
    main()
