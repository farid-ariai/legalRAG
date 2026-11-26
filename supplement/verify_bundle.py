"""Verify the saved comparisons, cohort, query records and release manifest offline."""

import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SYSTEMS = ['qwen38_kg', 'gemma4_kg', 'mistral_kg', 'terra_kg']


def rows(relative):
    with (ROOT / relative).open(newline='') as stream:
        return list(csv.DictReader(stream))


def load(relative):
    return json.loads((ROOT / relative).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_manifest():
    expected = load('MANIFEST.json')['files']
    actual = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*') if p.is_file() and p != ROOT / 'MANIFEST.json'}
    assert set(actual) == set(expected), 'Archive membership differs from the manifest'
    for name, path in actual.items():
        assert not path.is_symlink()
        raw = path.read_bytes()
        assert len(raw) == expected[name]['bytes'] and digest(raw) == expected[name]['sha256'], name
    return len(actual)


def key(row):
    stage = 'parent' if row['stage'] == 'initial' else row['stage']
    return row['model'], row['extractor'], row['method'], stage, int(row['k'])


def verify_accuracy(data):
    fields = {d + '_' + k for d in ['mbe', 'housing'] for k in ['ids', 'groups', 'columns', 'values']}
    fields.add('housing_gold')
    assert set(data.files) == fields
    positions = {}
    totals = {}
    for dataset, questions, clusters in [('mbe', 1071, 844), ('housing', 4584, 225)]:
        values = data[dataset + '_values']
        assert values.shape == (questions, 630) and set(np.unique(values)) == {0, 1}
        assert len(set(data[dataset + '_ids'])) == questions
        assert len(data[dataset + '_groups']) == questions and len(set(data[dataset + '_groups'])) == clusters
        positions[dataset] = {key(json.loads(str(c))): i for i, c in enumerate(data[dataset + '_columns'])}
        assert len(positions[dataset]) == 630
        for condition, i in positions[dataset].items():
            totals[(dataset,) + condition] = int(values[:, i].sum())
    fixed = rows('statistics/all_fixed_stage_accuracy.csv')
    assert len(fixed) == len(totals) == 1260
    assert {(r['dataset'],) + key(r) for r in fixed} == set(totals)
    for row in fixed:
        total = totals[(row['dataset'],) + key(row)]
        assert int(row['correct']) == total
        assert abs(float(row['accuracy_percent']) - 100 * total / int(row['questions'])) < 1e-10
    return positions, totals


def verify_holm(family, field):
    previous = 0.
    for rank, row in enumerate(sorted(family, key=lambda r: float(r['p_raw']))):
        previous = min(1., max(previous, (len(family) - rank) * float(row['p_raw'])))
        assert abs(previous - float(row[field])) < 1e-12


def verify_writer_summaries(totals):
    comparisons = defaultdict(Counter)
    frames = defaultdict(dict)
    for (dataset, model, extractor, writer, stage, depth), correct in totals.items():
        if writer not in SYSTEMS:
            continue
        frames[dataset, model, stage, depth][writer, extractor] = correct
        for comparator in ['qwen3_dense', 'bm25']:
            reference = totals[dataset, model, 'not_applicable', comparator, stage, depth]
            outcome = 'higher' if correct > reference else 'lower' if correct < reference else 'tied'
            comparisons[dataset, writer, comparator][outcome] += 1
    result = rows('review_diagnostics/writer_comparisons.csv')
    assert len(result) == len(comparisons) == 16
    for row in result:
        counts = comparisons.pop((row['dataset'], row['writer'], row['comparator']))
        assert sum(counts.values()) == int(row['settings']) == 112
        assert all(counts[key] == int(row[key]) for key in ['higher', 'tied', 'lower'])

    differences = defaultdict(list)
    for (dataset, model, stage, depth), scores in frames.items():
        assert len(scores) == 8
        for extractor in ['qwen', 'llama']:
            for a, b in combinations(SYSTEMS, 2):
                differences[dataset, 'writer'].append(abs(scores[a, extractor] - scores[b, extractor]))
        for writer in SYSTEMS:
            differences[dataset, 'extractor'].append(abs(scores[writer, 'qwen'] - scores[writer, 'llama']))
    medians = load('review_diagnostics/answer_evidence.json')['median_changes']
    for (dataset, component), values in differences.items():
        saved = medians[dataset][component]
        questions = 1071 if dataset == 'mbe' else 4584
        assert len(values) == saved['pairs'] == (672 if component == 'writer' else 224)
        assert round(100 * statistics.median(values) / questions, 2) == saved['median_absolute_change_pp']


def verify_contrasts(data, positions, totals):
    contrasts = rows('statistics/paired_KG_vs_baselines.csv')
    assert len(contrasts) == 1792
    conditions = {(r['dataset'], r['comparator']) + key(r) for r in contrasts}
    assert len(conditions) == len(contrasts)
    summaries = load('review_diagnostics/revision_numbers.json')
    splits = {}
    for dataset in ['mbe', 'housing']:
        family = [r for r in contrasts if r['dataset'] == dataset]
        assert len(family) == 896
        values = data[dataset + '_values']
        verify_holm(family, 'p_holm_896')
        selected = 'terra_kg' if dataset == 'mbe' else 'mistral_kg'
        for comparator in ['bm25', 'qwen3_dense']:
            subfamily = [r for r in family if r['comparator'] == comparator]
            assert len(subfamily) == 448
            verify_holm(subfamily, 'p_holm_448')
            selected_rows = [r for r in subfamily if r['method'] == selected]
            significant = [r for r in selected_rows if float(r['p_holm_896']) < .05 and float(r['effect_pp']) > 0]
            above = sum(int(r['KG_correct']) > totals[(dataset, r['model'], 'not_applicable', 'no_retrieval', 'parent', 0)] for r in significant)
            below = sum(int(r['KG_correct']) < totals[(dataset, r['model'], 'not_applicable', 'no_retrieval', 'parent', 0)] for r in significant)
            tag = dataset + '/' + comparator
            saved = summaries['selected_holm_positive'][tag]
            assert (len(significant), above, below) == (saved['n'], saved['above_own_no_retrieval'], saved['below_own_no_retrieval'])
            splits[tag] = [len(significant), above, below]
        for r in family:
            kg = key(r)
            baseline = r['model'], 'not_applicable', r['comparator'], kg[3], kg[4]
            a, b = values[:, positions[dataset][kg]], values[:, positions[dataset][baseline]]
            assert int(a.sum()) == int(r['KG_correct']) and int(b.sum()) == int(r['baseline_correct'])
            assert int(np.count_nonzero((a == 1) & (b == 0))) == int(r['KG_only_correct'])
            assert int(np.count_nonzero((a == 0) & (b == 1))) == int(r['baseline_only_correct'])
            assert abs(100 * (int(a.sum()) - int(b.sum())) / len(a) - float(r['effect_pp'])) < 1e-10
    assert splits == {'mbe/bm25': [88, 88, 0], 'mbe/qwen3_dense': [11, 11, 0],
                      'housing/bm25': [0, 0, 0], 'housing/qwen3_dense': [27, 4, 23]}
    for name, method, comparator in [('Terra_vs_BM25', 'terra_kg', 'bm25'),
                                      ('Terra_vs_Qwen_dense', 'terra_kg', 'qwen3_dense'),
                                      ('Mistral_vs_Qwen_dense', 'mistral_kg', 'qwen3_dense')]:
        effects = [float(r['effect_pp']) for r in contrasts if
                   (r['dataset'], r['method'], r['comparator']) == ('mbe', method, comparator)]
        expected = summaries['mbe_fixed_stage'][name]
        assert len(effects) == 112 and sum(e > 0 for e in effects) == expected['positive']
        assert sum(e < 0 for e in effects) == expected['negative']
        assert abs(statistics.median(effects) - expected['median_pp']) < 1e-10
    return splits


def verify_housing(data, positions, totals):
    cohort = rows('datasets/housing_cohort.csv')
    index = {r['question_id']: r for r in cohort}
    assert set(index) == set(data['housing_ids']) and len(cohort) == 4584
    assert len({r['jurisdiction'] for r in cohort}) == 47
    occurrences = [v for r in cohort for v in r['reference_statute_ids'].split()]
    assert len(occurrences) == 9973 and len(set(occurrences)) == 672
    labels = data['housing_gold']
    for q, group, label in zip(data['housing_ids'], data['housing_groups'], labels):
        assert index[q]['label'] == label and index[q]['question_template'] == group


def verify_queries(data):
    fingerprints = {}
    for dataset in ['mbe', 'housing']:
        for generator in SYSTEMS:
            with (ROOT / 'queries' / f'{dataset}_{generator}.jsonl').open() as stream:
                items = [json.loads(line) for line in stream]
            assert {r['question_id'] for r in items} == set(data[dataset + '_ids'])
            for r in items:
                tag = dataset, generator, r['question_id']
                assert tag not in fingerprints
                fingerprints[tag] = r['original_query_sha256']
                if dataset == 'housing':
                    assert set(r) == {'question_id', 'original_query_sha256', 'query'}
                    assert digest(r['query'].encode()) == r['original_query_sha256']
                else:
                    assert set(r) == {'question_id', 'original_query_sha256', 'structure', 'structure_sha256'}
                    assert digest(r['structure'].encode()) == r['structure_sha256']
    assert len(fingerprints) == 22620
    counted = defaultdict(Counter)
    for r in rows('queries/relationship_flags.csv'):
        assert fingerprints.pop((r['dataset'], r['generator'], r['question_id'])) == r['original_query_sha256']
        for f in ['mentions_element', 'scores_element', 'scores_area', 'scores_condition']:
            counted[r['dataset']][f] += int(r[f])
    assert not fingerprints
    assert counted['mbe']['mentions_element'] == 4284 and counted['mbe']['scores_element'] == 0
    assert counted['housing']['scores_area'] == 17026 and counted['housing']['scores_condition'] == 1


def verify_mbe_diagnostics(data, positions):
    saved = load('review_diagnostics/revision_numbers.json')
    counts = defaultdict(Counter)
    seen = defaultdict(set)
    raw = rows('review_diagnostics/mbe_initial_top10_source_counts_by_question.csv')
    assert len(raw) == 3213
    for r in raw:
        q = r['question_id']
        assert q not in seen[r['method']] and int(r['k']) == 10
        seen[r['method']].add(q)
        total = {k: int(r[k]) for k in ['mbe', 'caselaw', 'wex']}
        assert sum(total.values()) == 10 and min(total.values()) >= 0
        counts[r['method']].update(total)
    assert set(seen) == {'terra_qwen', 'terra_llama', 'qwen_dense'}
    assert all(s == set(data['mbe_ids']) for s in seen.values())
    assert counts == saved['mbe_initial_top10']
    assert sum(saved['mbe_corpus_sources'].values()) == 772001
    assert saved['mbe_unique_reference_sources'] == {'mbe': 1026}
    assert saved['mbe_question_weighted_reference_sources'] == {'mbe': 1071}
    hits = rows('review_diagnostics/mbe_reranked_k10_reference_membership_by_question.csv')
    lookup = {r['question_id']: (int(r['terra_qwen_requested_reference']), int(r['qwen_dense_requested_reference'])) for r in hits}
    assert len(hits) == len(lookup) == 1071 and set(lookup) == set(data['mbe_ids'])
    labels = {(0, 0): 'neither_hit', (1, 0): 'KG_only_hit', (0, 1): 'dense_only_hit', (1, 1): 'both_hit'}
    kg = positions['mbe'][('gemma3_27b_it', 'qwen', 'terra_kg', 'reranked', 10)]
    dense = positions['mbe'][('gemma3_27b_it', 'not_applicable', 'qwen3_dense', 'reranked', 10)]
    prior = positions['mbe'][('gemma3_27b_it', 'not_applicable', 'no_retrieval', 'parent', 0)]
    by_stratum = defaultdict(Counter)
    neither_by_prior = defaultdict(Counter)
    for q, outcomes in zip(data['mbe_ids'], data['mbe_values'][:, [kg, dense, prior]]):
        a, b, without_passages = map(int, outcomes)
        stratum = labels[lookup[q]]
        counts = {'questions': 1, 'KG_correct': a, 'dense_correct': b, 'net_correct': a-b,
                  'KG_only_correct': int(a > b), 'dense_only_correct': int(b > a)}
        by_stratum[stratum].update(counts)
        if stratum == 'neither_hit':
            group = 'no_retrieval_correct' if without_passages else 'no_retrieval_wrong'
            neither_by_prior[group].update(counts)
    expected = saved['mbe_gemma27_k10_terra_qwen_vs_dense_by_reference_in_requested_lists']
    assert dict(by_stratum) == expected
    assert by_stratum['neither_hit']['net_correct'] == 49 and sum(v['net_correct'] for v in by_stratum.values()) == 57
    evidence = load('review_diagnostics/answer_evidence.json')
    example = evidence['mbe_gemma27_reranked_10']
    assert example['questions_per_arm'] == len(hits)
    assert example['reference_groups'] == dict(by_stratum)
    assert example['neither_hit_by_no_retrieval'] == dict(neither_by_prior)
    # The lexical and prompt-delivery audits require source texts outside this review copy.
    phrases = evidence['mbe_terra_option_phrases']
    bins = phrases['questions_by_number_of_options_matched']
    assert set(bins) == {'0', '1', '2', '3', '4'} and min(bins.values()) >= 0
    assert sum(bins.values()) == phrases['questions'] == len(hits)
    assert phrases['questions_with_any_qualifying_phrase'] == len(hits) - bins['0']
    assert phrases['sole_option_correct'] + phrases['sole_option_distractor'] == bins['1']


def verify_controls(data):
    conditions = defaultdict(dict)
    for row in rows('review_diagnostics/control_correctness.csv'):
        col = conditions[row['dataset'], row['condition']]
        assert row['question_id'] not in col and row['correct'] in {'0', '1'}
        col[row['question_id']] = int(row['correct'])
    effect = {row['dataset']: row for row in rows('expansion/paired_expansion_effects.csv')}
    reported = {(row['dataset'], row['arm']): int(row['correct']) for row in rows('expansion/qa_metrics.csv')}
    assert len(conditions) == 6
    for dataset, raw_cue in [('mbe', 61.72), ('housing', 70.59)]:
        for arm in ['raw_passage_cues', 'raw', 'expanded']:
            assert set(conditions[dataset, arm]) == set(data[dataset + '_ids'])
        cue = conditions[dataset, 'raw_passage_cues']
        assert round(100 * sum(cue.values()) / len(cue), 2) == raw_cue
        raw = conditions[dataset, 'raw']
        expanded = conditions[dataset, 'expanded']
        assert reported[dataset, 'raw'] == sum(raw.values()) and reported[dataset, 'expanded'] == sum(expanded.values())
        assert sum(expanded[i] > raw[i] for i in raw) == int(effect[dataset]['expanded_only_correct'])
        assert sum(expanded[i] < raw[i] for i in raw) == int(effect[dataset]['raw_only_correct'])


def verify_revision(totals):
    """25 November revision: pooled contrasts, option-phrase strata, neither-found breakdown,
    tie-order bounds and the Table 3 / Fig. 3 values (deterministic arithmetic only)."""
    rev = load('revision_20261125/final_statistics.json')
    models = sorted({k[1] for k in totals if k[3] == 'terra_kg'})
    assert len(models) == 7
    pooled = rev['pooled_writer_contrasts']
    assert len(pooled) == 16
    for r in pooled:
        dataset = r['dataset']
        n = 1071 if dataset == 'mbe' else 4584
        cells = [totals[dataset, model, extractor, r['writer'], stage, k]
                 - totals[dataset, model, 'not_applicable', r['baseline'], stage, k]
                 for extractor in ['qwen', 'llama'] for model in models
                 for stage in ['parent', 'reranked'] for k in [1, 5, 10, 20]]
        assert len(cells) == r['cells'] == 112
        assert abs(100 * sum(cells) / (112 * n) - r['delta_pp']) < 1e-9
        assert (sum(c > 0 for c in cells), sum(c == 0 for c in cells), sum(c < 0 for c in cells)) == (
            r['cells_higher'], r['cells_tied'], r['cells_lower'])
        assert r['ci95_pp'][0] <= r['delta_pp'] <= r['ci95_pp'][1]
    verify_holm([{'p_raw': r['p_signflip'], 'p_holm_16': r['p_holm_16']} for r in pooled], 'p_holm_16')
    delta = {(r['dataset'], r['writer'], r['baseline']): r['delta_pp'] for r in pooled}
    for dataset, four in rev['four_writer_mean_descriptive'].items():
        for baseline, summary in four.items():
            mean = sum(delta[dataset, w, baseline] for w in SYSTEMS) / 4
            assert abs(mean - summary['delta_pp']) < 1e-9
    assert round(delta['mbe', 'terra_kg', 'qwen3_dense'], 2) == 3.38

    strata = rev['option_phrase_strata']
    audit = strata['audit']
    assert audit['stratum_counts'] == {'no_option_phrase': 477, 'sole_correct': 213,
                                       'sole_distractor': 176, 'multiple_options': 205}
    phrases = load('review_diagnostics/answer_evidence.json')['mbe_terra_option_phrases']
    assert audit['questions_by_number_of_options_matched'] == phrases['questions_by_number_of_options_matched']
    assert (audit['sole_option_correct'], audit['sole_option_distractor']) == (
        phrases['sole_option_correct'], phrases['sole_option_distractor'])
    for baseline in ['qwen3_dense', 'bm25']:
        part = [r for r in strata['rows'] if r['baseline'] == baseline]
        assert sum(r['questions'] for r in part) == 1071
        weighted = sum(r['questions'] * r['delta_pp'] for r in part) / 1071
        assert abs(weighted - delta['mbe', 'terra_kg', baseline]) < 1e-9
        assert all(r['ci95_pp'][0] <= r['delta_pp'] <= r['ci95_pp'][1] for r in part)
        by_name = {r['stratum']: r['delta_pp'] for r in part}
        contrast = strata['correct_minus_distractor'][baseline]
        assert abs(contrast['correct_minus_distractor_pp'] - (by_name['sole_correct'] - by_name['sole_distractor'])) < 1e-9

    found = rev['neither_found_breakdown']
    net, seen = Counter(), set()
    for r in rows('review_diagnostics/mbe_requested_reference_transitions.csv'):
        if r['prior_group'] == 'all':
            seen.add((r['model'], r['extractor'], r['stage'], int(r['k'])))
            net[r['reference_hit_group']] += int(r['net_correct'])
    assert len(seen) == 112 and dict(net) == found['net_additional_correct_by_reference_group']
    total = sum(net.values())
    assert total == found['net_additional_correct_total'] == sum(
        totals['mbe', m, e, 'terra_kg', s, k] - totals['mbe', m, 'not_applicable', 'qwen3_dense', s, k] for m, e, s, k in seen)
    assert abs(found['share_from_neither'] - net['neither_hit'] / total) < 1e-12
    assert (net['neither_hit'], total, round(100 * net['neither_hit'] / total, 2)) == (3905, 4059, 96.21)

    ties = {(r['dataset'], r['view'], r['system']): r for r in rows('raw_cues/tie_sensitivity_summary.csv')}
    columns = ['r_any10_tie_lower', 'r_any10_lexicographic', 'r_any10_random_tie_expected', 'r_any10_canonical', 'r_any10_tie_upper']
    quoted = {'qwen38_kg': (60, 65, 82.6, 135, 146), 'gemma4_kg': (59, 67, 84.1, 130, 154),
              'mistral_kg': (42, 47, 65.7, 105, 117), 'terra_kg': (76, 80, 95.4, 130, 138)}
    ratios, gains = [], []
    for system, expected in quoted.items():
        got = [1071 * float(ties['mbe', 'raw_text', system][c]) for c in columns]
        assert all(abs(round(g, 1) - e) < 1e-9 for g, e in zip(got, expected)) and got == sorted(got)
        ratios.append(got[3] / got[1])
        gains.append(got[2] / got[1] - 1)
    assert (round(min(ratios), 1), round(max(ratios), 1)) == (1.6, 2.2)
    assert (round(100 * min(gains)), round(100 * max(gains))) == (19, 40)
    percent = {s: [round(100 * float(ties['mbe', 'raw_text', s][c]), 2) for c in columns[1:4]] for s in quoted}
    assert percent['qwen38_kg'] == [6.07, 7.71, 12.61] and percent['terra_kg'] == [7.47, 8.91, 12.14]

    accuracy = {(r['dataset'], r['method'], r['extractor'], r['stage'], int(r['k']), r['model']): float(r['accuracy_percent'])
                for r in rows('statistics/all_fixed_stage_accuracy.csv')}
    ir = {(r['dataset'], r['method'], r['extractor'], r['stage'], int(r['k'])): float(r['value'])
          for r in rows('review_diagnostics/ir_figure_values.csv')}
    values = rows('revision_20261125/paper_table3_fig3_values.csv')
    assert len(values) == 48
    for r in values:
        if r['stage'] == 'not_applicable':
            mean = statistics.mean(accuracy[r['dataset'], r['method'], 'not_applicable', 'parent', 0, m] for m in models)
            assert abs(mean - float(r['table3_mean_accuracy_7_models_4_depths'])) < 1e-6
            continue
        stage = 'parent' if r['stage'] == 'initial' else 'reranked'
        cell = r['dataset'], r['method'], r['extractor'], stage
        all_depths = statistics.mean(accuracy[cell + (k, m)] for k in [1, 5, 10, 20] for m in models)
        at_ten = statistics.mean(accuracy[cell + (10, m)] for m in models)
        assert abs(all_depths - float(r['table3_mean_accuracy_7_models_4_depths'])) < 1e-6
        assert abs(at_ten - float(r['fig3_mean_accuracy_7_models_k10'])) < 1e-6
        assert abs(ir[cell + (10,)] - float(r['fig3_primary_ir_at_10'])) < 1e-6
    return {'pooled_contrasts': 16, 'option_strata': 8, 'neither_found_comparisons': 112,
            'tie_order_writers': 4, 'table3_fig3_rows': len(values)}


def verify_revision_20261126(data, positions, totals):
    """26 November additions: point estimates recomputed from saved correctness (deterministic)."""
    base = 'revision_20261126/'
    models = sorted({k[1] for k in totals if k[3] == 'terra_kg'})
    pairs = load(base + 'writer_pairs.json')
    assert len(pairs) == 12
    for r in pairs:
        n = 1071 if r['dataset'] == 'mbe' else 4584
        cells = [totals[r['dataset'], m, e, r['writer_a'], s, k] - totals[r['dataset'], m, e, r['writer_b'], s, k]
                 for e in ['qwen', 'llama'] for m in models for s in ['parent', 'reranked'] for k in [1, 5, 10, 20]]
        assert abs(100 * sum(cells) / (112 * n) - r['delta_pp']) < 1e-9
        assert r['ci95_pp'][0] <= r['delta_pp'] <= r['ci95_pp'][1]
    verify_holm([{'p_raw': r['p_signflip'], 'p_holm_12': r['p_holm_12']} for r in pairs], 'p_holm_12')
    gm = [r for r in pairs if (r['dataset'], r['writer_a'], r['writer_b']) == ('mbe', 'gemma4_kg', 'mistral_kg')][0]
    assert round(gm['delta_pp'], 2) == 4.82 and max(r['p_holm_12'] for r in pairs if r['dataset'] == 'housing') > 0.05
    assert min(r['p_holm_12'] for r in pairs if r['dataset'] == 'housing') > 0.05

    pilot = load(base + 'housing_pilot_question_ids.json')
    excluded = {str(i) for part in ['dev', 'test'] for i in pilot[part]}
    ids = [str(i) for i in data['housing_ids']]
    keep = np.array([i not in excluded for i in ids])
    assert int((~keep).sum()) == 384
    values = data['housing_values'].astype(float)
    result = load(base + 'housing_without_pilot.json')
    shifts = []
    for subset, mask in [('all_4584', np.ones(len(ids), bool)), ('without_pilot_4200', keep)]:
        assert result[subset]['questions'] == int(mask.sum())
        for r in result[subset]['contrasts']:
            diffs = np.column_stack([values[:, positions['housing'][(m, e, r['writer'], s, k)]]
                                     - values[:, positions['housing'][(m, 'not_applicable', r['baseline'], s, k)]]
                                     for e in ['qwen', 'llama'] for m in models for s in ['parent', 'reranked'] for k in [1, 5, 10, 20]])
            assert abs(100 * diffs.mean(axis=1)[mask].mean() - r['delta_pp']) < 1e-9
    for a, b in zip(result['all_4584']['contrasts'], result['without_pilot_4200']['contrasts']):
        shifts.append(abs(b['delta_pp'] - a['delta_pp']))
    assert round(max(shifts), 2) == 0.16

    raw = load(base + 'raw_text_intervals.json')
    control = defaultdict(dict)
    for r in rows('review_diagnostics/control_correctness.csv'):
        control[r['dataset'], r['condition']][r['question_id']] = int(r['correct'])
    for r in raw['contrasts']:
        dataset = r['dataset']
        order = [str(i) for i in data[dataset + '_ids']]
        cue = np.array([control[dataset, 'raw_passage_cues'][q] for q in order], dtype=float)
        if r['comparator'] == 'qwen_dense':
            col = positions[dataset][('gemma3_27b_it', 'not_applicable', 'qwen3_dense', 'parent', 10)]
        else:
            col = positions[dataset][('gemma3_27b_it', 'qwen', r['raw_text_writer'], 'parent', 10)]
        comp = data[dataset + '_values'][:, col].astype(float)
        assert abs(100 * (cue - comp).mean() - r['delta_pp']) < 1e-9
        assert r['ci95_pp'][0] <= r['delta_pp'] <= r['ci95_pp'][1]
    return {'writer_pairs': 12, 'pilot_excluded_questions': 384, 'raw_text_contrasts': len(raw['contrasts'])}


def main():
    if not __debug__:
        raise RuntimeError('Run with the normal Python interpreter so assertions are enabled.')
    files = check_manifest()
    with np.load(ROOT / 'statistics/validated_inputs.npz', allow_pickle=False) as data:
        positions, totals = verify_accuracy(data)
        verify_writer_summaries(totals)
        significant = verify_contrasts(data, positions, totals)
        verify_housing(data, positions, totals)
        verify_queries(data)
        verify_mbe_diagnostics(data, positions)
        verify_controls(data)
        revision = verify_revision(totals)
        revision26 = verify_revision_20261126(data, positions, totals)
    rq2 = json.loads(subprocess.check_output([sys.executable, '-B', str(ROOT / 'housing_rq2/verify.py')], text=True))
    assert rq2['status'] == 'PASS'
    print(json.dumps({'status': 'PASS', 'manifest_members': files, 'fixed_conditions': 1260,
                      'paired_contrasts': 1792, 'revision_20261125': revision, 'revision_20261126': revision26,
                      'writer_comparison_rows': 16, 'component_change_medians': 4,
                      'mbe_neither_hit_no_context_strata': 2,
                      'housing_rq2': rq2['housing_first_stage_qwen_extraction'],
                      'alabama_scores': {w: {sid: t['total'] for sid, t in hits.items()}
                                         for w, hits in rq2['alabama_scores'].items()},
                      'housing_full_queries': 18336, 'mbe_redacted_query_structures': 4284,
                      'significant_selected_gains_then_above_below_no_retrieval': significant,
                      'new_model_calls': 0, 'new_retrieval_runs': 0, 'new_randomisation_draws': 0}))


if __name__ == '__main__':
    main()
