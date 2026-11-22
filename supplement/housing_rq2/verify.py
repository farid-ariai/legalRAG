"""Reproduce the HousingQA RQ2 count and the Alabama example from saved records."""

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WRITERS = ('gemma4_kg', 'mistral_kg', 'qwen38_kg', 'terra_kg')


def checked_query_hashes(cohort, question_id):
    hashes, example_queries = {}, {}
    for writer in WRITERS:
        with (ROOT / 'queries' / f'housing_{writer}.jsonl').open() as stream:
            for query in map(json.loads, stream):
                qid = query['question_id']
                key = (qid, writer)
                assert qid in cohort and key not in hashes
                digest = hashlib.sha256(query['query'].encode()).hexdigest()
                assert digest == query['original_query_sha256']
                hashes[key] = digest
                if qid == question_id:
                    example_queries[writer] = query['query']
    assert len(hashes) == len(WRITERS) * len(cohort)
    return hashes, example_queries


def check_winners(hashes, question_id):
    grouped = defaultdict(list)
    example_orders = {}
    with (HERE / 'top_two.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            qid, writer = row['question_id'], row['writer']
            assert row['query_sha256'] == hashes.pop((qid, writer))
            first, second = row['first_statute_id'], row['second_statute_id']
            a, b = int(row['first_score']), int(row['second_score'])
            tied = int(row['statutes_tied_at_first'])
            assert first and second and first != second and a >= b
            assert tied >= 1 and (a > b) == (tied == 1)
            grouped[qid].append(first if a > b else None)
            if qid == question_id:
                example_orders[writer] = [(first, a), (second, b)]
    assert not hashes and len(grouped) == 4584
    eligible = 0
    disagree = 0
    for winners in grouped.values():
        assert len(winners) == len(WRITERS)
        unique = [statute for statute in winners if statute is not None]
        eligible += len(unique) >= 2
        disagree += len(set(unique)) >= 2
    total = len(grouped)
    assert (disagree, eligible - disagree, total - eligible) == (2866, 751, 967)
    return {
        'questions': total, 'writer_rankings': total * len(WRITERS),
        'at_least_two_writers_with_a_unique_first': eligible,
        'different_unique_first_statutes': disagree, 'percent_of_all_questions': round(100 * disagree / total, 2),
        'eligible_questions_where_unique_winners_agree': eligible - disagree,
        'questions_with_fewer_than_two_unique_winners': total - eligible,
    }, example_orders


def example_predicates(query):
    """Read every additive CASE term in the two published Alabama score expressions."""
    one_line = ' '.join(query.split())
    expression = one_line.split('WITH s, p, ', 1)[1].split(' AS score ', 1)[0]
    predicates = []
    for term in expression.split(' + '):
        clause = re.fullmatch(r'\(CASE WHEN (.+) THEN (\d+) ELSE 0 END\)', term)
        if clause is None:
            raise ValueError(f'Unrecognised score term: {term}')
        condition, weight = clause[1], int(clause[2])
        phrase = re.fullmatch(r"toLower\(p\.(name|rule)\) CONTAINS '([^']+)'", condition)
        types = re.fullmatch(r"p\.type IN \[\s*('[^']+'(?:,\s*'[^']+')*)\s*\]", condition)
        exact_area = re.fullmatch(r"EXISTS \{ MATCH \(p\)-\[:BELONGS_TO\]->\(:AreaOfLaw \{\s*name: '([^']+)'\s*\}\) \}", condition)
        area_terms = re.fullmatch(r"EXISTS \{ MATCH \(p\)-\[:BELONGS_TO\]->\(a:AreaOfLaw\) WHERE toLower\(a\.name\) CONTAINS '([^']+)' OR toLower\(a\.name\) CONTAINS '([^']+)' \}", condition)
        if phrase:
            predicates.append((phrase[1], 'contains', [phrase[2]], weight))
        elif types:
            predicates.append(('type', 'equals', re.findall(r"'([^']+)'", types[1]), weight))
        elif exact_area:
            predicates.append(('area', 'equals', [exact_area[1]], weight))
        elif area_terms:
            predicates.append(('area', 'contains', list(area_terms.groups()), weight))
        else:
            raise ValueError(f'Unrecognised Alabama predicate: {condition}')
    return predicates


def score_provision(unit, predicates):
    points = {'phrases': 0, 'area': 0, 'type': 0}
    matches = []
    for field, operator, values, weight in predicates:
        value = unit[field]
        matched = any(term in value.lower() for term in values) if operator == 'contains' else value in values
        if matched:
            group = 'phrases' if field in ('name', 'rule') else field
            points[group] += weight
            matches.append({'field': field, 'operator': operator, 'terms': values, 'points': weight})
    return {'provision': unit['name'], 'uid': unit['uid'], **points, 'total': sum(points.values()), 'matches': matches}


def check_example(example, queries, orders):
    expected = {'mistral_kg': {'431263': (68, 12, 0), '431286': (50, 12, 8)},
                'terra_kg': {'431263': (40, 12, 0), '431286': (40, 12, 8)}}
    traces = {}
    assert {s['statute_id']: len(s['provisions']) for s in example['statutes']} == {'431263': 16, '431286': 6}
    assert len(example['statutes']) == 2
    for writer, breakdowns in expected.items():
        predicates = example_predicates(queries[writer])
        winners = {}
        for statute in example['statutes']:
            sid = statute['statute_id']
            scores = [score_provision(unit, predicates) for unit in statute['provisions']]
            top = max(v['total'] for v in scores)
            winner, = [v for v in scores if v['total'] == top]
            unit, = [u for u in statute['provisions'] if u['uid'] == winner['uid']]
            assert hashlib.sha256(unit['rule'].encode()).hexdigest() == example['logged_best_rule_sha256'][writer][sid]
            assert (winner['phrases'], winner['area'], winner['type']) == breakdowns[sid]
            winners[sid] = winner
        ranked = sorted(((sid, v['total']) for sid, v in winners.items()), key=lambda p: -p[1])
        assert set(winners) == set(breakdowns) and ranked == orders[writer]
        traces[writer] = winners
    return traces


def verify():
    if not __debug__:
        raise RuntimeError('Use the normal Python interpreter with assertions enabled.')
    with (ROOT / 'datasets/housing_cohort.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cohort = {r['question_id']: r for r in rows}
    assert len(cohort) == len(rows) == 4584 and len({r['jurisdiction'] for r in rows}) == 47
    example = json.loads((HERE / 'alabama_records.json').read_text())
    qid = example['question_id']
    assert cohort[qid]['jurisdiction'] == example['jurisdiction'] == 'Alabama'
    assert cohort[qid]['reference_statute_ids'] == example['reference_statute_id'] == '431263'
    hashes, queries = checked_query_hashes(cohort, qid)
    summary, orders = check_winners(hashes, qid)
    return {'status': 'PASS', 'housing_first_stage_qwen_extraction': summary,
            'alabama_scores': check_example(example, queries, orders)}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
