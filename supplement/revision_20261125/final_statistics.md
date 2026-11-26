# Final secondary statistics (saved outputs only)

Script: `pooled_statistics.py`; machine-readable output: `final_statistics.json`. Seed 20261125 (streams {'mbe_pooled': 1, 'housing_pooled': 2, 'mbe_strata': 3}); 100,000 whole-cluster bootstrap draws for pointwise 95% percentile intervals; 200,000 cluster-level sign flips with plus-one correction for two-sided p. Clusters: 844 MBE fact-pattern groups (1071 questions) and 225 HousingQA question-text groups (4584 questions). Endpoint: ordinary accuracy, with unparseable responses counted as errors. HousingQA is analysed in ordinary accuracy only, and no class-specific quantity is read or reported. No model calls, retrieval or network access. Runtime 14 s, numpy 1.26.4.

Smallest attainable raw p = 1/200,001 ≈ 0.000005. Holm adjustment is over the 16 pooled contrasts below. These analyses are secondary to the paper's 896-contrast per-cell family.

## 1. Pooled writer contrasts (graph minus baseline, mean over 112 matched cells: 2 extractors × 7 answerers × 4 depths × 2 stages)

| Dataset | Writer | Baseline | Δ (pp) | 95% CI | p (sign flip) | p Holm (16) | Cells higher/tied/lower |
|---|---|---|---:|---|---:|---:|---|
| MBE | Qwen 3.8 | Qwen dense | -0.05 | [-1.16, +1.09] | 0.938 | 1.000 | 50/3/59 |
| MBE | Qwen 3.8 | BM25 | +3.68 | [+2.39, +4.96] | <0.001 | <0.001 | 111/1/0 |
| MBE | Gemma 4 | Qwen dense | +0.80 | [-0.24, +1.85] | 0.140 | 0.978 | 75/1/36 |
| MBE | Gemma 4 | BM25 | +4.52 | [+3.28, +5.76] | <0.001 | <0.001 | 112/0/0 |
| MBE | Mistral | Qwen dense | -4.02 | [-5.11, -2.93] | <0.001 | <0.001 | 2/0/110 |
| MBE | Mistral | BM25 | -0.30 | [-1.50, +0.91] | 0.631 | 1.000 | 38/4/70 |
| MBE | Terra | Qwen dense | +3.38 | [+2.31, +4.48] | <0.001 | <0.001 | 112/0/0 |
| MBE | Terra | BM25 | +7.11 | [+5.89, +8.33] | <0.001 | <0.001 | 112/0/0 |
| HousingQA | Qwen 3.8 | Qwen dense | +3.39 | [+1.78, +4.98] | <0.001 | <0.001 | 105/0/7 |
| HousingQA | Qwen 3.8 | BM25 | +0.24 | [-1.02, +1.51] | 0.714 | 1.000 | 72/0/40 |
| HousingQA | Gemma 4 | Qwen dense | +3.50 | [+1.63, +5.32] | <0.001 | 0.003 | 102/0/10 |
| HousingQA | Gemma 4 | BM25 | +0.35 | [-1.07, +1.74] | 0.631 | 1.000 | 70/2/40 |
| HousingQA | Mistral | Qwen dense | +4.29 | [+2.24, +6.30] | <0.001 | <0.001 | 103/0/9 |
| HousingQA | Mistral | BM25 | +1.14 | [-0.44, +2.71] | 0.163 | 0.980 | 93/3/16 |
| HousingQA | Terra | Qwen dense | +3.07 | [+1.61, +4.51] | <0.001 | <0.001 | 105/0/7 |
| HousingQA | Terra | BM25 | -0.08 | [-1.42, +1.24] | 0.901 | 1.000 | 63/0/49 |

The higher/tied/lower counts reproduce manuscript Table 3 exactly.

## 2. Four-writer mean (descriptive only, no test)

| Dataset | Baseline | Δ (pp) | 95% interval (same draws) |
|---|---|---:|---|
| MBE | Qwen dense | +0.03 | [-0.87, +0.94] |
| MBE | BM25 | +3.75 | [+2.67, +4.84] |
| HousingQA | Qwen dense | +3.56 | [+1.89, +5.19] |
| HousingQA | BM25 | +0.41 | [-0.88, +1.67] |

## 3. MBE option-phrase strata (Terra graphs, both extractors, pooled over 112 cells)

The strata use the archived rule in the archived option-phrase rule (also described in the 22 November README). A complete scored CONTAINS literal of at least two word tokens appears in exactly one available choice and nowhere in the facts or stem, after case and whitespace normalisation. Every query was checked against its released SHA-256. Counts: none 477, sole correct 213, sole distractor 176, two or more options 205. Distribution by number of options matched: {'0': 477, '1': 389, '2': 167, '3': 33, '4': 5}. The sole option is correct in 213/389 cases (54.76%). Only aggregates are reported, because item-level phrases and answer keys are restricted.

### Terra minus Qwen dense

| Stratum | n | Δ (pp) | 95% CI | p (sign flip vs 0) | Cells positive / tied (of 112) |
|---|---:|---:|---|---:|---|
| No option-specific phrase | 477 | +2.85 | [+1.26, +4.49] | <0.001 | 104 / 2 |
| Phrases point only to the correct option | 213 | +2.49 | [+0.04, +5.05] | 0.052 | 87 / 6 |
| Phrases point only to a distractor | 176 | +3.66 | [+1.08, +6.30] | 0.006 | 98 / 4 |
| Phrases point to two or more options | 205 | +5.32 | [+2.80, +7.90] | <0.001 | 109 / 1 |

Correct-only minus distractor-only: -1.16 pp [-4.69, +2.36], two-sided p = 0.516. This p comes from bootstrap inversion over the same draws, because the contrast compares disjoint strata.

### Terra minus BM25

| Stratum | n | Δ (pp) | 95% CI | p (sign flip vs 0) | Cells positive / tied (of 112) |
|---|---:|---:|---|---:|---|
| No option-specific phrase | 477 | +6.60 | [+4.93, +8.30] | <0.001 | 112 / 0 |
| Phrases point only to the correct option | 213 | +8.45 | [+5.74, +11.19] | <0.001 | 111 / 1 |
| Phrases point only to a distractor | 176 | +4.65 | [+1.35, +8.00] | 0.007 | 100 / 2 |
| Phrases point to two or more options | 205 | +9.01 | [+6.00, +12.10] | <0.001 | 112 / 0 |

Correct-only minus distractor-only: +3.79 pp [-0.43, +8.00], two-sided p = 0.078. This p comes from bootstrap inversion over the same draws, because the contrast compares disjoint strata.

## 4. MBE net additional correct answers, Terra (both extractors) vs Qwen dense, all 112 cells

Source: `review_diagnostics/mbe_requested_reference_transitions.csv` (prior_group = all). All 112 cells were verified against the saved correctness indicators: per-cell Terra and Qwen-dense correct totals match `validated_inputs.npz`. Membership is defined as "annotated passage present in the requested top-k list of each system at the cell stage and depth (before any trimming to the answerer input budget)". Depth- and stage-specific membership is available for all 112 cells.

| Annotated passage in requested list of | Question–cell pairs | Net additional correct (Terra − dense) |
|---|---:|---:|
| Neither | 93,793 | +3,905 |
| Graph only | 3,731 | +405 |
| Both | 6,363 | +172 |
| Dense only | 16,065 | -423 |
| **Total** | 119,952 | **+4,059** |

Share of the net gain from questions where neither system listed the annotated passage: **96.21%**. Cross-check against the manuscript cell (Gemma 3 27B, reranked, k = 10, Terra/Qwen): 763 neither-questions (the membership file gives 763), with net 49 of 57, as in the paper.

## 5. Comparison with earlier draft values (presentation review, 25 Nov)

Point estimates are unchanged. Intervals moved by at most 0.1 pp at full draws:
- MBE Terra − Qwen dense: +3.38 [2.31, 4.48] (draft 10k: [2.31, 4.47]).
- Terra − BM25: +7.11 [5.89, 8.33] (draft [5.90, 8.31]).
- Strata: sole-correct [+0.04, +5.05] (drafts [0.00/0.15, 5.07]); distractor [+1.08, +6.30] (drafts [1.11, 6.20/6.29]).
- Correct − distractor −1.16 (drafts quoted −1.17, from subtracting rounded stratum means; the exact value is −1.1635).
- HousingQA four-writer mean vs Qwen dense: +3.56 [1.89, 5.19] (draft [1.92, 5.15]).
- Net breakdown: +3,905 of +4,059, identical to the draft.
- New in this run: against BM25, the sole-correct stratum gain exceeds the distractor stratum gain by +3.79 pp [−0.43, 8.00], p = 0.078 (not significant). The direction differs from the Qwen-dense contrast.
