# Legal retrieval review materials

This supplement supports verification of the reported legal retrieval and answer results. The 22 November 2026 revision adds the HousingQA rank-one comparison and two-writer Alabama score check to the 20 November release. Both datasets use ordinary answer accuracy. The default check runs offline on saved outcomes.

## Verify the results

Extract the entire archive to an empty directory. Open its top-level directory and, using a Python environment with NumPy, run:

```sh
python verify_bundle.py
```

The arithmetic was checked with Python 3.11 and NumPy 1.26.4. The command prints a JSON receipt with status `PASS`. It checks every data file and the verifier against the integrity manifest. The manifest itself is excluded from its own hash. Record the ZIP hash when acquiring the archive. The default command only reads files and performs deterministic arithmetic.

The verifier recomputes all 1,260 accuracy totals, 1,792 graph–baseline contrasts, both Holm adjustments and 630 HousingQA class summaries. It also checks the 16 four-writer summary rows, four median absolute changes, the MBE reference-hit and no-retrieval breakdowns, and the HousingQA yes/no comparison. It verifies query coverage, checksums and relationships used in scoring. The raw Monte Carlo probabilities and bootstrap intervals are from the saved runs. An optional statistical replay is available with `python statistics/reproduce_statistics.py`.

## HousingQA query writers at rank one

From the supplement root, run `python housing_rq2/verify.py` for just this check. It needs only the Python standard library. The full verifier also runs it.

[The first two hits](housing_rq2/top_two.csv) give the original first- and second-ranked statute identifiers and scores for each of four query writers and each of 4,584 questions (18,336 rows). These are initial rankings on the Qwen-extracted graph, restricted to the question's jurisdiction. Each row includes its query fingerprint and the number of statutes tied at the top in the archived full-corpus replay. The checker joins the rows to the released cohort and queries, and verifies that a positive score gap agrees with a tie count of one.

At least two writers each have a unique top statute on 3,617 questions. They choose different statutes on **2,866 of all 4,584 questions (62.52%)**. On 751 questions the writers with unique winners agree, and on 967 fewer than two writers have a unique winner. Each question contributes once. Writers with a tied first place do not establish a disagreement.

[The Alabama records](housing_rq2/alabama_records.json) contain the scored fields of all 16 Qwen-extracted provisions in §35-9A-141 and all six in §35-9A-461 for HousingQA question 0. The checker reads the weighted conditions directly from the released Mistral and Terra query texts, evaluates each provision and takes the maximum within each statute. It prints the winning provision, its matched conditions and their points, and checks the scores, order and fingerprints of the winning rule texts against the saved graph results. In statute order §141 / §461, Mistral scores 80 / 70 and Terra scores 52 / 60. Both Terra provisions earn 40 phrase points and 12 legal-area points. Only the proceedings provision earns the eight-point type bonus. This local example and the cohort count use archived data and require no graph server or model calls. [Offline checker](housing_rq2/verify.py)

## Four writers and the highlighted answer comparisons

- [Writer comparisons](review_diagnostics/writer_comparisons.csv) counts higher, tied and lower ordinary accuracy for each writer against each baseline at matched stage, answerer and depth. Every row covers 112 settings including both extractors.
- [Answer evidence](review_diagnostics/answer_evidence.json) gives the median changes across every pair of writers and across the two extractors, the MBE answer transitions by annotated-passage presence, and the HousingQA yes/no counts. The MBE example is Gemma 3 27B with ten reranked passages from Terra on Qwen-extracted records or from Qwen dense. The HousingQA example is Gemma 3 12B with ten reranked passages from Mistral on Llama-extracted records or from Qwen dense.
- The MBE option-phrase counts and confirmation of complete prompt delivery come from an audit of restricted original queries and prompt records. The release includes the matching rule and aggregate results. The public verifier reproduces the answer transitions from saved correctness indicators and reference-membership flags, and checks the arithmetic of the option counts. Independently checking individual phrases or delivered prompt texts requires the restricted originals.
- [Raw-passage tie sensitivity](raw_cues/tie_sensitivity_summary.csv) includes identifier order, corpus order and expected Hit@10 under uniform tie-breaking. The raw-passage QA control selected writers using the historical corpus ordering and supplied passages using identifier ordering.

## What the files contain

| Directory | Contents |
| --- | --- |
| `statistics/` | Correctness matrices, question IDs, cluster groups, condition definitions, accuracy and paired tests. HousingQA includes yes/no labels for its class diagnostics. The MBE matrix contains a binary indicator of correctness for each item and configuration. |
| `queries/` | The complete text of all **18,336** HousingQA query instances. For all **4,284** MBE instances, a program structure preserves Cypher operations, schema fields, branch logic and numeric weights. Each distinct string literal within that query is replaced consistently with a placeholder. Non-schema variable names are generalised. The original full-query SHA-256 fingerprints let an authorised holder check exact identity. The masked MBE structures cannot reproduce historical rankings on their own. An executable-syntax example from the fixed MBE prompt is in `pipeline/prompts/mbe_v1.json`. |
| `review_diagnostics/` | HousingQA class counts and predicted yes rates, figure values, source-category counts, selected fixed-stage effect summaries, requested-reference hit strata and list lengths. All no-retrieval comparisons and hit-stratified summaries here are descriptive. The historical strata use requested passage lists. For the highlighted reranked MBE example, the prompt audit confirmed that all ten passages on each list reached the answerer. |
| `datasets/` | The two evaluated cohorts' definitions. The HousingQA mapping contains all public question IDs, question-text groups, jurisdictions, benchmark labels and corrected statute IDs for the retained cohort. Its rows link to the upstream text through the original identifiers. |
| `pipeline/` | Extraction instructions and schema source, query-generation prompts, and selected code snapshots for inspecting validation, recovery and graph construction. The scripts document the study's implementation. Rebuilding a full corpus/index requires upstream data, dependencies, model access and appropriate compute. |
| `generation_audit/` | Counts of returned responses, static validator rejections and execution recovery, plus recorded partial extraction cost. The 8.23 H100 allocation-hours cover the 66,906-passage HousingQA Llama completion only. |
| `expansion/`, `raw_cues/`, `extraction_quality/` | The saved raw-cue and dense-expansion control results, 225 HousingQA expansion texts, fingerprints of 1,071 MBE expansions, and the rubric and 64 extractor assessments from the matched sample of 32 HousingQA documents. Original per-document text can be looked up using the listed public corpus IDs. |

Source data, licence notices and access instructions appear in [DATA_ACCESS.md](DATA_ACCESS.md). The benchmark owner provides the original exam inputs. The historical MBE search programs and expansions were generated in this study. There is currently no public access route to their withheld text. A reader who already lawfully holds an original program can compare its SHA-256 fingerprint. The fingerprint does not supply its text. Item-specific MBE answer letters, choices and original question/passage text are omitted. Full graph stores, weights and raw answer-generation outputs are also outside this compact verification package. One proprietary query generator is recorded under the alias `gpt-5.6-terra`; the historical logs do not identify a dated provider snapshot.
