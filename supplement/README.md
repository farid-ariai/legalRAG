# Legal retrieval review materials

This supplement supports verification of the reported legal retrieval and answer results. Both datasets use ordinary answer accuracy. Everything in the default check runs offline on saved outcomes.

## Verify the results

Extract the entire archive to an empty directory. Open its top-level directory and, using a Python environment with NumPy, run:

```sh
python verify_bundle.py
```

The arithmetic was checked with Python 3.11 and NumPy 1.26.4. The command prints a JSON receipt with status `PASS`. It checks every data file and the verifier against the integrity manifest. The manifest itself is excluded from its own hash. Record the ZIP hash when acquiring the archive. The default command only reads files and performs deterministic arithmetic.

The verifier recomputes all 1,260 fixed-stage or reference-condition totals from correctness indicators, all 1,792 graph–baseline differences and discordant-pair counts, both Holm adjustments from the saved raw values, and all 630 HousingQA class summaries. It checks the selected comparisons against each answer model's no-retrieval baseline. It also verifies query coverage, checksums and the use of scoring relationships. Raw Monte Carlo probabilities and bootstrap intervals refer to the saved runs. An **optional** statistical replay is available with `python statistics/reproduce_statistics.py`; it needs more memory and time. The normal check does not run this replay or perform inference, query generation or retrieval.

## What the files contain

| Directory | Contents |
| --- | --- |
| `statistics/` | Correctness matrices, question IDs, cluster groups, condition definitions, accuracy and paired tests. HousingQA includes yes/no labels for its class diagnostics. The MBE matrix contains a binary indicator of correctness for each item and configuration. |
| `queries/` | The complete text of all **18,336** HousingQA query instances. For all **4,284** MBE instances, a program structure preserves Cypher operations, schema fields, branch logic and numeric weights. Each distinct string literal within that query is replaced consistently with a placeholder. Non-schema variable names are generalised. The original full-query SHA-256 fingerprints let an authorised holder check exact identity. The masked MBE structures cannot reproduce historical rankings on their own. An executable-syntax example from the fixed MBE prompt is in `pipeline/prompts/mbe_v1.json`. |
| `review_diagnostics/` | HousingQA class counts and predicted yes rates, figure values, source-category counts, selected fixed-stage effect summaries, requested-reference hit strata and list lengths. All no-retrieval comparisons and hit-stratified summaries here are descriptive. The MBE joint-miss example concerns one answerer, extractor, stage and depth, and uses the requested passage list before possible token trimming. |
| `datasets/` | The two evaluated cohorts' definitions. The HousingQA mapping contains all public question IDs, question-text groups, jurisdictions, benchmark labels and corrected statute IDs for the retained cohort. Its rows link to the upstream text through the original identifiers. |
| `pipeline/` | Extraction instructions and schema source, query-generation prompts, and selected code snapshots for inspecting validation, recovery and graph construction. The scripts document the study's implementation. Rebuilding a full corpus/index requires upstream data, dependencies, model access and appropriate compute. |
| `generation_audit/` | Counts of returned responses, static validator rejections and execution recovery, plus recorded partial extraction cost. The 8.23 H100 allocation-hours cover the 66,906-passage HousingQA Llama completion only. |
| `expansion/`, `raw_cues/`, `extraction_quality/` | The saved raw-cue and dense-expansion control results, 225 HousingQA expansion texts, fingerprints of 1,071 MBE expansions, and the rubric and 64 extractor assessments from the matched sample of 32 HousingQA documents. Original per-document text can be looked up using the listed public corpus IDs. |

Source data, licence notices and access instructions appear in [DATA_ACCESS.md](DATA_ACCESS.md). The benchmark owner provides the original exam inputs. The historical MBE search programs and expansions were generated in this study. There is currently no public access route to their withheld text. A reader who already lawfully holds an original program can compare its SHA-256 fingerprint. The fingerprint does not supply its text. Item-specific MBE answer letters, choices and original question/passage text are omitted. Full graph stores, weights and raw answer-generation outputs are also outside this compact verification package. One proprietary query generator is recorded under the alias `gpt-5.6-terra`; the historical logs do not identify a dated provider snapshot.
