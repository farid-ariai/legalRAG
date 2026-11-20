# From legal text to queryable graphs

These review materials contain the saved results, queries and verification code for the MBE and HousingQA experiments. The 20 November 2026 revision extends the 18 November supplement. The [supplement guide](supplement/README.md) describes the files and the scope of the checks.

| To inspect | Open |
| --- | --- |
| Four query writers against Qwen dense and BM25 | [Higher, tied and lower accuracy at matched settings](supplement/review_diagnostics/writer_comparisons.csv) |
| Writer and extractor changes, MBE questions where neither retriever found the reference, and the HousingQA yes/no comparison | [Aggregate counts and definitions](supplement/review_diagnostics/answer_evidence.json) |
| Answer accuracy by model, stage and retrieval depth | [Full results](supplement/statistics/all_fixed_stage_accuracy.csv) |
| Graph comparisons against BM25 and Qwen dense | [Paired differences and Holm adjustments](supplement/statistics/paired_KG_vs_baselines.csv) |
| HousingQA performance relative to no retrieval | [Class counts and reference conditions](supplement/review_diagnostics/housing_class_diagnostics.csv) |
| How the generated queries use graph relationships | [Query census](supplement/queries/relationship_census.csv) and [query coverage](supplement/queries/coverage.json) |
| Cohorts, source data and MBE access terms | [Data access](supplement/DATA_ACCESS.md) |

## Verify the results offline

Download the repository as a ZIP and extract it. Open the `supplement` directory and run the following command in a Python environment with NumPy installed.

```sh
python verify_bundle.py
```

The verifier checks 1,260 accuracy totals, 1,792 paired comparisons, 630 HousingQA class summaries, all 16 writer comparison rows and the MBE answer transitions. It also checks file integrity and query coverage.

The [standalone verification ZIP](ESWC2027_review_candidate_20261118.zip) contains the same 71 files as the browsable supplement. Its download link retains the 18 November filename. The enclosed manifest identifies the 20 November revision. Extract the ZIP and run the command from its top-level directory. The revised ZIP's SHA-256 digest is `8dd815b1e7460119125d0105a284a0472135111f5e707eb59ac6b9cc1e91a04b`.
