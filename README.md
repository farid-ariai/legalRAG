# From legal text to queryable graphs

These review materials contain the saved results, queries and verification code for the MBE and HousingQA experiments. The [supplement guide](supplement/README.md) describes the files and the scope of the checks.

| To inspect | Open |
| --- | --- |
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

The verifier checks 1,260 accuracy totals, 1,792 paired comparisons and 630 HousingQA class summaries, along with file integrity and query coverage. It uses the saved results and requires no model calls.

The [standalone verification ZIP](ESWC2027_review_candidate_20261118.zip) contains the same 69 supplement files. If you download this file directly, extract it and run the command from its top-level directory. Its SHA-256 digest is `1e64b363419e16b6f84c77ac4dc26361357008d6920d825b1ea72e0a527771f4`.
