# From legal text to queryable graphs

The 26 November 2026 supplement contains saved results, queries and verification code for the MBE and HousingQA experiments. The [supplement guide](supplement/README.md) describes the files and the scope of the checks.

| To inspect | Open |
| --- | --- |
| Differences between query writers on the same graphs | [Paired writer comparisons](supplement/revision_20261126/writer_pairs.json) |
| HousingQA results after excluding the prompt-pilot questions and rates of unparseable answers | [Pilot exclusion](supplement/revision_20261126/housing_without_pilot.json) and [unparseable rates](supplement/revision_20261126/housing_unparseable.json) |
| Raw-text control, graph ranking ties and MBE passage composition | [Paired raw-text intervals](supplement/revision_20261126/raw_text_intervals.json), [graph tie results](supplement/revision_20261126/graph_ties_results.json) and [passage composition](supplement/revision_20261126/passage_content_results.json) |
| Pooled graph comparisons against Qwen dense and BM25 | [Pooled results and intervals](supplement/revision_20261125/final_statistics.md) |
| HousingQA writers choosing different unique top statutes and the Alabama score reversal | [Saved first and second hits](supplement/housing_rq2/top_two.csv), [Alabama provisions](supplement/housing_rq2/alabama_records.json) and [offline check](supplement/housing_rq2/verify.py) |
| Answer accuracy by model, stage and retrieval depth | [Full results](supplement/statistics/all_fixed_stage_accuracy.csv) |
| Graph comparisons against BM25 and Qwen dense at each setting | [Paired differences and Holm adjustments](supplement/statistics/paired_KG_vs_baselines.csv) |
| How the generated queries use graph relationships | [Query census](supplement/queries/relationship_census.csv) and [query coverage](supplement/queries/coverage.json) |
| Cohorts, source data and MBE access terms | [Data access](supplement/DATA_ACCESS.md) |

## Verify the results offline

Download and extract the repository ZIP, then open the `supplement` directory and run this command in a Python environment with NumPy installed.

```sh
python verify_bundle.py
```

The focused HousingQA check runs with `python housing_rq2/verify.py` and needs only the Python standard library.

The [standalone verification ZIP](ESWC2027_review_candidate_20261118.zip) contains the same 85 files as the browsable supplement. The download link retains the original filename, and the enclosed manifest identifies the 26 November revision. Extract the ZIP and run the same command from its top-level directory. The ZIP's SHA-256 digest is `da24fb1456d43beef73989ef54a4a3746a270f042b76c2d80c5ba0d00f2ec22d`.
