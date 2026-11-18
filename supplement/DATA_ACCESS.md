# Data sources and access

The upstream benchmarks are by Lucia Zheng, Neel Guha, Javokhir Arifov, Sarah Zhang, Michal Skreta, Christopher D. Manning, Peter Henderson and Daniel E. Ho. See [A Reasoning-Focused Legal Retrieval Benchmark](https://doi.org/10.1145/3709025.3712219) and the [authors’ resource page](https://reglab.github.io/legal-rag-benchmarks/).

## Bar Exam QA

Use the dataset owner's [Bar Exam QA card](https://huggingface.co/datasets/reglab/barexam_qa) for the current access process and applicable conditions. Request access from the releasing team where required. The evaluation combines the released historical MBE training and test exports: 954 + 117 questions and 686,324 + 85,677 passages. The validation partition was excluded. Match the `idx` field to the released item IDs. The files in this package contain no MBE multiple-choice answer letters, answer predictions, choice text or raw exam questions.

The upstream catalogue card displays CC BY-SA 4.0 metadata and describes multiple-choice fields. Appendix C of the [benchmark paper](https://arxiv.org/html/2505.03970v1) separately describes researcher-gated distribution of the compilation of historical multiple-choice options and answers. This review edition directs readers to the releasing team for the underlying material. As a conservative packaging choice, item-dependent MBE query literals and expansion text are also withheld. The benchmark authors describe separate source terms for their passage collections. A SHA-256 fingerprint in this supplement identifies an artefact without redistributing its contents.

## HousingQA

The dataset owner's [HousingQA card](https://huggingface.co/datasets/reglab/housing_qa) describes the `questions` test split (6,853 source questions) and the `statutes` corpus split (1,837,403 passages). Use the annotated `questions` data to join on the public question ID. The historical benchmark concerns housing laws as of 2021.

The upstream HousingQA dataset is attributed to the authors named above and is available under [Creative Commons Attribution–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). Underlying question information was compiled from the [Legal Services Corporation Eviction Laws Database](https://www.lsc.gov/initiatives/effect-state-local-laws-evictions/lsc-eviction-laws-database), with statutes sourced from Justia as described on the dataset card. The HousingQA cohort mapping and any shared adaptations of HousingQA material are distributed under the same licence. None of the upstream organisations is represented as endorsing this study.

The mapping changes the selection and resolves reference identifiers by jurisdiction, citation and normalised excerpt. It retains 4,584 questions in 47 jurisdictions, excluding 2,260 unresolved-reference questions and nine conflicting-answer questions. The retained 9,973 reference occurrences identify 672 distinct corpus rows. Each record in `datasets/housing_cohort.csv` contains a public question ID, jurisdiction, question-template ID, yes/no label and its resolved statute IDs. Source questions, statute text and the original annotated passages remain available from the dataset owner. The accompanying study-generated queries and result files identify the evaluated conditions.
