"""Reproduce all paired statistics locally from the released saved outcomes."""

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    start = time.monotonic()
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, name
    contract = json.loads((HERE / "contract.json").read_text())
    arrays = np.load(HERE / "validated_inputs.npz", allow_pickle=False)
    with (HERE / "paired_KG_vs_baselines.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    checked = 0
    for dataset in ["mbe", "housing"]:
        columns = [json.loads(str(item)) for item in arrays[dataset + "_columns"]]
        lookup = {(c["model"], c["method"], c["extractor"], c["stage"], c["k"]): i for i, c in enumerate(columns)}
        answers = arrays[dataset + "_values"]
        family = [r for r in rows if r["dataset"] == dataset]
        differences = []
        for row in family:
            stage = "parent" if row["stage"] == "initial" else row["stage"]
            graph = answers[:, lookup[(row["model"], row["method"], row["extractor"], stage, int(row["k"]))]].astype(np.float64)
            baseline = answers[:, lookup[(row["model"], row["comparator"], "not_applicable", stage, int(row["k"]))]].astype(np.float64)
            assert int(graph.sum()) == int(row["KG_correct"])
            assert int(baseline.sum()) == int(row["baseline_correct"])
            assert abs(100 * (graph - baseline).mean() - float(row["effect_pp"])) < 1e-12
            differences.append(graph - baseline)
        differences = np.column_stack(differences)
        groups, inverse = np.unique(arrays[dataset + "_groups"], return_inverse=True)
        counts = np.bincount(inverse).astype(float)
        totals = np.zeros((len(groups), len(family)))
        np.add.at(totals, inverse, differences)
        rng = np.random.default_rng(np.random.SeedSequence([contract["seed"], contract["dataset_streams"][dataset]]))
        bootstrap = np.empty((100000, len(family)))
        for first in range(0, 100000, 500):
            weights = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=500)
            bootstrap[first:first + 500] = (weights @ totals) / (weights @ counts)[:, None]
        lower, upper = np.quantile(bootstrap, [.025, .975], axis=0)
        del bootstrap
        extremes = np.zeros(len(family), dtype=np.int64)
        observed = np.abs(differences.sum(axis=0))
        for _ in range(400):
            signs = rng.integers(0, 2, size=(500, len(groups)), dtype=np.int8).astype(float) * 2 - 1
            extremes += (np.abs(signs @ totals) >= observed[None, :] - 1e-12).sum(axis=0)
        for index, row in enumerate(family):
            assert abs((extremes[index] + 1) / 200001 - float(row["p_raw"])) < 1e-12
            assert abs(lower[index] * 100 - float(row["ci95_low_pp"])) < 1e-5
            assert abs(upper[index] * 100 - float(row["ci95_high_pp"])) < 1e-5
        for subset, column in [(family, "p_holm_896"),
                               ([r for r in family if r["comparator"] == "bm25"], "p_holm_448"),
                               ([r for r in family if r["comparator"] == "qwen3_dense"], "p_holm_448")]:
            running = 0.0
            for rank, row in enumerate(sorted(subset, key=lambda r: float(r["p_raw"]))):
                running = min(1.0, max(running, (len(subset) - rank) * float(row["p_raw"])))
                assert abs(running - float(row[column])) < 1e-12
        checked += len(family)
        print(json.dumps({"dataset": dataset, "verified_contrasts": len(family)}), flush=True)
    print(json.dumps({"status": "PASS", "verified_contrasts": checked,
                      "seconds": time.monotonic() - start, "new_model_calls": 0,
                      "new_retrieval_runs": 0, "numpy": np.__version__}))


if __name__ == "__main__":
    main()
