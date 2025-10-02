# AutoXAI (Research Reproducibility Scaffold)

This repository provides a clean, minimal, and reproducible implementation of the AutoXAI recommendation framework described in the manuscript.

> Scope. The goal of this repo is to make the selection logic of AutoXAI fully reproducible:
> 1) computing dataset similarity in the joint (X, Y) space using an OT-inspired distance between class-conditional Gaussians (Bures–Wasserstein),
> 2) retrieving candidate explanation techniques evaluated on the most similar dataset in a knowledge base,
> 3) performing multi-objective selection via Pareto non-dominated sorting under user-selected metrics (M1–M8), and
> 4) returning the final recommendation with a transparent rationale.

The repository is intentionally modular so you can plug in your preferred explanation libraries (e.g., LIME, Anchors, RuleFit, RuleMatrix, SHAP). For double-blind or licensing reasons, we do not vendor third-party explainer implementations here; instead, the interfaces are provided and simple mock adapters are included for end-to-end demos.

---

## Features

- OT-inspired dataset similarity (`autoxai/similarity.py`): class-conditional Gaussian fit per label and Bures–Wasserstein distance (closed-form).
- Multi-objective selection (`autoxai/pareto.py`): fast non-dominated sorting and deterministic tie-breaking.
- Metric suite (`autoxai/metrics.py`): M1–M8 operationalization (identity, separability, correctness, entropy ratio, KL, Gini, compactness, speed).
- Recommender (`autoxai/recommender.py`): offline KB builder + online recommender as described in the paper.
- Repro scripts (`examples/`): build a small knowledge base and run recommendations with the same metric sets used in the paper’s evaluation table (randomly sampled metric sets).

Note: For clarity/reproducibility, the example experiments simulate "explanation outputs" as feature-importance vectors or rule lists returned by adapters. If you install LIME/Anchors/RuleFit/RuleMatrix, you can swap the adapters to compute real outputs.

---

## Install

```bash
pip install -e .
```

`requirements.txt` lists: numpy, pandas, scikit-learn.

---

## Reproduce the paper’s workflow

1. Build the knowledge base (offline phase):

```bash
python examples/01_build_kb.py --out data/kb.json
```

2. Run recommendations (online phase):

```bash
python examples/02_recommend.py --kb data/kb.json --dataset_name "Abalone"
```

Both scripts write trace artifacts (.json) that include:
- matched dataset,
- metric set used,
- Pareto-optimal set with metric values, and
- final choice + rationale.

---

## Project structure

```
autoxai/
  __init__.py
  similarity.py
  pareto.py
  metrics.py
  recommender.py
  adapters/
    __init__.py
    mock_explainers.py
examples/
  01_build_kb.py
  02_recommend.py
data/               # (created at runtime)
configs/            # (optional)
```

---

## License

See `LICENSE` (MIT). Please also respect the licenses of any third-party libraries you choose to integrate.

---

## Citation

If you find this repository useful, please cite:

@article{{autoxai-repo,
  title   = {{AutoXAI}: Automated Recommendation of Global Explanation Techniques},
  author  = {{Anonymous}},
  year    = {2025},
  note    = {Code: GitHub repo URL upon publication}
}
