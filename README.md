# Neuro-Symbolic Causal Inference — Corr2Cause

Semester project: extract cause–effect relations from text, assemble them into a DAG, and answer
intervention queries with a Bayesian network (pgmpy). Phase 1 = classical ML, Phase 2 = DistilBERT.
See [PLAN.md](PLAN.md) for the full plan, results and viva notes.

## Layout
```
src/causal_nlp/
  parsing.py       Stage 1: text -> facts / hypothesis triple (regex rules)
  causal_rules.py  Stage 2: PC algorithm (skeleton, v-structures, Meek) + equivalent DAGs + support
  features.py      Stage 3: feature table (text facts, PC graph, support)
  graph_stage.py   Stage 5: acyclic DAG building, noisy-OR Bayesian network, do-queries
  data.py          Corr2Cause loaders (corrected labels, robustness test sets)
notebooks/
  01_phase1_ml_pipeline.ipynb   Phase 1 deliverable (executed, with outputs)
  build_phase1_notebook.py      regenerates the notebook
tests/test_pipeline.py          unit tests
results/                        metrics + test triples saved for the Phase 2 comparison
tools/build_visualiser.py       builds docs/pipeline_visualiser.html (step-by-step pipeline tracer)
docs/pipeline_visualiser.html   open in a browser: real examples traced through every stage
```

## Setup (Python 3.11)
```bash
uv venv --python 3.11 .venv && uv pip install --python .venv -r requirements.txt
# or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Data
The code expects this folder (it is **not** in git: the train CSVs are 366 MB each, over GitHub's 100 MB limit):
```
data/
  corrected/  train.csv dev.csv test.csv          labels from zluke/corr2cause-corrected (used by default)
  raw/        train.csv dev.csv test.csv          official causal-nlp/corr2cause (original labels)
              perturbation_by_paraphrasing_test.csv
              perturbation_by_refactorization_test.csv
```
Two ways to get it:
1. **Automatic**: run the tests or the notebook with internet access; `src/causal_nlp/data.py` downloads any
   missing file from Hugging Face into `data/`.
2. **Offline**: unzip the shared `DLProject_2026_data.zip` in the project root so the `data/` folder sits next to `src/`.

## Run
```bash
.venv/bin/python -m pytest -q                                   # 14 unit tests
.venv/bin/jupyter lab notebooks/01_phase1_ml_pipeline.ipynb     # Phase 1 notebook (~1-2 min on a laptop CPU)
.venv/bin/python tools/build_visualiser.py                      # rebuild docs/pipeline_visualiser.html
```
