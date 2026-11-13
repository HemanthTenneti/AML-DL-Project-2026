# Neuro-Symbolic Causal Inference — Corr2Cause

Semester project: extract cause–effect relations from text, assemble them into a DAG, and answer
intervention queries with a Bayesian network (pgmpy). Phase 1 = classical ML, Phase 2 = DistilBERT.
See the [Phase 1 report](report/Phase1_Report.pdf) and the [Phase 2 report](report/Phase2_Report.pdf) for the method and results.

## Layout
```
src/causal_nlp/
  parsing.py       Stage 1: text -> facts / hypothesis triple (regex rules)
  causal_rules.py  Stage 2: PC algorithm (skeleton, v-structures, Meek) + equivalent DAGs + support
  features.py      Stage 3: feature table (text facts, PC graph, support)
  graph_stage.py   Stage 5: acyclic DAG building, noisy-OR Bayesian network, do-queries
  data.py          Corr2Cause loaders (corrected labels, robustness test sets)
  dl_common.py     Phase 2: DistilBERT training loop, prediction, save / load
  dl_end2end.py    Phase 2A: end-to-end DistilBERT classifier + variable-renaming augmentation
  dl_parser.py     Phase 2B: DistilBERT hypothesis parser (drop-in for the regex parser)
  paraphrase_grammar.py  Phase 2B: grammar that writes the parser's training sentences
notebooks/
  01_phase1_ml_pipeline.ipynb   Phase 1 deliverable (executed, with outputs)
  02_phase2_dl_pipeline.ipynb   Phase 2 deliverable (executed, with outputs)
  build_phase*_notebook.py      regenerate the notebooks
tests/                          unit tests (Phase 1 + Phase 2)
results/                        metrics, test triples and report numbers saved for the Phase 2 comparison
report/
  Phase1_Report.tex / .pdf      Phase 1 report (LaTeX source + compiled PDF)
  Phase1_Presentation.pdf       Phase 1 slides (7 slides + 6 backup slides for the viva)
  Phase2_Report.tex / .pdf      Phase 2 report (DistilBERT 2A / 2B, ablations, where each reference is used)
  figures/                      figures used by the report
tools/
  make_report_figures.py        re-runs the pipeline and redraws report/figures/
  train_phase2.py               trains the three DistilBERT models into models/ (not in git)
  build_visualiser.py           builds docs/pipeline_visualiser.html (step-by-step pipeline tracer)
docs/
  pipeline_visualiser.html      open in a browser: real examples traced through every stage
  papers/                       reference papers (BERT)
```

## Setup (Python 3.11)
```bash
uv venv --python 3.11 .venv && uv pip install --python .venv -r requirements.txt
# or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Data
**Dataset: Corr2Cause** (Jin et al., ICLR 2024, "Can Large Language Models Infer Causation from Correlation?").
Each row gives a premise listing the statistical (in)dependence relations among 2-6 variables and a causal
hypothesis, labelled 1 if the hypothesis follows and 0 if it does not.

| Split | Rows | Source |
|---|---:|---|
| train | 205,734 | [zluke/corr2cause-corrected](https://huggingface.co/datasets/zluke/corr2cause-corrected) |
| dev | 1,076 | same |
| test | 1,162 | same |
| paraphrase test | 2,246 | [causal-nlp/corr2cause](https://huggingface.co/datasets/causal-nlp/corr2cause), labels re-mapped to the corrected set |
| variable-refactor test | 2,246 | same |

Columns: `input` (premise + hypothesis text), `label` (0/1), `num_variables`, `template` (relation type, e.g.
`has_collider`). The corrected release keeps the official text and fixes 6,943 labels broken by a bug in the
original generator. Splits are grouped by premise, so no premise appears in two splits.

The Corr2Cause dataset is in `data/`, so a fresh clone runs offline. The two train splits are 366 MB each,
over GitHub's 100 MB limit, so they are stored as `train.csv.gz` (about 4.4 MB) and read directly by the loader:
```
data/
  corrected/  train.csv.gz dev.csv test.csv       labels from zluke/corr2cause-corrected (used by default)
  raw/        train.csv.gz dev.csv test.csv       official causal-nlp/corr2cause (original labels)
              perturbation_by_paraphrasing_test.csv
              perturbation_by_refactorization_test.csv
```
If a file is missing, `src/causal_nlp/data.py` downloads it from Hugging Face into `data/`. See [data/README.md](data/README.md) for details.

## Run
```bash
.venv/bin/python -m pytest -q                                   # unit tests (Phase 1 + Phase 2)
.venv/bin/jupyter lab notebooks/01_phase1_ml_pipeline.ipynb     # Phase 1 notebook (~1-2 min on a laptop CPU)
.venv/bin/python tools/train_phase2.py                          # Phase 2 models (~1 h on a laptop GPU)
.venv/bin/jupyter lab notebooks/02_phase2_dl_pipeline.ipynb     # Phase 2 notebook (trains any missing model first)
.venv/bin/python tools/build_visualiser.py                      # rebuild docs/pipeline_visualiser.html
.venv/bin/python tools/make_report_figures.py                   # redraw report figures (~5 min)
cd report && pdflatex Phase1_Report.tex && pdflatex Phase1_Report.tex   # rebuild the report PDF
```
