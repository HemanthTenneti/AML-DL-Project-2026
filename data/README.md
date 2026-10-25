# Corr2Cause data

Corr2Cause (Jin et al., ICLR 2024, "Can Large Language Models Infer Causation from Correlation?") tests
whether a model can derive causal claims from correlational statements. Each row has a premise listing the
statistical (in)dependence relations in a closed system of 2-6 variables, and a causal hypothesis about two
of them. The label is 1 if the hypothesis follows from the premise and 0 if it does not.

## Folders

| Folder | Source | Used for |
|---|---|---|
| `corrected/` | [zluke/corr2cause-corrected](https://huggingface.co/datasets/zluke/corr2cause-corrected) | train, dev and test (default) |
| `raw/` | [causal-nlp/corr2cause](https://huggingface.co/datasets/causal-nlp/corr2cause) | original labels and the two robustness test sets |

The corrected release keeps the official text and fixes 6,943 labels broken by a bug in the original
generator. Splits are grouped by premise, so no premise appears in two splits.

## Files

| File | Rows | Notes |
|---|---:|---|
| `corrected/train.csv.gz` | 205,734 | gzipped (366 MB uncompressed, over GitHub's 100 MB limit) |
| `corrected/dev.csv` | 1,076 | |
| `corrected/test.csv` | 1,162 | |
| `raw/train.csv.gz`, `raw/dev.csv`, `raw/test.csv` | same as above | original labels, for comparison with the corrected set |
| `raw/perturbation_by_paraphrasing_test.csv` | 2,246 | test hypotheses re-worded (e.g. "directly causes" -> "directly affects") |
| `raw/perturbation_by_refactorization_test.csv` | 2,246 | test problems with variables renamed (e.g. A, B, C -> Z, Y, X) |

The split files have the columns `input` (premise and hypothesis text), `label` (0/1), `num_variables` and
`template` (relation type, e.g. `has_collider`). The paraphrasing file also has separate `premise`, `hypothesis`
and `relation` columns. `src/causal_nlp/data.py` maps both robustness sets onto the corrected labels.

## Loading

```python
from causal_nlp.data import load_split, load_paraphrase_test, load_refactorization_test

train = load_split("train")                    # corrected labels
test_original = load_split("test", corrected=False)
```

pandas reads the `.gz` files directly. If a file is missing, the loader downloads it from Hugging Face.
