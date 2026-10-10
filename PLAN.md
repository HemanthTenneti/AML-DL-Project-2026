# Project plan — Neuro-Symbolic Causal Inference (Corr2Cause)

## Core idea

Split the problem into three layers and keep the interfaces fixed, so Phase 2 only swaps one layer:

```
TEXT ──[1. language layer]──▶ facts + hypothesis triple ──[2. reasoning layer]──▶ valid triples ──[3. graph layer]──▶ DAG / BN / do-queries
       Phase 1: regex rules         (relation, x, y)        PC rules + support         (premise_id, relation,      networkx + pgmpy
       Phase 2: DistilBERT                                  + decision tree             x, y, prob)
```

| Interface | Producer | Consumer |
|---|---|---|
| `hypothesis_parser(text) -> Hypothesis(relation, x, y) or None` | `parsing.parse_hypothesis` (P1) / DistilBERT parser (P2) | `features.build_feature_table(inputs, hypothesis_parser=...)` |
| triples table `premise_id, relation, x, y, prob, label` | any extractor | `graph_stage.*` |

Evaluation is identical for every system: same splits, same metrics, same code.

| Level | Metric | Data |
|---|---|---|
| Relation extraction | precision / recall / **F1 of the valid class** | test (+ dev, refactorization, paraphrase) |
| Graph | edge precision / recall vs gold direct-cause edges | 21 test premises |
| Causal queries | accuracy on known do-queries | 242 queries over the test premises |

## Phase 1 — ML (done)

Notebook: `notebooks/01_phase1_ml_pipeline.ipynb`

| Step | Method | Why it is explainable |
|---|---|---|
| Data | Corr2Cause with corrected labels; official premise-grouped split | 0 premises shared between train and test |
| 1. Extraction | regular expressions → `correlates(X,Y)`, `independent(X,Y | S)`, hypothesis triple | one pattern per sentence template |
| 2. Reasoning | PC algorithm: skeleton → v-structures → Meek rules → PDAG; enumerate equivalent DAGs | textbook causal discovery (Pearl / Spirtes) |
| 3. Features | 4 relation + 8 text-fact + 9 PC-graph + 2 support features | each feature is one sentence to explain |
| 4. Classifier | Logistic Regression, **Decision Tree (depth 4)**, Random Forest | final tree is printable |
| 5. Graph | greedy acyclic DAG, noisy-OR Bayesian network in pgmpy, do-queries, MLE fit demo | standard BN workflow |

Phase 1 results (F1 of the valid class):

| System | dev | test | refactorization | paraphrase | edge P / R | query acc. |
|---|---|---|---|---|---|---|
| TF-IDF + LR (text only) | 0.61 | 0.65 | 0.32 | 0.25 | – | – |
| Rules + PC + Decision Tree | 1.00 | 1.00 | 1.00 | **0.00** | 1.00 / 1.00 | 0.975 |

Key findings to carry into Phase 2:
* Reasoning, not wording, drives performance (ablation: text facts ≈ 0.5 → + PC graph ≈ 0.93 → + support ≈ 1.0).
* The text-only model learns a **shortcut** (variables are named in causal order) and collapses on renamed variables.
* The rule parser reads **0 %** of paraphrased hypotheses → the whole system scores 0 there. This is the gap for DL.
* The 2.5 % query misses are causal chains through edges whose direction correlations cannot identify (not an extraction error).

## Phase 2 — DL (planned)

Two transformer systems, each paired with its Phase 1 counterpart:

| | text only (no reasoning layer) | language layer + reasoning layer (neuro-symbolic) |
|---|---|---|
| ML (Phase 1) | TF-IDF + LR | Rules + PC + Decision Tree |
| DL (Phase 2) | **2A: DistilBERT end-to-end** | **2B: DistilBERT parser + PC + Decision Tree** |

### 2A. DistilBERT end-to-end classifier (fair rival of TF-IDF)
* Model: `distilbert-base-uncased` (or `typeform/distilbert-base-uncased-mnli`), sequence-pair classification.
* Input: `[CLS] hypothesis [SEP] premise [SEP]`, `max_length=512`, `truncation="only_second"`.
  About 25 % of training inputs exceed 512 tokens; putting the hypothesis first means only the tail of
  long premises is cut. Report how many test inputs are truncated (test max ≈ 404 words, so few).
* Training data: premise-grouped subsample of train (≈ 40k rows, stratified by relation and label),
  2–3 epochs, lr 2e-5, batch 16, weight decay 0.01, early stopping on dev F1. Free Colab T4: about 1 hour.
* Class imbalance (≈ 21 % valid): weighted loss or threshold tuned on dev.
* One simple, explainable fix to test: **variable-renaming augmentation** (randomly permute letters in
  training examples) to remove the "causal order = alphabetical order" shortcut. Compare with and without.
* Expected: clearly better than TF-IDF in-distribution; drop on refactorization without augmentation
  (the Corr2Cause paper reports this); weak on paraphrase.

### 2B. DistilBERT as the language layer (the neuro-symbolic system)
* Task: read the hypothesis and output its relation type (direct / indirect / collider / confounder / none).
  Variables `x`, `y` = the capital-letter tokens in order of appearance (same as the regex parser).
* Training data: hypotheses from the train split (the original templates) **plus our own hand-written
  paraphrases** (≈ 10 per relation, e.g. "X has a direct influence on Y"). The official paraphrase test
  wordings must **not** be used for training — they stay a held-out test.
* Plug-in: `build_feature_table(inputs, hypothesis_parser=distilbert_parser)` — nothing downstream changes.
* Report: (i) relation-type accuracy on paraphrased hypotheses, (ii) end-to-end F1 on all four sets,
  (iii) edge precision / query accuracy via `graph_stage`.
* Expected: in-distribution ≈ Phase 1 (the reasoning layer is identical); paraphrase F1 rises from 0 to
  roughly the parser's accuracy; refactorization stays robust because features ignore names.
* "Feed only reviewed predictions into the graph stage": triples go to the graph only if `prob ≥ 0.5`;
  optionally also show a stricter threshold (0.9) and the precision/recall trade-off.

### Final comparison table (Phase 2 deliverable)
Rows: TF-IDF + LR, Rules + PC + DT, DistilBERT end-to-end (± renaming augmentation), DistilBERT parser + PC + DT.
Columns: dev / test / refactorization / paraphrase F1, edge precision / recall, intervention-query accuracy,
training time, number of parameters.

### Phase 2 to-do
1. `src/causal_nlp/dl.py` — dataset class, tokenisation, train / predict helpers (Hugging Face `Trainer`).
2. `notebooks/02_phase2_dl_pipeline.ipynb` — 2A, 2B, final comparison; load `results/phase1_metrics.csv`.
3. Write the paraphrase augmentation list (kept separate from the test wordings).
4. Error analysis: 20 wrong predictions per DL system, grouped by relation and by number of variables.
5. Optional only if time remains: PubMed abstracts with spaCy (domain extension mentioned in the brief).

## Viva notes (likely questions)
* *Why Corr2Cause and not raw news text?* Labelled cause/effect reasoning with a known answer, so extraction
  and graph quality can be measured exactly; PubMed is an optional extension.
* *Why corrected labels?* Our own PC reasoner disagreed with ~40 % of training premises on the original labels;
  a published bug fix (6,943 labels) brought agreement to 99.98 %. We report both label versions.
* *Why a decision tree if support already decides?* The tree **learns** the decision threshold
  ("valid only if true in every equivalent DAG") and a dataset quirk for confounders from data; it stays
  readable and gives probabilities for the graph stage.
* *Why noisy-OR probabilities?* Corr2Cause gives structure, not measurements. The CPDs are illustrative;
  section 5.3 shows that with data pgmpy learns them (fitted ACE 0.51 vs reference 0.49).
* *Observational vs interventional?* Example in section 5.2: P(F|E) difference 0.73 vs do-effect 0.49 because
  E and F share common causes.
