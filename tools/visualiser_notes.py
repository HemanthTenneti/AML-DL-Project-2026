"""Hand-written explanations for the pipeline visualiser.

NOTES[snippet_id]["lines"] is a list of (substring, note). Each note attaches to the next
line (after the previous match) that contains the substring, so the code itself is never copied
by hand - it is always read from the real files.
"""

# --------------------------------------------------------------------------- stages
STAGE_TEXT = {
    "load": {
        "name": "Load the row",
        "plain": "Think of the dataset as a big quiz book. Each question has a short story (the premise), "
                 "a claim about the story (the hypothesis), and an answer key (the label: 1 = the claim must be true, "
                 "0 = the story does not prove it). Step one is simply opening the book at one question.",
        "technical": "Each CSV row has `input` = 'Premise: … Hypothesis: …' plus `label`. load_split() downloads the "
                     "corrected-label CSV, splits `input` into premise and hypothesis columns, and gives every distinct "
                     "premise an integer premise_id so later stages can group hypotheses by premise.",
        "why": "Splitting by premise matters: one premise appears with up to 90 hypotheses, so if the same premise were in "
               "train and test the model could memorise it (text leakage). The official split keeps test premises unseen.",
    },
    "parse": {
        "name": "Read the sentences",
        "plain": "A computer cannot reason with English sentences, so we turn each sentence into a tidy note. "
                 "'A correlates with C' becomes the note correlated(A, C). It is like a clerk copying a messy letter into a form. "
                 "The clerk only knows a few fixed sentence shapes; a new wording leaves the form empty.",
        "technical": "Regular expressions match each sentence template: _HEADER (variables), _CORR (correlation), "
                     "_MARG_INDEP (independent with no condition, sepset {}), _COND_INDEP (independent given S, sepset S). "
                     "The hypothesis is matched against 5 patterns and becomes Hypothesis(relation, x, y), or None.",
        "why": "This is the language layer. Everything after it works on facts only, so in Phase 2 we can replace just this "
               "step with DistilBERT and keep the rest identical, which makes the ML vs DL comparison fair.",
    },
    "skeleton": {
        "name": "PC step 1: skeleton",
        "plain": "Draw a line between two variables if nothing can ever 'explain away' their link. If A and B look related "
                 "only because of something else (like ice cream and drowning, both caused by hot weather), then once you "
                 "account for that something, the link disappears, so there is no direct line between them.",
        "technical": "X – Y is kept iff no conditioning set S makes X ⊥ Y | S, i.e. the pair has no separating set (sepset). "
                     "Under the faithfulness assumption, a pair that is never independent must be directly causally linked.",
        "why": "This finds WHO is directly connected to whom, but not the direction of the arrows yet.",
    },
    "vstruct": {
        "name": "PC step 2: v-structures",
        "plain": "Talent and luck are unrelated, but both cause success. Among successful people, if someone was not lucky "
                 "they were probably talented: knowing the common effect links its causes. So if X and Y are unrelated, "
                 "both touch Z, and Z is NOT one of the things that separates them, Z must be their common effect: X → Z ← Y.",
        "technical": "For every non-adjacent pair X, Y with a common neighbour Z: if Z is in none of X, Y's sepsets, orient "
                     "X → Z ← Y (an unshielded collider / v-structure). If Z IS in a sepset, Z is a chain or fork node, "
                     "which conditioning blocks, so it cannot be a collider.",
        "why": "Colliders are the only pattern that correlations alone can point at directly. They give the first arrow directions.",
    },
    "meek": {
        "name": "PC step 3: Meek rules",
        "plain": "Once some arrows are known, others follow by logic, like Sudoku: if one direction would create a pattern "
                 "the facts rule out (a new collider, or a loop), the other direction must be right.",
        "technical": "R1: W → U – V with W, V not adjacent ⇒ U → V (else a new collider at U). "
                     "R2: U → W → V and U – V ⇒ U → V (else a cycle). "
                     "R3: U – W1 → V, U – W2 → V, W1, W2 not adjacent ⇒ U → V. Repeat until nothing changes.",
        "why": "The output is a PDAG: arrows the data proves, dashed lines the data cannot decide.",
    },
    "dags": {
        "name": "Possible worlds",
        "plain": "Dashed lines could point either way. We list every way to point them that still fits the facts (no loops, "
                 "no extra colliders). Each one is a 'possible world'. A claim is proven only if it is true in every possible "
                 "world. The share of worlds where it is true is called support.",
        "technical": "Enumerate all 2^k orientations of the k undirected edges; keep those that are acyclic and create no new "
                     "v-structure. These are the Markov-equivalent DAGs (same independences). support = fraction of them in "
                     "which relation_holds(dag, relation, x, y).",
        "why": "Corr2Cause labels a claim 1 only if it holds in all equivalent DAGs. Support turns that rule into one number.",
    },
    "features": {
        "name": "Turn it into numbers",
        "plain": "A machine-learning model reads a row of numbers, not graphs. So we describe the claim with about two dozen "
                 "numbers: what kind of claim it is, what the sentences said about the pair, what the graph shows, "
                 "and the support score.",
        "technical": "pair_features() builds 4 relation one-hot + 8 text-fact + 9 PC-graph + 2 support features. None uses "
                     "variable names, so renaming A→Z cannot change them.",
        "why": "Splitting features into groups lets the notebook show how much each reasoning step adds (ablation).",
    },
    "tree": {
        "name": "Decision tree decides",
        "plain": "A decision tree is a flowchart of yes/no questions learned from 205,734 solved examples. It found that one "
                 "question does almost all the work: 'is the claim true in (nearly) every possible world?'",
        "technical": "DecisionTreeClassifier(max_depth=4) trained on ALL_FEATURES. Each internal node tests feature ≤ threshold; "
                     "the leaf's share of valid training rows is P(valid). Prediction = P(valid) ≥ 0.5.",
        "why": "We never hard-coded 'support must be 1'; the tree learned that rule from data, and it stays printable for the viva.",
    },
    "graph": {
        "name": "Graph, network, what-if",
        "plain": "Take every 'X directly causes Y' claim the tree accepted and draw it as an arrow. Then give the drawing some "
                 "probabilities so we can ask what-if questions: if we FORCE X on, how much does Y change? That differs from "
                 "just SEEING X on, because seeing can also hint at X's hidden causes.",
        "technical": "build_dag() adds edges by confidence and skips cycles. noisy_or_network() attaches CPDs "
                     "P(child=1 | k active parents) = 1 − 0.9 · 0.3^k. CausalInference answers P(Y | do(X)); "
                     "ACE = P(Y=1|do(X=1)) − P(Y=1|do(X=0)). Observational effect uses ordinary conditioning.",
        "why": "This is what text extraction alone cannot give: a guaranteed acyclic graph, probabilities, and intervention answers.",
    },
}

# ------------------------------------------------------------------- stage → code shown
STAGE_CODE = {
    "load": ["nb:3", "py:data.load_split", "py:data._with_columns", "py:parsing.split_input", "py:data._download"],
    "parse": ["nb:9", "blk:parsing:DIRECT, INDIRECT, COLLIDER, CONFOUNDER =|VAR = r",
              "blk:parsing:_HEADER = re.compile|_COND_INDEP = re.compile", "py:parsing.parse_premise",
              "py:parsing._split_var_list", "blk:parsing:HYPOTHESIS_PATTERNS = [|]", "py:parsing.parse_hypothesis",
              "py:parsing.Premise", "nb:11"],
    "skeleton": ["py:causal_rules.skeleton", "py:causal_rules.PDAG"],
    "vstruct": ["py:causal_rules.orient_v_structures"],
    "meek": ["py:causal_rules.apply_meek_rules", "py:causal_rules._forced", "py:causal_rules.pc_from_premise"],
    "dags": ["py:causal_rules.equivalent_dags", "py:causal_rules._v_structures", "py:causal_rules.relation_holds",
             "py:causal_rules._reachable_avoiding", "py:causal_rules.support", "nb:14"],
    "features": ["nb:16", "py:features.build_feature_table", "py:features.analyse_premise", "py:features.pair_features"],
    "tree": ["nb:20", "nb:22", "nb:25"],
    "graph": ["nb:29", "py:graph_stage.dags_from_triples", "py:graph_stage.build_dag", "nb:31",
              "py:graph_stage.edge_metrics", "py:graph_stage.noisy_or_network", "py:graph_stage.interventional_effect",
              "py:graph_stage.observational_effect", "nb:34", "py:graph_stage.known_intervention_queries",
              "py:graph_stage.answer_queries", "nb:37", "nb:39"],
}

# ---------------------------------------------------------------------- feature meaning
FEATURE_MEANING = {
    "rel_direct": "1 if the claim is 'X directly causes Y'",
    "rel_indirect": "1 if the claim is 'X causes Y through something else'",
    "rel_collider": "1 if the claim is 'X and Y have a common effect'",
    "rel_confounder": "1 if the claim is 'X and Y have a common cause'",
    "n_vars": "how many variables the story has (2 to 6)",
    "marginally_correlated": "1 if the text says X correlates with Y",
    "adjacent": "1 if X and Y are joined in the skeleton (never independent)",
    "n_sepsets": "how many different sets make X and Y independent",
    "min_sepset_size": "size of the smallest such set (-1 if there is none)",
    "degree_x": "how many variables X is directly joined to",
    "degree_y": "how many variables Y is directly joined to",
    "n_common_neighbours": "how many variables are joined to both X and Y",
    "edge_x_to_y": "1 if the graph has a proven arrow X → Y",
    "edge_y_to_x": "1 if the graph has a proven arrow Y → X",
    "edge_undirected": "1 if X – Y is joined but the direction is unknown",
    "x_ancestor_of_y": "1 if proven arrows lead from X to Y",
    "y_ancestor_of_x": "1 if proven arrows lead from Y to X",
    "x_possible_ancestor_of_y": "1 if a path X to Y exists when dashed lines may point either way",
    "y_possible_ancestor_of_x": "1 if a path Y to X exists when dashed lines may point either way",
    "n_common_children": "how many variables both X and Y point to (proven arrows)",
    "n_common_parents": "how many variables point to both X and Y (proven arrows)",
    "support": "share of possible worlds in which the claim is true (1.0 = proven)",
    "n_equivalent_dags": "how many possible worlds fit the facts",
}

# ------------------------------------------------------------------------- line notes
NOTES = {
    # ===================================================================== notebook
    "nb:1": {"summary": "Setup: tell Python where our code lives and load every library the notebook uses.", "lines": [
        ("import sys, time, warnings", "Built-in helpers: sys (Python's settings), time (stopwatch), warnings (to silence noisy messages)."),
        ("from pathlib import Path", "Path = a handy object for file and folder paths."),
        ("warnings.filterwarnings", "Hide library warnings so the output stays readable."),
        ("ROOT = ", "Find the project folder, whether the notebook is started from /notebooks or from the project root."),
        ("sys.path.insert", "Let Python import our own package from src/causal_nlp."),
        ("import numpy as np", "numpy = fast arrays, pandas = tables, networkx = graphs, matplotlib = plots."),
        ("TfidfVectorizer", "Turns text into word-frequency numbers (used by the text-only baseline)."),
        ("ColumnTransformer", "Applies a different transformer to different table columns (premise vs hypothesis)."),
        ("make_pipeline", "Chains steps (transform → model) into one object with .fit and .predict."),
        ("LogisticRegression", "A linear classifier: weighted sum of features → probability."),
        ("StandardScaler", "Rescales each feature to mean 0, spread 1 (helps logistic regression)."),
        ("DecisionTreeClassifier", "Flowchart-style classifier; export_text prints its rules."),
        ("RandomForestClassifier", "Many decision trees voting together."),
        ("precision_score", "Scoring functions: precision, recall, F1, accuracy, and a confusion-matrix plot."),
        ("from causal_nlp.data", "Our loaders for the dataset and its two robustness test sets."),
        ("from causal_nlp.parsing", "Stage 1: the regex reader."),
        ("from causal_nlp.causal_rules", "Stage 2: the PC algorithm, possible worlds and support."),
        ("from causal_nlp.features", "Stage 3: builds the feature table and the lists of feature names per group."),
        ("from causal_nlp import graph_stage", "Stage 5: DAG building, Bayesian network and what-if queries."),
        ("SEED = 42", "Fixed random seed so every run gives the same models and numbers."),
        ("RESULTS = ", "Folder where results are saved for the Phase 2 comparison."),
        ("pd.set_option", "Show longer text in table cells."),
        ("plt.rcParams", "Plot style: sharper figures, no top/right box lines."),
    ]},
    "nb:3": {"summary": "Load all five datasets and print a size summary of each.", "lines": [
        ("data = {", "A dictionary: name → table. Each value is a pandas DataFrame, one row per (premise, hypothesis)."),
        ('"train": load_split("train")', "205,734 training rows (with corrected labels)."),
        ('"dev": load_split("dev")', "Small development set, used only as a second check."),
        ('"test": load_split("test")', "The held-out test set: 21 premises, 1,162 hypotheses. Headline numbers come from here."),
        ('"refactorization"', "Same questions with variables renamed (A, B, C → Z, Y, X). Tests if a model relies on names."),
        ('"paraphrase"', "Same questions with hypotheses re-worded. Tests if a model relies on exact wording."),
        ("pd.DataFrame({name:", "Build a summary table: for each set, number of rows, number of distinct premises,"),
        ('"% valid"', "and the percentage of claims that are valid (label 1)."),
        ("for name, df in data.items()}).T", ".T flips the table so each dataset is one row."),
    ]},
    "nb:5": {"summary": "Leakage check: make sure test premises never appear in training.", "lines": [
        ("premise_sets =", "For train, dev, test: the set of distinct premise texts."),
        ('print("train ∩ test', "∩ = overlap. 0 shared premises means the test is truly unseen."),
        ('print("train ∩ dev', "Dev shares 2 premises with train, which is why dev is only a secondary check."),
        ('print("Variables per premise in train', "How many rows have 2, 3, 4, 5, 6 variables. Train is mostly 6-variable."),
        ('print("Variables per premise in test', "Test covers 2 to 6 variables."),
    ]},
    "nb:6": {"summary": "Print one small example premise with three of its hypotheses and labels.", "lines": [
        ("example = ", "Pick test rows with 3 variables, then rows 0, 3 and 5 of those."),
        ("for _, row in example.iterrows()", "Loop over the three chosen rows."),
        ('print(f"[label=', "Show each hypothesis with its gold label."),
        ('print("\\nPremise:', "Show the premise they all share."),
    ]},
    "nb:7": {"summary": "Count how many labels the published bug-fix changed.", "lines": [
        ("for split in", "For each of the three official splits:"),
        ("original = load_split(split, corrected=False)", "load the ORIGINAL (buggy) labels,"),
        ("print(f\"{split:5s}", "and count rows where original label ≠ corrected label."),
    ]},
    "nb:9": {"summary": "Stage 1 demo: pick a teaching premise and show the facts the regex reader extracts from it.", "lines": [
        ("def mixed(p):", "Helper: does this premise make a graph with BOTH proven arrows and undecided (dashed) edges?"),
        ("g = pc_from_premise(parse_premise(p))", "Read the premise, then run the PC algorithm on it."),
        ("return len(g.nodes) <= 5", "Keep it small (≤ 5 variables) and require both edge types."),
        ("text = next(", "Take the first test premise that passes the check."),
        ("facts = parse_premise(text)", "Stage 1: sentences → Premise object (variables, correlated pairs, sepsets)."),
        ('print("variables', "The variable names from the header sentence."),
        ('print("correlated', "Pairs the text says are correlated, written as 'AC' etc."),
        ('print("independent', "For each pair: every set of variables that makes the pair independent ({} = no condition)."),
        ('print("\\nhypothesis', "Parse one example hypothesis into Hypothesis(relation, x, y)."),
    ]},
    "nb:11": {"summary": "How much text can the regex reader understand in each test set?", "lines": [
        ("coverage = {name:", "For each set: share of hypotheses where parse_hypothesis returns something (not None)."),
        ('if name != "train"', "Skip train (it is all template text anyway)."),
        ("pd.Series(coverage", "Show as percentages. Paraphrase = 0%: the reader knows none of the new wordings."),
    ]},
    "nb:14": {"summary": "Stage 2 demo: draw the PDAG for the teaching premise and compute support for a few claims.", "lines": [
        ("def draw_pdag(pdag, ax, title):", "Helper that draws a PDAG: solid arrows = proven, dashed lines = undecided."),
        ("g = nx.DiGraph()", "Empty directed graph holding just the nodes, so networkx can place them."),
        ("pos = nx.circular_layout", "Put the variables on a circle (sorted, so A is always in the same place)."),
        ("draw_networkx_nodes", "Draw the circles."),
        ("draw_networkx_labels", "Write the letters inside."),
        ("edgelist=list(pdag.directed)", "Draw proven arrows."),
        ("g.to_undirected()", "Draw the undecided edges as grey dashed lines (no arrow heads)."),
        ("ax.set_title(title); ax.axis", "Add the title and hide the axes."),
        ("pdag = pc_from_premise(facts)", "Run PC (skeleton → v-structures → Meek) on the facts from cell 9."),
        ("dags = equivalent_dags(pdag)", "List every possible world (DAG) the PDAG stands for."),
        ("fig, ax = plt.subplots", "Make a blank figure."),
        ("draw_pdag(pdag, ax, f\"PDAG", "Draw it, with the number of possible worlds in the title."),
        ("(a, b), (c, d) =", "Pick one proven arrow (a → b) and one dashed edge (c – d) to test claims on."),
        ("for rel, x, y in", "Four claims: the proven arrow, both directions of the dashed edge, and an indirect claim."),
        ('print(f"support(', "support = share of possible worlds where the claim holds. 1.00 means proven."),
    ]},
    "nb:16": {"summary": "Stage 3: turn every row of every dataset into a row of numbers (features).", "lines": [
        ("t0 = time.time()", "Start a stopwatch."),
        ("features = {name: build_feature_table", "For each dataset, build the feature table (premises are cached, so this takes ~11 s for 212k rows)."),
        ('print(f"Built features', "Report how many rows and how long it took."),
        ('features["test"][', "Peek at a few columns of the first six test rows."),
    ]},
    "nb:18": {"summary": "Text-only baseline: TF-IDF word statistics + logistic regression, with no reasoning at all.", "lines": [
        ("def scores(y_true, y_pred):", "Helper returning the four metrics for the 'valid' class."),
        ('"precision":', "Precision: of the claims we called valid, how many really are."),
        ('"recall":', "Recall: of the truly valid claims, how many we found."),
        ('"f1":', "F1: one number balancing precision and recall (their harmonic mean)."),
        ('"accuracy":', "Accuracy: share of all rows predicted correctly (misleading here: 83% of rows are 0)."),
        ("EVAL_SETS =", "The four sets we always report."),
        ("token = r", "Count one-letter words too; the default pattern would throw away the variable names A, B, C."),
        ("tfidf_lr = make_pipeline(", "Build the baseline model as a pipeline:"),
        ('("premise", TfidfVectorizer', "Premise text → TF-IDF numbers for words and 1-3 word phrases (ignore rare ones, keep top 50k)."),
        ('("hypothesis", TfidfVectorizer', "Hypothesis text → its own separate TF-IDF numbers."),
        ('LogisticRegression(max_iter=2000, class_weight="balanced")', "Linear classifier; 'balanced' makes the rare valid class count more."),
        ("tfidf_lr.fit(", "Train on all 205k training rows (~40 s)."),
        ("tfidf_results =", "Predict and score each evaluation set."),
        ('print(f"trained in', "Report training time."),
        ("pd.DataFrame(tfidf_results)", "Show the score table. Watch the drop on refactorization: the model learned letter names."),
    ]},
    "nb:20": {"summary": "Stage 4: train 3 models × 3 feature groups (the ablation) and score each.", "lines": [
        ("def predict(model, feats, cols):", "Helper: predict for rows the reader understood; rows it could not read get 0 (not valid)."),
        ("pred = np.zeros(", "Start with all zeros."),
        ("ok = (feats.parsed == 1)", "Mask of rows whose hypothesis was parsed."),
        ("pred[ok] = model.predict", "Fill in model predictions for those rows only."),
        ("feature_groups = {", "Three feature sets, each adding one reasoning step:"),
        ('"relation + text facts"', "only what the sentences say,"),
        ('"+ PC graph"', "plus what the PC graph shows,"),
        ('"+ support (all)"', "plus the possible-worlds support score."),
        ("model_makers = {", "Three model types, created fresh for each feature set:"),
        ('"Logistic Regression"', "scaled features → linear model,"),
        ('"Decision Tree (depth 4)"', "a flowchart at most 4 questions deep,"),
        ('"Random Forest"', "300 trees voting."),
        ("rows, trained = [], {}", "Collect result rows and keep every trained model."),
        ("y_train = ", "Training labels."),
        ("for group, cols in feature_groups.items()", "Loop over feature sets…"),
        ("for name, make in model_makers.items()", "…and model types (9 combinations)."),
        ("model = make().fit(", "Train on the training features."),
        ("trained[(group, name)] = model", "Remember the model."),
        ("row.update(", "Score F1 on dev, test and refactorization."),
        ("ablation = pd.DataFrame(rows)", "The ablation table: about 0.5 → 0.93 → 1.0 as reasoning is added."),
    ]},
    "nb:22": {"summary": "Pick the final model (depth-4 tree on all features) and print its rules.", "lines": [
        ("final_cols = ALL_FEATURES", "The final model uses every feature."),
        ("final_model = trained[", "Reuse the depth-4 decision tree already trained in cell 20."),
        ("print(export_text(", "Print the whole tree as indented if/else rules."),
    ]},
    "nb:23": {"summary": "Bar chart of which features the tree actually uses.", "lines": [
        ("importance = pd.Series(", "Feature importance = how much each feature reduced impurity across the tree's splits."),
        ("importance[importance > 0].plot.barh", "Plot only the features the tree used; support dominates."),
    ]},
    "nb:25": {"summary": "Final model on the test set: scores per relation type and a confusion matrix.", "lines": [
        ("test_feats = features", "Test-set features."),
        ("test_pred = predict(", "Final predictions on test."),
        ("per_relation = pd.DataFrame({rel:", "Score each relation type (direct, indirect, collider, confounder) separately."),
        ('per_relation["n valid"]', "How many truly valid claims of each type exist in test."),
        ("display(per_relation.round(3))", "Show the table."),
        ("ConfusionMatrixDisplay.from_predictions", "2×2 grid: true vs predicted (valid / not valid)."),
        ('plt.title("Decision tree', "Title and show."),
    ]},
    "nb:27": {"summary": "Summary table: text-only baseline vs our pipeline on every test set, plus original labels.", "lines": [
        ("final_results = {s:", "Score the final model on all four evaluation sets."),
        ("original_test = load_split", "Also load the test set with the ORIGINAL (buggy) labels, for comparison with the paper."),
        ("summary = pd.DataFrame({", "One row per system, one column per test set (F1)."),
        ('summary["test (original labels)"]', "Extra column: F1 against the original labels."),
    ]},
    "nb:29": {"summary": "Stage 5 start: collect the test predictions as triples, build one DAG per premise, score edges.", "lines": [
        ("test = data[\"test\"]", "Shortcut to the test table."),
        ("triples = pd.DataFrame({", "One row per claim: premise id, relation, x, y, gold label, and the tree's P(valid)."),
        ('"prob": final_model.predict_proba', "predict_proba gives [P(not valid), P(valid)]; [:, 1] keeps P(valid)."),
        ("premise_info = {pid: analyse_premise", "For each test premise, its parsed facts (cached from stage 1)."),
        ("nodes = {pid:", "Each premise's list of variables."),
        ("predicted_dags = gs.dags_from_triples", "Build a DAG per premise from 'direct' claims with P(valid) ≥ 0.5."),
        ("edge_scores = gs.edge_metrics", "Compare predicted arrows with the gold arrows: precision, recall, F1."),
        ("pd.Series(edge_scores)", "Show the edge scores."),
    ]},
    "nb:31": {"summary": "Show the cycle check working on a deliberately bad set of candidate arrows.", "lines": [
        ("candidates = [", "Four candidate arrows with confidences; C → A would close a loop, B → B is a self-loop."),
        ("dag, rejected = gs.build_dag", "Add arrows from most to least confident, skipping any that would make a loop."),
        ('print("kept', "Arrows that made it into the DAG."),
        ('print("rejected', "Arrows refused, and why."),
    ]},
    "nb:32": {"summary": "Draw PC output vs predicted DAG vs gold arrows side by side for one large premise.", "lines": [
        ("gold_direct = ", "Gold arrows: 'direct' claims labelled valid."),
        ("pid = max(nodes", "Choose the premise with the most variables and arrows (a rich picture)."),
        ("fig, axes = plt.subplots(1, 3", "Three panels side by side."),
        ("draw_pdag(pc_from_premise(", "Panel 1: the PC output, dashed edges included."),
        ("for ax, (title, edges) in zip(", "Panels 2 and 3: predicted arrows, then gold arrows."),
        ("g = nx.DiGraph(edges)", "Make a graph from the arrow list (keep isolated variables too)."),
        ("pos = nx.circular_layout", "Same circle layout, so the panels line up."),
        ("nx.draw_networkx(", "Draw nodes, labels and arrows in one call."),
        ("plt.tight_layout()", "Tidy spacing and show."),
    ]},
    "nb:34": {"summary": "Ask every 'known' what-if question with the Bayesian network and measure accuracy.", "lines": [
        ("correlated = {pid:", "For each premise: its variables and correlated pairs (needed to build 'no effect' questions)."),
        ("queries = gs.known_intervention_queries", "Questions 'does forcing X change Y?' whose true answer we know from labels and text."),
        ("answered = gs.answer_queries", "Answer each one with a noisy-OR Bayesian network on the predicted DAG."),
        ('answered["correct"]', "Did the network's yes/no match the known answer?"),
        ("query_accuracy =", "Share answered correctly (0.975)."),
        ('print(f"{len(answered)}', "Report the count and accuracy."),
        ("answered.groupby(\"gold_effect\")", "Accuracy split by true answer: 'no' questions are all right; misses are 'yes' questions."),
    ]},
    "nb:35": {"summary": "List the what-if questions the network got wrong.", "lines": [
        ("answered[~answered.correct]", "~ means NOT: show only the wrong rows. All are causal chains through an undecided edge."),
    ]},
    "nb:37": {"summary": "Find the clearest example where seeing X and forcing X give different answers.", "lines": [
        ("best = None", "Will hold the example with the biggest gap."),
        ("for pid, dag in predicted_dags.items()", "Look at every premise's DAG…"),
        ("bn = gs.noisy_or_network(dag)", "…turn it into a Bayesian network…"),
        ("for x, y in dag.edges", "…and check every arrow X → Y."),
        ("if set(dag.predecessors(x))", "Only pairs with a shared cause (a confounder), where seeing ≠ forcing."),
        ("gap = gs.observational_effect", "Gap = (effect of seeing X) − (effect of forcing X)."),
        ("if best is None or gap > best[0]", "Keep the biggest gap so far."),
        ("gap, pid, x, y, bn = best", "Unpack the winner."),
        ("z = sorted(", "The shared causes, for printing."),
        ('print(f"  observational', "Seeing: P(Y=1 | X=1) − P(Y=1 | X=0). Includes the confounder's influence."),
        ('print(f"  interventional', "Forcing: P(Y=1 | do(X=1)) − P(Y=1 | do(X=0)). Only X's own effect."),
    ]},
    "nb:39": {"summary": "Show that pgmpy can LEARN the probabilities from data, using simulated samples.", "lines": [
        ("from pgmpy.models import DiscreteBayesianNetwork", "pgmpy's Bayesian network class."),
        ("samples = bn.simulate(5000", "Draw 5,000 rows of fake 0/1 data from the reference network (Corr2Cause has no real measurements)."),
        ("fitted = DiscreteBayesianNetwork(", "A fresh network with the predicted structure but no numbers yet."),
        ("fitted.add_nodes_from", "Include variables that have no arrows."),
        ("fitted.fit(samples)", "Learn every probability table by counting in the data (maximum likelihood)."),
        ('print(f"ACE of', "The learned network gives nearly the same causal effect as the reference (0.51 vs 0.49)."),
    ]},
    "nb:41": {"summary": "Save the Phase 1 results so the Phase 2 notebook can compare against them.", "lines": [
        ("phase1 = pd.DataFrame([", "One row per system with all its metrics."),
        ('"model": "TF-IDF + LR', "Baseline row: only text-level F1 scores."),
        ('"model": "Rules + PC', "Our pipeline: F1 scores plus edge precision/recall and what-if accuracy."),
        ("phase1.to_csv(", "Save metrics."),
        ("triples.to_csv(", "Save every test prediction (Phase 2 will produce the same table)."),
    ]},

    # ========================================================================= data.py
    "py:data._download": {"summary": "Download a CSV from Hugging Face once, then reuse the local copy.", "lines": [
        ("target = DATA_DIR", "Where the file should live locally: data/<folder>/<file>."),
        ("if not target.exists()", "Only download if we do not have it yet."),
        ("target.parent.mkdir", "Make the folder if needed."),
        ("hf_hub_download(", "Fetch the file from the Hugging Face dataset repository and copy it here."),
        ("return target", "Give back the local path."),
    ]},
    "py:data._with_columns": {"summary": "Split the raw 'input' text into premise and hypothesis, and number the premises.", "lines": [
        ("parts = df[\"input\"].map(split_input)", "For every row, cut 'Premise: … Hypothesis: …' into two pieces."),
        ("premise = parts.str[0]", "First piece = premise text."),
        ("return df.assign(", "Add columns: premise, hypothesis,"),
        ("premise_id=pd.factorize", "and premise_id: every distinct premise text gets a number 0, 1, 2…"),
    ]},
    "py:data.load_split": {"summary": "Load train, dev or test, with corrected labels by default.", "lines": [
        ("repo, folder = ", "Corrected labels come from the fixed copy; original labels from the official repo."),
        ("df = pd.read_csv(_download(", "Download if needed, then read the CSV into a table."),
        ("return _with_columns(", "Keep the useful columns and add premise / hypothesis / premise_id."),
    ]},

    # ====================================================================== parsing.py
    "py:parsing.split_input": {"summary": "Cut one raw input string into (premise, hypothesis).", "lines": [
        ("premise, hypothesis = text.split", "Split at the line break before 'Hypothesis:' (only once)."),
        ("return premise.replace", "Remove the 'Premise:' label and surrounding spaces from both parts."),
    ]},
    "blk:parsing:DIRECT, INDIRECT, COLLIDER, CONFOUNDER =|VAR = r": {"title": "relation names + variable pattern", "summary": "Our small vocabulary: four relation types, and what a variable looks like.", "lines": [
        ("DIRECT, INDIRECT, COLLIDER, CONFOUNDER =", "The four kinds of causal claim the system understands."),
        ("RELATIONS = ", "The same four, as a list to loop over."),
        ("VAR = r", "Regex for a variable: exactly one capital letter, captured in a group so we can read it."),
    ]},
    "blk:parsing:_HEADER = re.compile|_COND_INDEP = re.compile": {"title": "premise sentence patterns", "summary": "One regex per sentence shape in the premise.", "lines": [
        ("_HEADER =", "Reads 'closed system of N variables, A, B and C.' → captures N and the list of names."),
        ("_CORR =", "Reads 'X correlates with Y.' → captures X and Y."),
        ("_MARG_INDEP =", "Reads 'X is independent of Y.' → independent with no condition."),
        ("_COND_INDEP =", "Reads 'X and Y are independent given …' → captures X, Y and the conditioning list."),
    ]},
    "py:parsing._split_var_list": {"summary": "Pull single capital letters out of a phrase like 'A, B and C'.", "lines": [
        ("return re.findall", "\\b…\\b = whole word, so only standalone capital letters are taken."),
    ]},
    "py:parsing.parse_premise": {"summary": "Turn the whole premise paragraph into a Premise object of facts.", "lines": [
        ("header = _HEADER.search(text)", "Find the header sentence."),
        ("if header is None", "If it is missing, this is not a Corr2Cause premise: stop with a clear error."),
        ("n_vars, var_list =", "Number of variables, and their names."),
        ("if len(var_list) != n_vars", "Sanity check: the count must match the names found."),
        ("premise = Premise(variables=var_list)", "Start an empty fact sheet."),
        ("premise.sepsets = {pair: []", "Every pair starts with an empty list of separating sets (= never independent)."),
        ("body = text[header.end():]", "The text after the header holds the facts."),
        ("for a, b in _CORR.findall(body)", "Every 'X correlates with Y' …"),
        ("premise.correlated.add", "… is stored as an unordered pair {X, Y}."),
        ("for a, b in _MARG_INDEP.findall(body)", "Every 'X is independent of Y' …"),
        ("premise.sepsets[frozenset((a, b))].append(frozenset())", "… adds the empty set {} to that pair's separating sets."),
        ("for a, b, given in _COND_INDEP.findall(body)", "Every 'X and Y are independent given S' …"),
        ("premise.sepsets[frozenset((a, b))].append(frozenset(_split_var_list", "… adds S to that pair's separating sets."),
        ("return premise", "The finished fact sheet."),
    ]},
    "blk:parsing:HYPOTHESIS_PATTERNS = [|]": {"title": "HYPOTHESIS_PATTERNS", "summary": "The five hypothesis wordings the reader knows, each mapped to a relation type.", "lines": [
        ("HYPOTHESIS_PATTERNS = [", "Each entry: (regex, relation, which captured letter is x, which is y)."),
        ("directly causes", "'X directly causes Y.' → direct."),
        ("causes something else which causes", "'X causes something else which causes Y.' → indirect."),
        ("is a cause for", "'X is a cause for Y, but not a direct one.' → indirect too."),
        ("collider", "'There exists at least one collider … of X and Y.' → collider."),
        ("confounder", "'There exists at least one confounder … of X and Y.' → confounder."),
    ]},
    "py:parsing.parse_hypothesis": {"summary": "Turn the hypothesis sentence into Hypothesis(relation, x, y), or None.", "lines": [
        ("text = text.strip()", "Remove stray spaces."),
        ("for pattern, relation, gx, gy in HYPOTHESIS_PATTERNS", "Try each known wording in turn."),
        ("m = pattern.match(text)", "Does the whole sentence fit this pattern?"),
        ("return Hypothesis(relation, m.group(gx), m.group(gy))", "Yes: return the relation and the two variables."),
        ("return None", "No pattern fits (a new wording): return None. Phase 2 replaces this function with DistilBERT."),
    ]},
    "py:parsing.Premise": {"summary": "The fact sheet that stage 1 fills in.", "lines": [
        ("variables: list[str]", "Variable names, e.g. ['A', 'B', 'C']."),
        ("correlated: set[frozenset]", "Pairs that are correlated. frozenset = unordered, so {A, C} = {C, A}."),
        ("sepsets: dict", "For each pair, every set of variables that makes the pair independent."),
        ("def pairs(self)", "All unordered pairs of variables."),
    ]},

    # ================================================================= causal_rules.py
    "py:causal_rules.PDAG": {"summary": "A partly directed graph: some arrows known, some edges undecided.", "lines": [
        ("nodes: list[str]", "The variables."),
        ("directed: set", "Proven arrows, stored as (a, b) meaning a → b."),
        ("undirected: set", "Undecided edges, stored as {a, b}."),
        ("def adjacent(self, a: str, b: str)", "Are a and b joined at all, by an arrow either way or a dashed edge?"),
        ("def neighbours(self, a: str)", "Everything joined to a."),
        ("def parents(self, a: str)", "Proven causes of a (arrows into a)."),
        ("def children(self, a: str)", "Proven effects of a (arrows out of a)."),
        ("def orient(self, a: str, b: str)", "Turn the dashed edge a – b into the arrow a → b."),
    ]},
    "py:causal_rules.skeleton": {"summary": "PC step 1: join every pair that nothing can make independent.", "lines": [
        ("pdag = PDAG(nodes=", "Start with the variables and no edges."),
        ("for pair in premise.pairs()", "Look at every pair once."),
        ("if not premise.sepsets[pair]", "No separating set at all → the pair is never independent → directly linked."),
        ("pdag.undirected.add(pair)", "Add a dashed (direction unknown) edge."),
        ("return pdag", "The skeleton."),
    ]},
    "py:causal_rules.orient_v_structures": {"summary": "PC step 2: find common effects (colliders) and draw their arrows.", "lines": [
        ("for x, y in combinations(pdag.nodes, 2)", "Look at every pair X, Y …"),
        ("if pdag.adjacent(x, y)", "… but only pairs that are NOT joined (skip joined ones)."),
        ("sepsets = premise.sepsets", "The sets that separate X and Y."),
        ("for z in pdag.neighbours(x) & pdag.neighbours(y)", "Every Z joined to both: the shape X – Z – Y."),
        ("if all(z not in s for s in sepsets)", "Z is in NONE of the separating sets → knowing Z does not separate X and Y → Z is a common effect."),
        ("pdag.orient(x, z)", "Draw X → Z …"),
        ("pdag.orient(y, z)", "… and Y → Z."),
    ]},
    "py:causal_rules.apply_meek_rules": {"summary": "PC step 3: keep applying the Meek rules until no edge changes.", "lines": [
        ("changed = True", "Repeat while something changed in the last pass."),
        ("for edge in list(pdag.undirected)", "Look at every still-undecided edge."),
        ("for u, v in ((a, b), (b, a))", "Try both directions, u → v and v → u."),
        ("if frozenset((u, v)) not in pdag.undirected", "Stop if the edge was already oriented a moment ago."),
        ("if _forced(pdag, u, v)", "Does a Meek rule force u → v?"),
        ("pdag.orient(u, v)", "Yes: draw the arrow and note that something changed."),
    ]},
    "py:causal_rules._forced": {"summary": "Checks the three Meek rules for one edge u – v.", "lines": [
        ("# R1:", "Rule 1: some W → U, W not joined to V. If V pointed into U we would get a new collider W → U ← V, which the facts did not show, so U → V."),
        ("if any(not pdag.adjacent(w, v)", "Check rule 1 for every parent W of U."),
        ("# R2:", "Rule 2: there is already U → W → V. Then V → U would make a loop, so U → V."),
        ("if pdag.children(u) & pdag.parents(v)", "Is there a W that U points to and that points to V?"),
        ("# R3:", "Rule 3: U is dashed-joined to two unjoined W1, W2 that both point to V. Then U → V (the other way creates a loop or a collider)."),
        ("undirected_nbrs =", "Variables joined to U by a dashed edge."),
        ("candidates = sorted(", "Those that also point into V."),
        ("return any(not pdag.adjacent(w1, w2)", "Rule 3 fires if two of them are not joined to each other."),
    ]},
    "py:causal_rules.pc_from_premise": {"summary": "The whole PC algorithm in one line: step 1, then 2, then 3.", "lines": [
        ("return apply_meek_rules(", "skeleton → v-structures → Meek rules, applied inside-out."),
    ]},
    "py:causal_rules._v_structures": {"summary": "List all colliders A → Z ← B where A and B are not joined.", "lines": [
        ("for z in dag.nodes", "For every variable Z …"),
        ("for a, b in combinations(sorted(dag.predecessors(z)), 2)", "… every pair of arrows into Z …"),
        ("if not adjacency.has_edge(a, b)", "… whose tails A, B are not joined …"),
        ("found.add((a, z, b))", "… is a collider. Record it."),
    ]},
    "py:causal_rules.equivalent_dags": {"summary": "List every possible world: each way to point the dashed edges that still fits the facts.", "lines": [
        ("undirected = [", "The dashed edges, as (a, b) pairs."),
        ("base = nx.DiGraph(list(pdag.directed))", "Start from the proven arrows only."),
        ("adjacency = base.to_undirected()", "Who is joined to whom (direction ignored), dashed edges included."),
        ("required = _v_structures(base, adjacency)", "The colliders the facts showed. Every world must have exactly these."),
        ("for flips in product((False, True)", "Every combination of directions: 2 × 2 × … for each dashed edge."),
        ("dag = base.copy()", "Copy the proven arrows,"),
        ("dag.add_edges_from(", "then point each dashed edge one way or the other."),
        ("if nx.is_directed_acyclic_graph(dag) and _v_structures", "Keep the world only if it has no loop and no extra collider."),
        ("return dags", "All the possible worlds."),
    ]},
    "py:causal_rules._reachable_avoiding": {"summary": "Who can be reached from a variable without passing through another one.", "lines": [
        ("view = nx.subgraph_view(", "A view of the graph with the 'avoid' variable hidden."),
        ("return nx.ancestors(view, start) if reverse", "Going backwards (causes of causes) or forwards (effects of effects)."),
    ]},
    "py:causal_rules.relation_holds": {"summary": "Is a claim true in ONE particular world (DAG)?", "lines": [
        ("if relation == DIRECT", "direct: is there an arrow X → Y?"),
        ("if relation == INDIRECT", "indirect: does X point to some Z (not Y) from which Y can be reached?"),
        ("if relation == COLLIDER", "collider: is there a Z reachable from X (without passing Y) AND from Y (without passing X)? That Z is a common effect."),
        ("if relation == CONFOUNDER", "confounder: is there a Z that reaches X (not via Y) AND reaches Y (not via X)? That Z is a common cause."),
        ("raise ValueError(relation)", "Unknown relation name: stop with an error."),
    ]},
    "py:causal_rules.support": {"summary": "Share of possible worlds in which the claim is true.", "lines": [
        ("return sum(relation_holds(d, relation, x, y) for d in dags) / len(dags)", "Count worlds where it holds (True counts as 1) and divide by the number of worlds."),
    ]},

    # ===================================================================== features.py
    "py:features.build_feature_table": {"summary": "Make one row of numbers for every input row.", "lines": [
        ("rows = []", "Collect one dictionary per input row."),
        ("for text in inputs", "For each raw input string:"),
        ("premise_text, hyp_text = split_input(text)", "cut it into premise and hypothesis,"),
        ("premise, pdag, definite, possible, dags = analyse_premise", "stages 1-2 for the premise (cached: done once per premise),"),
        ("hyp = hypothesis_parser(hyp_text)", "stage 1 for the hypothesis. The parser is a parameter, so Phase 2 can pass DistilBERT here."),
        ("if hyp is None", "Unreadable hypothesis:"),
        ('rows.append({"parsed": 0})', "mark it parsed=0 with no features (it will be predicted 'not valid')."),
        ('rows.append({"parsed": 1', "Otherwise store the relation, x, y and all pair features."),
        ("return pd.DataFrame(rows, index=inputs.index)", "Turn the list into a table aligned with the input rows."),
    ]},
    "py:features.analyse_premise": {"summary": "Everything about one premise, computed once and remembered (cached).", "lines": [
        ("premise = parse_premise(premise_text)", "Stage 1: sentences → facts."),
        ("pdag = pc_from_premise(premise)", "Stage 2: facts → PDAG."),
        ("definite = nx.DiGraph(list(pdag.directed))", "Graph of proven arrows only (for 'definite ancestor' checks)."),
        ("possible = definite.copy()", "Graph where dashed edges count in both directions (for 'possible ancestor' checks)."),
        ("possible.add_edges_from([(a, b), (b, a)])", "Add both directions of each dashed edge."),
        ("return premise, pdag, definite, possible, equivalent_dags(pdag)", "Also list the possible worlds."),
    ]},
    "py:features.pair_features": {"summary": "The feature numbers for one claim about the pair (x, y).", "lines": [
        ("x, y = hyp.x, hyp.y", "The two variables the claim is about."),
        ("sepsets = premise.sepsets", "Their separating sets."),
        ("feats = {f\"rel_{r}\"", "One-hot relation: four 0/1 columns, exactly one is 1."),
        ('"n_vars":', "Text facts start here: number of variables,"),
        ('"marginally_correlated":', "does the text say X correlates with Y,"),
        ('"adjacent":', "are they joined in the skeleton,"),
        ('"n_sepsets":', "how many separating sets,"),
        ('"min_sepset_size":', "smallest separating set (-1 if none),"),
        ('"degree_x":', "how many neighbours X and Y have,"),
        ('"n_common_neighbours":', "and how many neighbours they share."),
        ('"edge_x_to_y":', "PC graph features: proven arrow X → Y, Y → X, or dashed edge,"),
        ('"x_ancestor_of_y":', "proven arrow path X ⇒ Y or Y ⇒ X,"),
        ('"x_possible_ancestor_of_y":', "path if dashed edges may point either way,"),
        ('"n_common_children":', "shared proven effects and shared proven causes."),
        ('"support":', "Support: share of possible worlds where the claim holds."),
        ('"n_equivalent_dags":', "Number of possible worlds."),
    ]},

    # ================================================================== graph_stage.py
    "py:graph_stage.build_dag": {"summary": "Add arrows from most to least confident, refusing any that would create a loop.", "lines": [
        ("dag, rejected = nx.DiGraph(), []", "Empty graph and an empty list of refused arrows."),
        ("dag.add_nodes_from(nodes)", "Every variable appears, even with no arrows."),
        ("for cause, effect, score in sorted(", "Highest confidence first, so a weak arrow can never block a strong one."),
        ("if cause == effect or nx.has_path(dag, effect, cause)", "Self-loop, or effect already leads back to cause (adding it would close a loop)?"),
        ("rejected.append(", "Then refuse it."),
        ("dag.add_edge(cause, effect, weight=score)", "Otherwise add the arrow, remembering its confidence."),
    ]},
    "py:graph_stage.dags_from_triples": {"summary": "Build one DAG per premise from the accepted 'direct' claims.", "lines": [
        ("direct = triples[", "Keep claims of type 'direct' with P(valid) ≥ threshold (0.5): the 'reviewed' predictions."),
        ("for pid, nodes in nodes_by_premise.items()", "For each premise …"),
        ("rows = direct[direct.premise_id == pid]", "… its accepted arrows …"),
        ("dags[pid], _ = build_dag(", "… become a DAG (loops refused)."),
    ]},
    "py:graph_stage.edge_metrics": {"summary": "Compare predicted arrows with gold arrows across all premises.", "lines": [
        ("gold = triples[", "Gold arrows: 'direct' claims whose true label is 1."),
        ("pred_edges = {", "Every arrow in every predicted DAG."),
        ("tp = len(pred_edges & gold_edges)", "Arrows that are both predicted and gold (true positives)."),
        ("precision = ", "Share of predicted arrows that are right."),
        ("recall = ", "Share of gold arrows we found."),
        ("f1 = ", "Harmonic mean of the two."),
    ]},
    "py:graph_stage.noisy_or_network": {"summary": "Give the DAG simple probabilities so it becomes a Bayesian network.", "lines": [
        ("bn = DiscreteBayesianNetwork(list(dag.edges))", "pgmpy network with the same arrows."),
        ("bn.add_nodes_from(dag.nodes)", "Include variables with no arrows."),
        ("for node in dag.nodes", "For each variable, build its probability table:"),
        ("parents = sorted(dag.predecessors(node))", "its direct causes,"),
        ("p_on = [1 - (1 - base)", "P(on) for every on/off combination of causes: 10% by itself, each active cause switches it on with 70% chance."),
        ("bn.add_cpds(TabularCPD(", "Store the table: row 0 = P(off), row 1 = P(on)."),
        ("bn.check_model()", "pgmpy checks every table sums to 1 and matches the arrows."),
    ]},
    "py:graph_stage.interventional_effect": {"summary": "What-if: how much does forcing X on (vs off) change the chance Y is on?", "lines": [
        ("if not nx.has_path(bn, x, y)", "No arrow path from X to Y → forcing X cannot touch Y → effect 0."),
        ("ci = CausalInference(bn)", "pgmpy's do-calculus engine."),
        ("p = [ci.query([y], do={x: v}", "P(Y=1 | do(X=0)) and P(Y=1 | do(X=1)): cut X off from its causes and set it."),
        ("return float(p[1] - p[0])", "The average causal effect (ACE)."),
    ]},
    "py:graph_stage.observational_effect": {"summary": "Just looking: how much does SEEING X on (vs off) change the chance Y is on?", "lines": [
        ("ve = VariableElimination(bn)", "Ordinary probability inference."),
        ("p = [ve.query([y], evidence={x: v}", "P(Y=1 | X=0) and P(Y=1 | X=1): condition on what we saw."),
        ("return float(p[1] - p[0])", "Difference. Larger than the ACE when X and Y share a cause."),
    ]},
    "py:graph_stage.known_intervention_queries": {"summary": "Build the what-if questions whose true answer we know.", "lines": [
        ("causes = triples[", "Claims 'X directly or indirectly causes Y' that are labelled valid."),
        ("yes = set(", "Those pairs: forcing X definitely changes Y."),
        ('rows = [{"premise_id": p, "x": x, "y": y, "gold_effect": True}', "Record each as a 'yes' question."),
        ("for x, y in product(nodes, nodes)", "Then check every other ordered pair:"),
        ("if (pid, y, x) in yes or frozenset((x, y)) not in correlated", "If Y causes X (effects cannot cause their causes) or they are not even correlated,"),
        ('rows.append({"premise_id": pid, "x": x, "y": y, "gold_effect": False})', "the true answer is 'no effect'."),
    ]},
    "py:graph_stage.answer_queries": {"summary": "Answer every question with the Bayesian network.", "lines": [
        ("networks = {pid: noisy_or_network(dag)", "One network per premise."),
        ("effects = [interventional_effect(", "Compute the ACE for each question."),
        ("predicted_effect=[e > EFFECT_THRESHOLD", "Say 'yes, it changes Y' if the ACE is above 0.01."),
    ]},
}
