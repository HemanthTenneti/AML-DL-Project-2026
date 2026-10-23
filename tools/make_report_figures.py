"""Re-run the Phase 1 pipeline and draw the report figures into report/figures/.

Colours match the report: navy #1F4E79 and blue #4A86C5 on a light-blue tint, grey for baselines.
The numbers behind the figures are written to results/phase1_report_stats.json.

Usage:  .venv/bin/python tools/make_report_figures.py      (about 5 minutes on a laptop CPU)
"""
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
STATS = ROOT / "results" / "phase1_report_stats.json"
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd, networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

from causal_nlp.data import load_split, load_refactorization_test, load_paraphrase_test
from causal_nlp.parsing import parse_premise, parse_hypothesis, RELATIONS
from causal_nlp.causal_rules import pc_from_premise, equivalent_dags, support
from causal_nlp.features import (build_feature_table, analyse_premise, RELATION_FEATURES,
                                 TEXT_FACT_FEATURES, PC_GRAPH_FEATURES, ALL_FEATURES)
from causal_nlp import graph_stage as gs

INK, NAVY, BLUE = "#222222", "#1F4E79", "#4A86C5"
GREY, TINT = "#A6ADB4", "#E3ECF6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E3E7EC", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
SEED = 42
stats = {}

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); print("saved", name)

# ------------------------------------------------------------------ data
t0 = time.time()
data = {"train": load_split("train"), "dev": load_split("dev"), "test": load_split("test"),
        "refactorization": load_refactorization_test(), "paraphrase": load_paraphrase_test()}
print(f"loaded in {time.time()-t0:.0f}s")
stats["splits"] = {k: {"rows": len(v), "premises": int(v.premise.nunique()), "pct_valid": round(100*v.label.mean(), 1)}
                   for k, v in data.items()}

# ------------------------------------------------------------------ features
features = {k: build_feature_table(v.input) for k, v in data.items()}
train_rel = features["train"].relation

# ---- EDA figure
fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.7), gridspec_kw={"width_ratios": [1.1, 1, 1.15]})
ax = axes[0]
names = ["train", "dev", "test", "refactorization", "paraphrase"]
short = ["train", "dev", "test", "refactor.", "paraphr."]
pv = [stats["splits"][n]["pct_valid"] for n in names]
bars = ax.bar(short, pv, color=[NAVY, NAVY, NAVY, BLUE, BLUE], width=0.62)
for b, n in zip(bars, names):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.6, f"{stats['splits'][n]['rows']:,}\nrows",
            ha="center", va="bottom", fontsize=7)
ax.set_ylim(0, 31); ax.set_ylabel("% hypotheses labelled valid")
ax.set_title("(a) Class balance per split"); ax.tick_params(axis="x", labelsize=7.5)

ax = axes[1]
nv = {s: data[s].num_variables.value_counts(normalize=True).sort_index() for s in ("train", "test")}
ks = np.arange(2, 7); w = 0.38
ax.bar(ks - w/2, [100*nv["train"].get(k, 0) for k in ks], w, color=NAVY, label="train")
ax.bar(ks + w/2, [100*nv["test"].get(k, 0) for k in ks], w, color=BLUE, label="test")
ax.set_xticks(ks); ax.set_xlabel("variables in the premise"); ax.set_ylabel("% of hypotheses")
ax.set_title("(b) Graph size"); ax.legend(loc="upper left")

ax = axes[2]
tr = pd.DataFrame({"rel": train_rel, "label": data["train"].label})
g = tr.groupby("rel").label.agg(["size", "mean"]).reindex(list(RELATIONS))
stats["train_relation_mix"] = {r: {"n": int(g.loc[r, "size"]), "pct_valid": round(100*g.loc[r, "mean"], 1)} for r in RELATIONS}
ax.barh(list(RELATIONS)[::-1], g["size"].values[::-1]/1000, color=TINT, edgecolor=NAVY)
ax.barh(list(RELATIONS)[::-1], (g["size"]*g["mean"]).values[::-1]/1000, color=NAVY, label="valid")
for i, r in enumerate(list(RELATIONS)[::-1]):
    ax.text(g.loc[r, "size"]/1000 + 1.5, i, f"{100*g.loc[r,'mean']:.0f}% valid", va="center", fontsize=7.5)
ax.set_xlabel("train hypotheses (thousands)"); ax.set_title("(c) Relation types in train")
ax.set_xlim(0, g["size"].max()/1000*1.38); ax.grid(axis="y", visible=False)
fig.tight_layout(w_pad=1.6); save(fig, "fig_eda")

# ------------------------------------------------------------------ graph drawing helpers
def draw_graph(ax, nodes, directed, undirected=(), highlight=(), title=None, node_col=TINT, edge_col=NAVY, rad=0.0):
    pos = nx.circular_layout(sorted(nodes))
    for n, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.17, facecolor=node_col, edgecolor=NAVY, lw=1.4, zorder=3))
        ax.text(x, y, n, ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=4)
    for a, b in undirected:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], ls=(0, (4, 3)), color=GREY, lw=1.8, zorder=1)
    for a, b in directed:
        c = BLUE if (a, b) in highlight else edge_col
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], arrowstyle="-|>", mutation_scale=14, lw=2.2 if c == BLUE else 1.5,
                                     color=c, shrinkA=13, shrinkB=13, zorder=2, connectionstyle=f"arc3,rad={rad}"))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
    if title: ax.set_title(title, fontsize=9.5)

# ---- PDAG + equivalent DAGs figure (same 5-variable premise as the notebook)
def mixed(p):
    gg = pc_from_premise(parse_premise(p)); return len(gg.nodes) <= 5 and gg.directed and gg.undirected
text = next(p for p in data["test"].premise.unique() if mixed(p))
facts = parse_premise(text); pdag = pc_from_premise(facts); dags = equivalent_dags(pdag)
stats["pdag_example"] = {"n_equivalent_dags": len(dags), "directed": sorted(pdag.directed),
                         "undirected": sorted(tuple(sorted(e)) for e in pdag.undirected),
                         "support": {f"{r}({x},{y})": support(dags, r, x, y) for r, x, y in
                                     [("direct", "A", "E"), ("direct", "A", "C"), ("direct", "C", "A"), ("indirect", "C", "E"), ("confounder", "C", "E")]}}
fig, axes = plt.subplots(1, 1 + len(dags), figsize=(2.35 * (1 + len(dags)), 2.6))
draw_graph(axes[0], pdag.nodes, sorted(pdag.directed), [tuple(sorted(e)) for e in pdag.undirected], title="PDAG (PC output)")
for i, d in enumerate(dags):
    ac = "A→C" if d.has_edge("A", "C") else "C→A"
    draw_graph(axes[i + 1], d.nodes, list(d.edges), highlight={("A", "C"), ("C", "A")},
               title=f"Equivalent DAG {i+1}  ({ac})", node_col="white")
fig.tight_layout(w_pad=0.2); save(fig, "fig_pdag")

# ------------------------------------------------------------------ TF-IDF baseline
def scores(y, p):
    return {"precision": precision_score(y, p, zero_division=0), "recall": recall_score(y, p, zero_division=0),
            "f1": f1_score(y, p, zero_division=0)}
EVAL = ["dev", "test", "refactorization", "paraphrase"]
tok = r"(?u)\b\w+\b"
tfidf = make_pipeline(ColumnTransformer([
    ("premise", TfidfVectorizer(token_pattern=tok, ngram_range=(1, 3), min_df=5, max_features=50_000, sublinear_tf=True), "premise"),
    ("hypothesis", TfidfVectorizer(token_pattern=tok, ngram_range=(1, 3), min_df=2, sublinear_tf=True), "hypothesis")]),
    LogisticRegression(max_iter=2000, class_weight="balanced"))
t0 = time.time(); tfidf.fit(data["train"], data["train"].label); stats["tfidf_train_s"] = round(time.time() - t0)
tfidf_res = {s: scores(data[s].label, tfidf.predict(data[s])) for s in EVAL}

# ------------------------------------------------------------------ ablation
def predict(model, feats, cols):
    pred = np.zeros(len(feats), dtype=int); ok = (feats.parsed == 1).values
    if ok.any(): pred[ok] = model.predict(feats.loc[ok, cols])
    return pred
groups = {"relation + text facts": RELATION_FEATURES + TEXT_FACT_FEATURES,
          "+ PC graph": RELATION_FEATURES + TEXT_FACT_FEATURES + PC_GRAPH_FEATURES,
          "+ support (all)": ALL_FEATURES}
makers = {"Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
          "Decision Tree (depth 4)": lambda: DecisionTreeClassifier(max_depth=4, random_state=SEED),
          "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)}
abl, trained = [], {}
for gname, cols in groups.items():
    for mname, mk in makers.items():
        m = mk().fit(features["train"][cols], data["train"].label); trained[(gname, mname)] = m
        abl.append({"features": gname, "model": mname,
                    **{s: f1_score(data[s].label, predict(m, features[s], cols)) for s in ["dev", "test", "refactorization"]}})
abl = pd.DataFrame(abl); stats["ablation"] = abl.round(3).to_dict(orient="records")
fig, ax = plt.subplots(figsize=(6.4, 2.6))
x = np.arange(3); w = 0.26
for i, (mname, c) in enumerate(zip(makers, [GREY, BLUE, NAVY])):
    v = abl[abl.model == mname].test.values
    b = ax.bar(x + (i - 1) * w, v, w, color=c, label=mname)
    for bb, vv in zip(b, v): ax.text(bb.get_x()+bb.get_width()/2, vv+0.015, f"{vv:.2f}", ha="center", fontsize=7)
ax.set_xticks(x, ["relation + text facts", "+ PC-graph features", "+ support features"])
ax.set_ylabel("test F1 (valid class)"); ax.set_ylim(0, 1.1); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
save(fig, "fig_ablation")

final = trained[("+ support (all)", "Decision Tree (depth 4)")]
final_res = {s: scores(data[s].label, predict(final, features[s], ALL_FEATURES)) for s in EVAL}
orig_test = load_split("test", corrected=False)
stats["tfidf"] = {s: {k: round(v, 3) for k, v in r.items()} for s, r in tfidf_res.items()}
stats["final"] = {s: {k: round(v, 3) for k, v in r.items()} for s, r in final_res.items()}
stats["orig_labels_test_f1"] = {"tfidf": round(f1_score(orig_test.label, tfidf.predict(orig_test)), 3),
                                "final": round(f1_score(orig_test.label, predict(final, features["test"], ALL_FEATURES)), 3)}
stats["parse_coverage"] = {s: round(float(data[s].hypothesis.map(lambda h: parse_hypothesis(h) is not None).mean()), 3) for s in EVAL}

# ---- robustness figure
fig, ax = plt.subplots(figsize=(6.4, 2.6))
x = np.arange(4); w = 0.36
for i, (nm, res, c) in enumerate([("TF-IDF + LR (text only)", tfidf_res, GREY), ("Rules + PC + Decision Tree", final_res, NAVY)]):
    v = [res[s]["f1"] for s in EVAL]
    b = ax.bar(x + (i - 0.5) * w, v, w, color=c, label=nm)
    for bb, vv in zip(b, v): ax.text(bb.get_x()+bb.get_width()/2, vv+0.015, f"{vv:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x, ["dev", "test", "refactorization\n(renamed vars)", "paraphrase\n(reworded)"])
ax.set_ylabel("F1 (valid class)"); ax.set_ylim(0, 1.1); ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5)
save(fig, "fig_robustness")

# ---- importance + confusion
test_pred = predict(final, features["test"], ALL_FEATURES)
cm = confusion_matrix(data["test"].label, test_pred); stats["confusion"] = cm.tolist()
stats["per_relation"] = {r: {k: round(v, 3) for k, v in scores(data["test"].label[features["test"].relation == r],
                                                               test_pred[(features["test"].relation == r).values]).items()} for r in RELATIONS}
imp = pd.Series(final.feature_importances_, index=ALL_FEATURES); imp = imp[imp > 0].sort_values()
stats["importance"] = imp.round(4).to_dict()
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.4), gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]
ax.barh(imp.index, imp.values, color=[NAVY if n in ("support", "n_equivalent_dags") else GREY for n in imp.index])
ax.set_xscale("log"); ax.set_xlabel("Gini importance (log scale)"); ax.set_title("(a) What the tree looks at")
ax.grid(axis="y", visible=False)
ax = axes[1]
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("t", ["#FFFFFF", NAVY])
ax.imshow(np.log1p(cm), cmap=cmap); ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if cm[i, j] > 100 else INK)
ax.set_xticks([0, 1], ["not valid", "valid"]); ax.set_yticks([0, 1], ["not valid", "valid"])
ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title("(b) Test confusion matrix")
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(w_pad=2); save(fig, "fig_tree_eval")

# ------------------------------------------------------------------ graph stage
test = data["test"]; tf = features["test"]
triples = pd.DataFrame({"premise_id": test.premise_id, "relation": tf.relation, "x": tf.x, "y": tf.y,
                        "label": test.label, "prob": final.predict_proba(tf[ALL_FEATURES])[:, 1]})
pinfo = {pid: analyse_premise(p)[0] for pid, p in test.groupby("premise_id").premise.first().items()}
nodes = {pid: f.variables for pid, f in pinfo.items()}
pred_dags = gs.dags_from_triples(triples, nodes)
stats["edges"] = gs.edge_metrics(pred_dags, triples)
queries = gs.known_intervention_queries(triples, {pid: (f.variables, f.correlated) for pid, f in pinfo.items()})
ans = gs.answer_queries(queries, pred_dags); ans["correct"] = ans.gold_effect == ans.predicted_effect
stats["queries"] = {"n": len(ans), "premises": int(ans.premise_id.nunique()), "acc": round(ans.correct.mean(), 4),
                    "by_gold": ans.groupby("gold_effect").correct.agg(["size", "mean"]).round(3).reset_index().to_dict(orient="records"),
                    "misses": ans[~ans.correct][["premise_id", "x", "y"]].values.tolist()}

gold = triples[(triples.relation == "direct") & (triples.label == 1)]
pid = max(nodes, key=lambda p: (len(nodes[p]), len(pred_dags[p].edges)))
stats["graph_example_pid"] = int(pid)
fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.9))
pd_ = pc_from_premise(pinfo[pid])
draw_graph(axes[0], pd_.nodes, sorted(pd_.directed), [tuple(sorted(e)) for e in pd_.undirected], title="(a) PC output: PDAG")
draw_graph(axes[1], nodes[pid], list(pred_dags[pid].edges), title="(b) Predicted DAG (from triples)", node_col=TINT)
draw_graph(axes[2], nodes[pid], list(zip(gold[gold.premise_id == pid].x, gold[gold.premise_id == pid].y)),
           title="(c) Gold direct-cause edges", node_col="white")
fig.tight_layout(w_pad=0.3); save(fig, "fig_graph")

# ---- observational vs interventional
best = None
for p_, dag in pred_dags.items():
    bn = gs.noisy_or_network(dag)
    for a, b in dag.edges:
        if set(dag.predecessors(a)) & set(dag.predecessors(b)):
            gap = gs.observational_effect(bn, a, b) - gs.interventional_effect(bn, a, b)
            if best is None or gap > best[0]: best = (gap, p_, a, b, bn)
gap, p_, a, b, bn = best
obs, ace = gs.observational_effect(bn, a, b), gs.interventional_effect(bn, a, b)
from pgmpy.models import DiscreteBayesianNetwork
samples = bn.simulate(5000, seed=SEED, show_progress=False)
fitted = DiscreteBayesianNetwork(list(pred_dags[p_].edges)); fitted.add_nodes_from(pred_dags[p_].nodes); fitted.fit(samples)
ace_fit = gs.interventional_effect(fitted, a, b)
conf = sorted(set(pred_dags[p_].predecessors(a)) & set(pred_dags[p_].predecessors(b)))
stats["do_example"] = {"premise": int(p_), "x": a, "y": b, "confounders": conf, "observational": round(obs, 3),
                       "interventional": round(ace, 3), "fitted_ace": round(ace_fit, 3)}
fig, ax = plt.subplots(figsize=(4.4, 2.2))
labels = [f"observe\nP({b}|{a}=1) − P({b}|{a}=0)", f"intervene\nP({b}|do({a}=1)) − P({b}|do({a}=0))", "intervene, CPDs\nlearned from 5k samples"]
vals = [obs, ace, ace_fit]
bb = ax.barh(labels[::-1], vals[::-1], color=[TINT, NAVY, GREY][::-1], edgecolor=[NAVY, NAVY, GREY][::-1], height=0.6)
for r, v in zip(bb, vals[::-1]): ax.text(v + 0.01, r.get_y() + r.get_height()/2, f"{v:.3f}", va="center", fontsize=8.5, fontweight="bold")
ax.set_xlim(0, 0.85); ax.set_xlabel(f"effect of {a} on {b}"); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=7.5)
save(fig, "fig_do")

json.dump(stats, open(STATS, "w"), indent=1, default=str)
print(json.dumps(stats, indent=1, default=str))
