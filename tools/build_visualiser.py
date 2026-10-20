"""Build docs/pipeline_visualiser.html: real data points traced through the real pipeline,
next to the real notebook / source code, with line-by-line notes.

Run:  .venv/bin/python tools/build_visualiser.py
"""
from __future__ import annotations

import copy
import inspect
import json
import sys
import warnings
from itertools import combinations, product
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import nbformat
import networkx as nx
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from causal_nlp import causal_rules as cr
from causal_nlp import data as data_mod
from causal_nlp import features as feat_mod
from causal_nlp import graph_stage as gs
from causal_nlp import parsing as ps
from causal_nlp.data import load_paraphrase_test, load_split
from causal_nlp.features import (ALL_FEATURES, PC_GRAPH_FEATURES, RELATION_FEATURES, SUPPORT_FEATURES,
                                 TEXT_FACT_FEATURES, build_feature_table)
from visualiser_notes import FEATURE_MEANING, NOTES, STAGE_CODE, STAGE_TEXT

SEED = 42
OUT = ROOT / "docs" / "pipeline_visualiser.html"
TEMPLATE = Path(__file__).with_name("visualiser_template.html")

PREMISE_3 = ("A correlates with C. B correlates with C. However, A is independent of B.")
PREMISE_5 = "A correlates with C. A correlates with D. A correlates with E. B correlates with E. C correlates with D."
PREMISE_MEEK = None  # found automatically below

JOURNEYS = [
    # id, title, which set, premise key, hypothesis, why it is interesting
    ("j1", "Forced collider", "test", "p3", "A directly causes C.",
     "The smallest case: three facts force A → C ← B."),
    ("j2", "Wrong direction", "test", "p3", "B directly causes A.",
     "Same premise, a claim the graph rejects."),
    ("j3", "Ambiguous edge", "test", "p5", "A directly causes C.",
     "True in only 1 of 3 possible worlds, so not valid."),
    ("j4", "Hidden path", "test", "p5", "C is a cause for E, but not a direct one.",
     "We never learn which path C uses, yet every world has one."),
    ("j5", "Almost a confounder", "test", "p5", "There exists at least one confounder (i.e., common cause) of C and E.",
     "Holds in 2 of 3 worlds: 'might be' is not 'must be'."),
    ("j6", "Meek rules fire", "test", "pm", None,
     "An arrow no v-structure gives directly; Meek rule R1 forces it."),
    ("j7", "Paraphrased claim", "paraphrase", "p5", "A directly affects E.",
     "True claim, new wording: the regex reader fails at stage 1."),
]

STAGES = ["load", "parse", "skeleton", "vstruct", "meek", "dags", "features", "tree", "graph"]


# ---------------------------------------------------------------------------- helpers
def edges_of(pdag) -> dict:
    return {"directed": sorted(pdag.directed), "undirected": sorted(tuple(sorted(e)) for e in pdag.undirected)}


def layout(nodes: list[str]) -> dict:
    pos = nx.circular_layout(sorted(nodes))
    return {n: [round(float(p[0]), 4), round(float(p[1]), 4)] for n, p in pos.items()}


def fmt_set(s) -> str:
    return "{" + ",".join(sorted(s)) + "}"


def sentence_rules(premise_text: str) -> list[dict]:
    """Which regex reads each premise sentence, and what it extracts."""
    header = ps._HEADER.search(premise_text)
    out = [{"text": premise_text[:header.end()].strip(), "rule": "_HEADER",
            "fact": f"{header.group(1)} variables: {', '.join(ps._split_var_list(header.group(2)))}"}]
    body = premise_text[header.end():].strip()
    for sent in [s.strip() for s in body.replace("However, ", "").split(". ") if s.strip()]:
        s = sent if sent.endswith(".") else sent + "."
        if m := ps._CORR.search(s):
            out.append({"text": s, "rule": "_CORR", "fact": f"correlated({m.group(1)}, {m.group(2)})"})
        elif m := ps._MARG_INDEP.search(s):
            out.append({"text": s, "rule": "_MARG_INDEP", "fact": f"sepset({m.group(1)}{m.group(2)}) += {{}}"})
        elif m := ps._COND_INDEP.search(s):
            given = ps._split_var_list(m.group(3))
            out.append({"text": s, "rule": "_COND_INDEP", "fact": f"sepset({m.group(1)}{m.group(2)}) += {fmt_set(given)}"})
        else:
            out.append({"text": s, "rule": None, "fact": "no rule"})
    return out


def hypothesis_rule(text: str) -> dict:
    for i, (pattern, relation, gx, gy) in enumerate(ps.HYPOTHESIS_PATTERNS):
        if m := pattern.match(text.strip()):
            return {"pattern_index": i, "pattern": pattern.pattern, "relation": relation,
                    "x": m.group(gx), "y": m.group(gy)}
    return {"pattern_index": None}


def skeleton_table(premise) -> list[dict]:
    rows = []
    for pair in premise.pairs():
        a, b = sorted(pair)
        seps = premise.sepsets[pair]
        rows.append({"pair": a + b, "correlated": pair in premise.correlated,
                     "sepsets": [fmt_set(s) for s in seps], "joined": not seps})
    return rows


def vstructure_table(skel, premise) -> list[dict]:
    rows = []
    for x, y in combinations(skel.nodes, 2):
        if skel.adjacent(x, y):
            continue
        seps = premise.sepsets[frozenset((x, y))]
        for z in sorted(skel.neighbours(x) & skel.neighbours(y)):
            in_sep = [fmt_set(s) for s in seps if z in s]
            rows.append({"x": x, "y": y, "z": z, "sepsets": [fmt_set(s) for s in seps],
                         "z_in": in_sep, "collider": not in_sep})
    return rows


def meek_trace(pdag) -> list[dict]:
    """Same loop as apply_meek_rules, but records which rule oriented which edge and why."""
    pdag = copy.deepcopy(pdag)
    steps, changed = [], True
    while changed:
        changed = False
        for edge in sorted(pdag.undirected, key=lambda e: tuple(sorted(e))):
            a, b = sorted(edge)
            for u, v in ((a, b), (b, a)):
                if frozenset((u, v)) not in pdag.undirected:
                    break
                why = None
                w1 = [w for w in sorted(pdag.parents(u)) if w != v and not pdag.adjacent(w, v)]
                if w1:
                    why = ("R1", f"{w1[0]} → {u} – {v}, and {w1[0]}, {v} are not joined: "
                                 f"{v} → {u} would make a new collider at {u}")
                elif w2 := sorted(pdag.children(u) & pdag.parents(v)):
                    why = ("R2", f"{u} → {w2[0]} → {v} already exists: {v} → {u} would close a cycle")
                else:
                    und = {w for w in pdag.nodes if frozenset((u, w)) in pdag.undirected}
                    cands = sorted(und & pdag.parents(v))
                    pair = next(((p, q) for p, q in combinations(cands, 2) if not pdag.adjacent(p, q)), None)
                    if pair:
                        why = ("R3", f"{u} – {pair[0]} → {v} and {u} – {pair[1]} → {v} with {pair[0]}, {pair[1]} not joined")
                if why:
                    pdag.orient(u, v)
                    steps.append({"edge": [u, v], "rule": why[0], "reason": why[1]})
                    changed = True
    return steps


def dag_candidates(pdag, hyp) -> list[dict]:
    """Every way to orient the undirected edges, with the reason each is kept or rejected."""
    undirected = [tuple(sorted(e)) for e in pdag.undirected]
    base = nx.DiGraph(list(pdag.directed)); base.add_nodes_from(pdag.nodes)
    adjacency = base.to_undirected(); adjacency.add_edges_from(undirected)
    required = cr._v_structures(base, adjacency)
    out = []
    for flips in product((False, True), repeat=len(undirected)):
        dag = base.copy()
        dag.add_edges_from((b, a) if f else (a, b) for (a, b), f in zip(undirected, flips))
        status, reason = "kept", ""
        if not nx.is_directed_acyclic_graph(dag):
            status, reason = "rejected", "contains a cycle"
        else:
            extra = cr._v_structures(dag, adjacency) - required
            if extra:
                a, z, b = sorted(extra)[0]
                status, reason = "rejected", f"new collider {a} → {z} ← {b} (the facts do not show one)"
        holds = cr.relation_holds(dag, hyp.relation, hyp.x, hyp.y) if (hyp and status == "kept") else None
        out.append({"edges": sorted(dag.edges), "status": status, "reason": reason, "holds": holds})
    return out


def tree_dict(model) -> dict:
    t = model.tree_
    nodes = []
    for i in range(t.node_count):
        leaf = t.children_left[i] == -1
        counts = t.value[i][0]
        nodes.append({"id": i, "leaf": bool(leaf),
                      "feature": None if leaf else ALL_FEATURES[t.feature[i]],
                      "threshold": None if leaf else round(float(t.threshold[i]), 2),
                      "left": int(t.children_left[i]), "right": int(t.children_right[i]),
                      "p_valid": round(float(counts[1] / counts.sum()), 3),
                      "samples": int(t.n_node_samples[i])})
    return {"nodes": nodes}


def bn_tables(dag) -> dict:
    bn = gs.noisy_or_network(dag)
    cpds = {}
    for node in sorted(dag.nodes):
        parents = sorted(dag.predecessors(node))
        rows = [{"parents": dict(zip(parents, states)), "p1": round(1 - 0.9 * 0.3 ** sum(states), 4)}
                for states in product((0, 1), repeat=len(parents))]
        cpds[node] = {"parents": parents, "rows": rows}
    queries = {}
    for x in sorted(dag.nodes):
        for y in sorted(dag.nodes):
            if x != y:
                queries[f"{x}{y}"] = {"ace": round(gs.interventional_effect(bn, x, y), 3),
                                      "obs": round(gs.observational_effect(bn, x, y), 3)}
    return {"cpds": cpds, "queries": queries}


# ------------------------------------------------------------------------- code extraction
def notebook_cells() -> list[dict]:
    nb = nbformat.read(ROOT / "notebooks" / "01_phase1_ml_pipeline.ipynb", 4)
    out, heading = [], ""
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown":
            first = c.source.strip().splitlines()[0]
            if first.startswith("#"):
                heading = first.lstrip("# ").strip()
            out.append({"cell": i, "kind": "markdown", "heading": heading, "source": c.source})
        else:
            out.append({"cell": i, "kind": "code", "heading": heading, "source": c.source})
    return out


def source_block(path: Path, start: str, end: str) -> tuple[list[str], int]:
    lines = path.read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if start in l)
    j = next(k for k in range(i, len(lines)) if end in lines[k])
    return lines[i:j + 1], i + 1


def snippet(sid: str) -> dict:
    """sid is 'nb:<cell>' or 'py:<module>.<name>' or 'blk:<module>:<start>|<end>'."""
    if sid.startswith("nb:"):
        cell = int(sid[3:])
        src = NB_BY_CELL[cell]["source"].splitlines()
        title, where, start = f"Notebook cell {cell}", "notebooks/01_phase1_ml_pipeline.ipynb", 1
    elif sid.startswith("py:"):
        mod_name, obj_name = sid[3:].split(".", 1)
        mod = MODULES[mod_name]
        src, start = inspect.getsourcelines(getattr(mod, obj_name))
        src = [l.rstrip("\n") for l in src]
        title, where = f"{obj_name}()" if inspect.isfunction(getattr(mod, obj_name)) else obj_name, f"src/causal_nlp/{mod_name}.py"
    else:
        _, mod_name, rng = sid.split(":", 2)
        a, b = rng.split("|")
        src, start = source_block(ROOT / "src" / "causal_nlp" / f"{mod_name}.py", a, b)
        title, where = NOTES.get(sid, {}).get("title", "module constants"), f"src/causal_nlp/{mod_name}.py"
    notes = attach_notes(sid, src)
    return {"id": sid, "title": NOTES.get(sid, {}).get("title", title), "where": where, "start": start,
            "lines": src, "notes": notes, "summary": NOTES.get(sid, {}).get("summary", "")}


def attach_notes(sid: str, lines: list[str]) -> dict:
    """Match each (substring, note) pair to the next line containing the substring."""
    out, cursor = {}, 0
    for sub, note in NOTES.get(sid, {}).get("lines", []):
        idx = next((k for k in range(cursor, len(lines)) if sub in lines[k]), None)
        if idx is None:
            raise ValueError(f"[{sid}] note anchor not found after line {cursor}: {sub!r}")
        out[idx] = (out[idx] + " " + note) if idx in out else note
        cursor = idx
    return out


MODULES = {"parsing": ps, "causal_rules": cr, "features": feat_mod, "graph_stage": gs, "data": data_mod}
NB = notebook_cells()
NB_BY_CELL = {c["cell"]: c for c in NB}


# ------------------------------------------------------------------------------- build
def main() -> None:
    print("loading data / training the depth-4 tree (same settings as the notebook)...")
    train, test, para = load_split("train"), load_split("test"), load_paraphrase_test()
    f_train = build_feature_table(train.input)
    tree = DecisionTreeClassifier(max_depth=4, random_state=SEED).fit(f_train[ALL_FEATURES], train.label)

    def premise_for(key: str) -> str:
        if key == "p3":
            return next(p for p in test.premise.unique() if p.endswith(PREMISE_3) and "3 variables" in p)
        if key == "p5":
            return next(p for p in test.premise.unique() if PREMISE_5 in p)
        # first test premise (<= 5 vars) where Meek orients an edge
        for p in test.premise.unique():
            pr = ps.parse_premise(p)
            if len(pr.variables) > 5:
                continue
            v = cr.orient_v_structures(cr.skeleton(pr), pr)
            if meek_trace(v):
                return p
        raise RuntimeError("no Meek example")

    premises = {k: premise_for(k) for k in ("p3", "p5", "pm")}
    journeys = []
    for jid, title, split, pkey, hyp_text, blurb in JOURNEYS:
        df = test if split == "test" else para
        prem_text = premises[pkey]
        if hyp_text is None:  # a 'directly causes' claim on an edge that only a Meek rule orients
            pr = ps.parse_premise(prem_text)
            steps = meek_trace(cr.orient_v_structures(cr.skeleton(pr), pr))
            have = set(df[df.premise == prem_text].hypothesis)
            hyp_text = next(t for st in steps if (t := f"{st['edge'][0]} directly causes {st['edge'][1]}.") in have)
        row = df[(df.premise == prem_text) & (df.hypothesis == hyp_text)].iloc[0]
        premise = ps.parse_premise(prem_text)
        hyp = ps.parse_hypothesis(hyp_text)
        skel = cr.skeleton(premise)
        vs = cr.orient_v_structures(copy.deepcopy(skel), premise)
        final = cr.apply_meek_rules(copy.deepcopy(vs))
        feats = build_feature_table(pd.Series([row.input])).iloc[0]
        j = {
            "id": jid, "title": title, "blurb": blurb, "split": split, "premise_key": pkey,
            "input": row.input, "premise": prem_text, "hypothesis": hyp_text, "label": int(row.label),
            "columns": {"input": "…", "label": int(row.label), "premise_id": int(row.premise_id)},
            "variables": premise.variables, "pos": layout(premise.variables),
            "sentences": sentence_rules(prem_text), "hyp_rule": hypothesis_rule(hyp_text),
            "correlated": sorted("".join(sorted(p)) for p in premise.correlated),
            "sepsets": {"".join(sorted(k)): [fmt_set(s) for s in v] for k, v in premise.sepsets.items()},
            "skeleton": edges_of(skel), "skeleton_table": skeleton_table(premise),
            "vstruct": edges_of(vs), "vstruct_table": vstructure_table(skel, premise),
            "meek": edges_of(final), "meek_steps": meek_trace(vs),
            "dags": dag_candidates(final, hyp),
            "hyp": None if hyp is None else {"relation": hyp.relation, "x": hyp.x, "y": hyp.y},
        }
        if hyp is not None:
            x_row = feats[ALL_FEATURES].to_frame().T.astype(float)
            path = [int(n) for n in tree.decision_path(x_row).indices]
            j["features"] = {k: float(feats[k]) for k in ALL_FEATURES}
            j["tree_path"] = path
            j["p_valid"] = float(tree.predict_proba(x_row)[0, 1])
            j["prediction"] = int(tree.predict(x_row)[0])
        else:
            j["features"], j["tree_path"], j["p_valid"], j["prediction"] = None, [], 0.0, 0
        journeys.append(j)

    # Stage 5: one block per premise (and the paraphrase version of p5)
    graph_stage = {}
    for key, prem_text in premises.items():
        for split, df in (("test", test), ("paraphrase", para)):
            if split == "paraphrase" and key != "p5":
                continue
            rows = df[df.premise == prem_text]
            f = build_feature_table(rows.input)
            prob = pd.Series(0.0, index=rows.index)
            ok = f.parsed == 1
            if ok.any():
                prob[ok] = tree.predict_proba(f.loc[ok, ALL_FEATURES])[:, 1]
            tri = pd.DataFrame({"premise_id": 0, "relation": f.get("relation"), "x": f.get("x"), "y": f.get("y"),
                                "label": rows.label, "prob": prob, "hypothesis": rows.hypothesis})
            premise = ps.parse_premise(prem_text)
            direct = tri[(tri.relation == ps.DIRECT) & (tri.prob >= 0.5)]
            dag, rejected = gs.build_dag(premise.variables, list(zip(direct.x, direct.y, direct.prob)))
            gold_rows = test[test.premise == prem_text]
            gf = build_feature_table(gold_rows.input)
            gold_tri = pd.DataFrame({"premise_id": 0, "relation": gf.relation, "x": gf.x, "y": gf.y,
                                     "label": gold_rows.label})
            gold_edges = sorted(zip(gold_tri[(gold_tri.relation == ps.DIRECT) & (gold_tri.label == 1)].x,
                                    gold_tri[(gold_tri.relation == ps.DIRECT) & (gold_tri.label == 1)].y))
            known = gs.known_intervention_queries(gold_tri, {0: (premise.variables, premise.correlated)})
            graph_stage[f"{key}:{split}"] = {
                "n_hypotheses": int(len(tri)),
                "n_parsed": int(ok.sum()),
                "valid": [{"hypothesis": h, "relation": r, "x": x, "y": y, "prob": round(float(p), 3), "label": int(l)}
                          for h, r, x, y, p, l in zip(tri.hypothesis, tri.relation, tri.x, tri.y, tri.prob, tri.label)
                          if p >= 0.5],
                "dag_edges": sorted(dag.edges), "rejected": [list(r[:2]) for r in rejected],
                "gold_edges": gold_edges,
                "known": {f"{a}{b}": bool(g) for a, b, g in zip(known.x, known.y, known.gold_effect)},
                **bn_tables(dag),
            }

    stage_code = {stage: [snippet(s) for s in sids] for stage, sids in STAGE_CODE.items()}
    nb_code = [{**c, "snippet": snippet(f"nb:{c['cell']}")} if c["kind"] == "code" else c for c in NB]

    payload = {
        "journeys": journeys, "graph_stage": graph_stage, "tree": tree_dict(tree),
        "stages": STAGES, "stage_code": stage_code, "notebook": nb_code,
        "feature_groups": {"relation": RELATION_FEATURES, "text facts": TEXT_FACT_FEATURES,
                           "PC graph": PC_GRAPH_FEATURES, "support": SUPPORT_FEATURES},
        "feature_meaning": FEATURE_MEANING, "stage_text": STAGE_TEXT,
    }
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(TEMPLATE.read_text().replace("/*__DATA__*/", blob))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
