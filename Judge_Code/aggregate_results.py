"""
aggregate_results.py
====================
Turns the raw judge runs in results/*.jsonl into the three configurations from the SOP
and computes stability metrics.

    python aggregate_results.py --results results --rubric rubric_9dim.json --out analysis

Configurations (per pair and dimension)
  single/<judge>     run 0 of that judge only            (baseline, like the paper)
  sc/<judge>         majority vote over that judge's N runs (self-consistency)
  ensemble_single    majority vote over the judges' run-0 verdicts   (SOP ensemble)
  ensemble_sc        majority vote over the judges' self-consistency verdicts (bonus)

Voting rule: strict majority wins. No strict majority (e.g. X / Y / tie, or X / Y) -> "tie",
which matches the paper's rule "if verdicts disagree, choose tie". Needs >= 2 valid votes.

Outputs (in --out):
  verdicts_long.csv    one row per (config, pair, dimension)
  category_scores.csv  paper-style Exploration / Insight / Action preference per config
  stability.csv        per judge x dimension: run agreement, order consistency, first-slot rate, ...
  judge_agreement.csv  pairwise agreement between judges (run-0 verdicts)
"""
import argparse
import glob
import json
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from esc_judge_pipeline import load_rubric


def majority(votes, min_valid=2):
    v = [x for x in votes if x is not None]
    if len(v) < min_valid:
        return None
    top = Counter(v).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "tie"
    return top[0][0]


def load_runs(results_dir):
    """Return best record per (judge, pid, model_x, model_y, run)."""
    best = {}
    for path in sorted(glob.glob(str(Path(results_dir) / "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (r["judge"], r["pid"], r["model_x"], r["model_y"], r["run"])
                if key not in best or r["n_parsed"] >= best[key]["n_parsed"]:
                    best[key] = r
    return list(best.values())


def to_xy(rec, winner):
    if winner is None:
        return None
    if winner == "tie":
        return "tie"
    return "x" if winner == rec["model_x"] else "y"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--rubric", default="rubric_9dim.json")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--min-valid", type=int, default=2, help="min valid votes needed for a majority")
    args = ap.parse_args(argv)

    dims = load_rubric(args.rubric)
    cat_of = {d.name: d.category for d in dims}
    recs = load_runs(args.results)
    if not recs:
        raise SystemExit(f"No results found in {args.results}/")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # votes[judge][(pid,mx,my)][dim] = {run: 'x'|'y'|'tie'|None}
    votes = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    first_slot = defaultdict(list)       # (judge, dim) -> list of 'A'/'B'/'Tie'/None
    for r in recs:
        pk = (r["pid"], r["model_x"], r["model_y"])
        for d in dims:
            votes[r["judge"]][pk][d.name][r["run"]] = to_xy(r, r["winners"].get(d.name))
            first_slot[(r["judge"], d.name)].append(r["verdicts_presented"].get(d.name))
    judges = sorted(votes)
    print(f"Loaded {len(recs)} runs from judges: {judges}")

    # ---------------- build verdicts per config ----------------
    rows = []
    sc_verdict, single_verdict = {}, {}
    for j in judges:
        for pk, dd in votes[j].items():
            for dim in (d.name for d in dims):
                runs = dd.get(dim, {})
                single_verdict[(j, pk, dim)] = runs.get(0)
                sc_verdict[(j, pk, dim)] = majority(list(runs.values()), args.min_valid)
    all_pk = {pk for j in judges for pk in votes[j]}
    for pk in sorted(all_pk):
        for d in dims:
            for j in judges:
                rows.append((f"single/{j}", pk, d.name, single_verdict.get((j, pk, d.name))))
                rows.append((f"sc/{j}", pk, d.name, sc_verdict.get((j, pk, d.name))))
            if len(judges) >= 2:
                rows.append(("ensemble_single", pk, d.name,
                             majority([single_verdict.get((j, pk, d.name)) for j in judges], args.min_valid)))
                rows.append(("ensemble_sc", pk, d.name,
                             majority([sc_verdict.get((j, pk, d.name)) for j in judges], args.min_valid)))

    long = pd.DataFrame([{
        "config": c, "pid": pk[0], "model_x": pk[1], "model_y": pk[2],
        "dimension": dim, "category": cat_of[dim], "winner_xy": w,
        "winner": None if w is None else (pk[1] if w == "x" else pk[2] if w == "y" else "tie"),
    } for c, pk, dim, w in rows])
    long.to_csv(Path(args.out) / "verdicts_long.csv", index=False)

    # ---------------- paper-style category scores ----------------
    score = {"x": 1.0, "y": 0.0, "tie": 0.5}
    long["w"] = long["winner_xy"].map(score)
    per_role = (long.dropna(subset=["w"])
                .groupby(["config", "model_x", "model_y", "pid", "category"])["w"].mean().reset_index())
    cat = (per_role.groupby(["config", "model_x", "model_y", "category"])["w"]
           .agg(S_x_over_y="mean", n_roles="count").reset_index())
    cat["preferred"] = cat["S_x_over_y"].apply(
        lambda s: "tie" if abs(s - 0.5) < 1e-9 else "model_x" if s > 0.5 else "model_y")
    cat.to_csv(Path(args.out) / "category_scores.csv", index=False)

    # ---------------- stability per judge x dimension ----------------
    st = []
    for j in judges:
        for d in dims:
            all_agree, pair_agree, order_cons, n_total, n_valid = [], [], [], 0, 0
            for pk, dd in votes[j].items():
                runs = dd.get(d.name, {})
                n_total += len(runs)
                vals = [v for v in runs.values() if v is not None]
                n_valid += len(vals)
                if len(vals) >= 2:
                    all_agree.append(len(set(vals)) == 1)
                    pair_agree += [a == b for a, b in itertools.combinations(vals, 2)]
                if runs.get(0) is not None and runs.get(1) is not None:   # run 0 vs 1 = opposite order
                    order_cons.append(runs[0] == runs[1])
            fs = [v for v in first_slot[(j, d.name)] if v is not None]
            nontie = [v for v in fs if v != "Tie"]
            st.append({
                "judge": j, "dimension": d.name, "category": d.category,
                "parse_rate": n_valid / n_total if n_total else None,
                "all_runs_agree": sum(all_agree) / len(all_agree) if all_agree else None,
                "mean_pairwise_run_agreement": sum(pair_agree) / len(pair_agree) if pair_agree else None,
                "order_consistency_run0_vs_run1": sum(order_cons) / len(order_cons) if order_cons else None,
                "tie_rate": fs.count("Tie") / len(fs) if fs else None,
                "first_slot_rate_among_nontie": nontie.count("A") / len(nontie) if nontie else None,
            })
    st = pd.DataFrame(st)
    st.to_csv(Path(args.out) / "stability.csv", index=False)

    # ---------------- agreement between judges (run-0 verdicts) ----------------
    ja = []
    for a, b in itertools.combinations(judges, 2):
        same, n = 0, 0
        for pk in votes[a]:
            if pk not in votes[b]:
                continue
            for d in dims:
                va, vb = single_verdict.get((a, pk, d.name)), single_verdict.get((b, pk, d.name))
                if va is not None and vb is not None:
                    n += 1
                    same += va == vb
        ja.append({"judge_a": a, "judge_b": b, "n": n, "agreement": same / n if n else None})
    pd.DataFrame(ja).to_csv(Path(args.out) / "judge_agreement.csv", index=False)

    # ---------------- console summary ----------------
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n== Stability per judge (mean over 9 dimensions) ==")
    print(st.groupby("judge")[["parse_rate", "all_runs_agree", "mean_pairwise_run_agreement",
                               "order_consistency_run0_vs_run1", "tie_rate",
                               "first_slot_rate_among_nontie"]].mean().round(3))
    print("\n== Category preference (S_x_over_y > 0.5 -> model_x preferred) ==")
    print(cat.pivot_table(index=["model_x", "model_y", "config"], columns="category",
                          values="S_x_over_y").round(3))
    print("\n== Judge agreement ==")
    print(pd.DataFrame(ja).round(3))
    print(f"\nWrote CSVs to {args.out}/")


if __name__ == "__main__":
    main()
