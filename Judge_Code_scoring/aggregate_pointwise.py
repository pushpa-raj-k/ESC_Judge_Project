"""
aggregate_pointwise.py
======================
Turns the raw pointwise scores in results_pointwise/*.jsonl into winners, the same
configurations as the pairwise study, and stability / agreement metrics.

    python aggregate_pointwise.py --results results_pointwise --rubric rubric_scored.json --out analysis_pointwise

How a winner is derived (per judge, persona, dimension)
  gap = (score_x - score_y) / spread        x, y = the two supporter models (x = alphabetically first)
  |gap| <= delta  -> "tie",   gap > 0 -> x wins,   gap < 0 -> y wins
  spread = the judge's standard deviation on that dimension (--normalize sd, default) or 1 (--normalize none),
  so delta is in "SD units" (default) or raw score points.  Choose delta BEFORE looking at human agreement,
  and read delta_sensitivity.csv to see how much it matters.

Configurations (same names as aggregate_results.py, so downstream code can treat both alike)
  single/<judge>      run 0 only (baseline)
  sc/<judge>          mean score over all runs (self-consistency)
  ensemble_single     majority vote of the judges' single-run winners
  ensemble_sc         majority vote of the judges' sc winners
  ensemble_sc_mean    average of the judges' sc gaps, then thresholded (uses the size of the gap)

Majority rule: strict majority wins; otherwise "tie"; needs >= --min-valid valid votes.

Outputs (in --out)
  verdicts_long.csv       one row per (config, pair, dimension), incl. the gap
  category_scores.csv     Exploration / Insight / Action preference per config, with bootstrap 95% CI
  stability.csv           per judge x dimension: parse rate, score spread, ICC, run-to-run winner agreement, ...
  mean_scores.csv         mean raw score per judge x supporter model x dimension
  judge_agreement.csv     pairwise judge agreement + Cohen's kappa + Spearman of score gaps
  judge_agreement_fleiss.csv   Fleiss' kappa across all judges
  delta_sensitivity.csv   tie rate / agreement as delta changes
"""
import argparse
import glob
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from esc_judge_pointwise import load_scored_rubric

LABELS = ("x", "y", "tie")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def majority(votes, min_valid=2):
    v = [x for x in votes if x is not None]
    if len(v) < min_valid:
        return None
    top = Counter(v).most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "tie"
    return top[0][0]


def to_winner(gap, delta):
    if gap is None:
        return None
    if abs(gap) <= delta:
        return "tie"
    return "x" if gap > 0 else "y"


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fleiss_kappa(rows):
    """rows: one list of category counts per item; every item must have the same number of raters."""
    N = len(rows)
    if N == 0:
        return float("nan")
    n = sum(rows[0])
    if n < 2:
        return float("nan")
    p_i = [(sum(c * c for c in r) - n) / (n * (n - 1)) for r in rows]
    pbar = sum(p_i) / N
    pj = [sum(r[i] for r in rows) / (N * n) for i in range(len(rows[0]))]
    pe = sum(p * p for p in pj)
    return (pbar - pe) / (1 - pe) if pe < 1 else float("nan")


def icc1(groups):
    """One-way random-effects ICC(1,1): share of score variance that is between conversations
    (1 = runs always reproduce the same score, 0 = scores are pure noise). Handles unequal group sizes."""
    groups = [g for g in groups if len(g) >= 2]
    g = len(groups)
    N = sum(len(x) for x in groups)
    if g < 2 or N - g <= 0:
        return None
    grand = sum(sum(x) for x in groups) / N
    ssb = sum(len(x) * (statistics.mean(x) - grand) ** 2 for x in groups)
    ssw = sum(sum((v - statistics.mean(x)) ** 2 for v in x) for x in groups)
    msb, msw = ssb / (g - 1), ssw / (N - g)
    k0 = (N - sum(len(x) ** 2 for x in groups) / N) / (g - 1)
    denom = msb + (k0 - 1) * msw
    return (msb - msw) / denom if denom > 0 else None


def boot_ci(vals, n_boot=1000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) < 2 or n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def load_runs(results_dir):
    """Best record per (judge, pid, supporter_model, run)."""
    best = {}
    for path in sorted(glob.glob(str(Path(results_dir) / "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "scores" not in r:
                    raise SystemExit(f"{path} has no 'scores' field - is it a pairwise results file? "
                                     "Point --results at the pointwise output folder.")
                key = (r["judge"], str(r["pid"]), r["supporter_model"], r["run"])
                if key not in best or r["n_parsed"] >= best[key]["n_parsed"]:
                    best[key] = r
    return list(best.values())


def judge_pair_stats(Wm, judges):
    out = []
    for a, b in itertools.combinations(judges, 2):
        keys = [k for k, v in Wm[a].items() if v is not None and Wm[b].get(k) is not None]
        xa, xb = [Wm[a][k] for k in keys], [Wm[b][k] for k in keys]
        n = len(keys)
        out.append({"judge_a": a, "judge_b": b, "n": n,
                    "agreement": sum(p == q for p, q in zip(xa, xb)) / n if n else float("nan"),
                    "cohen_kappa": cohen_kappa(xa, xb) if n else float("nan")})
    return out


def fleiss_for(Wm, judges):
    if len(judges) < 2:
        return float("nan"), 0
    keys = set.intersection(*[{k for k, v in Wm[j].items() if v is not None} for j in judges])
    rows = [[sum(Wm[j][k] == lab for j in judges) for lab in LABELS] for k in keys]
    return fleiss_kappa(rows), len(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results_pointwise")
    ap.add_argument("--rubric", default="rubric_scored.json")
    ap.add_argument("--out", default="analysis_pointwise")
    ap.add_argument("--normalize", choices=["sd", "none"], default="sd",
                    help="sd: divide each gap by the judge's SD on that dimension; none: raw score points")
    ap.add_argument("--delta", type=float, default=0.25,
                    help="tie if |gap| <= delta (SD units with --normalize sd, score points with none)")
    ap.add_argument("--delta-grid", default="0,0.1,0.25,0.5,0.75,1.0", help="deltas for delta_sensitivity.csv")
    ap.add_argument("--min-valid", type=int, default=2,
                    help="min valid runs for a conversation's mean score, and min valid votes for a majority")
    ap.add_argument("--allow-different-seeker", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples over personas (0 = skip CI)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    mv, delta = args.min_valid, args.delta

    rub = load_scored_rubric(args.rubric)
    dims = rub.dims
    cat_of = {d.name: d.category for d in dims}
    recs = load_runs(args.results)
    if not recs:
        raise SystemExit(f"No results found in {args.results}/")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # ---- collect scores: S[judge][(pid, supporter)][dim] = {run: score} ----
    S = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    raw = defaultdict(list)
    n_rec = Counter()
    seeker = {}
    for r in recs:
        j, cs = r["judge"], (str(r["pid"]), r["supporter_model"])
        n_rec[j] += 1
        seeker[cs] = r.get("seeker_model")
        for d in dims:
            v = r["scores"].get(d.name)
            if v is not None:
                S[j][cs][d.name][r["run"]] = float(v)
                raw[(j, d.name)].append(float(v))
    judges = sorted(S)
    print(f"Loaded {len(recs)} runs from judges: {judges}")

    # ---- pairs: same persona, two different supporter models (x = alphabetically first) ----
    sup_by_pid = defaultdict(set)
    for j in judges:
        for pid, sup in S[j]:
            sup_by_pid[pid].add(sup)
    pairs, skipped = [], 0
    for pid in sorted(sup_by_pid):
        for mx, my in itertools.combinations(sorted(sup_by_pid[pid]), 2):
            if not args.allow_different_seeker and seeker.get((pid, mx)) != seeker.get((pid, my)):
                skipped += 1
                continue
            pairs.append((pid, mx, my))
    print(f"{len(pairs)} pairs ({skipped} skipped: different seeker model)")

    # ---- per-judge spread used to normalise gaps ----
    spread = {}
    for j in judges:
        for d in dims:
            vals = raw.get((j, d.name), [])
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            spread[(j, d.name)] = sd if (args.normalize == "sd" and sd > 1e-9) else 1.0

    # ---- score gaps: single (run 0), sc (mean of runs), and per-run ----
    gap = {"single": defaultdict(dict), "sc": defaultdict(dict)}   # gap[mode][judge][(pk, dim)]
    rgap = {}                                                       # rgap[(judge, pk, dim)] = {run: gap}
    for j in judges:
        for pk in pairs:
            pid, mx, my = pk
            for d in dims:
                sx = S[j].get((pid, mx), {}).get(d.name, {})
                sy = S[j].get((pid, my), {}).get(d.name, {})
                sp = spread[(j, d.name)]
                rgap[(j, pk, d.name)] = {k: (sx[k] - sy[k]) / sp for k in sorted(set(sx) & set(sy))}
                if 0 in sx and 0 in sy:
                    gap["single"][j][(pk, d.name)] = (sx[0] - sy[0]) / sp
                if len(sx) >= mv and len(sy) >= mv:
                    gap["sc"][j][(pk, d.name)] = (statistics.mean(sx.values()) - statistics.mean(sy.values())) / sp

    def winners(mode, dl):
        return {j: {k: to_winner(v, dl) for k, v in gap[mode][j].items()} for j in judges}

    # ---- verdicts_long.csv ----
    W = {m: winners(m, delta) for m in gap}
    rows = []
    for pk in pairs:
        for d in dims:
            key = (pk, d.name)
            for j in judges:
                for mode in ("single", "sc"):
                    g = gap[mode][j].get(key)
                    rows.append((f"{mode}/{j}", pk, d.name, g, to_winner(g, delta)))
            if len(judges) >= 2:
                rows.append(("ensemble_single", pk, d.name, None,
                             majority([W["single"][j].get(key) for j in judges], mv)))
                rows.append(("ensemble_sc", pk, d.name, None,
                             majority([W["sc"][j].get(key) for j in judges], mv)))
                gs = [gap["sc"][j][key] for j in judges if key in gap["sc"][j]]
                gm = statistics.mean(gs) if len(gs) >= mv else None
                rows.append(("ensemble_sc_mean", pk, d.name, gm, to_winner(gm, delta)))
    long = pd.DataFrame([{
        "config": c, "pid": pk[0], "model_x": pk[1], "model_y": pk[2], "dimension": dim,
        "category": cat_of[dim], "score_gap": g, "winner_xy": w,
        "winner": None if w is None else (pk[1] if w == "x" else pk[2] if w == "y" else "tie"),
    } for c, pk, dim, g, w in rows])
    long.to_csv(Path(args.out) / "verdicts_long.csv", index=False)

    # ---- category_scores.csv (paper-style, plus bootstrap CI over personas) ----
    long["w"] = long["winner_xy"].map({"x": 1.0, "y": 0.0, "tie": 0.5})
    per_role = (long.dropna(subset=["w"])
                .groupby(["config", "model_x", "model_y", "pid", "category"])["w"].mean().reset_index())
    cat_rows = []
    for (cfg, mx, my, c), g in per_role.groupby(["config", "model_x", "model_y", "category"]):
        s = g["w"].mean()
        lo, hi = boot_ci(g["w"].values, args.bootstrap, args.seed)
        cat_rows.append({
            "config": cfg, "model_x": mx, "model_y": my, "category": c, "S_x_over_y": s,
            "ci_low": lo, "ci_high": hi, "n_roles": len(g),
            "preferred": "tie" if abs(s - 0.5) < 1e-9 else "model_x" if s > 0.5 else "model_y",
            "ci_excludes_0.5": bool(not (lo <= 0.5 <= hi)) if not np.isnan(lo) else None,
        })
    cat = pd.DataFrame(cat_rows)
    cat.to_csv(Path(args.out) / "category_scores.csv", index=False)

    # ---- stability.csv ----
    st = []
    for j in judges:
        for d in dims:
            vals = raw.get((j, d.name), [])
            groups = [list(S[j][cs][d.name].values()) for cs in S[j] if d.name in S[j][cs]]
            within = [statistics.stdev(g) for g in groups if len(g) >= 2]
            modal = Counter(vals).most_common(1)
            agree_all, agree_pair, n_ties, n_runs_w = [], [], 0, 0
            for pk in pairs:
                ws = [to_winner(v, delta) for v in rgap.get((j, pk, d.name), {}).values()]
                n_runs_w += len(ws)
                n_ties += ws.count("tie")
                if len(ws) >= 2:
                    agree_all.append(len(set(ws)) == 1)
                    agree_pair += [a == b for a, b in itertools.combinations(ws, 2)]
            st.append({
                "judge": j, "dimension": d.name, "category": d.category,
                "parse_rate": len(vals) / n_rec[j] if n_rec[j] else None,
                "mean_score": statistics.mean(vals) if vals else None,
                "sd_score": statistics.stdev(vals) if len(vals) > 1 else None,
                "modal_score": modal[0][0] if modal else None,
                "modal_score_share": modal[0][1] / len(vals) if vals else None,
                "within_conv_sd": statistics.mean(within) if within else None,
                "icc1": icc1(groups),
                "winner_all_runs_agree": sum(agree_all) / len(agree_all) if agree_all else None,
                "winner_pairwise_run_agreement": sum(agree_pair) / len(agree_pair) if agree_pair else None,
                "tie_rate_per_run": n_ties / n_runs_w if n_runs_w else None,
            })
    st = pd.DataFrame(st)
    st.to_csv(Path(args.out) / "stability.csv", index=False)

    # ---- mean_scores.csv ----
    ms = []
    for j in judges:
        for sup in sorted({cs[1] for cs in S[j]}):
            for d in dims:
                means = [statistics.mean(S[j][cs][d.name].values())
                         for cs in S[j] if cs[1] == sup and S[j][cs].get(d.name)]
                ms.append({"judge": j, "supporter_model": sup, "dimension": d.name, "category": d.category,
                           "n_conversations": len(means),
                           "mean_score": statistics.mean(means) if means else None,
                           "sd_across_conversations": statistics.stdev(means) if len(means) > 1 else None})
    pd.DataFrame(ms).to_csv(Path(args.out) / "mean_scores.csv", index=False)

    # ---- judge_agreement.csv (+ Fleiss) ----
    ja, fl = [], []
    for mode in ("single", "sc"):
        for row in judge_pair_stats(W[mode], judges):
            a, b = row["judge_a"], row["judge_b"]
            keys = [k for k in gap[mode][a] if k in gap[mode][b]]
            sp = float("nan")
            if len(keys) >= 3:
                sp = pd.Series([gap[mode][a][k] for k in keys]).corr(
                    pd.Series([gap[mode][b][k] for k in keys]), method="spearman")
            ja.append({"mode": mode, **row, "spearman_score_gap": sp})
        fk, nf = fleiss_for(W[mode], judges)
        fl.append({"mode": mode, "n_judges": len(judges), "n_items": nf, "fleiss_kappa": fk})
    ja, fl = pd.DataFrame(ja), pd.DataFrame(fl)
    ja.to_csv(Path(args.out) / "judge_agreement.csv", index=False)
    fl.to_csv(Path(args.out) / "judge_agreement_fleiss.csv", index=False)

    # ---- delta_sensitivity.csv ----
    sens = []
    for dl in [float(x) for x in args.delta_grid.split(",") if x.strip()]:
        Wm = winners("sc", dl)
        for j in judges:
            vals = [v for v in Wm[j].values() if v is not None]
            n = len(vals)
            sens.append({"delta": dl, "judge": j, "n": n,
                         "tie_rate": vals.count("tie") / n if n else None,
                         "x_win_rate": vals.count("x") / n if n else None,
                         "y_win_rate": vals.count("y") / n if n else None,
                         "mean_pairwise_agreement": None, "fleiss_kappa": None})
        if len(judges) >= 2:
            ps = judge_pair_stats(Wm, judges)
            fk, nf = fleiss_for(Wm, judges)
            sens.append({"delta": dl, "judge": "ALL", "n": nf, "tie_rate": None, "x_win_rate": None,
                         "y_win_rate": None,
                         "mean_pairwise_agreement": float(np.nanmean([p["agreement"] for p in ps])),
                         "fleiss_kappa": fk})
    pd.DataFrame(sens).to_csv(Path(args.out) / "delta_sensitivity.csv", index=False)

    # ---- console summary ----
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"\n(normalize={args.normalize}, delta={delta}, min_valid={mv})")
    print("\n== Stability per judge (mean over dimensions) ==")
    print(st.groupby("judge")[["parse_rate", "mean_score", "sd_score", "modal_score_share", "within_conv_sd",
                               "icc1", "winner_all_runs_agree", "winner_pairwise_run_agreement",
                               "tie_rate_per_run"]].mean().round(3))
    print("\n== Category preference (S_x_over_y > 0.5 -> model_x preferred) ==")
    if not cat.empty:
        print(cat.pivot_table(index=["model_x", "model_y", "config"], columns="category",
                              values="S_x_over_y").round(3))
    print("\n== Judge agreement ==")
    print(ja.round(3))
    print(fl.round(3))
    print(f"\nWrote CSVs to {args.out}/")


if __name__ == "__main__":
    main()
