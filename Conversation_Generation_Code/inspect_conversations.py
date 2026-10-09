"""
inspect_conversations.py  --  quick quality check of generated conversations (Step 2).

Prints a per-conversation report and, optionally, the FULL text of some conversations
(no 300-character cut-off), so you can judge them properly before the long runs.

Checks per conversation
  * number of messages (expected 15)
  * average words per seeker / supporter reply
  * seeker replies containing asterisks (stage directions that slipped through)
  * highest similarity between two seeker replies (loops; > 0.6 is suspicious)
  * number of 'thank you' / closure phrases in the seeker's replies (persona drift)
  * how many replies hit the token limit and were trimmed
  * markdown headings / list lines in supporter replies (listy style)

Usage
    python inspect_conversations.py output/exp_Qwen-Qwen2.5-3B-Instruct.jsonl            # report only
    python inspect_conversations.py output/exp_Qwen-Qwen2.5-3B-Instruct.jsonl --full 2   # + 2 full conversations
    python inspect_conversations.py FILE.jsonl --pid 3e9f2ab1_baseline                   # one specific persona
"""
import argparse
import difflib
import json
import re

CLOSURE = re.compile(r"thank you|thanks|grateful|closure|relieved|at peace|renewed|appreciate", re.I)


def words(t):
    return len(t.split())


def report(row):
    msgs = row["messages"]
    seeker = [m["content"] for m in msgs if m["role"] == "seeker"]
    sup = [m["content"] for m in msgs[1:] if m["role"] == "supporter"]
    sims = [difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            for i, a in enumerate(seeker) for b in seeker[i + 1:i + 2]]      # neighbouring seeker replies
    trunc = row.get("truncated", [])
    return {
        "pid": row["pid"],
        "msgs": len(msgs),
        "seeker_words": sum(map(words, seeker)) / max(1, len(seeker)),
        "supporter_words": sum(map(words, sup)) / max(1, len(sup)),
        "seeker_asterisks": sum("*" in t for t in seeker),
        "max_seeker_sim": max(sims) if sims else 0.0,
        "closure_hits": sum(bool(CLOSURE.search(t)) for t in seeker),
        "trimmed": sum(trunc),
        "list_lines": sum(bool(re.match(r"\s*(#+ |\d+\.|- |\* )", l)) for t in sup for l in t.split("\n")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--full", type=int, default=0, help="print N conversations in full")
    ap.add_argument("--pid", default=None, help="print this persona's conversation in full")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    print(f"{len(rows)} conversations in {args.jsonl}\n")
    head = f"{'pid':<26}{'msgs':>5}{'seekW':>7}{'supW':>7}{'ast':>5}{'maxSim':>8}{'thanks':>7}{'trim':>6}{'lists':>6}"
    print(head)
    print("-" * len(head))
    flagged = 0
    for r in rows:
        x = report(r)
        flag = (x["msgs"] != 15 or x["seeker_asterisks"] or x["max_seeker_sim"] > 0.6 or x["closure_hits"] >= 5
                or x["trimmed"] > 0 or x["supporter_words"] > 180 or x["supporter_words"] < 50
                or x["seeker_words"] > 130 or x["seeker_words"] < 20)
        flagged += bool(flag)
        print(f"{x['pid']:<26}{x['msgs']:>5}{x['seeker_words']:>7.0f}{x['supporter_words']:>7.0f}"
              f"{x['seeker_asterisks']:>5}{x['max_seeker_sim']:>8.2f}{x['closure_hits']:>7}"
              f"{x['trimmed']:>6}{x['list_lines']:>6}{'   <-- check' if flag else ''}")
    print(f"\n{flagged}/{len(rows)} flagged "
          "(not 15 messages, stage directions, repeated seeker replies, >=5 thank-you/closure replies,\n"
          " a reply still cut off, or average length outside ~50-180 words supporter / ~20-130 words seeker)")
    tot = sum(sum(r.get("truncated", [])) for r in rows)
    print(f"replies trimmed because they hit the token limit: {tot}")

    show = [r for r in rows if r["pid"] == args.pid] if args.pid else rows[:args.full]
    for r in show:
        print("\n" + "=" * 90)
        print(f"{r['pid']}  |  seeker={r['seeker_model']}  supporter={r['supporter_model']}")
        print("=" * 90)
        for i, m in enumerate(r["messages"]):
            cut = " [trimmed]" if r.get("truncated", [False] * 99)[i] else ""
            print(f"\n[{i}] {m['role'].upper()}{cut} ({words(m['content'])} words)\n{m['content']}")


if __name__ == "__main__":
    main()
