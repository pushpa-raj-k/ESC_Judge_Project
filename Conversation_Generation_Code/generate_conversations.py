"""
generate_conversations.py  --  Step 2 of the project (conversation generation).

Replaces langchain_persona_based_chat.py from the ESC-Judge repo. Same simulation protocol:
  * the supporter opens with "Hey! how's it going?"
  * 14 further turns alternate seeker, supporter, ... (7 each): 15 messages in total
  * seeker = help-seeker model playing an original ESC-Judge persona (roles-v1.json)
  * supporter = emotional_supporter_nodir.txt ("You are a helpful emotional support expert.")

Differences from the original, all needed because small open-weight models are used:
  1. No OpenAI / langchain: local Hugging Face models (llm_utils.py).
  2. Two runs, one supporter each, same personas:   --supporter qwen   and   --supporter mistral
  3. All personas advance one turn at a time in a batch (fast).
  4. Replies must look like the paper's conversations (plain paragraphs of normal length, ending
     naturally). Small models otherwise write 400-word essays that hit the token limit mid-sentence.
       - SUPPORTER_STYLE is appended to the supporter prompt (identical for both supporters;
         it asks for plain conversational paragraphs of ~80-150 words; it gives NO counselling method).
       - SEEKER_RULES are appended to the seeker prompt (short, in character, no stage directions,
         no wrapping up).
       - A reply that was cut off by the token limit, is too short, or (seeker) repeats its previous
         message is regenerated (up to --retries times). Only if it still fails is it trimmed to the
         last complete sentence.
  5. Supporter history starts with a neutral user turn ("[The session has started.]") because
     Mistral's chat template rejects a history that starts with an assistant message.

Examples
    python generate_conversations.py --supporter qwen --limit 3              # smoke test
    python generate_conversations.py --supporter qwen                        # run 1 (all personas)
    python generate_conversations.py --supporter mistral                     # run 2 (same personas)
"""
import argparse
import difflib
import json
import os
import re
from typing import Dict, List

from llm_utils import MockLLM, resolve_model_name

# ----------------------------------------------------------------------------- prompts
# First paragraph = the original repo's seeker prompt. The rules after it are our addition.
SEEKER_PROMPT_TEMPLATE = """Here is a description of your role:

{description}

You are talking to an emotional support expert about your ongoing challenge. Focus on your problem and engage in the conversation. make sure you are consistent with your designated behavioral traits and role description.

Additional rules:
- Reply in 2 to 5 sentences (roughly 40 to 90 words) of plain spoken dialogue, the way a real person talks in a chat. Always finish your last sentence.
- Do not write actions or stage directions (nothing between asterisks).
- Never repeat or rephrase something you already said, and do not keep thanking the counselor. Add something new about your situation or feelings in every message.
- Do not ask the counselor for lists, tips or exercises; you came to talk about how you feel.
- The conversation will go on for many more messages. Do not wrap up, say goodbye, or announce that you feel relieved, hopeful, at peace or that you have closure. One chat does not solve your problems; keep struggling with them realistically.
- React according to your traits: if you are resistant, defensive, avoidant or emotionally reactive, do not suddenly become open, calm and agreeable.
"""

# Appended to emotional_supporter_nodir.txt for BOTH supporters. It only fixes the format
# (length, plain paragraphs, finished sentences); it does not tell the model how to counsel.
SUPPORTER_STYLE = """Reply the way a caring person talks in a conversation: one or two short paragraphs of plain text, no headings, no bold text, no lists. Keep each reply to 4 to 7 sentences (about 80 to 140 words) and always finish your last sentence."""

GREETING = "Hey! how's it going?"
SESSION_START = "[The session has started.]"

_LABEL = re.compile(r"^\s*(client|seeker|help[- ]?seeker|therapist|counselor|counsellor|supporter|assistant|user)\s*:\s*",
                    re.IGNORECASE)
_DANGLING = re.compile(r"\s*(\d+[.)]|[-*\u2022])\s*(\*\*[^*]*\*\*:?)?\s*")


# ------------------------------------------------------------------------ text cleaning
def _action_repl(text):
    def repl(m):
        inner = m.group(1)
        start = m.start()
        # single word in the middle of a sentence = emphasis ("I *really* do") -> keep the word
        if " " not in inner.strip() and "," not in inner and start > 0 and text[start - 1] not in "\n.!?":
            return inner
        return ""
    return repl


def strip_actions(text: str) -> str:
    """Remove stage directions like '*nods, looking hopeful*' from a help-seeker reply."""
    cleaned = re.sub(r"\*([^*\n]{1,200})\*", _action_repl(text), text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned if cleaned else text.replace("*", "").strip()


def finalize_reply(text: str, hit_limit: bool) -> str:
    """Last resort for a reply that STILL hit the token limit after all retries: drop the
    unfinished sentence/line and any dangling heading or intro line."""
    if not hit_limit or not text:
        return text
    lines = text.rstrip().split("\n")
    last = lines[-1]
    if last and last[-1] not in ".!?\"')":
        cut = max(last.rfind(c) for c in ".!?")
        if cut >= 0:
            lines[-1] = last[:cut + 1]
        else:
            lines.pop()
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("#")
                     or lines[-1].rstrip().endswith(":") or _DANGLING.fullmatch(lines[-1])):
        lines.pop()
    result = "\n".join(lines).strip()
    return result if result and len(result) >= 0.4 * len(text) else text


def clean_reply(text: str, seeker: bool = False, hit_limit: bool = False) -> str:
    text = (text or "").strip()
    for _ in range(2):
        text = _LABEL.sub("", text, count=1).strip()
    if seeker:
        text = strip_actions(text)
    text = finalize_reply(text, hit_limit)
    return text if text else "..."


def n_words(t: str) -> int:
    return len(t.split())


def too_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold


def save_chat_txt(messages: List[Dict], path: str) -> None:
    """Same format as the original save_chat(): one 'role: content' entry per message."""
    with open(path, "w") as f:
        for m in messages:
            f.write(f"{m['role']}: {m['content']}\n")


# ------------------------------------------------------------------------ persona selection
def category_of(row: dict) -> str:
    """Stressor category of a persona (needs persona_traits.py; 'unknown' without it)."""
    try:
        from persona_traits import CHALLENGE_TO_CATEGORY
        return CHALLENGE_TO_CATEGORY.get(row["challenge"].strip().lower(), "other")
    except ImportError:
        return "unknown"


def select_personas(rows: List[dict], n: int, seed: int) -> List[dict]:
    """Pick n personas, balanced over the 6 stressor categories of the paper (round-robin over
    categories, random inside each). Deterministic for a given file/n/seed, so the Qwen run and
    the Mistral run use exactly the same personas. n <= 0 means: all personas, file order."""
    import random
    if n <= 0 or n >= len(rows):
        return rows
    try:
        from persona_traits import CHALLENGE_TO_CATEGORY
        cat = lambda r: CHALLENGE_TO_CATEGORY.get(r["challenge"].strip().lower(), "other")
    except ImportError:
        cat = lambda r: "all"
    rng = random.Random(seed)
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(cat(r), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    picked: List[dict] = []
    while len(picked) < n and any(groups.values()):
        for g in groups.values():
            if g and len(picked) < n:
                picked.append(g.pop())
    return picked


# ------------------------------------------------------------------------ one batch of turns
def generate_replies(llm, convs, gen: dict, seeker: bool, min_words: int, retries: int, prev=None,
                     max_words: int = 0):
    """One reply per conversation. Re-generates replies that were cut off by the token limit,
    are too short or too long (max_words, 0 = no limit), or (seeker) repeat the previous seeker message.
    Returns (replies, still_cut_off)."""
    n = len(convs)
    raws = list(llm.generate(convs, verbose=False, **gen))
    hits = list(getattr(llm, "last_hit_limit", [False] * n))

    def bad(i):
        text = clean_reply(raws[i], seeker=seeker)
        return (hits[i] or n_words(text) < min_words or (max_words and n_words(text) > max_words)
                or (prev is not None and prev[i] and too_similar(text, prev[i])))

    pending = [i for i in range(n) if bad(i)]
    for attempt in range(1, retries + 1):
        if not pending:
            break
        g = dict(gen, temperature=max(0.5, gen.get("temperature", 0.8) - 0.05 * attempt))
        if seeker:
            g["repetition_penalty"] = gen.get("repetition_penalty", 1.0) + 0.05 * attempt
        new = list(llm.generate([convs[i] for i in pending], verbose=False, **g))
        new_hits = list(getattr(llm, "last_hit_limit", [False] * len(pending)))
        for i, o, h in zip(pending, new, new_hits):
            raws[i], hits[i] = o, h
        pending = [i for i in pending if bad(i)]

    replies = [clean_reply(raws[i], seeker=seeker, hit_limit=hits[i]) for i in range(n)]
    return replies, hits


def simulate_batch(personas: List[dict], seeker_llm, supporter_llm, supporter_sys: str, n_turns: int,
                   seeker_gen: dict, supporter_gen: dict, min_seeker: int, min_supporter: int,
                   retries: int, max_seeker: int = 0, max_supporter: int = 0, verbose: bool = True):
    """Returns (conversations, cut_off_flags): cut_off_flags[i][j] is True if message j of conversation i
    still hit the token limit after all retries (and was trimmed)."""
    n = len(personas)
    recorded = [[{"role": "supporter", "content": GREETING}] for _ in range(n)]
    cut = [[False] for _ in range(n)]
    sup_hist = [[{"role": "user", "content": SESSION_START},
                 {"role": "assistant", "content": GREETING}] for _ in range(n)]
    seek_hist = [[{"role": "user", "content": GREETING}] for _ in range(n)]
    seek_sys = [SEEKER_PROMPT_TEMPLATE.format(description=p["role"]) for p in personas]

    for turn in range(n_turns):
        seeker_turn = (turn % 2 == 0)           # turn 0 = seeker, 1 = supporter, ...
        if verbose:
            print(f"turn {turn + 1}/{n_turns} ({'seeker' if seeker_turn else 'supporter'})")
        if seeker_turn:
            convs = [[{"role": "system", "content": seek_sys[i]}] + seek_hist[i] for i in range(n)]
            prev = [next((m["content"] for m in reversed(recorded[i]) if m["role"] == "seeker"), None)
                    for i in range(n)]
            outs, hits = generate_replies(seeker_llm, convs, seeker_gen, True, min_seeker, retries, prev, max_seeker)
            for i, o in enumerate(outs):
                recorded[i].append({"role": "seeker", "content": o})
                cut[i].append(bool(hits[i]))
                seek_hist[i].append({"role": "assistant", "content": o})
                sup_hist[i].append({"role": "user", "content": o})
        else:
            convs = [[{"role": "system", "content": supporter_sys}] + sup_hist[i] for i in range(n)]
            outs, hits = generate_replies(supporter_llm, convs, supporter_gen, False, min_supporter, retries,
                                          None, max_supporter)
            for i, o in enumerate(outs):
                recorded[i].append({"role": "supporter", "content": o})
                cut[i].append(bool(hits[i]))
                sup_hist[i].append({"role": "assistant", "content": o})
                seek_hist[i].append({"role": "user", "content": o})
    return recorded, cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeker_personas_file", default="roles-v1.json", help="original ESC-Judge personas (JSONL)")
    ap.add_argument("--supporter_persona_file", default="emotional_supporter_nodir.txt")
    ap.add_argument("--no_supporter_style", action="store_true",
                    help="do NOT append SUPPORTER_STYLE (supporters then write long essays and get cut off)")
    ap.add_argument("--seeker", default="llama", help="help-seeker model (alias or HF id)")
    ap.add_argument("--supporter", required=True, help="supporter model: qwen | mistral | HF id")
    ap.add_argument("--output_dir", default="/kaggle/working/conv")
    ap.add_argument("--output_prefix", default="exp")
    ap.add_argument("--n_turns", type=int, default=14)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=600, help="supporter token limit (safety net)")
    ap.add_argument("--seeker_max_new_tokens", type=int, default=300)
    ap.add_argument("--seeker_repetition_penalty", type=float, default=1.1)
    ap.add_argument("--min_supporter_words", type=int, default=40, help="shorter replies are regenerated")
    ap.add_argument("--min_seeker_words", type=int, default=15)
    ap.add_argument("--max_supporter_words", type=int, default=200, help="longer replies are regenerated (0 = off)")
    ap.add_argument("--max_seeker_words", type=int, default=140, help="longer replies are regenerated (0 = off)")
    ap.add_argument("--supporter_repetition_penalty", type=float, default=1.0,
                    help="1.0 = off. Try 1.05-1.1 only if a supporter keeps repeating the same phrases")
    ap.add_argument("--retries", type=int, default=3, help="regenerations for cut-off / too short / repeated replies")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=5, help="personas per checkpoint chunk")
    ap.add_argument("--n_personas", type=int, default=40,
                    help="personas to use, balanced over the 6 stressor categories (0 = all). Same value + --seed "
                         "in both supporter runs gives the same personas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="only the first N of those (smoke test)")
    ap.add_argument("--seeker_4bit", action="store_true", help="load the seeker in 4-bit (single-GPU memory saver)")
    ap.add_argument("--dry_run", action="store_true", help="mock models: tests the pipeline, no GPU")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    supporter_sys = open(args.supporter_persona_file).read().strip()
    if not args.no_supporter_style:
        supporter_sys = supporter_sys + "\n\n" + SUPPORTER_STYLE
    with open(args.seeker_personas_file) as f:
        personas = [json.loads(l) for l in f if l.strip()]
    personas = select_personas(personas, args.n_personas, args.seed)
    if args.limit:
        personas = personas[:args.limit]
    min_sup, min_seek = (0, 0) if args.dry_run else (args.min_supporter_words, args.min_seeker_words)
    max_sup, max_seek = (0, 0) if args.dry_run else (args.max_supporter_words, args.max_seeker_words)

    supporter_repo = resolve_model_name(args.supporter)
    seeker_repo = resolve_model_name(args.seeker)
    tag = supporter_repo.replace("/", "-")

    # write down exactly which personas this run uses (same file content in the Qwen and Mistral runs)
    import csv
    sel_name = f"selected_personas_n{args.n_personas}" + (f"_limit{args.limit}" if args.limit else "") + ".csv"
    with open(os.path.join(args.output_dir, sel_name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "pid", "stressor_category", "challenge"])
        for i, p in enumerate(personas, 1):
            w.writerow([i, p["pid"], category_of(p), p["challenge"]])
    print(f"selected personas written to {os.path.join(args.output_dir, sel_name)}")

    def txt_path(pid):
        return os.path.join(args.output_dir, f"{args.output_prefix}_{tag}_{pid}.txt")

    todo = [p for p in personas if not os.path.exists(txt_path(p["pid"]))]
    print(f"{len(personas)} personas, {len(personas) - len(todo)} already done, {len(todo)} to run")
    print(f"seeker={seeker_repo}  supporter={supporter_repo}")
    print("supporter prompt:\n" + supporter_sys + "\n")
    if not todo:
        return

    if args.dry_run:
        seeker_llm, supporter_llm = MockLLM(), MockLLM()
    else:
        import torch
        from llm_utils import LocalLLM
        two_gpus = torch.cuda.device_count() >= 2
        seeker_llm = LocalLLM(args.seeker, load_in_4bit=True if args.seeker_4bit else None, device=0)
        supporter_llm = LocalLLM(args.supporter, device=1 if two_gpus else 0)

    seeker_gen = dict(max_new_tokens=args.seeker_max_new_tokens, temperature=args.temperature,
                      batch_size=args.batch_size, repetition_penalty=args.seeker_repetition_penalty)
    supporter_gen = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                         batch_size=args.batch_size, repetition_penalty=args.supporter_repetition_penalty)

    jsonl_path = os.path.join(args.output_dir, f"{args.output_prefix}_{tag}.jsonl")
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c:c + args.chunk]
        convs, cut = simulate_batch(chunk, seeker_llm, supporter_llm, supporter_sys, args.n_turns,
                                    seeker_gen, supporter_gen, min_seek, min_sup, args.retries, max_seek, max_sup)
        with open(jsonl_path, "a") as jf:
            for p, msgs, ct in zip(chunk, convs, cut):
                save_chat_txt(msgs, txt_path(p["pid"]))
                jf.write(json.dumps({
                    "pid": p["pid"], "base_pid": p.get("base_pid"), "factor": p.get("factor"),
                    "challenge": p.get("challenge"),
                    "seeker_model": seeker_repo, "supporter_model": supporter_repo,
                    "messages": msgs,
                    "truncated": ct,        # True = still cut off after all retries (trimmed to a clean end)
                }) + "\n")
        print(f"saved chunk {c // args.chunk + 1}: {len(chunk)} conversations")

    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    total = sum(len(r["messages"]) - 1 for r in rows)
    cut_n = sum(sum(r["truncated"]) for r in rows)
    print(f"DONE: {len(rows)} conversations in {jsonl_path}")
    print(f"replies still cut off by the token limit after retries (trimmed): {cut_n}/{total}")
    seeker_llm.unload()
    supporter_llm.unload()


if __name__ == "__main__":
    main()
