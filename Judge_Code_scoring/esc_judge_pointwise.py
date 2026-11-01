"""
esc_judge_pointwise.py
======================
Pointwise (absolute-scoring) variant of the ESC-Judge reliability pipeline.

Instead of showing a judge two conversations and asking "A, B or Tie?", every
conversation is shown ALONE and the judge gives a 1-10 score on each of the 9
E-I-A dimensions (all 9 in ONE call).  Winners are derived later, in
`aggregate_pointwise.py`, by comparing the two scores of a persona.

Why this design
---------------
* No position bias: the judge never sees two conversations, so there is no
  "first slot".  Run order does not matter, any number of runs is fine.
* One call per (conversation, run) instead of per (pair, run).
* Scores are numbers, so self-consistency = average over runs.

It re-uses the backends (hf / openai-compatible / mock), JudgeConfig and PRESETS
from `esc_judge_pipeline.py`, which must be importable (same folder).

Output: one JSON line per (judge, conversation, run) in <out_dir>/<judge>.jsonl
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from esc_judge_pipeline import (          # re-used building blocks
    PAPER_9_DIMENSIONS, JudgeConfig, PRESETS, get_judge, make_backend,
    format_conversation, strip_reasoning, _extract_json_obj, _match_key, _norm, _read_jsonl,
)

__all__ = ["load_scored_rubric", "load_conversations", "run_judge_pointwise",
           "build_score_prompt", "parse_scores", "PRESETS", "JudgeConfig", "get_judge"]


# --------------------------------------------------------------------------- #
# 1. Rubric with scoring anchors
# --------------------------------------------------------------------------- #

@dataclass
class ScoredDimension:
    category: str                      # Exploration | Insight | Action
    name: str
    description: str
    anchors: Dict[str, str] = field(default_factory=dict)   # {"1-2": "...", "3-4": "...", ...}


@dataclass
class Rubric:
    dims: List[ScoredDimension]
    lo: int = 1
    hi: int = 10
    guidance: List[str] = field(default_factory=list)


def load_scored_rubric(path: str, only: Optional[Sequence[str]] = tuple(PAPER_9_DIMENSIONS)) -> Rubric:
    """
    Load rubric_scored.json:
      {"scale": {"min":1,"max":10}, "scale_guidance": [...],
       "<Category>": {"<Dimension>": {"description": "...", "anchors": {"1-2": "...", ...}}}}
    `only` keeps just those dimensions (in that order); only=None keeps all.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))
    data = json.loads(text)
    scale = data.pop("scale", {"min": 1, "max": 10})
    guidance = data.pop("scale_guidance", [])
    dims = [ScoredDimension(cat, name, spec["description"], dict(spec.get("anchors", {})))
            for cat, d in data.items() for name, spec in d.items()]
    if only:
        by_norm = {_norm(d.name): d for d in dims}
        missing = [n for n in only if _norm(n) not in by_norm]
        if missing:
            raise ValueError(f"Dimensions not found in {path}: {missing}")
        dims = [by_norm[_norm(n)] for n in only]
    return Rubric(dims, int(scale["min"]), int(scale["max"]), list(guidance))


# --------------------------------------------------------------------------- #
# 2. Conversations (one record per persona x supporter model)
# --------------------------------------------------------------------------- #

@dataclass
class Conversation:
    pid: str
    supporter_model: str
    seeker_model: Optional[str]
    base_pid: Optional[str]
    factor: Optional[str]
    challenge: Optional[str]
    messages: List[dict]
    truncated: Optional[list] = None


def load_conversations(conv_files: Sequence[str], pids: Optional[Sequence[str]] = None,
                       skip_truncated: bool = False, require_partner: bool = True) -> List[Conversation]:
    """
    Read the per-model conversation .jsonl files (same format as esc_judge_pipeline.build_pairs).
    With require_partner=True, personas that have fewer than 2 supporter models are dropped
    (they can never form a pair, so scoring them would waste calls).
    """
    keep = set(map(str, pids)) if pids else None
    by_key: Dict[Tuple[str, str], Conversation] = {}
    for path in conv_files:
        for rec in _read_jsonl(path):
            pid = str(rec["pid"])
            if keep and pid not in keep:
                continue
            sup = rec.get("supporter_model") or Path(path).stem
            if skip_truncated and any(rec.get("truncated") or []):
                continue
            if (pid, sup) in by_key:
                print(f"[warn] duplicate (pid={pid}, supporter={sup}) in {path}; keeping the last one")
            by_key[(pid, sup)] = Conversation(
                pid=pid, supporter_model=sup, seeker_model=rec.get("seeker_model"),
                base_pid=rec.get("base_pid"), factor=rec.get("factor"), challenge=rec.get("challenge"),
                messages=rec["messages"], truncated=rec.get("truncated"))
    convs = [by_key[k] for k in sorted(by_key)]
    if require_partner:
        n_sup = defaultdict(int)
        for c in convs:
            n_sup[c.pid] += 1
        dropped = {p for p, n in n_sup.items() if n < 2}
        if dropped:
            print(f"[data] dropping {len(dropped)} persona(s) with only one supporter model")
        convs = [c for c in convs if c.pid not in dropped]
    print(f"[data] {len(convs)} conversations from {len({c.pid for c in convs})} personas")
    return convs


# --------------------------------------------------------------------------- #
# 3. Prompt
# --------------------------------------------------------------------------- #

def build_score_prompt(conversation: str, rubric: Rubric) -> str:
    lo, hi = rubric.lo, rubric.hi
    blocks = []
    for i, d in enumerate(rubric.dims, 1):
        anchors = "\n".join(f"   - {band}: {txt}" for band, txt in d.anchors.items())
        blocks.append(f"{i}. {d.name} ({d.category}): {d.description}\n   Scoring anchors:\n{anchors}")
    schema = "{\n" + ",\n".join(
        f'  "{d.name}": {{"reasoning": "<1-2 short sentences citing evidence>", "score": <integer {lo}-{hi}>}}'
        for d in rubric.dims) + "\n}"
    guidance = "\n".join(f"- {g}" for g in rubric.guidance)
    return (
        "You are an expert in counselling psychology, trained in Clara Hill's Exploration-Insight-Action "
        "helping-skills model.\n"
        f"You will read ONE conversation between an emotional-support chatbot (the \"supporter\") and a "
        f"help-seeker. Each line starts with the speaker's role; identify which side is the supporter. "
        f"Rate the supporter on {len(rubric.dims)} skills, each from {lo} (very poor) to {hi} (outstanding).\n\n"
        "# Scoring guidance\n" + guidance + "\n\n"
        "# Conversation\n" + conversation + "\n\n"
        f"# Skills to rate (integer {lo}-{hi})\n" + "\n\n".join(blocks) + "\n\n"
        "# Output format\n"
        "Return ONLY one JSON object (no markdown fences, no text before or after it). "
        "Use exactly these keys, in this order. In each \"reasoning\" write 1-2 short sentences BEFORE the "
        f"\"score\"; the score must be a single integer from {lo} to {hi}.\n" + schema + "\n"
    )


# --------------------------------------------------------------------------- #
# 4. Parsing
# --------------------------------------------------------------------------- #

_NUM_FULL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:(?:/|out of)\s*\d+(?:\.\d+)?)?\s*$", re.I)


def norm_score(v, lo: float, hi: float) -> Optional[float]:
    """7, 7.0, "7", "7/10", "7 out of 10" -> 7.0.  Anything else (or out of range) -> None.
    Strings must be a bare number so a copied placeholder like "<integer 1-10>" is rejected."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        m = _NUM_FULL.match(str(v))
        if not m:
            return None
        x = float(m.group(1))
    return x if lo <= x <= hi else None


def parse_scores(raw: str, rubric: Rubric) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    """Returns (scores, reasoning) keyed by dimension name; scores[dim] is a float or None."""
    dims, lo, hi = rubric.dims, rubric.lo, rubric.hi
    text = strip_reasoning(raw or "")
    scores: Dict[str, Optional[float]] = {d.name: None for d in dims}
    reasoning: Dict[str, str] = {d.name: "" for d in dims}

    # attempt 1: JSON
    blob = _extract_json_obj(text)
    obj = None
    if blob:
        for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
            try:
                obj = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
    if isinstance(obj, dict):
        containers = [obj] + [v for v in obj.values() if isinstance(v, dict)]
        for cont in containers:
            hit = 0
            for d in dims:
                k = _match_key(d.name, list(cont.keys()))
                if k is None:
                    continue
                val = cont[k]
                if isinstance(val, dict):
                    sc = next((val[key] for key in ("score", "Score", "rating", "Rating") if key in val), None)
                    reasoning[d.name] = str(val.get("reasoning", val.get("Reasoning", "")))
                    s = norm_score(sc, lo, hi)
                else:
                    s = norm_score(val, lo, hi)
                if s is not None:
                    scores[d.name] = s
                    hit += 1
            if hit:
                break

    # attempt 2: regex fallback for dimensions still missing
    if any(v is None for v in scores.values()):
        pos = {}
        for d in dims:
            m = re.search(re.escape(d.name), text, flags=re.I)
            if m:
                pos[d.name] = m.start()
        ordered = sorted(pos.items(), key=lambda kv: kv[1])
        for i, (name, start) in enumerate(ordered):
            if scores[name] is not None:
                continue
            end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
            m = re.search(r"score\W{0,6}(\d+(?:\.\d+)?)\b", text[start:end], flags=re.I)
            if m:
                scores[name] = norm_score(m.group(1), lo, hi)
    return scores, reasoning


# --------------------------------------------------------------------------- #
# 5. Mock backend (dry run without a model)
# --------------------------------------------------------------------------- #

class ScoreMockBackend:
    """
    Random but well-formed score JSON. Each conversation gets a stable 'quality' derived from its
    prompt hash, plus small per-run noise, so repeated runs mostly agree and different
    conversations get different scores. Lets you test the whole pipeline with no model.
    """
    def __init__(self, cfg: JudgeConfig, rubric: Rubric):
        self.rng, self.rubric, self.name = random.Random(f"{cfg.seed}-{cfg.name}"), rubric, cfg.name

    def generate(self, prompts: List[str]) -> List[str]:
        lo, hi = self.rubric.lo, self.rubric.hi
        outs = []
        for p in prompts:
            h = int(hashlib.md5((self.name + p).encode()).hexdigest(), 16)   # differs per judge
            base = 4 + h % 5
            obj = {}
            for i, d in enumerate(self.rubric.dims):
                offset = (h >> (8 + 4 * i)) % 3 - 1
                s = min(hi, max(lo, base + offset + self.rng.choice([-1, 0, 0, 1])))
                obj[d.name] = {"reasoning": "mock", "score": s}
            outs.append("```json\n" + json.dumps(obj) + "\n```")
        return outs

    def close(self):
        pass


def make_score_backend(cfg: JudgeConfig, rubric: Rubric):
    if cfg.backend == "mock":
        return ScoreMockBackend(cfg, rubric)
    return make_backend(cfg, rubric.dims)      # "hf" or "openai"; dims are unused by those


# --------------------------------------------------------------------------- #
# 6. Running a judge
# --------------------------------------------------------------------------- #

def _load_done(path: Path, n_dims: int) -> set:
    done = set()
    if path.exists():
        for rec in _read_jsonl(str(path)):
            if rec.get("n_parsed") == n_dims:
                done.add((str(rec["pid"]), rec["supporter_model"], rec["run"]))
    return done


def run_judge_pointwise(
    judge: JudgeConfig,
    convs: Sequence[Conversation],
    rubric: Rubric,
    out_dir: str = "results_pointwise",
    n_runs: int = 4,
    limit: Optional[int] = None,
    max_chars_per_message: Optional[int] = None,
) -> Path:
    """
    Score every conversation n_runs times with one judge. Resumable: finished runs are skipped.
    `limit` keeps only the first N personas (both supporters of each), handy for a quick test.
    """
    from tqdm import tqdm

    if judge.temperature <= 0 and n_runs > 1:
        print(f"[warn] {judge.name}: temperature=0, so all {n_runs} runs will be (almost) identical")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{judge.name}.jsonl"
    convs = list(convs)
    if limit:
        keep = set(sorted({c.pid for c in convs})[:limit])
        convs = [c for c in convs if c.pid in keep]
    n_dims = len(rubric.dims)
    done = _load_done(out_path, n_dims)

    # run-major order: if you stop early, every conversation already has its first runs
    tasks = [(c, k) for k in range(n_runs) for c in convs if (c.pid, c.supporter_model, k) not in done]
    print(f"[{judge.name}] {len(tasks)} runs to do ({len(done)} already finished) -> {out_path}")
    if not tasks:
        return out_path

    backend = make_score_backend(judge, rubric)
    try:
        step = max(1, judge.batch_size) if judge.backend == "hf" else 1
        with tqdm(total=len(tasks), desc=judge.name) as bar:
            for i in range(0, len(tasks), step):
                chunk = tasks[i:i + step]
                prompts = [build_score_prompt(format_conversation(c.messages, max_chars_per_message), rubric)
                           for c, _ in chunk]
                raws = backend.generate(prompts)
                results = [parse_scores(r, rubric) + (r, 1) for r in raws]   # (scores, reasoning, raw, attempts)

                # re-ask for outputs that could not be fully parsed; keep the best attempt
                for j, (scores, reasoning, raw, attempts) in enumerate(results):
                    n_ok = sum(v is not None for v in scores.values())
                    while n_ok < n_dims and attempts <= judge.max_retries:
                        raw2 = backend.generate([prompts[j]])[0]
                        s2, r2 = parse_scores(raw2, rubric)
                        attempts += 1
                        n2 = sum(v is not None for v in s2.values())
                        if n2 >= n_ok:
                            scores, reasoning, raw, n_ok = s2, r2, raw2, n2
                    results[j] = (scores, reasoning, raw, attempts)

                with open(out_path, "a", encoding="utf-8") as f:
                    for (c, k), (scores, reasoning, raw, attempts) in zip(chunk, results):
                        f.write(json.dumps(_make_record(judge, c, k, rubric, scores, reasoning, raw, attempts),
                                           ensure_ascii=False) + "\n")
                bar.update(len(chunk))
    finally:
        backend.close()
    return out_path


def _make_record(judge: JudgeConfig, conv: Conversation, run: int, rubric: Rubric,
                 scores, reasoning, raw, attempts) -> dict:
    return {
        "judge": judge.name, "judge_model": judge.model,
        "judge_is_supporter": judge.model.lower() == conv.supporter_model.lower(),
        "pid": conv.pid, "base_pid": conv.base_pid, "factor": conv.factor, "challenge": conv.challenge,
        "seeker_model": conv.seeker_model, "supporter_model": conv.supporter_model,
        "run": run, "scale": [rubric.lo, rubric.hi],
        "scores": scores,                    # {dimension: float | None}  (None = could not be parsed)
        "reasoning": reasoning,
        "n_parsed": sum(v is not None for v in scores.values()),
        "attempts": attempts,
        "truncated": conv.truncated,
        "raw_output": raw,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
