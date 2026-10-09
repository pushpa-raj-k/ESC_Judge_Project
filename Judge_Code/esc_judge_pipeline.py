"""
esc_judge_pipeline.py
=====================
Level-3 (judge reliability) pipeline for the open-weight ESC-Judge extension.

For every conversation pair (same persona `pid`, two different supporter models)
and for every judge model, this module:

  1. builds the prompt with all 9 E-I-A dimensions in ONE call,
  2. runs the judge `n_runs` times (default 3), swapping which conversation is
     shown first (position-bias control),
  3. parses the A / B / Tie verdict for each dimension,
  4. maps the verdict back to the real model (so order no longer matters),
  5. appends one JSON line per (judge, pair, run) to  <out_dir>/<judge_name>.jsonl

Majority voting (self-consistency / ensemble) is done afterwards in
`aggregate_results.py`, so the raw runs are always kept.

Backends
--------
  "hf"      local Hugging Face transformers model (Colab T4, optional 4-bit)
  "openai"  any OpenAI-compatible endpoint (Groq, Gemini-OpenAI endpoint,
            Ollama, vLLM, ...)
  "mock"    random verdicts, for a dry run without any model
"""
from __future__ import annotations

import difflib
import gc
import hashlib
import itertools
import json
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 1. Rubric
# --------------------------------------------------------------------------- #

# The 9 dimensions used in the ESC-Judge paper (Table 2), in paper order.
PAPER_9_DIMENSIONS = [
    "Empathic Understanding",
    "Encouragement of Emotional Expression",
    "Exploration of Thoughts and Narratives",
    "Establish a Trusting Foundation",
    "Assess Readiness for Insight",
    "Use Gentle Challenges and Interpretations",
    "Clarify the Desired Change",
    "Ensure Readiness and Collaboration",
    "Brainstorm and Evaluate Options",
]


@dataclass
class Dimension:
    category: str      # Exploration | Insight | Action
    name: str
    description: str


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_rubric(path: str, only: Optional[Sequence[str]] = tuple(PAPER_9_DIMENSIONS)) -> List[Dimension]:
    """
    Load a rubric file in the repo's nested format
        {"Exploration": {"<dimension>": "<description>", ...}, "Insight": {...}, "Action": {...}}
    Lines starting with // are ignored. `only` keeps just those dimensions (in that
    order); pass only=None to use every dimension in the file.
    """
    text = Path(path).read_text(encoding="utf-8")
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("//"))
    data = json.loads(text)
    dims = [Dimension(cat, name, desc) for cat, d in data.items() for name, desc in d.items()]
    if only:
        by_norm = {_norm(d.name): d for d in dims}
        missing = [n for n in only if _norm(n) not in by_norm]
        if missing:
            raise ValueError(f"Dimensions not found in {path}: {missing}")
        dims = [by_norm[_norm(n)] for n in only]
    return dims


# --------------------------------------------------------------------------- #
# 2. Conversation data -> pairs
# --------------------------------------------------------------------------- #

@dataclass
class Pair:
    pid: str
    base_pid: Optional[str]
    factor: Optional[str]
    challenge: Optional[str]
    seeker_model: Optional[str]
    model_x: str                 # canonical order = alphabetical by supporter model name
    model_y: str
    conv_x: List[dict]
    conv_y: List[dict]
    truncated_x: Optional[list] = None
    truncated_y: Optional[list] = None

    @property
    def key(self) -> str:
        return f"{self.pid}|{self.model_x}|{self.model_y}"


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{ln} bad JSON line skipped ({e})")
    return out


def build_pairs(
    conv_files: Sequence[str],
    pids: Optional[Sequence[str]] = None,
    require_same_seeker: bool = True,
    skip_truncated: bool = False,
) -> List[Pair]:
    """
    Read the per-model .jsonl files and pair up conversations that share a `pid`
    but come from different supporter models.
    """
    by_pid: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for path in conv_files:
        for rec in _read_jsonl(path):
            sup = rec.get("supporter_model") or Path(path).stem
            pid = str(rec["pid"])
            if pids and pid not in set(map(str, pids)):
                continue
            if sup in by_pid[pid]:
                print(f"[warn] duplicate (pid={pid}, supporter={sup}) in {path}; keeping the last one")
            by_pid[pid][sup] = rec

    pairs, skipped_seeker, skipped_trunc = [], 0, 0
    for pid in sorted(by_pid):
        models = sorted(by_pid[pid])
        for mx, my in itertools.combinations(models, 2):
            rx, ry = by_pid[pid][mx], by_pid[pid][my]
            if require_same_seeker and rx.get("seeker_model") != ry.get("seeker_model"):
                skipped_seeker += 1
                continue
            if skip_truncated and (any(rx.get("truncated") or []) or any(ry.get("truncated") or [])):
                skipped_trunc += 1
                continue
            pairs.append(Pair(
                pid=pid, base_pid=rx.get("base_pid"), factor=rx.get("factor"),
                challenge=rx.get("challenge"), seeker_model=rx.get("seeker_model"),
                model_x=mx, model_y=my, conv_x=rx["messages"], conv_y=ry["messages"],
                truncated_x=rx.get("truncated"), truncated_y=ry.get("truncated"),
            ))
    print(f"[data] {len(pairs)} pairs from {len(by_pid)} personas "
          f"(skipped: {skipped_seeker} different-seeker, {skipped_trunc} truncated)")
    return pairs


def format_conversation(messages: List[dict], max_chars_per_message: Optional[int] = None) -> str:
    lines = []
    for m in messages:
        text = str(m["content"]).strip()
        if max_chars_per_message and len(text) > max_chars_per_message:
            text = text[:max_chars_per_message] + " ..."
        lines.append(f"[{m['role']}]: {text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. Prompt
# --------------------------------------------------------------------------- #

def build_prompt(conv_a: str, conv_b: str, dims: Sequence[Dimension]) -> str:
    dim_list = "\n".join(f"{i}. {d.name} ({d.category}): {d.description}" for i, d in enumerate(dims, 1))
    schema = "{\n" + ",\n".join(
        f'  "{d.name}": {{"reasoning": "<1-2 short sentences comparing A and B>", "verdict": "A or B or Tie"}}'
        for d in dims) + "\n}"
    return (
        "You are an expert in counselling psychology. You are comparing two emotional-support chatbots "
        "(the \"supporter\" in each conversation) using Clara Hill's Exploration-Insight-Action helping-skills model.\n\n"
        "Both conversations were held with the same help-seeker role, so differences come from the supporter only. "
        "Judge ONLY the supporter's behaviour. Do not favour a conversation because of its position (first or second) "
        "or because it is longer. Answer \"Tie\" only if neither supporter is clearly better on a dimension.\n\n"
        "# Conversation 1 (Support Model A)\n" + conv_a + "\n\n"
        "# Conversation 2 (Support Model B)\n" + conv_b + "\n\n"
        "# Dimensions\nFor each dimension, decide which supporter does better.\n" + dim_list + "\n\n"
        "# Output format\n"
        "Return ONLY one JSON object (no markdown fences, no text before or after it). "
        "Use exactly these keys, in this order. In each \"reasoning\" write 1-2 short sentences BEFORE the \"verdict\"; "
        "the verdict must be exactly \"A\", \"B\" or \"Tie\".\n" + schema + "\n"
    )


# --------------------------------------------------------------------------- #
# 4. Parsing the judge output
# --------------------------------------------------------------------------- #

def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks (DeepSeek-R1 style). Handles a missing opening tag."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "<think>" in text:                      # unfinished thinking (output was cut off)
        text = text.split("<think>")[0]
    return text.strip()


def norm_verdict(v) -> Optional[str]:
    s = re.sub(r"[^a-z0-9 ]", " ", str(v).lower()).strip()
    if not s:
        return None
    if any(w in s.split() for w in ("tie", "draw", "equal", "equally")):
        return "Tie"
    has_a = bool(re.search(r"\b(a|model a|conversation 1|support model a)\b", s))
    has_b = bool(re.search(r"\b(b|model b|conversation 2|support model b)\b", s))
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    return None


def _extract_json_obj(text: str) -> Optional[str]:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _match_key(name: str, keys: Sequence[str]) -> Optional[str]:
    nk = {_norm(k): k for k in keys}
    if _norm(name) in nk:
        return nk[_norm(name)]
    close = difflib.get_close_matches(_norm(name), list(nk), n=1, cutoff=0.85)
    return nk[close[0]] if close else None


def parse_judgement(raw: str, dims: Sequence[Dimension]) -> Tuple[Dict[str, Optional[str]], Dict[str, str]]:
    """
    Returns (verdicts, reasoning) keyed by dimension name.
    verdicts[dim] in {"A", "B", "Tie", None}  (None = could not be parsed).
    """
    text = strip_reasoning(raw or "")
    verdicts: Dict[str, Optional[str]] = {d.name: None for d in dims}
    reasoning: Dict[str, str] = {d.name: "" for d in dims}

    # --- attempt 1: proper JSON -------------------------------------------------
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
                    v = norm_verdict(val.get("verdict", val.get("Verdict", "")))
                    reasoning[d.name] = str(val.get("reasoning", val.get("Reasoning", "")))
                else:
                    v = norm_verdict(val)
                if v:
                    verdicts[d.name] = v
                    hit += 1
            if hit:
                break

    # --- attempt 2: regex fallback for dimensions still missing ----------------
    if any(v is None for v in verdicts.values()):
        pos = {}
        for d in dims:
            m = re.search(re.escape(d.name), text, flags=re.I)
            if m:
                pos[d.name] = m.start()
        ordered = sorted(pos.items(), key=lambda kv: kv[1])
        for i, (name, start) in enumerate(ordered):
            if verdicts[name] is not None:
                continue
            end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)
            seg = text[start:end]
            m = re.search(r"verdict\W{0,6}(?:model\s*)?(a|b|tie)\b", seg, flags=re.I)
            if m:
                verdicts[name] = norm_verdict(m.group(1))
    return verdicts, reasoning


# --------------------------------------------------------------------------- #
# 5. Judge configuration + backends
# --------------------------------------------------------------------------- #

@dataclass
class JudgeConfig:
    name: str                          # short label -> results/<name>.jsonl
    model: str                         # HF repo id, or model name for the API
    backend: str = "hf"                # "hf" | "openai" | "mock"
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 1500         # raise to ~4096 for reasoning models (R1 distill)
    batch_size: int = 1                # HF only; >1 needs more GPU memory
    max_retries: int = 2               # re-ask when the output can't be parsed
    seed: int = 0
    # --- HF options
    load_in_4bit: bool = False
    dtype: str = "float16"             # "float16" (T4) | "bfloat16" | "auto"
    trust_remote_code: bool = False
    bnb_compute_dtype: Optional[str] = None   # 4-bit only; default = same as `dtype`. Use "bfloat16" for Gemma 3
    attn_implementation: Optional[str] = None  # e.g. "eager"; None = transformers default
    remove_invalid_values: bool = False        # InfNanRemoveLogitsProcessor: hides NaN/inf instead of fixing it (not recommended)
    sanity_check: bool = True                  # fail fast with a clear message if logits are NaN/inf
    # --- API options
    base_url: Optional[str] = None     # e.g. https://api.groq.com/openai/v1  or  http://localhost:11434/v1
    api_key_env: str = "OPENAI_API_KEY"
    min_interval_s: float = 0.0        # sleep between calls (free-tier rate limits)
    json_mode: bool = False            # response_format={"type": "json_object"} (not for reasoning models)
    api_retries: int = 5


# Ready-made configs: pick by name, or build your own JudgeConfig(...).
PRESETS: Dict[str, JudgeConfig] = {
    "qwen2.5-3b": JudgeConfig(name="qwen2.5-3b", model="Qwen/Qwen2.5-3B-Instruct"),
    "phi-4-mini": JudgeConfig(name="phi-4-mini", model="microsoft/Phi-4-mini-instruct"),
    # gated on Hugging Face: accept the licence and `huggingface-cli login` first
    # Gemma 3 overflows in float16 (NaN/inf logits) -> keep bfloat16 even on a T4
    "gemma-3-4b": JudgeConfig(name="gemma-3-4b", model="google/gemma-3-4b-it", dtype="bfloat16"),
    # reasoning model: long <think> output, recommended temperature 0.6
    "deepseek-r1-distill-qwen-7b": JudgeConfig(
        name="deepseek-r1-distill-qwen-7b", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        temperature=0.6, max_new_tokens=4096, load_in_4bit=True),
}


def get_judge(name_or_cfg) -> JudgeConfig:
    return name_or_cfg if isinstance(name_or_cfg, JudgeConfig) else PRESETS[name_or_cfg]


class HFBackend:
    def __init__(self, cfg: JudgeConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.cfg = torch, cfg
        torch.manual_seed(cfg.seed)
        hf_token = os.environ.get("HF_TOKEN")
        self.tok = AutoTokenizer.from_pretrained(cfg.model,token=hf_token,trust_remote_code=cfg.trust_remote_code)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kwargs = dict(device_map="auto", trust_remote_code=cfg.trust_remote_code)
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_compute_dtype or
                                               ("float32" if cfg.dtype == "auto" else cfg.dtype)))
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        dtype = "auto" if cfg.dtype == "auto" else getattr(torch, cfg.dtype)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype,token=hf_token, **kwargs)
        except TypeError:                      # older transformers use torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype,token=hf_token, **kwargs)
        self.model.eval()
        if cfg.sanity_check:
            self._sanity_check()

    def _sanity_check(self):
        """One short forward pass: raise a clear error if the logits are not finite."""
        msgs = [{"role": "user", "content": "Reply with one short sentence about the weather."}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits
        if not self.torch.isfinite(logits).all():
            raise RuntimeError(
                f"{self.cfg.model}: NaN/inf logits with dtype={self.cfg.dtype}. The model overflows in this "
                "precision. Use dtype='bfloat16' (or float32 across 2 GPUs), not float16.")

    def generate(self, prompts: List[str]) -> List[str]:
        torch, cfg = self.torch, self.cfg
        from transformers import LogitsProcessor, LogitsProcessorList

        class _FiniteGuard(LogitsProcessor):
            """Raise a normal Python error BEFORE sampling if logits are NaN/inf. This replaces the
            device-side CUDA assert from torch.multinomial, which corrupts the CUDA context."""
            def __call__(self, input_ids, scores):
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("NaN/inf logits during generation (precision overflow). "
                                             "Use dtype='bfloat16' for this model.")
                return scores
        outs: List[str] = []
        for i in range(0, len(prompts), max(1, cfg.batch_size)):
            chunk = prompts[i:i + cfg.batch_size]
            texts = [self.tok.apply_chat_template([{"role": "user", "content": p}],
                                                  tokenize=False, add_generation_prompt=True) for p in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            gen_kwargs = dict(max_new_tokens=cfg.max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if cfg.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
            else:
                gen_kwargs.update(do_sample=False)
            gen_kwargs["logits_processor"] = LogitsProcessorList([_FiniteGuard()])
            if cfg.remove_invalid_values:
                gen_kwargs["remove_invalid_values"] = True
            with torch.no_grad():
                gen = self.model.generate(**enc, **gen_kwargs)
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(self.tok.batch_decode(new_tokens, skip_special_tokens=True))
        return outs

    def close(self):
        del self.model
        gc.collect()
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass


class OpenAICompatBackend:
    def __init__(self, cfg: JudgeConfig):
        from openai import OpenAI
        key = os.environ.get(cfg.api_key_env) or ("EMPTY" if cfg.base_url else None)
        if not key:
            raise RuntimeError(f"Set the environment variable {cfg.api_key_env}")
        self.client = OpenAI(base_url=cfg.base_url, api_key=key)
        self.cfg = cfg

    def generate(self, prompts: List[str]) -> List[str]:
        cfg, outs = self.cfg, []
        for p in prompts:
            text = ""
            for attempt in range(cfg.api_retries):
                try:
                    kw = dict(model=cfg.model, messages=[{"role": "user", "content": p}],
                              temperature=cfg.temperature, top_p=cfg.top_p, max_tokens=cfg.max_new_tokens)
                    if cfg.json_mode:
                        kw["response_format"] = {"type": "json_object"}
                    resp = self.client.chat.completions.create(**kw)
                    text = resp.choices[0].message.content or ""
                    break
                except Exception as e:                      # rate limit, timeout, ...
                    wait = min(60, 2 ** (attempt + 1))
                    print(f"[api] {type(e).__name__}: {str(e)[:120]} -> retry in {wait}s")
                    time.sleep(wait)
            outs.append(text)
            if cfg.min_interval_s:
                time.sleep(cfg.min_interval_s)
        return outs

    def close(self):
        pass


class MockBackend:
    """Random but well-formed answers: lets you test the whole pipeline with no model."""
    def __init__(self, cfg: JudgeConfig, dims: Sequence[Dimension]):
        self.rng, self.dims = random.Random(cfg.seed), dims

    def generate(self, prompts: List[str]) -> List[str]:
        outs = []
        for _ in prompts:
            obj = {d.name: {"reasoning": "mock", "verdict": self.rng.choice(["A", "B", "Tie"])} for d in self.dims}
            outs.append("```json\n" + json.dumps(obj) + "\n```")
        return outs

    def close(self):
        pass


def make_backend(cfg: JudgeConfig, dims: Sequence[Dimension]):
    if cfg.backend == "hf":
        return HFBackend(cfg)
    if cfg.backend == "openai":
        return OpenAICompatBackend(cfg)
    if cfg.backend == "mock":
        return MockBackend(cfg, dims)
    raise ValueError(f"unknown backend {cfg.backend!r}")


# --------------------------------------------------------------------------- #
# 6. Running the judge
# --------------------------------------------------------------------------- #

def run_orders(pair_key: str, judge_name: str, n_runs: int) -> List[str]:
    """
    "XY" = model_x is shown first (as Model A);  "YX" = model_y is shown first.
    Orders alternate between runs. The starting order is chosen by a stable hash of
    (pair, judge), so over the whole dataset (and across judges) the first slot is
    balanced even though 3 runs cannot be perfectly balanced for a single pair.
    """
    h = int(hashlib.md5(f"{pair_key}|{judge_name}".encode()).hexdigest(), 16) % 2
    return ["XY" if (h + k) % 2 == 0 else "YX" for k in range(n_runs)]


def _load_done(path: Path, n_dims: int) -> set:
    done = set()
    if path.exists():
        for rec in _read_jsonl(str(path)):
            if rec.get("n_parsed") == n_dims:
                done.add((rec["pid"], rec["model_x"], rec["model_y"], rec["run"]))
    return done


def run_judge(
    judge: JudgeConfig,
    pairs: Sequence[Pair],
    dims: Sequence[Dimension],
    out_dir: str = "results",
    n_runs: int = 3,
    limit: Optional[int] = None,
    max_chars_per_message: Optional[int] = None,
) -> Path:
    """Judge all pairs with one judge model. Resumable: finished runs are skipped."""
    from tqdm import tqdm

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{judge.name}.jsonl"
    pairs = list(pairs)[:limit] if limit else list(pairs)
    done = _load_done(out_path, len(dims))

    tasks = []
    for pair in pairs:
        for k, order in enumerate(run_orders(pair.key, judge.name, n_runs)):
            if (pair.pid, pair.model_x, pair.model_y, k) not in done:
                tasks.append((pair, k, order))
    print(f"[{judge.name}] {len(tasks)} runs to do ({len(done)} already finished) -> {out_path}")
    if not tasks:
        return out_path

    backend = make_backend(judge, dims)
    n_dims = len(dims)
    try:
        step = max(1, judge.batch_size) if judge.backend == "hf" else 1
        with tqdm(total=len(tasks), desc=judge.name) as bar:
            for i in range(0, len(tasks), step):
                chunk = tasks[i:i + step]
                prompts = []
                for pair, k, order in chunk:
                    first, second = (pair.conv_x, pair.conv_y) if order == "XY" else (pair.conv_y, pair.conv_x)
                    prompts.append(build_prompt(format_conversation(first, max_chars_per_message),
                                                format_conversation(second, max_chars_per_message), dims))
                raws = backend.generate(prompts)
                results = [parse_judgement(r, dims) + (r, 1) for r in raws]   # (verdicts, reasoning, raw, attempts)

                # re-ask for outputs that could not be fully parsed; keep the best attempt
                for j, (verdicts, reasoning, raw, attempts) in enumerate(results):
                    n_ok = sum(v is not None for v in verdicts.values())
                    while n_ok < n_dims and attempts <= judge.max_retries:
                        raw2 = backend.generate([prompts[j]])[0]
                        v2, r2 = parse_judgement(raw2, dims)
                        attempts += 1
                        n2 = sum(v is not None for v in v2.values())
                        if n2 >= n_ok:
                            verdicts, reasoning, raw, n_ok = v2, r2, raw2, n2
                    results[j] = (verdicts, reasoning, raw, attempts)

                with open(out_path, "a", encoding="utf-8") as f:
                    for (pair, k, order), (verdicts, reasoning, raw, attempts) in zip(chunk, results):
                        f.write(json.dumps(_make_record(judge, pair, k, order, verdicts, reasoning, raw, attempts),
                                           ensure_ascii=False) + "\n")
                bar.update(len(chunk))
    finally:
        backend.close()
    return out_path


def _make_record(judge, pair: Pair, run: int, order: str, verdicts, reasoning, raw, attempts) -> dict:
    first, second = (pair.model_x, pair.model_y) if order == "XY" else (pair.model_y, pair.model_x)
    winners = {}
    for dim, v in verdicts.items():
        winners[dim] = None if v is None else (first if v == "A" else second if v == "B" else "tie")
    return {
        "judge": judge.name, "judge_model": judge.model,
        "judge_in_pair": judge.model.lower() in {pair.model_x.lower(), pair.model_y.lower()},
        "pid": pair.pid, "base_pid": pair.base_pid, "factor": pair.factor,
        "challenge": pair.challenge, "seeker_model": pair.seeker_model,
        "model_x": pair.model_x, "model_y": pair.model_y,
        "run": run, "order": order, "first_shown": first,
        "verdicts_presented": verdicts,      # what the judge literally said (A = first conversation shown)
        "winners": winners,                  # mapped back to real model names, or "tie"
        "reasoning": reasoning,
        "n_parsed": sum(v is not None for v in verdicts.values()),
        "attempts": attempts,
        "truncated_x": pair.truncated_x, "truncated_y": pair.truncated_y,
        "raw_output": raw,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
