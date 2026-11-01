"""
run_pointwise.py
================
Score every conversation with one or more judges (pointwise, 1-10 per dimension).

Examples
--------
# dry run, no model needed
python run_pointwise.py --conv demo_data/*.jsonl --judges demo --backend mock --n-runs 4 --limit 3

# local Hugging Face judges (Colab T4)
python run_pointwise.py --conv data/conv_qwen.jsonl data/conv_mistral.jsonl \
       --judges gemma-3-4b phi-4-mini deepseek-r1-distill-qwen-7b --n-runs 4

# API judges (Groq / Gemini OpenAI-compatible endpoint / Ollama ...)
export GROQ_API_KEY=...
python run_pointwise.py --conv data/*.jsonl \
       --api-judge llama8b=llama-3.1-8b-instant --base-url https://api.groq.com/openai/v1 \
       --api-key-env GROQ_API_KEY --min-interval 2 --n-runs 4

Then:  python aggregate_pointwise.py --results results_pointwise --rubric rubric_scored.json
"""
import argparse
import dataclasses
import sys

from esc_judge_pointwise import (JudgeConfig, get_judge, load_conversations, load_scored_rubric,
                                 run_judge_pointwise)


def build_judges(args):
    judges = []
    for name in args.judges:
        try:
            cfg = get_judge(name)
        except KeyError:
            if args.backend == "mock":                      # any name is fine for a dry run
                cfg = JudgeConfig(name=name, model=name, backend="mock")
            else:
                raise SystemExit(f"Unknown judge preset '{name}'. Known presets: see PRESETS in esc_judge_pipeline.py")
        judges.append(cfg)
    for spec in args.api_judge:
        if "=" not in spec:
            raise SystemExit(f"--api-judge expects NAME=MODEL, got '{spec}'")
        name, model = spec.split("=", 1)
        judges.append(JudgeConfig(
            name=name, model=model, backend="openai", base_url=args.base_url,
            api_key_env=args.api_key_env, min_interval_s=args.min_interval, json_mode=args.json_mode))
    overrides = {}
    if args.backend:
        overrides["backend"] = args.backend
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.max_new_tokens:
        overrides["max_new_tokens"] = args.max_new_tokens
    return [dataclasses.replace(j, **overrides) for j in judges]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conv", nargs="+", required=True, help="per-supporter-model conversation .jsonl files")
    ap.add_argument("--rubric", default="rubric_scored.json")
    ap.add_argument("--judges", nargs="*", default=[], help="preset names from PRESETS")
    ap.add_argument("--api-judge", action="append", default=[], metavar="NAME=MODEL",
                    help="OpenAI-compatible API judge; repeatable")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between API calls")
    ap.add_argument("--json-mode", action="store_true", help="response_format=json_object (not for reasoning models)")
    ap.add_argument("--backend", choices=["hf", "openai", "mock"], default=None, help="override every judge's backend")
    ap.add_argument("--temperature", type=float, default=None, help="must be > 0 for runs to differ")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--n-runs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N personas (quick test)")
    ap.add_argument("--pids", nargs="*", default=None)
    ap.add_argument("--max-chars", type=int, default=None, help="truncate each message to this many characters")
    ap.add_argument("--skip-truncated", action="store_true")
    ap.add_argument("--out", default="results_pointwise")
    args = ap.parse_args(argv)

    judges = build_judges(args)
    if not judges:
        ap.error("give at least one --judges or --api-judge")
    rubric = load_scored_rubric(args.rubric)
    convs = load_conversations(args.conv, pids=args.pids, skip_truncated=args.skip_truncated)
    if not convs:
        sys.exit("No conversations to score.")
    for j in judges:
        run_judge_pointwise(j, convs, rubric, out_dir=args.out, n_runs=args.n_runs,
                            limit=args.limit, max_chars_per_message=args.max_chars)
    print(f"\nDone. Next: python aggregate_pointwise.py --results {args.out} --rubric {args.rubric}")


if __name__ == "__main__":
    main()
