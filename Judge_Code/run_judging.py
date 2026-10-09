"""
run_judging.py  --  edit the CONFIG block, then run:

    python run_judging.py                       # all judges in JUDGES
    python run_judging.py --judges qwen2.5-3b   # only some (names from JUDGES)
    python run_judging.py --limit 3             # smoke test on the first 3 pairs
    python run_judging.py --dry-run             # mock judge, no model needed

Results: results/<judge_name>.jsonl   (one line per judge x pair x run)
In Colab/Jupyter you can instead do:  from run_judging import main; main(["--limit", "3"])
"""
import argparse

from esc_judge_pipeline import (JudgeConfig, PRESETS, build_pairs, load_rubric,
                                run_judge)

# ============================== CONFIG ====================================== #

# # Conversation files, one per supporter model (jsonl, format from your generation step).
# CONV_FILES = [
#     "mistralai-Mistral-7B-Instruct-v0.3.jsonl",
#     "Qwen-Qwen2.5-3B-Instruct.jsonl",
# ]

# RUBRIC_PATH = "rubric_9dim.json"   # or your exploration_rubric file (9 paper dims are selected automatically)
# OUT_DIR = "results"
# N_RUNS = 3                         # runs per pair per judge (order alternates between runs)

# # Judges. Use a preset by name ...
# JUDGES = [
#     PRESETS["qwen2.5-3b"],
#     PRESETS["phi-4-mini"],
#     PRESETS["gemma-3-4b"],
#     # PRESETS["deepseek-r1-distill-qwen-7b"],

#     # ... or define your own. Any HF causal-LM chat model:
#     # JudgeConfig(name="llama-3.2-3b", model="meta-llama/Llama-3.2-3B-Instruct", temperature=0.7),
#     #
#     # Free-tier / hosted / local servers through an OpenAI-compatible API (no GPU needed):
#     # JudgeConfig(name="groq-judge", model="<model name on Groq>", backend="openai",
#     #             base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY", min_interval_s=3),
#     # JudgeConfig(name="gemini-judge", model="<gemini model name>", backend="openai",
#     #             base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
#     #             api_key_env="GEMINI_API_KEY", min_interval_s=6),
#     # JudgeConfig(name="ollama-gemma", model="gemma3:4b", backend="openai",
#     #             base_url="http://localhost:11434/v1"),
# ]

# # Optional: restrict to certain personas / quality filters
# PIDS = None                        # e.g. ["9b", "12a"]  (None = all)
# REQUIRE_SAME_SEEKER = True         # only pair conversations produced with the same seeker model
# SKIP_TRUNCATED = False             # drop pairs where any conversation has truncated=True

# ============================================================================ #
# ============================== CONFIG ====================================== #

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Conversation files
CONV_FILES = [
    BASE_DIR / "mistralai-Mistral-7B-Instruct-v0.3.jsonl",
    BASE_DIR / "Qwen-Qwen2.5-7B-Instruct.jsonl",
]

RUBRIC_PATH = BASE_DIR / "rubric_9dim.json"

# Kaggle input is read-only, so save results in working directory
OUT_DIR = "/kaggle/working/results"

N_RUNS = 2

# Judges
JUDGES = [
    PRESETS["phi-4-mini"],
    PRESETS["gemma-3-4b"],
    PRESETS["qwen2.5-3b"],
    PRESETS["deepseek-r1-distill-qwen-7b"],

]

# Optional filters
PIDS = None
REQUIRE_SAME_SEEKER = True
SKIP_TRUNCATED = False

# ============================================================================ #

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="*", help="names of judges in JUDGES to run (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs")
    ap.add_argument("--dry-run", action="store_true", help="use the mock backend (no model)")
    args = ap.parse_args(argv)

    dims = load_rubric(RUBRIC_PATH)
    pairs = build_pairs(CONV_FILES, pids=PIDS, require_same_seeker=REQUIRE_SAME_SEEKER,
                        skip_truncated=SKIP_TRUNCATED)

    judges = [j for j in JUDGES if not args.judges or j.name in args.judges]
    if args.judges and len(judges) != len(args.judges):
        raise SystemExit(f"Unknown judge name(s). Available: {[j.name for j in JUDGES]}")

    for judge in judges:                          # one model loaded at a time (saves GPU memory)
        if args.dry_run:
            judge = JudgeConfig(**{**judge.__dict__, "backend": "mock", "name": judge.name + "-mock"})
        run_judge(judge, pairs, dims, out_dir=OUT_DIR, n_runs=N_RUNS, limit=args.limit)


if __name__ == "__main__":
    main()
