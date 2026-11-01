# Pointwise ESC-Judge (score each conversation 1-10, then compare)

Each judge reads ONE conversation and scores the supporter 1-10 on the 9 E-I-A dimensions (one call, JSON out).
For a persona, the two supporter models' scores are compared per judge to get a winner (x / y / tie);
judges are then compared with each other (and later with the human labels).

## Files
| File | Purpose |
|---|---|
| `rubric_scored.json` | 9 dimensions, each with a description and 5 scoring anchors (1-2, 3-4, 5-6, 7-8, 9-10) + scale guidance |
| `esc_judge_pointwise.py` | rubric loader, prompt, parser, mock backend, resumable runner (re-uses `esc_judge_pipeline.py`) |
| `run_pointwise.py` | command-line runner for one or more judges |
| `aggregate_pointwise.py` | scores -> winners -> single / self-consistency / ensemble configs, stability, judge agreement |
| `compare_human.py` | agreement of every config with human labels (accuracy, paper-style match rate, Cohen's kappa) |
| `make_demo_data.py` | tiny fake conversations for a dry run |
| `test_pointwise_parser.py` | parser unit tests |

Needs `esc_judge_pipeline.py` in the same folder. Python deps: pandas, numpy, tqdm (+ torch/transformers/bitsandbytes for HF judges, openai for API judges).

## Quick start
```bash
# 1. dry run (no model)
python make_demo_data.py --n-personas 6 --out demo_data
python run_pointwise.py --conv demo_data/*.jsonl --judges j1 j2 j3 --backend mock --n-runs 4
python aggregate_pointwise.py

# 2. real run (local HF judges, 4 runs each)
python run_pointwise.py --conv data/conv_qwen.jsonl data/conv_mistral.jsonl \
       --judges gemma-3-4b phi-4-mini deepseek-r1-distill-qwen-7b --n-runs 4
python aggregate_pointwise.py --results results_pointwise --rubric rubric_scored.json --out analysis_pointwise

# 3. against human labels
python compare_human.py --verdicts analysis_pointwise/verdicts_long.csv --human human_labels.csv
```
API judges (Groq etc.): `--api-judge llama8b=llama-3.1-8b-instant --base-url https://api.groq.com/openai/v1 --api-key-env GROQ_API_KEY --min-interval 2`.
Runs are resumable; re-running skips finished runs, and raising `--n-runs` only adds the missing ones.
Keep `--temperature` > 0 (presets use 0.6-0.7), otherwise all runs are identical.

## Winner rule
`gap = (score_x - score_y) / spread`; tie if `|gap| <= delta`. x = alphabetically first supporter model.
`--normalize sd` (default): spread = that judge's SD on that dimension, delta in SD units (default 0.25).
`--normalize none`: raw score points. Fix delta before looking at human agreement; `delta_sensitivity.csv` shows its effect.

## Configurations (same names as the pairwise study)
`single/<judge>` = run 0 · `sc/<judge>` = mean over runs · `ensemble_single`, `ensemble_sc` = strict-majority vote across judges
(no majority -> tie) · `ensemble_sc_mean` = average of judges' gaps, then thresholded.

## Outputs of aggregate_pointwise.py
* `verdicts_long.csv` - one row per config x pair x dimension (with `score_gap`, `winner_xy`, `winner`)
* `category_scores.csv` - paper-style S(x over y) per category, with bootstrap 95% CI over personas and `ci_excludes_0.5`
* `stability.csv` - `parse_rate`, `mean_score`, `sd_score`, `modal_score_share` (clustering on one score), `within_conv_sd`
  (run-to-run SD of the same conversation), `icc1` (share of variance that is real differences between conversations),
  `winner_all_runs_agree`, `winner_pairwise_run_agreement`, `tie_rate_per_run`
* `mean_scores.csv` - mean raw score per judge x supporter model x dimension
* `judge_agreement.csv`, `judge_agreement_fleiss.csv` - agreement, Cohen's / Fleiss' kappa, Spearman of score gaps
* `delta_sensitivity.csv` - tie rate, win rates and judge agreement for a grid of deltas

## Human labels format (`compare_human.py`)
```
pid,model_x,model_y,dimension,annotator,winner
12,Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,Empathic Understanding,yash,x
```
`winner` = `x`, `y`, `tie` or the model name. Do not use A/B (position-dependent). Consensus = strict majority, else tie.

## Things to report honestly
* Judges scoring a conversation written by their own model family (`judge_is_supporter` flag in the raw records) may favour it.
* The rubric rule "a skill that is never shown scores 1-2" is a methodological choice; mention it.
* The mock judge only tests the plumbing; it says nothing about real judge quality.
