# ESC-Judge with Small Open-Weight Models: Can Small LLMs Judge Emotional-Support Conversations?

An extension of the **ESC-Judge** framework (Madani & Srihari, EMNLP 2025) that replaces the proprietary GPT-class models with **open-weight LLMs** at every stage: the help-seeker, the emotional supporters, and the judges. The project asks three questions:

1. **Which of two small supporters, Qwen2.5-7B-Instruct or Mistral-7B-Instruct-v0.3, gives better emotional support** according to Clara Hill's Exploration-Insight-Action (E-I-A) model?
2. **Can small open-weight LLMs act as *reliable* judges?** We test pairwise judges with `n = 2` and `n = 4` runs, and a large judge (GPT-OSS-120B) as a reference.
3. **Does a pointwise scoring judge** (score each conversation 1-10, then compare scores) **make judging more stable?**

> **TL;DR**
> - Small pairwise judges (3-7B) are **unstable**: their verdict changes when the presentation order is swapped (order consistency 28-54 %), they favour the conversation shown first (56-71 % of non-tie verdicts) and they agree with each other at about chance level. More runs (`n = 2` to `n = 4`) help only slightly, and majority voting collapses 81-97 % of decisions into ties.
> - A **large judge (GPT-OSS-120B)** is clearly steadier (64 % run agreement, no first-slot bias) and prefers **Qwen** on all three categories.
> - Our **final solution, pointwise scoring**, removes position bias by construction, gives decisive verdicts (64 % non-tie vs 3-25 %), has the highest agreement with the 120B reference (42 %, Cohen's kappa 0.07 vs 33-35 %) and recovers the same direction (Qwen better, strongest on Action and Insight).
> - **Caveat:** there are no human labels yet. "Agreement with the 120B judge" is a proxy, not accuracy, and all kappa values are in the "slight" range. The improvement is real in direction but modest.

---

## Table of Contents

1. [Background](#1-background)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Method](#3-method)
4. [Results](#4-results)
5. [Ablation](#5-ablation)
6. [Discussion](#6-discussion)
7. [Limitations](#7-limitations)
8. [Repository Structure](#8-repository-structure)
9. [Reproducing the Experiments](#9-reproducing-the-experiments)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. Background

ESC-Judge evaluates emotional-support chatbots by having two supporters talk to the *same* simulated help-seeker persona, then asking an LLM judge which supporter did better on each dimension of Hill's three-stage helping-skills model:

| Stage | Dimensions (9 total, from the paper's Table 2) |
|---|---|
| **Exploration** | Empathic Understanding · Encouragement of Emotional Expression · Exploration of Thoughts and Narratives |
| **Insight** | Establish a Trusting Foundation · Assess Readiness for Insight · Use Gentle Challenges and Interpretations |
| **Action** | Clarify the Desired Change · Ensure Readiness and Collaboration · Brainstorm and Evaluate Options |

The original paper uses one small reasoning judge (o1-mini) sampled twice and lists larger judges, more sampling and cross-model ensembling as untested future work. This project tests those ideas with open models only (Kaggle/Colab-class GPUs, 4-bit quantisation for the 7B models, Groq free tier for the 120B judge).

Rubrics: [`Judge_Code/rubric_9dim.json`](Judge_Code/rubric_9dim.json) (pairwise) and [`Judge_Code_scoring/rubric_scored.json`](Judge_Code_scoring/rubric_scored.json) (pointwise, with five scoring anchors per dimension).

## 2. Pipeline Overview

```
 40 personas (ESC-Judge roles-v1.json)
        |
        v
 +--------------------------------+
 | Level 2: Conversation generation|   seeker     = Llama-3.2-3B-Instruct
 |  15-message simulated chats     |   supporters = Qwen2.5-7B-Instruct
 +--------------------------------+                Mistral-7B-Instruct-v0.3
        |   40 paired conversations (same persona, two supporters)
        v
 +--------------------------------------------------------------+
 | Level 3: Judging (four tracks)                                |
 |  A  pairwise, 3 small judges, n = 2 runs                      |
 |  B  pairwise, 3 small judges, n = 4 runs                      |
 |  C  + large judge GPT-OSS-120B (Groq), pairwise, n = 4        |
 |  D  pointwise 1-10 scoring, 3 small judges, n = 4  (FINAL)    |
 +--------------------------------------------------------------+
        |
        v
 Aggregation: single / self-consistency / ensemble / mean score gap
        |
        v
 Analysis_Output/*  and  Judge_Code_scoring/analysis_pointwise/*
```

## 3. Method

### 3.1 Conversation generation

Code: [`Conversation_Generation_Code/`](Conversation_Generation_Code)

| Item | Setting |
|---|---|
| Personas | 40 personas from the ESC-Judge set (`selected_personas_n40.csv`): 7 career/academic, 7 identity/discrimination, 7 health, 7 environmental/societal, 6 personal loss, 6 financial |
| Help-seeker model | `meta-llama/Llama-3.2-3B-Instruct` (same for every conversation) |
| Supporter A | `Qwen/Qwen2.5-7B-Instruct` |
| Supporter B | `mistralai/Mistral-7B-Instruct-v0.3` |
| Supporter prompt | `"You are a helpful emotional support expert."` (non-directive prompt from the original repo) plus a short *format-only* style note. No counselling method is specified |
| Conversation length | 15 messages: the supporter opens with "Hey! how's it going?", then 14 alternating turns |
| Sampling | temperature 0.8, seed 42 |
| Quality controls | Truncated, too short/long or (for the seeker) repetitive replies are regenerated up to 3 times, then trimmed to the last full sentence |

All 80 conversations (40 per supporter) completed with 0 truncated messages flagged.

### 3.2 Judging tracks

Code: [`Judge_Code/`](Judge_Code) (pairwise) and [`Judge_Code_scoring/`](Judge_Code_scoring) (pointwise).

| Track | Judges | Runs | Prompt |
|---|---|:-:|---|
| **A** pairwise | Phi-4-mini, Gemma-3-4B, DeepSeek-R1-Distill-Qwen-7B | 2 | both conversations, A/B/Tie for 9 dimensions in one JSON call |
| **B** pairwise | same three | 4 | same prompt, order alternates every run |
| **C** large judge | `openai/gpt-oss-120b` served by **Groq** (`groq-judge`) | 4 | same pairwise prompt |
| **D** pointwise (final) | same three small judges | 4 | ONE conversation, supporter scored 1-10 on 9 dimensions with anchored rubric |

> **Naming note.** Folders and files called `groq-judge` or `with_grok` refer to the **Groq**-hosted GPT-OSS-120B model, not xAI's Grok.

| Small judge | Model | Notes |
|---|---|---|
| `phi-4-mini` | `microsoft/Phi-4-mini-instruct` | temp 0.7 |
| `gemma-3-4b` | `google/gemma-3-4b-it` | bfloat16 (float16 overflows) |
| `deepseek-r1-distill-qwen-7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | reasoning model, 4-bit, temp 0.6, up to 4096 new tokens, `<think>` blocks stripped |

- **Position-bias control (pairwise).** Order is swapped between runs; verdicts are mapped back to the real model.
- **Output parsing.** JSON is extracted from the raw output with retries. Unparseable dimensions are *missing*, never counted as ties.
- **Pointwise winner rule.** For judge `j` and dimension `d`, `gap = (score_x - score_y) / SD(j, d)`. If `|gap| <= delta` the result is a tie, otherwise the higher score wins. `delta = 0.25` was fixed before looking at agreement. The `ensemble_sc_mean` config averages the judges' gaps and then applies the same threshold.

### 3.3 Aggregation configurations

| Config | Definition |
|---|---|
| `single/<judge>` | Run 0 of one judge (baseline, as in the paper) |
| `sc/<judge>` | Majority vote (pairwise) or mean score (pointwise) over that judge's runs |
| `ensemble_single` | Strict-majority vote over judges' run-0 verdicts |
| `ensemble_sc` | Strict-majority vote over judges' self-consistency verdicts |
| `ensemble_sc_mean` | (pointwise only) average of the judges' score gaps, then thresholded |

No strict majority gives `Tie` (the paper's rule). At least two valid votes are required.

**Category score.** A win counts 1, a tie 0.5, a loss 0, averaged over the dimensions of a category within a persona, then over personas. Here `x` = Qwen2.5-7B and `y` = Mistral-7B, so a score above 0.5 favours Qwen. Pointwise outputs include bootstrap 95 % CIs over personas (the report adds CIs for the pairwise configs too).

**Accuracy proxy.** Without human labels, we report 3-class agreement and Cohen's kappa of each configuration with the GPT-OSS-120B self-consistency verdict (`sc/groq-judge`).

## 4. Results

Numbers come from the CSVs in [`Analysis_Output/`](Analysis_Output) and [`Judge_Code_scoring/analysis_pointwise/`](Judge_Code_scoring/analysis_pointwise).

### 4.1 Stability of pairwise judges (Tracks A, B, C)

Averages over the 9 dimensions (`stability.csv`):

| Judge | Run agreement `n=2` | Run agreement `n=4` | All 4 runs agree | Order consistency | First-slot win rate (non-tie) | Tie rate | Parse rate |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| phi-4-mini | 0.54 | 0.60 | 0.34 | 0.51 | 0.56 | 0.72 | 0.83 |
| gemma-3-4b | 0.34 | 0.42 | 0.11 | 0.34 | 0.66 | 0.11 | 1.00 |
| deepseek-r1-distill-qwen-7b | 0.28 | 0.35 | 0.07 | 0.28 | 0.71 | 0.33 | 1.00 |
| **gpt-oss-120b (Groq)** | - | **0.64** | **0.40** | **0.56** | 0.39 | 0.23 | 1.00 |

- Chance level for a 3-class verdict is about 0.33. Run 0 and run 1 show the conversations in opposite orders, so order consistency measures how often the verdict survives an order swap.
- **More runs help only a little.** Going from `n=2` to `n=4` raises mean run agreement by 0.03-0.08, and the share of cases where all four runs agree is 7 %, 11 % and 34 %.
- **Position bias.** Small judges favour the first-shown conversation (0.50 would be unbiased). The 120B judge leans slightly towards the second (0.39).
- **Phi-4-mini's steadiness is largely an artifact of ties** (72 %) and it fails to produce complete output in about 17 % of runs.

### 4.2 Do judges agree with each other?

Agreement of run-0 verdicts (`judge_agreement.csv`):

- Pairwise small judges: **22-37 %** (at or below the 33 % chance level).
- Large judge vs small judges: **29-38 %**.
- Pointwise small judges: **32-40 %** (Fleiss kappa about 0.04), a small gain but still slight.

### 4.3 Does majority voting help? Final ensemble verdicts (360 decisions)

| Configuration | Tie | Qwen wins | Mistral wins |
|---|:-:|:-:|:-:|
| Pairwise `n=2`, ensemble of self-consistency | 350 (97.2 %) | 8 (2.2 %) | 2 (0.6 %) |
| Pairwise `n=4`, ensemble | 291 (80.8 %) | 31 (8.6 %) | 38 (10.6 %) |
| Pairwise `n=4` + large judge in the vote | 270 (75.0 %) | 55 (15.3 %) | 35 (9.7 %) |
| Pointwise ensemble (vote) | 161 (44.7 %) | 130 (36.1 %) | 69 (19.2 %) |
| **Pointwise ensemble (mean gap, final)** | **131 (36.4 %)** | **149 (41.4 %)** | **80 (22.2 %)** |

Voting needs voters with signal. Averaging near-chance, position-driven judges mostly averages noise into ties.

### 4.4 Which supporter is better? Category scores (Qwen over Mistral)

| Configuration | Exploration | Insight | Action |
|---|:-:|:-:|:-:|
| Small ensemble, `n=2` | 0.50 [0.49, 0.52] | 0.51 [0.50, 0.53] | 0.51 [0.50, 0.52] |
| Small ensemble, `n=4` | 0.50 [0.45, 0.55] | 0.48 [0.42, 0.53] | 0.49 [0.44, 0.54] |
| GPT-OSS-120B, `n=4` | 0.74 [0.65, 0.83] | 0.74 [0.65, 0.83] | 0.62 [0.53, 0.71] |
| **Pointwise ensemble (final)** | 0.52 [0.44, 0.60] | 0.61 [0.50, 0.71] | 0.66 [0.55, 0.76] |

(0.50 = no preference; brackets are bootstrap 95 % CIs over personas.) Small pairwise ensembles cannot separate the supporters. The 120B judge and the pointwise ensemble both prefer Qwen, with CIs above 0.5 for Insight and Action in the pointwise case. Per-judge category scores are in `category_scores.csv`.

### 4.5 Pointwise scoring (final solution)

- **Position bias cannot occur**, because each conversation is scored alone.
- **ICC(1)** (share of score variance due to real differences between conversations): Gemma 0.30, Phi 0.15, DeepSeek negative, so DeepSeek's run-to-run noise exceeds its signal and it should be down-weighted.
- **Judges differ in direction:** mean Qwen-minus-Mistral score gap is +0.44 (Phi), +0.20 (Gemma), -0.13 (DeepSeek) points, which is why averaging gaps across judges is useful.
- **Threshold `delta`:** raising it increases ties and inflates raw agreement (0.61 at `delta = 1`) while Fleiss kappa stays near zero. See `delta_sensitivity.csv`.

### 4.6 Progression across stages

| Stage | Run agreement | Decisive (non-tie) verdicts | Agreement with 120B reference |
|---|:-:|:-:|:-:|
| S1 single small judge, `n=2` | 0.39 | 3 % | 0.33 |
| S2 self-consistency `n=4` | 0.46 | 19 % | 0.35 |
| S3 + large judge | 0.64 | 25 % | (is the reference) |
| **S4 pointwise ensemble** | 0.41 | **64 %** | **0.42** |

Run agreement is highest for the large judge, so **stability comes mainly from model scale**. Pointwise scoring adds decisiveness, the best agreement with the reference, and no position bias.

### 4.7 Possible confound: reply length

Mean supporter reply is about **79 words for Qwen** versus **140 for Mistral**. Verbosity bias would favour Mistral, yet the stronger judges prefer Qwen, so length does not explain the preference.

## 5. Ablation

Agreement and Cohen's kappa with the GPT-OSS-120B reference (small-judge averages where applicable):

| Variant | Agreement | kappa |
|---|:-:|:-:|
| Pairwise single run | 0.34 | -0.00 |
| Pairwise self-consistency `n=4` | 0.35 | 0.04 |
| Pairwise ensemble `n=2` | 0.33 | -0.02 |
| Pairwise ensemble `n=4` | 0.35 | 0.02 |
| Pointwise single run | 0.36 | 0.00 |
| Pointwise self-consistency (mean of 4) | 0.40 | 0.05 |
| **Pointwise ensemble, mean gap** | **0.42** | **0.07** |

More runs help little in pairwise mode, voting over pairwise judges does not help, and pointwise scoring with averaged runs and averaged gaps gives the best value. All kappa values remain "slight".

## 6. Discussion

- **Instrument versus finding.** For small pairwise judges the honest reading is "the instrument could not tell", not "the supporters are equal". Only the 120B judge and the pointwise ensemble give a consistent signal, and both prefer Qwen2.5-7B.
- **Why pointwise helps.** A nine-dimension pairwise prompt over two long transcripts overloads small models, which lean on the first conversation. A single transcript with anchored bands is an easier task, and averaging runs reduces sampling noise before the comparison.
- **Practical takeaway.** Use small pointwise judges for bulk screening and a large judge to audit, and calibrate against human-labelled pairs before trusting any of them.
- **Versus the base paper.** ESC-Judge reports 83-86 % match with human annotators on non-tie cases. Our numbers are not comparable (no human labels, 3-class agreement that counts ties), but they confirm that the limitation the authors raise (a single small judge sampled twice) is real for 3-7B judges.

## 7. Limitations

- **No human ground truth.** Reliability is measured, validity is not. The 120B judge is a proxy and can itself be wrong. The planned human-annotated validation set is the key next step.
- **Small sample.** 40 personas, one conversation per persona and supporter; confidence intervals are wide.
- **Large judge only pairwise.** The 120B judge was not run in pointwise mode.
- **Uneven run counts.** Some judge files contain incomplete runs (for example DeepSeek pointwise has 39 of 80 fourth runs; Phi-4-mini parses fewer outputs). These are treated as missing data.
- **Judge-family bias.** Pointwise judges may favour conversations from their own model family (flag `judge_is_supporter` in the raw records).
- **Design choices.** `delta`, the "never shown skill scores 1-2" rule and the single-call nine-dimension prompt are methodological choices.
- **Quantisation and simulator.** The 7B models run in 4-bit, and the help-seeker is a 3B model of limited realism.
- **Scope.** English only, one theory (Hill), no safety audit, no clinical validation.

## 8. Repository Structure

```
ESC_Judge_Project/
├── Conversation_Generation_Code/
│   ├── generate_conversations.py     # Simulates seeker <-> supporter conversations
│   ├── inspect_conversations.py      # Quick inspection / sanity checks of generated chats
│   ├── llm_utils.py                  # Local HF model wrapper (4-bit support, mock LLM for dry runs)
│   ├── persona_traits.py             # Behavioural-trait catalogue from the ESC-Judge repo
│   ├── roles-v1.json                 # Original ESC-Judge personas
│   ├── selected_personas_n40.csv     # The 40 personas used
│   └── emotional_supporter_nodir.txt # Non-directive supporter system prompt
├── Judge_Code/                       # Pairwise judging (Tracks A, B, C)
│   ├── esc_judge_pipeline.py         # Prompt, judge backends (hf / openai / mock), parsing
│   ├── run_judging.py                # Runs all judges over all pairs (config block at top)
│   ├── aggregate_results.py          # Single / self-consistency / ensemble + stability metrics
│   ├── rubric_9dim.json              # 9-dimension E-I-A rubric
│   └── requirements.txt
├── Judge_Code_scoring/               # Pointwise scoring (Track D, final solution)
│   ├── esc_judge_pointwise.py        # Rubric loader, prompt, parser, resumable runner
│   ├── run_pointwise.py              # Command-line runner
│   ├── aggregate_pointwise.py        # Scores -> winners -> configs, stability, agreement, CIs
│   ├── esc_judge_pipeline.py         # Shared backends (copy of the pairwise module)
│   ├── rubric_scored.json            # 9 dimensions x 5 scoring anchors
│   ├── README_pointwise.md           # Detailed notes on the pointwise method
│   ├── results_pointwise/            # Raw pointwise outputs (3 judges x 4 runs)
│   └── analysis_pointwise/           # verdicts_long, category_scores (with CI), stability,
│                                     # mean_scores, judge_agreement(+fleiss), delta_sensitivity
├── DATA/
│   ├── Generated_Converstion/                 # 2 x 40 generated conversations (.jsonl)
│   ├── Judging_Result_Model_wise/             # Raw pairwise outputs, n = 2 (3 small judges)
│   └── Judging_results_model_wise_nruns-4/    # Raw pairwise outputs, n = 4 (3 small judges + groq-judge)
├── Analysis_Output/
│   ├── analysis_(nruns=2)/                    # Stage 1: small judges, n = 2
│   ├── analysis-with_out_grok(nruns=4)/       # Stage 2: small judges, n = 4
│   └── analysis-with_grok(nruns=4)/           # Stage 3: + Groq-hosted GPT-OSS-120B
│       (each: verdicts_long.csv, category_scores.csv, stability.csv, judge_agreement.csv)
└── Report/                                    # (add) ESC_Judge_Report.pdf and LaTeX source
```

## 9. Reproducing the Experiments

### Setup

```bash
git clone https://github.com/pushpa-raj-k/ESC_Judge_Project
cd ESC_Judge_Project
pip install -r Judge_Code/requirements.txt
```

A GPU is needed for the local runs (developed on Kaggle/Colab-class GPUs). Gemma 3 is gated on Hugging Face: accept its licence and set `HF_TOKEN` (or run `huggingface-cli login`). The Groq judge needs `export GROQ_API_KEY=...`.

### Step 1: generate conversations

```bash
cd Conversation_Generation_Code
python generate_conversations.py --supporter qwen --dry_run                     # pipeline test, no GPU
python generate_conversations.py --supporter Qwen/Qwen2.5-7B-Instruct           # 40 personas
python generate_conversations.py --supporter mistralai/Mistral-7B-Instruct-v0.3 # same personas
```

> The `qwen` alias in `llm_utils.py` points to `Qwen2.5-3B-Instruct`. Results here use **Qwen2.5-7B-Instruct**, so pass the full model id.

### Step 2: pairwise judging (Tracks A, B, C)

Edit the config block in `Judge_Code/run_judging.py` (conversation paths, `OUT_DIR`, and `N_RUNS = 2` or `4`; defaults point to Kaggle paths). To add the large judge, append to `JUDGES`:

```python
JudgeConfig(name="groq-judge", model="openai/gpt-oss-120b", backend="openai",
            base_url="https://api.groq.com/openai/v1", api_key_env="GROQ_API_KEY",
            min_interval_s=3)
```

```bash
cd Judge_Code
python run_judging.py --dry-run     # mock judge, no model needed
python run_judging.py --limit 3     # smoke test on 3 pairs
python run_judging.py               # all judges
```

### Step 3: aggregate pairwise results (no GPU needed)

```bash
cd Judge_Code
# Stage 1 (n = 2)
python aggregate_results.py --results ../DATA/Judging_Result_Model_wise --rubric rubric_9dim.json --out analysis_n2
# Stage 3 (n = 4, includes groq-judge)
python aggregate_results.py --results ../DATA/Judging_results_model_wise_nruns-4 --rubric rubric_9dim.json --out analysis_n4_with_large
# Stage 2 (n = 4, small judges only): copy the three small-judge .jsonl files into a new folder and point --results there
```

### Step 4: pointwise scoring (final solution)

```bash
cd Judge_Code_scoring
python run_pointwise.py --conv ../DATA/Generated_Converstion/*.jsonl \
       --judges gemma-3-4b phi-4-mini deepseek-r1-distill-qwen-7b --n-runs 4
python aggregate_pointwise.py --results results_pointwise --rubric rubric_scored.json --out analysis_pointwise
```

Runs are resumable (finished runs are skipped). Keep `--temperature` above 0, otherwise all runs are identical. Raw outputs from the reported runs are included, so `aggregate_pointwise.py` can be re-run **without a GPU**. Details: [`Judge_Code_scoring/README_pointwise.md`](Judge_Code_scoring/README_pointwise.md).

## 10. Acknowledgements

- The **ESC-Judge** authors (Navid Madani and Rohini K. Srihari) for the framework, persona set and behavioural-trait catalogue this project builds on: <https://github.com/navidmdn/ESC-Judge>.
- Clara Hill's **Exploration-Insight-Action** model of helping skills.
- The open-weight model authors: Meta (Llama 3.2), Alibaba (Qwen2.5), Mistral AI, Microsoft (Phi-4-mini), Google (Gemma 3), DeepSeek and OpenAI (GPT-OSS-120B), served through Groq.
- The scope and write-up were discussed with an AI assistant (Claude, Anthropic); all experiments and final decisions are the team's own.

