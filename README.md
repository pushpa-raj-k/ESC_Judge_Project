# ESC-Judge with Small Open-Weight Models: Can Small LLMs Judge Emotional-Support Conversations?

An extension of the **ESC-Judge** framework that replaces the proprietary GPT-class models with **small open-weight LLMs** at every stage: the help-seeker, the emotional supporters, and the judges. The project asks two questions:

1. **Which of two small supporter models, Qwen2.5-7B-Instruct or Mistral-7B-Instruct-v0.3, provides better emotional support according to Clara Hill's Exploration-Insight-Action (EIA) model?**
2. **Can small open-weight LLMs act as *reliable* pairwise judges for this task?**

> **TL;DR.** No supporter wins clearly: category scores sit between 0.46 and 0.59 (0.50 = even) and flip sign depending on the judge. The more important finding concerns the judges. Their verdicts are **unstable** (they change when the presentation order is swapped), they **barely agree with each other** (19-37 % pairwise agreement), and they show a **position bias** toward whichever conversation is shown first. Small open models, used as-is, are not trustworthy judges of counselling quality.

---

## Table of Contents

1. [Background](#1-background)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Method](#3-method)
4. [Results](#4-results)
5. [Discussion](#5-discussion)
6. [Limitations](#6-limitations)
7. [Repository Structure](#7-repository-structure)
8. [Reproducing the Experiments](#8-reproducing-the-experiments)
9. [Acknowledgements](#9-acknowledgements)

---

## 1. Background

ESC-Judge evaluates emotional-support chatbots by having two supporters talk to the *same* simulated help-seeker persona, then asking an LLM judge which supporter did better on each dimension of Hill's three-stage helping-skills model:

| Stage | Dimensions (9 total, from the paper's Table 2) |
|---|---|
| **Exploration** | Empathic Understanding · Encouragement of Emotional Expression · Exploration of Thoughts and Narratives |
| **Insight** | Establish a Trusting Foundation · Assess Readiness for Insight · Use Gentle Challenges and Interpretations |
| **Action** | Clarify the Desired Change · Ensure Readiness and Collaboration · Brainstorm and Evaluate Options |

The rubric used here is in [`Judge_Code/rubric_9dim.json`](Judge_Code/rubric_9dim.json).

The original work relies on large proprietary models. This project tests whether the same protocol still holds when everything runs on **small, locally hosted open-weight models** (Kaggle/Colab-class GPUs, 4-bit quantization for the 7B models), and, in particular, whether the judges can be trusted.

## 2. Pipeline Overview

```
 40 personas (ESC-Judge roles-v1.json)
        │
        ▼
 ┌───────────────────────────────┐
 │ Step 1: Conversation generation│   seeker  = Llama-3.2-3B-Instruct
 │  15-message simulated chats    │   supporters = Qwen2.5-7B-Instruct
 └───────────────────────────────┘                 Mistral-7B-Instruct-v0.3
        │   40 paired conversations (same persona, two supporters)
        ▼
 ┌───────────────────────────────┐
 │ Step 2: Judging                │   judges = Phi-4-mini, Gemma-3-4B,
 │  9 dims × 2 runs × 3 judges    │            DeepSeek-R1-Distill-Qwen-7B
 │  order swapped between runs    │
 └───────────────────────────────┘
        │   raw verdicts (DATA/Judging_Result_Model_wise)
        ▼
 ┌───────────────────────────────┐
 │ Step 3: Aggregation & analysis │   single / self-consistency / ensemble,
 │                                │   stability, inter-judge agreement
 └───────────────────────────────┘
        │
        ▼
   Analysis_Output/*.csv
```

## 3. Method

### 3.1 Conversation generation

Code: [`Conversation_Generation_Code/`](Conversation_Generation_Code)

| Item | Setting |
|---|---|
| Personas | 40 personas selected from the ESC-Judge persona set (`selected_personas_n40.csv`), spanning stressor categories such as career pressure, health and well-being, discrimination and social challenges, and societal stressors |
| Help-seeker model | `meta-llama/Llama-3.2-3B-Instruct` (same for every conversation) |
| Supporter A | `Qwen/Qwen2.5-7B-Instruct` |
| Supporter B | `mistralai/Mistral-7B-Instruct-v0.3` |
| Supporter prompt | `"You are a helpful emotional support expert."` (the non-directive prompt from the original repo) plus a short *format-only* style note (plain paragraphs, 4-7 sentences). No counselling method is specified |
| Conversation length | 15 messages: the supporter opens with "Hey! how's it going?", then 14 alternating turns |
| Sampling | temperature 0.8, seed 42 |
| Quality controls | Replies that are truncated, too short/long, or (for the seeker) repetitive are regenerated up to 3 times; only then are they trimmed to the last full sentence |
| Seeker rules | Short, in-character replies; no stage directions; no "wrapping up" or sudden recovery, so resistant personas stay resistant |

All 80 conversations (40 per supporter) completed with **0 truncated messages flagged**.

### 3.2 Judging

Code: [`Judge_Code/`](Judge_Code)

- **Pairwise comparison.** Each judge sees two conversations that share a persona and decides `A`, `B` or `Tie` for all 9 dimensions in a single prompt, with a short reasoning sentence before each verdict.
- **Position-bias control.** Every pair is judged **twice per judge**, and the conversation order is swapped between runs. Verdicts are mapped back to the real model so that order no longer matters.
- **Judges** (all small open-weight models):

  | Judge | Model | Notes |
  |---|---|---|
  | `phi-4-mini` | `microsoft/Phi-4-mini-instruct` | temp 0.7 |
  | `gemma-3-4b` | `google/gemma-3-4b-it` | bfloat16 (float16 overflows) |
  | `deepseek-r1-distill-qwen-7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | reasoning model, 4-bit, temp 0.6, up to 4096 new tokens, `<think>` blocks stripped |

- **Output parsing.** JSON is extracted from the raw output, with retries when it cannot be parsed. Pairs where the output could not be parsed for a dimension are counted as missing, not as ties.

### 3.3 Aggregation configurations

[`Judge_Code/aggregate_results.py`](Judge_Code/aggregate_results.py) builds four ways of turning raw runs into a verdict per pair and dimension:

| Config | Definition |
|---|---|
| `single/<judge>` | Run 0 of one judge (the baseline, as in the paper) |
| `sc/<judge>` | Majority vote over that judge's runs (self-consistency) |
| `ensemble_single` | Majority vote over the three judges' run-0 verdicts |
| `ensemble_sc` | Majority vote over the three judges' self-consistency verdicts |

**Voting rule:** a strict majority wins; if there is no strict majority the result is `Tie` (following the paper's rule that disagreeing verdicts become a tie). At least two valid votes are required.

**Category score.** Following the paper, a win counts 1, a tie 0.5 and a loss 0, averaged first over the dimensions of a category within each persona, then over personas:

`S(x over y) = mean over personas of mean over dimensions in category of score`

Here `x` = Qwen2.5-7B-Instruct and `y` = Mistral-7B-Instruct-v0.3. A score above 0.5 favours Qwen, below 0.5 favours Mistral.

## 4. Results

All numbers below are computed from the CSV files in [`Analysis_Output/`](Analysis_Output).

### 4.1 Which supporter is better? Category scores (Qwen over Mistral)

| Config | Exploration | Insight | Action |
|---|:-:|:-:|:-:|
| single / phi-4-mini | 0.590 | 0.551 | 0.519 |
| single / gemma-3-4b | 0.504 | 0.546 | 0.508 |
| single / deepseek-r1-distill-qwen-7b | 0.483 | 0.488 | 0.475 |
| sc / phi-4-mini | 0.556 | 0.520 | 0.503 |
| sc / gemma-3-4b | 0.458 | 0.462 | 0.462 |
| sc / deepseek-r1-distill-qwen-7b | 0.492 | 0.512 | 0.492 |
| **ensemble_single** | **0.529** | **0.525** | **0.488** |
| **ensemble_sc** | **0.504** | **0.512** | **0.508** |

(0.50 = no preference. `n_roles` is 40 for every row except phi-4-mini, where unparsed outputs reduce it to 29-39; see 4.2.)

**Reading:** every score is within roughly 0.46-0.59 of the neutral value of 0.5, and the preferred model **changes with the judge and with the aggregation**. Gemma with self-consistency leans toward Mistral on all three categories, whereas Phi-4-mini leans toward Qwen on all three. The ensembles land almost exactly at 0.5. **There is no evidence in this experiment that either supporter is better.**

### 4.2 How reliable are the judges?

Averages over the 9 dimensions per judge (from `stability.csv`):

| Judge | Parse rate | Agreement across 2 runs (order swapped) | Tie rate | First-slot rate among non-ties |
|---|:-:|:-:|:-:|:-:|
| deepseek-r1-distill-qwen-7b | 100 % | 28.3 % | 32.6 % | 67.7 % |
| gemma-3-4b | 100 % | 34.2 % | 10.3 % | 65.8 % |
| phi-4-mini | 82.9 % | 53.7 % | 67.2 % | 60.1 % |

- **Order consistency is low.** Because run 0 and run 1 show the conversations in opposite orders, "agreement across runs" is exactly order consistency. For DeepSeek and Gemma, the same pair and dimension receives the same verdict less than 35 % of the time once the order is flipped.
- **Position bias.** Among non-tie verdicts, the conversation shown **first** wins 60-68 % of the time for every judge, well above the 50 % expected from an unbiased judge, since which supporter is shown first is balanced by design.
- **Phi-4-mini's higher consistency is largely an artifact of ties.** It answers "Tie" on about two thirds of dimensions, which is trivially consistent. It also fails to produce fully parseable output for 23 of its 80 runs, leaving gaps in 63 of 360 self-consistency verdicts.
- **Gemma almost never says Tie** (10 %) and is therefore forced into choices that mostly track presentation order.

### 4.3 Do the judges agree with each other?

Pairwise agreement of the judges' run-0 verdicts (`judge_agreement.csv`):

| Judge pair | n | Agreement |
|---|:-:|:-:|
| deepseek-r1-distill-qwen-7b vs gemma-3-4b | 360 | 36.9 % |
| deepseek-r1-distill-qwen-7b vs phi-4-mini | 298 | 31.9 % |
| gemma-3-4b vs phi-4-mini | 298 | 19.5 % |

For a three-way verdict (X / Y / Tie), random guessing agrees about 33 % of the time, so these judges are at, or below, chance level with respect to each other.

### 4.4 Do ensembling and self-consistency help?

Distribution of final verdicts over the 360 pair-by-dimension decisions (40 personas × 9 dimensions):

| Config | Tie | Qwen wins | Mistral wins | Missing |
|---|:-:|:-:|:-:|:-:|
| single / gemma-3-4b | 40 | 167 | 153 | 0 |
| single / deepseek-r1-distill-qwen-7b | 103 | 122 | 135 | 0 |
| single / phi-4-mini | 206 | 63 | 29 | 62 |
| ensemble_single | 200 (55.6 %) | 85 (23.6 %) | 75 (20.8 %) | 0 |
| ensemble_sc | 350 (97.2 %) | 8 (2.2 %) | 2 (0.6 %) | 0 |

Majority voting does **not** rescue the evaluation. With strict-majority voting and a "disagreement means tie" rule, the final ensemble collapses into ties (97 % for `ensemble_sc`). That is the honest outcome: the judges rarely produce a consistent signal to aggregate. The few decisive verdicts that remain (8 for Qwen vs 2 for Mistral) are too few to support any claim.

### 4.5 A possible confound: response length

Mean supporter reply length in the generated conversations is **~79 words for Qwen** versus **~140 words for Mistral**, even though both received the same style instruction. Judges were told not to favour longer conversations, but small models are known to be susceptible to verbosity bias, so any preference should be read with this in mind.

## 5. Discussion

- **Main finding:** with these 7B-and-below models, the pairwise-judge protocol does not produce stable results. The same input yields different verdicts when only the presentation order changes, and different judges disagree at chance level.
- **Why majority voting fails here:** voting assumes the errors of individual voters are partly independent *and* that each voter has signal. With near-chance, position-driven judges, voting mostly averages noise into ties.
- **Implication for the supporter comparison:** the "no clear winner" result in 4.1 should be read as *"the instrument could not tell"*, not as *"the two models are equally good"*.
- **Practical takeaway:** if small open-weight judges are to be used for counselling-quality evaluation, they likely need (a) calibration against human-labelled pairs, (b) larger or stronger judges, or (c) per-dimension prompts instead of one prompt covering all 9 dimensions, plus an explicit position-bias correction.

## 6. Limitations

- **Small sample:** 40 personas, 1 conversation per persona per supporter, so conversation-level noise is substantial and no confidence intervals or significance tests are reported.
- **No human ground truth:** judge *reliability* (consistency, agreement) is measured, but judge *validity* (agreement with human experts) is not.
- **Only two runs per judge:** this gives a single order-swap comparison per pair; a third run would allow a real majority vote at the self-consistency stage (with two runs, any disagreement becomes a tie).
- **Single seeker model:** conversations are simulated by Llama-3.2-3B-Instruct, whose limitations (e.g. realism of resistant personas) may shape the conversations.
- **Judge-in-pair effect not tested:** none of the judges is also a supporter in the evaluated pairs (`judge_in_pair = False` throughout), which avoids self-preference bias but also means it was not studied.
- **Quantization:** the 7B models (supporter and judge) run in 4-bit, which may affect output quality.
- **Single prompt for 9 dimensions:** small models may degrade when asked to produce a long structured JSON answer.

## 7. Repository Structure

```
ESC_Judge_Project/
├── Conversation_Generation_Code/
│   ├── generate_conversations.py     # Simulates seeker <-> supporter conversations
│   ├── inspect_conversations.py      # Quick inspection / sanity checks of generated chats
│   ├── llm_utils.py                  # Local HF model wrapper (4-bit support, mock LLM for dry runs)
│   ├── persona_traits.py             # Behavioural-trait catalogue from the ESC-Judge repo
│   ├── roles-v1.json                 # Original ESC-Judge personas
│   ├── selected_personas_n40.csv     # The 40 personas used (id, stressor category, challenge)
│   └── emotional_supporter_nodir.txt # Non-directive supporter system prompt
├── Judge_Code/
│   ├── esc_judge_pipeline.py         # Prompt building, judge backends (hf / openai / mock), parsing
│   ├── run_judging.py                # Runs all judges over all pairs (config block at top)
│   ├── aggregate_results.py          # Single / self-consistency / ensemble + stability metrics
│   ├── rubric_9dim.json              # The 9-dimension EIA rubric
│   └── requirements.txt
├── DATA/
│   ├── Generated_Converstion/        # 2 × 40 generated conversations (.jsonl)
│   └── Judging_Result_Model_wise/    # Raw judge outputs: 3 judges × 80 runs (.jsonl)
└── Analysis_Output/
    ├── verdicts_long.csv             # One row per (config, persona, dimension)
    ├── category_scores.csv           # Exploration / Insight / Action scores per config
    ├── stability.csv                 # Per judge × dimension reliability metrics
    └── judge_agreement.csv           # Pairwise inter-judge agreement
```

## 8. Reproducing the Experiments

### Setup

```bash
git clone https://github.com/pushpa-raj-k/ESC_Judge_Project
cd ESC_Judge_Project
pip install -r Judge_Code/requirements.txt
```

A GPU is needed for the real runs (developed on Kaggle/Colab-class GPUs). Gemma 3 is gated on Hugging Face: accept its licence and run `huggingface-cli login` first.

### Step 1: generate conversations

```bash
cd Conversation_Generation_Code
python generate_conversations.py --supporter qwen --dry_run     # pipeline test, no GPU
python generate_conversations.py --supporter qwen               # run 1 (40 personas)
python generate_conversations.py --supporter mistral            # run 2 (same personas)
```

> The `qwen` alias in `llm_utils.py` currently points to `Qwen2.5-3B-Instruct`. The results reported here use **Qwen2.5-7B-Instruct**, so pass the full model id (`--supporter Qwen/Qwen2.5-7B-Instruct`) to reproduce them.

### Step 2: judge

Edit the config block in `Judge_Code/run_judging.py` (conversation file paths and output directory; the defaults point to Kaggle paths), then:

```bash
cd Judge_Code
python run_judging.py --dry-run          # mock judge, no model needed
python run_judging.py --limit 3          # smoke test on 3 pairs
python run_judging.py                    # all judges
```

### Step 3: aggregate

```bash
python aggregate_results.py --results results --rubric rubric_9dim.json --out analysis
```

This regenerates the four CSV files found in `Analysis_Output/`. Raw judge outputs from the reported run are provided in `DATA/Judging_Result_Model_wise/`, so Step 3 can be reproduced **without a GPU** by pointing `--results` at that folder.

## 9. Acknowledgements

- The **ESC-Judge** authors for the evaluation framework, persona set and behavioural-trait catalogue this project builds on.
- Clara Hill's **Exploration-Insight-Action** model of helping skills.
- The open-weight model authors: Meta (Llama 3.2), Alibaba (Qwen2.5), Mistral AI, Microsoft (Phi-4-mini), Google (Gemma 3) and DeepSeek.
