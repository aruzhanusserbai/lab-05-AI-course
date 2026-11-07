# Lab 05 — The harness, not just the model

Content: Agents, Memory and Skills, Harness and Reliable AI Systems · Narxoz University

You are a junior analyst at Dala Insight, a small fictional firm in Almaty that writes economic
briefs from regional statistics. Your colleague is a small open language model that can call five
functions on the firm's data. In this lab you **write its agent loop**, measure how often it is
right, put protective layers around it (a *harness*) **one at a time**, and measure again. You
predict each step before you see it; the interesting part is where you were wrong.

**No API key. No cost.** One open model (`Qwen2.5-3B-Instruct`) runs inside the notebook. The
download is about 6.2 GB, once per Colab session. Every region, indicator and number is invented.

## Setup

Open the notebook in Google Colab — one click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unreal-kz/lab-05-AI-course/blob/main/lab05_harness.ipynb)

1. **Runtime → Change runtime type → T4 GPU.** On a CPU the model is far too slow.
2. Run the first code cell once. It installs the two Python packages the lab was prepared with.
3. Run the cells in order, one at a time. A ✏️ cell is a cell where **you** write inside the quotes.
   Write each prediction *before* you run the cell that reveals the answer.

The reference run (below) took 640 s for the whole notebook on an Apple-silicon Mac (MPS, float16).
**A Colab T4 run has not been timed**, and it has not been checked that the 3B model fits and runs
there; the weights are about 6.2 GB in float16 and a T4 has 16 GB.

## What the notebook does

- **Part 0 — the data and 24 questions.** 655 rows: 20 regions, five indicators, 2019–2025. Three
  regions (Abai, Jetisu, Ulytau) were created in 2022 and have no earlier data. Each question's
  answer is **computed from the CSV**, never typed: 6 `lookup`, 5 `rank`, 4 `growth`, 3 `pp`
  (percentage points), 2 `nodata`, and 4 `alias`. An alias question names a region as people really
  write it (an old name, another spelling); the tools accept only the names in `list_regions`.
  ✏️ Predict 0.
- **Part 1 — workflow or agent.** A fixed pipeline (one model call extracts the parameters, plain
  code does the rest) against an agent loop, on the 15 questions whose shape is known in advance.
  ✏️ Predict 1, and what you see.
- **Part 2 — write the loop.** Three TODOs, about eight lines. `check_loop` compares your loop with
  the reference loop on three questions: with greedy decoding a correct loop gives the same final
  reply and the same number of tool results.
- **Part 3 — three layers, one at a time.** (1) tool errors go back to the model instead of
  stopping the program; (2) a step budget and a forced final reply; (3) grounding: every number
  in the answer must appear in a tool output. Results are *correct*, *wrong* or *no answer*, with
  95 % Wilson intervals. ✏️ Predict 3, and what you see.
- **Part 4 — memory.** You write the firm's notes (conventions and procedures, never data values)
  and measure what they change. ✏️ what you see.

## What you hand in

Everything is committed to **your fork** of this repository. Submit the URL of your last commit in
Canvas. Late commits are flagged, not hidden.

1. **`lab05_harness.ipynb`**: run from top to bottom, with all outputs saved and every ✏️ cell filled
   in. *File → Download → .ipynb*, and replace the template in your fork. The last cell lists any ✏️
   cell that is still empty.
2. **`CONCLUSION.md`**, three short answers, **typed by you without any AI tool**. That means no
   chatbot, no AI assistant, no AI autocomplete, no "rewrite my text" tool, not even for grammar.
   The instructor reads it personally. Rules, also written inside the file:
   - about 80–200 words per question; plain English that is yours beats polished English that is not;
   - every number you quote must be one **your own notebook printed**, copied exactly; a number your
     notebook did not print counts as wrong;
   - keep the three headings and the last line.

The notebook and `CONCLUSION.md` are separate on purpose: the notebook shows that you ran and
measured; `CONCLUSION.md` shows what you think the measurements mean.

## Reference run

Every number below is from an actual run (`reference_run.py`, `extra_checks.py`;
`Qwen/Qwen2.5-3B-Instruct`, `transformers` 5.18.0, `torch` 2.14.1, MPS float16, greedy decoding,
Python 3.12.13, 2026-11-02), not from memory. Decoding is greedy, so the notebook reproduced the
reference run on all 24 questions in every configuration, down to the raw model text, on that
machine. On Colab a T4 computes in a different numerical order, so expect your counts to differ from
these by a question or two; that has not been measured. Differences between configurations are
**small next to the intervals**: with 24 questions, two configurations whose intervals overlap are
not yet shown to differ.

| Configuration | correct / wrong / no answer (of 24) | 95 % interval |
|---|---|---|
| bare | 16 / 4 / 4 | 0.47–0.82 |
| + layer 1 | 17 / 7 / 0 | 0.51–0.85 |
| + layer 2 | 17 / 7 / 0 | 0.51–0.85 |
| + layer 3 | 18 / 4 / 2 | 0.55–0.88 |
| layer 3 + the reference notes | 20 / 1 / 3 | 0.64–0.93 |

## Extension tasks

Every number under "Expect" is from the reference run above. The Core tasks use only what the
notebook already holds; the Advanced one needs a little code of your own.

### Core — no new tools

**1. What a layer costs.**
Do: for every configuration in `RUNS`, add up the prompt and generated tokens over the 24 questions
(`r.prompt_tokens + r.new_tokens`) and divide by the number of correct answers.
Hand in: the table, and one sentence on which layer you would pay for.
Expect (tokens per correct answer): bare 2,471; layer 1 2,804; layer 2 2,804; layer 3 3,366; layer 3
with notes 3,672.
Trap: this number rewards correct answers and ignores wrong ones. Layers 1 and 2 have the most wrong
answers (7 each, against 4 for bare). The four `alias` questions explain it: the bare loop stopped
with *no answer* on all four after the first tool error; layer 1 let the model carry on, and it
answered one correctly (`q23`) and three wrongly (`q21` with a number, `q22` and `q24` with
`NO DATA`). A cost that does not count wrong answers cannot tell you which layer is safe.

**2. The step budget.**
Do: run layers 1 and 2 as `Harness(validate=True, limit=True, max_steps=n)` for `n` = 2, 3, 5, 8 on all
24 questions.
Hand in: the four (correct, wrong, no answer) triples, and one sentence on the budget you would ship.
Expect: n = 2 → 16 / 8 / 0 (7 runs hit the limit); n = 3, 5, 8 → 17 / 7 / 0 each (model calls 61 for
all three; tokens 46,768 / 47,674 / 47,674).
Trap: from 3 steps up the budget changes nothing, so on these questions it has never been *tested*:
a protection that nothing has triggered is a guess, not a result. At 2 steps it makes things worse.
The arithmetic questions need three model replies (`get_series`, `calculate`, the answer; look at
the trace of `q12`), so a budget of 2 cuts them off.

**3. Notes that are wrong.**
Do: set `NOTES` to exactly these two lines and run Part 4's cell:

```
The city of Almaty is Almaty Region.
compare_regions lists the regions lowest first.
```

Hand in: the counts before and after, and the questions whose outcome changed.
Expect: layer 3 alone 18 / 4 / 2 → with these notes 14 / 7 / 3. Six questions lose a correct answer
(`q06`, `q10`, `q13`, `q14`, `q16`, `q24`) and two gain one (`q18`, `q22`).
Trap: the notes pass the leak check, because it looks for *answers*, not for *mistakes*. And the
damage is not tidy. Two questions got better, and `q13` and `q14` (percent changes, which neither
line mentions) lost their answers; the reasons were not investigated. A false line in memory is a
bug you cannot find by reading one question's trace.

### Advanced — extra credit

**4. A second opinion.**
Do: on the 15 fixed-shape questions, compare the layer 3 agent's answer with the workflow's answer
(`WORKFLOW` from Part 1) and flag every question where they disagree or either has no answer.
Hand in: how many of the agent's non-correct answers you flagged, how many correct ones you flagged,
and whether you would let this run without a human.
Expect: the layer 3 agent is correct on 11 of the 15, wrong on 3 (`q08`, `q11`, `q21`) and gives no
answer on 1 (`q22`). All 4 are flagged; none of the 11 correct answers is.
Trap: 15 questions are few, and the agent and the workflow are the same model. This worked because the
workflow was right or silent exactly where the agent was wrong. Where the two make the *same* mistake
they agree and nothing is flagged, as with Lab 04's arithmetic checks. It also needs a workflow, and
a workflow exists only for questions of a known shape (15 of 24).

## Files

| File | What it is |
|---|---|
| `lab05_harness.ipynb` | The lab: 41 cells, Parts 0–4, eight ✏️ cells, no outputs saved |
| `lab05_core.py` | The toolbox, pasted unchanged into one notebook cell |
| `data/` | `regional_stats.csv`, `indicators.csv` (synthetic) |
| `CONCLUSION.md` | The template you replace with your own text |
| `README.md` | This file |

Instructor-side, not needed by students and kept in the instructor's course repository rather than
in this one: `build_notebook.py` (generates the notebook), `reference_run.py` → `reference_run.json`,
`run_log.txt` (the recorded run behind the numbers above), `extra_checks.py` → `extra_checks.json`,
`extra_checks_log.txt` (the extension-task numbers), `check_notebook.py` (the notebook reproduces the
reference run), `test_lab05_core.py` (47 unit tests of the toolbox, no model needed).
