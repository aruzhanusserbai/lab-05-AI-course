# Lab 05 — Conclusion

<!--
Read this before you write.

This file is graded by your instructor, by hand, and it is worth 6 of the 15 points.

- Type it yourself. Do not use a chatbot, an AI assistant, an AI autocomplete or a "rewrite my
  text" tool for any part of it, not even for grammar. English that is plain and a little
  clumsy but yours is better than polished text that is not.
- Use your own numbers. Every number you quote must be one your own notebook printed. Copy
  the number exactly as it appeared. A number your notebook did not print is marked as wrong.
- Answer in your own words, about 80-200 words per question. A short answer that uses your
  numbers beats a long answer that could have been written by anyone.
- Delete these comment lines if you like. Keep the three headings and the last line.
-->

## 1. Which layer changed your results the most?

Give the numbers before -> after for that layer (correct / wrong / no answer, out of 20) and say
whether it changed what the agent *can do* or only what it *admits*. If no layer changed the
number of correct answers, say so and explain what the layers did change instead.

Layer_1 changed my results the most, before it was 16 / 4 / 4 -> 18 / 6 / 0 and this is after. It was the only model 
that increased the correct answer's number. Without Layer_1 4 alias questions answer was no answer, because the loop stopped
after it threw ToolError. With Layer_1 instead of ending loop with error, it sent the error message back to the model, so 
it can see the error message like why it stopped and what shoud it do to handle it, in the message it says call list_region.
So tha is how it got correct answers to the questions number 23 and 24. For the questions number 21 and 22 it got wrong answer, 
because it picked the wrong region names.
(There are actually 24 questions but in your question you asking out of 20, so I am consuming that this is just a mistake, if it is not, then maybe you didn't
count the alias questions because they are the ones that changed its results. So Layer_1 and others didn't changed any results into correct)
Layer_2 changed nothing, before 18 / 6 / 0 -> 18 / 6 / 0 after.
Layer_3 changed 18 / 6 / 0 -> 18 / 4 / 2. It did not added any correct answers, but it turned q18 and q22 that were wrong into no answer. So it changed only what the agent admits, not what it can do.

## 2. One failure the full harness did not fix

Pick one question that the full harness (layer 3) still gets wrong or refuses. Give its id
(q01 ... q20). Open its trace in the notebook: at which step did it go wrong, and what did
the model do instead? Name one thing you would add to the harness, and what it would cost
(extra model calls, tokens, time, or a new way to fail).

q18. Model first called get_series for Jambyl, cpi_yoy, 2022-2025 and got the right values, it was 16.1 in 2022 and 10.6 in 2025. 
It went wrong at the second step, so instead of calculating 16.1 - 10.6 (5.5 percentage points), it called calculate with (16.1 - 10.6)/100 and got 0.055. 
Then it wrote ANSWER: 0.55, which is not in any tool output. Layer 3 sent it back, but model repeated the same wrong calculation, Layer 2 stopped the repeated call, the model wrote 0.55 again, and the answer became UNVERIFIED. 
The harness stopped the wrong number, but it couldn't tell the model that the formula was wrong.
I would add a self check step. After the final answer, the harness asks the model to reread the question and check that its formula matches what it was asking. The cost is one extra model call for every question, so it will be about 24 more calls, more tokens and more time, even when the answer is already right. It also add a new way to fail, model may doubt a correct answer and change it to a wrong one.

## 3. Workflow or agent for the monthly ranking?

Dala Insight wants to publish, every month, "which region is highest and which is lowest on one
indicator in one year". Would you ship the fixed workflow or the agent? Use the accuracy, the
number of model calls, the tokens and the time you measured in Part 1. Then say in one sentence
what kind of question would make you choose the other one.

I would ship the fixed workflow one. The monthly ranking always has the same shape, which is exactly what the workflow do. In Part 1, on the 15
fixed shape questions, the workflow got 13/15 right and the agent only 9/15. Workflow also used 15 model calls instead of 26 model calls, fewer tokens about 5 times(3852 then 19835) and was about 3 times faster (16.5 s then 48.7 s). Also, the workflow got all 5 rank questions right, because
plain code picks the first or the last region from the sorted list. The agent got q08 and q11 wrong even with all three layers on, because the model picked the region by himself.
I would choose the agent for questions whose shape is not fixed, like a percent change between two years, which needs several steps like getting two values and then calculating it.

---

I wrote this file myself, without any AI tool.
