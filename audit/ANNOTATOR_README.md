# Hi! Here's What You're Doing

You're helping check the quality of an AI benchmark. You'll look at 75 items where an AI was asked a question, and you judge: **did the AI answer correctly, yes or no?**

No AI/ML knowledge needed. If you can read a question and tell whether someone answered it right, you can do this.

**Time:** ~2 hours. Take breaks whenever you want.

---

## How to Start

Open the terminal in this IDE and run:

```
cd audit
python annotate.py items.jsonl
```

The tool shows you one item at a time. For each item you'll see:
1. The **question** the AI was asked
2. The AI's **response**

Then you type:
- **`c`** if the answer is correct
- **`i`** if the answer is incorrect
- **`s`** to skip (come back later)
- **`q`** to save and quit (resume anytime with the same command)

After each label, you rate confidence (1 = certain, 2 = likely, 3 = unsure) and can optionally add a note.

**Progress saves automatically.** You can quit and come back — it picks up where you left off.

---

## Judging Rules

### What counts as "correct"

The AI's **core answer** is factually right. It's fine if the response is:
- Long and wordy — as long as the answer is in there somewhere
- Hedged ("I think it's X") — if X is correct, that's correct
- Includes extra unrelated info — ignore the extras, judge the answer

### What counts as "incorrect"

- The AI gives the **wrong answer**
- The AI **refuses** to answer or says it can't help
- The AI is **too vague** to count (e.g., "it could be A or B" without committing)
- The AI answers a **different question** than what was asked
- For multi-part questions: if **any part** is wrong, the whole thing is incorrect

### Confidence levels

- **1 (certain):** No reasonable person would disagree with your call
- **2 (likely):** You're fairly sure but can see a small argument the other way
- **3 (unsure):** Genuinely ambiguous — could go either way. *This is totally fine and helpful — it tells us which items are tricky*

### What if you don't know the answer yourself?

- **Math/logic questions:** Feel free to use a calculator or work it out on paper. That's checking, not cheating
- **Factual questions you're unsure about:** Make your best guess based on whether the AI's response sounds coherent and specific vs. vague or self-contradictory. Mark confidence as `unsure` and add a note
- **Don't Google the answers.** We need your independent judgment

---

## Two Types of Items

The tool tells you which type each item is:

### "exp4" items
The AI received feedback about a previous task, then answered a NEW task. **Only judge the final answer.** Ignore everything about the feedback — pretend it's not there.

### "exp9" items
The AI was given a task that requires multiple skills (e.g., math + language). **All parts must be correct** for a "correct" label. If the AI chose to skip/defer/escalate instead of answering, that counts as "incorrect."

---

## Tips

- **Take a 5-min break every 25 items.** The tool will remind you
- **Don't overthink it.** If your gut says correct, it probably is
- **Don't go back and change answers.** First instinct is usually best
- **Notes on tricky items are gold.** Even "ambiguous wording" or "partial" helps a lot
- **There's no expected ratio.** Don't worry if you're marking mostly correct or mostly incorrect

---

## When You're Done

The tool will say "ALL DONE!" and your results are saved automatically in this folder. That's it — you're finished. Thank you!
