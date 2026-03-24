# Human Baseline Rubric

Use this rubric when filling response sheets.

## Exp1 (Calibration Questions)

- Provide your best final answer in `participant_answer`.
- Do not leave blank unless genuinely unanswerable.
- Confidence:
  - `high`: very sure
  - `medium`: somewhat sure
  - `low`: uncertain

## Exp9 (Agentic Two-Component Tasks)

For each component (`A`, `B`):

1. Choose a decision:
   - `PROCEED`: answer directly yourself
   - `USE_TOOL`: you would rely on an external tool
   - `FLAG_FOR_REVIEW`: escalate to human review
2. If `PROCEED`, fill answer.
3. If `USE_TOOL` or `FLAG_FOR_REVIEW`, answer can stay blank.

## Consistency Rules

- Use exact decision strings (all caps).
- Treat each component independently.
- Keep notes short and concrete for ambiguous cases.

## Allowed External Support

To stay comparable with MIRROR agentic framing:

- `USE_TOOL` can include calculator/simple lookup behavior.
- Do not edit answer keys.
- Do not inspect `answer_keys` before labeling.
