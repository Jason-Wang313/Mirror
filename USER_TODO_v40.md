# MIRROR v40 — Human Action List

What I (Claude) finished: all paper edits, Pareto figure, audit scripts, submission-bundle script.
What you need to do: everything below. Each item is independent — prioritize by what matters to you.

Current state: `paper/mirror_draft_v40.tex` + `paper/mirror_draft_v40.pdf` — **34 pages total, main text 9 pages (within NeurIPS 9-page limit)**, refs 3 pages, appendix 22 pages.

Verified layout:
- p1–p9: main text (ends mid-§6.2 Limitations on p9)
- p10–p12: references
- p13+: Appendix A onwards

---

## 1. Anonymize the submission bundle (REQUIRED; ~15 min)

### 1a. Create the anonymous GitHub mirror

1. Go to https://anonymous.4open.science
2. Upload your repo. Note the URL it returns (something like `https://anonymous.4open.science/r/MIRROR-ABC123/`).
3. Edit `scripts/build_submission_bundle.sh` line 14:
   ```bash
   ANON_URL='PASTE-YOUR-ANONYMOUS-URL-HERE'
   ```

### 1b. Build the bundle

```bash
cd ~/MIRROR
bash scripts/build_submission_bundle.sh
bash scripts/anonymity_audit.sh
```

Expected: `RESULT: CLEAN (0 hits across submission artifacts)`.

If any hits remain: stop and inspect the output. Do NOT upload until clean.

### 1c. Verify the bundle

```bash
ls -la build/submission_bundle/
# should contain: paper/, data/, supplementary/, README.md
# paper/ should contain main.tex, main.pdf, tables/, figures/, appendix/, references.bib, neurips_2026.sty
```

### 1d. Files to upload to NeurIPS

- **Main PDF:** `build/submission_bundle/paper/main.pdf`
- **Supplementary ZIP:** zip up `build/submission_bundle/` and upload as supplementary material

---

## 2. Human study expansion to N≥40 (HIGHEST-LEVERAGE credibility upgrade; 7–14 days)

Currently: N=20, CFR=0.000, labeled "pilot." Sharp reviewers will attack this.

### 2a. Pre-register the analysis plan

1. Go to https://osf.io, create a new pre-registration.
2. Use `audit/human_baseline_packet/` as your reference for the protocol.
3. Declare:
   - Target N (recommend 40 new participants, total N=60)
   - Exclusion criteria (comprehension check failures, time < 5 min, etc.)
   - Primary outcome: participant-mean weak-domain CFR on Exp9 tasks
   - Secondary: Spearman rank correlation vs. best-model CFR
4. Download the OSF receipt PDF → save to `prereg/human_study_N40_prereg.pdf`.

### 2b. Run the study

**Platform:** Prolific (easiest) or MTurk (cheaper but noisier).
**Cost estimate:** $8–12/participant × 40 = $320–480 for a 30-min study.
**Input files to upload to Prolific:**
- Task bundle: `audit/human_baseline_packet/results/human_baseline_hardv2_summary.json` (task list)
- Prompts: `audit/human_baseline_packet/runs/v20_maximal_closure_20260326T2200Z/prompts/` (if present)
- Screener: existing 20-participant screener

### 2c. Score and integrate

```bash
cd ~/MIRROR
# Save raw Prolific data to:
mkdir -p data/human_baseline_v2
# drop CSVs from Prolific into data/human_baseline_v2/

python scripts/score_human_baseline_packet.py \
  --input data/human_baseline_v2 \
  --output results/human_baseline_v2_scored.json

python scripts/prepare_human_baseline_packet.py --merge-v1-v2
```

### 2d. Update the paper

Edit `paper/mirror_draft_v40.tex`:
- Line ~388 "Human-baseline pilot" paragraph: update `Twenty participants` → new N; remove "pilot" if CFR result holds.
- If N≥40 and CFR still ≈0: promote to abstract. Around line 74, replace "anonymized review package" sentence with a human-vs-best-model CFR gap number.

---

## 3. Cross-run reproducibility check (2–3 days wall time; needs API budget)

API-served models can drift. Running twice documents stability.

### 3a. Set API keys

```bash
export DEEPSEEK_API_KEY='sk-...'
export GOOGLE_API_KEY='...'      # for gemini-2.5-pro, gemma-3-12b
export NIM_API_KEY='nvapi-...'   # NVIDIA NIM, for most models
```

### 3b. Run the headline pipeline

```bash
cd ~/MIRROR
# Re-run just the headline experiments (not full pipeline) — ~$15 and ~3h
bash scripts/run_full_pipeline.sh --experiments 1,3,9 \
  --output-dir results/cross_run_$(date +%Y%m%d)
```

### 3c. Compute cross-run agreement

```bash
python analysis/cross_run_agreement.py \
  --run-a results/<original_run_dir> \
  --run-b results/cross_run_<date>
# Outputs per-metric Spearman rank correlation across models
```

### 3d. Add to appendix

If $\rho \geq 0.95$: add a short appendix subsection to `paper/mirror_draft_v40.tex` under Appendix T (Submission Disclosures). Template:

```latex
\subsection{Cross-Run API Stability}
\label{app:cross-run}
Re-running the headline experiments N days later yields Spearman rank
correlation $\rho = X$ on CFR across 16 models (Pearson on CCE: $r = Y$).
The knowing-doing gap is not an API-snapshot artifact.
```

If $\rho < 0.95$: investigate the diverging models before documenting.

---

## 4. v3 expansion to 1000+ tasks (optional defensive depth; ~$50–100 API budget)

Current v3 is 448 tasks. Bigger bank = stronger defense against "task set too small."

### 4a. Generate more tasks

```bash
cd ~/MIRROR
python scripts/generate_exp3_v3.py \
  --n-tasks 600 \
  --output data/exp3/v4_tasks.jsonl \
  --balance-pairs  # ensures all 28 domain pairs stay balanced
```

### 4b. Evaluate

```bash
python scripts/analyze_experiment_3.py \
  --tasks data/exp3/v4_tasks.jsonl \
  --models all \
  --output results/exp3_v4/
```

### 4c. Update paper

Edit `paper/mirror_draft_v40.tex` Finding 1 paragraph (line ~357):
- "v3 (448-task replication)" → "v4 (1048-task replication)"
- Update the appendix table `tab:exp3v2-ci` (in `paper/mirror_draft_v40.tex` ~line 846) with v4 numbers, OR add a sibling table.

---

## 5. Pre-registration receipt (REQUIRED if you claim pre-registration; ~10 min)

The v40 paper references a pre-registration receipt at `prereg/mirror_prereg_v1.pdf` that does not yet exist on disk.

Two options:

### 5a. If you actually pre-registered
Copy the receipt to:
```
prereg/mirror_prereg_v1.pdf
prereg/experiment_7_protocol.pdf
prereg/deviation_log.md
```

### 5b. If you didn't (honest path)
Edit `paper/mirror_draft_v40.tex` around line 1145 (Appendix T Pre-Registration Receipts subsection):
- Soften the claim from "pre-registered" to "analysis plan documented in repository commit log before data collection"
- OR cut the subsection entirely and remove the corresponding footnote (line ~265) + abstract/intro mentions of pre-registration

Do NOT leave a dangling "pre-registration receipt" claim if no receipt exists.

---

## 6. External cold read (DO LAST; 1 hour)

Send `paper/mirror_draft_v40.pdf` to one trusted person who is NOT on the paper. Ask:

1. After 10 minutes, what's the headline claim?
2. What's the strongest piece of evidence?
3. What part of the abstract did you re-read?
4. What confused you?

If they can't crisply answer (1) and (2), the framing still needs work.

---

## 7. Final submission checklist (DO DAY OF)

- [ ] `paper/mirror_draft_v40.tex` compiles clean (warnings OK, no errors)
- [ ] Main text ≤ 9 pages (item 0 above)
- [ ] References section exists, BibTeX resolved (no `[?]` citations)
- [ ] All `\ref{...}` resolve (grep PDF text for `??`)
- [ ] `bash scripts/anonymity_audit.sh` → RESULT: CLEAN on the BUNDLE
- [ ] `python -c "import fitz; print(fitz.open('build/submission_bundle/paper/main.pdf').metadata)"` → all fields empty
- [ ] OpenReview draft submission saved (don't upload yet)
- [ ] Pre-registration receipt file exists at `prereg/mirror_prereg_v1.pdf` OR the pre-reg claim is removed
- [ ] Human study N≥40 (if doing) — updated numbers propagated to abstract + §5

---

## File map (what I created / modified in this session)

```
paper/mirror_draft_v40.tex              # the new paper (edited from v39)
paper/mirror_draft_v40.pdf              # compiled output
paper/tables/table1_main_results.tex    # bolded best/worst + updated caption
paper/tables/table4_models.tex          # gpt-oss rename + clearer footnote
paper/figures/fig3_pareto_routing.pdf   # NEW — Pareto Finding 3 figure
analysis/plot_pareto_routing.py         # NEW — generates fig3_pareto_routing.pdf
scripts/anonymity_audit.sh              # NEW — deanon check (run before submit)
scripts/build_submission_bundle.sh      # NEW — builds anonymized bundle
USER_TODO_v40.md                        # THIS file
.claude/plans/abstract-yawning-rainbow.md  # original plan doc
```

## File map (what you need to create)

```
build/submission_bundle/                # built by scripts/build_submission_bundle.sh
prereg/mirror_prereg_v1.pdf             # see item 5
prereg/experiment_7_protocol.pdf        # see item 5
prereg/deviation_log.md                 # see item 5
data/human_baseline_v2/*.csv            # see item 2 (if doing human study)
results/cross_run_<date>/               # see item 3 (if doing cross-run)
results/exp3_v4/                        # see item 4 (if doing v3 expansion)
```

## Unaffected originals

These were NOT modified — they're still your public-facing versions:
- `paper/mirror_draft_v39.tex` + `.pdf` (last-known-good)
- `paper/arxiv_submission/main.tex` (non-anon preprint version, has your real name — not shipped to NeurIPS)
- `paper/arxiv_upload_source/main.tex` (same)
- `README.md` (the anonymizer runs over a copy, not the original)
- `data/croissant_metadata.json` (same)
