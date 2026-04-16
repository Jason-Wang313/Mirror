#!/usr/bin/env bash
# Anonymity audit for NeurIPS 2026 E&D submission artifacts.
# Run before bundling any artifact for upload.
# Exit code: 0 = clean; nonzero = at least one hit (do NOT upload).

set -u

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# Patterns that would deanonymize the author.
# Add to this list whenever new identifying terms appear.
PATTERNS=(
  'Jason-Wang313'
  'Jason Z Wang'
  'Jason Wang'
  'jasonhearlte'
  'wangz'
)

# Roots that get bundled for the NeurIPS submission. These MUST be clean.
# Explicitly excludes arxiv_submission/ and arxiv_upload_source/ which are
# intentionally non-anonymous (post-acceptance preprint variants).
SUBMISSION_ROOTS=(
  "$REPO_ROOT/paper/mirror_draft_v40.tex"
  "$REPO_ROOT/paper/mirror_draft_v40.pdf"
  "$REPO_ROOT/paper/tables"
  "$REPO_ROOT/paper/figures"
  "$REPO_ROOT/paper/appendix"
  "$REPO_ROOT/paper/references.bib"
  "$REPO_ROOT/paper/supplementary"
  "$REPO_ROOT/data/croissant_metadata.json"
  "$REPO_ROOT/README.md"
  "$REPO_ROOT/REPRODUCE.md"
  "$REPO_ROOT/SETUP.md"
  "$REPO_ROOT/LICENSE"
  "$REPO_ROOT/setup.py"
  "$REPO_ROOT/requirements.txt"
)

EXCLUDE_DIRS=(
  '.git'
  '__pycache__'
  'arxiv_submission'
  'arxiv_upload_source'
  'mirror_arxiv_source.zip'
  'mirror_bench.egg-info'
)

EX_ARGS=()
for d in "${EXCLUDE_DIRS[@]}"; do
  EX_ARGS+=( --exclude-dir="$d" )
done

PAT_ARGS=()
for p in "${PATTERNS[@]}"; do
  PAT_ARGS+=( -e "$p" )
done

HITS=0
echo "== Anonymity audit =="
echo "Repo root: $REPO_ROOT"
echo "Patterns: ${PATTERNS[*]}"
echo

for root in "${SUBMISSION_ROOTS[@]}"; do
  if [[ ! -e "$root" ]]; then
    echo "  [skip] $root (not present)"
    continue
  fi
  printf "  scanning %s ... " "$root"
  if [[ -d "$root" ]]; then
    out=$(grep -rn "${EX_ARGS[@]}" "${PAT_ARGS[@]}" "$root" 2>/dev/null || true)
  else
    out=$(grep -n "${PAT_ARGS[@]}" "$root" 2>/dev/null || true)
  fi
  if [[ -n "$out" ]]; then
    n=$(printf '%s\n' "$out" | wc -l)
    HITS=$((HITS + n))
    printf 'HITS\n'
    printf '%s\n' "$out"
  else
    printf 'clean\n'
  fi
done

echo
echo "== PDF metadata check =="
for pdf in "$REPO_ROOT/paper/mirror_draft_v40.pdf" "$REPO_ROOT/paper/figures"/*.pdf; do
  [[ -f "$pdf" ]] || continue
  meta=$(python -c "
import sys, fitz
try:
    d = fitz.open(sys.argv[1])
    m = d.metadata or {}
    for k in ('author','title','subject','keywords','creator','producer'):
        v = (m.get(k) or '').strip()
        if v:
            for p in ${PATTERNS[*]@Q}.split():
                pass
        print(f'  {k}: {v!r}')
except Exception as e:
    print(f'  [error] {e}')
" "$pdf" 2>/dev/null)
  echo "  $pdf:"
  echo "$meta"
done

echo
if [[ "$HITS" -eq 0 ]]; then
  echo "RESULT: CLEAN (0 hits across submission artifacts)."
  exit 0
else
  echo "RESULT: $HITS hit(s). DO NOT UPLOAD until resolved."
  exit 1
fi
