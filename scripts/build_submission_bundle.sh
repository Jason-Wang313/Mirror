#!/usr/bin/env bash
# Build the NeurIPS 2026 E&D anonymized submission bundle.
# Produces a clean copy under build/submission_bundle/ with:
#  - the v40 paper PDF
#  - tables/, figures/, appendix/, references.bib
#  - an anonymized README and croissant_metadata.json
#  - the supplementary/ directory
#  - the Croissant metadata
# DOES NOT mutate any source files. Run anonymity_audit.sh against the bundle
# before uploading.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BUNDLE="$REPO_ROOT/build/submission_bundle"

# Anonymized GitHub URL — swap to anonymous.4open.science upload after creation.
# Until that URL exists, point reviewers at the supplementary archive.
ANON_URL='https://anonymous.4open.science/r/Mirror-NEURIPS2026  (see supplementary archive)'

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/paper" "$BUNDLE/data" "$BUNDLE/supplementary"

# Paper artifacts
cp "$REPO_ROOT/paper/mirror_draft_v40.tex" "$BUNDLE/paper/main.tex"
[[ -f "$REPO_ROOT/paper/mirror_draft_v40.pdf" ]] && \
  cp "$REPO_ROOT/paper/mirror_draft_v40.pdf" "$BUNDLE/paper/main.pdf"
cp -r "$REPO_ROOT/paper/tables" "$BUNDLE/paper/"
cp -r "$REPO_ROOT/paper/figures" "$BUNDLE/paper/"
cp -r "$REPO_ROOT/paper/appendix" "$BUNDLE/paper/"
cp "$REPO_ROOT/paper/references.bib" "$BUNDLE/paper/"
cp "$REPO_ROOT/paper/neurips_2026.sty" "$BUNDLE/paper/" 2>/dev/null || true

# Supplementary
cp -r "$REPO_ROOT/paper/supplementary"/* "$BUNDLE/supplementary/" 2>/dev/null || true

# Anonymized README
sed "s|https://github.com/Jason-Wang313/Mirror\.git|${ANON_URL}|g; s|https://github.com/Jason-Wang313/Mirror|${ANON_URL}|g" \
  "$REPO_ROOT/README.md" > "$BUNDLE/README.md"

# Anonymized Croissant metadata
sed "s|https://github.com/Jason-Wang313/Mirror|${ANON_URL}|g" \
  "$REPO_ROOT/data/croissant_metadata.json" > "$BUNDLE/data/croissant_metadata.json"

# Strip PDF metadata from any included PDFs as a defense-in-depth layer
python <<'PY'
import os, sys
try:
    import fitz
except Exception:
    sys.exit(0)
for root, _, files in os.walk(os.environ['BUNDLE']):
    for f in files:
        if f.lower().endswith('.pdf'):
            p = os.path.join(root, f)
            try:
                d = fitz.open(p)
                d.set_metadata({k: '' for k in ('title','author','subject','keywords','creator','producer')})
                d.save(p, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
            except Exception as e:
                print(f'  [warn] {p}: {e}')
PY

echo "Bundle built at: $BUNDLE"
echo "Run: bash $REPO_ROOT/scripts/anonymity_audit.sh after pointing it at the bundle"
