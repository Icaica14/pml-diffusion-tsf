#!/usr/bin/env bash
# Build the Electricity slide deck from deck_electricity_it.md.
#
#   Outputs (next to the source, in docs/presentation/):
#     deck_electricity_it.pptx   PowerPoint — speaker notes in the notes pane, emoji render fine
#     deck_electricity_it.pdf    beamer/xelatex — self-contained, Unicode font, slides auto-shrink
#
#   Usage (from anywhere; the script cd's to the repo root itself):
#     bash docs/presentation/build_deck.sh
#
#   Deps: pandoc (>=3); for the PDF also a TeX with xelatex (MacTeX/TeX Live)
#         and the "Arial Unicode MS" font (ships with macOS). If you only need
#         the .pptx, the PDF step can be skipped — open the .pptx and export.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
export PATH="/Library/TeX/texbin:$PATH"   # MacTeX default location (harmless if already on PATH)

SRC="docs/presentation/deck_electricity_it.md"
OUT_PPTX="docs/presentation/deck_electricity_it.pptx"
OUT_PDF="docs/presentation/deck_electricity_it.pdf"

# --- PowerPoint -------------------------------------------------------------
# Emoji render fine in PowerPoint; ::: notes -> notes pane; images embedded.
pandoc "$SRC" -o "$OUT_PPTX"
echo "built $OUT_PPTX"

# --- PDF (beamer) -----------------------------------------------------------
# xelatex's text fonts have no color emoji, and the HTML build-comment would
# become a blank frame, so we transform a throwaway copy for the PDF only:
#   * drop the HTML comment block (else an empty leading slide)
#   * strip the two emoji (no text font covers them): ⭐ -> nothing, ⚠️ -> "Nota: "
#   * add [shrink] to every slide so dense frames auto-fit (no overflow)
# A Unicode-rich font (Arial Unicode MS) covers Greek / math / arrows / Greek-Extended.
if command -v xelatex >/dev/null 2>&1; then
  TMP="$(mktemp -t deck_pdf_XXXX).md"
  perl -0777 -CSD -pe '
    s/<!--.*?-->//s;                 # drop HTML comment (else blank frame)
    s/\x{2B50}//g;                   # star emoji -> (nothing)
    s/\x{26A0}\x{FE0F}? ?/Nota: /g;  # warning emoji -> "Nota: "
    s/^(# .*)$/$1 {.shrink}/mg;      # auto-shrink every slide to the frame
  ' "$SRC" > "$TMP"
  pandoc "$TMP" -t beamer --pdf-engine=xelatex \
    -V mainfont="Arial Unicode MS" -V sansfont="Arial Unicode MS" -V monofont="Arial Unicode MS" \
    -o "$OUT_PDF"
  rm -f "$TMP"
  echo "built $OUT_PDF"
else
  echo "xelatex not found — skipped $OUT_PDF (open the .pptx and export to PDF instead)" >&2
fi
