#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v pdflatex >/dev/null || { echo 'pdflatex is required.' >&2; exit 2; }
mkdir -p "$ROOT/results"
cd "$ROOT/paper"
for pass in 1 2 3; do
  if ! pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex \
       > "$ROOT/results/build-pass${pass}.log" 2>&1; then
    tail -80 "$ROOT/results/build-pass${pass}.log" >&2
    exit 1
  fi
done
cp main.log "$ROOT/results/latex.log"
if grep -E 'Overfull|Underfull|LaTeX Warning|Package .* Warning|undefined|multiply defined' main.log; then
  echo 'The final LaTeX pass did not meet the clean-build check.' >&2
  exit 1
fi
printf 'PASS: paper/main.pdf compiled without warnings or box errors.\n'
