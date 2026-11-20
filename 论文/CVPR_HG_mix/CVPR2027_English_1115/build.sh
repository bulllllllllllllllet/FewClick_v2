#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export TEXMFHOME="$PWD/texmf"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build_main.log 2>&1
bibtex main > build_bibtex.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build_main.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build_main.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build_main.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex > build_supplementary.log 2>&1
bibtex supplementary > build_supplementary_bibtex.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex >> build_supplementary.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex >> build_supplementary.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error supplementary.tex >> build_supplementary.log 2>&1
