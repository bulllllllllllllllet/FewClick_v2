#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export TEXMFHOME="$PWD/texmf"
xelatex -interaction=nonstopmode -halt-on-error main_ch.tex > build_main_ch.log 2>&1
bibtex main_ch > build_main_ch_bibtex.log 2>&1
xelatex -interaction=nonstopmode -halt-on-error main_ch.tex >> build_main_ch.log 2>&1
xelatex -interaction=nonstopmode -halt-on-error main_ch.tex >> build_main_ch.log 2>&1
xelatex -interaction=nonstopmode -halt-on-error main_ch.tex >> build_main_ch.log 2>&1
