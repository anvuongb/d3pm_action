# AAAI LaTeX Paper (ITW / D3PM)

AAAI-26 format paper (named authors) converted from [`../paper.md`](../paper.md). Title: *Active Data Acquisition with Side Information via Discrete Diffusion Priors*.

## Prerequisites

- `pdflatex`
- `bibtex`
- `make`

## Build

```bash
cd docs/latex
make
```

Output: `main.pdf`

```bash
make clean   # remove auxiliary files and PDFs
```

## Files

| File | Role |
|------|------|
| `main.tex` | Full paper source |
| `references.bib` | Bibliography |
| `aaai2026.sty` | AAAI-26 style (vendored) |
| `aaai2026.bst` | AAAI-26 BibTeX style (vendored) |
| `figures/pipeline.tex` | TikZ training pipeline diagram |

## Style file provenance

`aaai2026.sty` and `aaai2026.bst` are from the **AAAI-26 Author Kit** (TemplateVersion 2026.1). Official download:

- [AAAI-26 Submission Instructions](https://aaai.org/conference/aaai/aaai-26/submission-instructions/) (Author Kit link)

Vendored copies match the public AAAI 2026 template distribution. If formatting requirements change, replace these files from the latest official kit.

## Authorship

The paper uses named authors (Oregon State University and East Tennessee State University) with `\usepackage{aaai2026}` (no `submission` option). For anonymous review, add the `submission` option and replace `\author{}` / `\affiliations{}` with the anonymous placeholder per the Author Kit.

## Notes

- Do **not** add `\bibliographystyle{aaai2026}`; `aaai2026.sty` sets it automatically.
- Avoid forbidden packages (`hyperref`, `geometry`, etc.) per AAAI guidelines.
