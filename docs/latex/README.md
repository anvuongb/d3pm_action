# ICLR LaTeX Paper (ITW / D3PM)

ICLR 2027 format paper (double-blind; authors hidden unless `\iclrfinalcopy` is set) converted from [`../paper.md`](../paper.md). Title: *Active Data Acquisition with Side Information via Discrete Diffusion Priors*.

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
| `iclr2027_conference.sty` | ICLR 2027 style (vendored) |
| `iclr2027_conference.bst` | ICLR 2027 BibTeX style (vendored) |
| `natbib.sty`, `fancyhdr.sty` | Required by the ICLR style (vendored) |
| `aaai2026.sty`, `aaai2026.bst` | Unused; kept from the earlier AAAI draft |
| `figures/pipeline.tex` | TikZ training pipeline diagram |

## Style file provenance

`iclr2027_conference.sty`, `iclr2027_conference.bst`, `natbib.sty` and `fancyhdr.sty` come from the **ICLR 2027 style files** distributed by the conference. Limits: **9 pages** of main text for the initial submission (10 for rebuttal/camera-ready), unlimited pages for citations, appendix after the references. The AI use statement is **required** and does not count toward the limit.

## Anonymity

Submissions must be anonymous: leave `\iclrfinalcopy` commented out and the style prints "Anonymous authors / Paper under double-blind review". The real author block in `main.tex` is only used once `\iclrfinalcopy` is uncommented for camera-ready.

## Notes

- Avoid `geometry`; the style sets the page dimensions.
