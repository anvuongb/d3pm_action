.PHONY: all clean figures

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

all: figures $(MAIN).pdf

figures: figures/pipeline.pdf

figures/pipeline.pdf: figures/pipeline.tex
	cd figures && $(LATEX) -interaction=nonstopmode pipeline.tex
	cd figures && $(LATEX) -interaction=nonstopmode pipeline.tex

$(MAIN).pdf: $(MAIN).tex references.bib aaai2026.sty aaai2026.bst figures/pipeline.pdf
	$(LATEX) -interaction=nonstopmode $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN)
	$(LATEX) -interaction=nonstopmode $(MAIN)

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f figures/pipeline.pdf figures/pipeline.aux figures/pipeline.log
