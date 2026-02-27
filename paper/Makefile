LATEX = /Library/TeX/texbin/pdflatex
BIBTEX = /Library/TeX/texbin/bibtex
PAPER = paper_backbone

.PHONY: pdf clean

pdf: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER).tex
	$(LATEX) -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f $(PAPER).{aux,bbl,blg,log,out,toc,pdf}
