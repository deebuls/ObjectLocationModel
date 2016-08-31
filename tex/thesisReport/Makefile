.PHONY: all deploy
.ONESHELL:
all:deploy
	pdflatex NairDS-MTR.tex
	bibtex NairDS-MTR.aux
	pdflatex NairDS-MTR.tex
	pdflatex NairDS-MTR.tex
	pdflatex NairDS-MTR.tex 
	$(MAKE) clean

clean:
	rm -f *.aux
	rm -f *.toc
	rm -f *.log
	rm -f *.lof
	rm -f *.lot
	rm -f *.out
	rm -f *.bbl
	rm -f *.blg
