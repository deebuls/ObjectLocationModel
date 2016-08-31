.PHONY: all deploy
.ONESHELL:
all:deploy
	latex NairDS-MTR.tex
	bibtex NairDS-MTR.aux
	latex NairDS-MTR.tex
	latex NairDS-MTR.tex
	pdflatex NairDS-MTR.tex 
	$(MAKE) clean

clean:
	rm -f *.aux
	rm -f *.toc
	rm -f *.log
	rm -f *.lof
	rm -f *.lot
	rm -f *.out
