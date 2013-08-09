main: main.pdf clean
main.pdf: main.tex all-refs.bib files/*.tex
	pdflatex main
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main
clean:
	rm -f main.aux main.bbl main.log main.blg main.out
flush: clean
	rm -f main.pdf


