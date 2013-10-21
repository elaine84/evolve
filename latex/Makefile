main: main.pdf clean
onepage: onepage.pdf clean
main.pdf: main.tex all-refs.bib files/*.tex latex_macros.tex
	pdflatex main
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main
onepage.pdf:
	pdflatex onepage
	pdflatex onepage
	bibtex onepage
	pdflatex onepage
	pdflatex onepage
clean:
	rm -f main.aux main.bbl main.log main.blg main.out
flush: clean
	rm -f main.pdf


