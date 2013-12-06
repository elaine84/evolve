main: main.pdf clean
onepage: onepage.pdf clean
main.pdf: main.tex all-refs.bib files/*.tex latex_macros.tex
	pdflatex main
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main
onepage.pdf: onepage.tex all-refs.bib latex_macros.tex
	pdflatex onepage
	pdflatex onepage
	bibtex onepage
	pdflatex onepage
	pdflatex onepage
clean:
	rm -f main.aux main.bbl main.log main.blg main.out
	rm -f onepage.aux onepage.bbl onepage.log onepage.blg onepage.out
flush: clean
	rm -f main.pdf
	rm -f onepage.pdf


