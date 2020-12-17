#!/bin/bash
pdflatex project-proposal -synctex=1 -interaction=nonstopmode %.tex #-aux-directory=metadata 
biber project-proposal % #-output-directory=metadata -input-directory=metadata %

pdflatex project-proposal -synctex=1 -interaction=nonstopmode %.tex #-aux-directory=metadata 

