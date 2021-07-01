#!/bin/sh


iconv -f UTF-8-MAC -t UTF-8 $1.md | pandoc -f markdown -o $2.pdf -V documentclass=ltjarticle --pdf-engine=lualatex
