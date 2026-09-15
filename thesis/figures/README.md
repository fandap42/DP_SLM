# Obrázky a schémata práce (Figures)

Tato složka slouží k uchovávání **vlastních schémat, diagramů a ilustrací** vytvořených přímo pro text diplomové práce (např. architektura Transformeru, schéma tokenizační pipeline, vývojový diagram experimentu).

## Odkazování na experimentální grafy

Grafy z trénování, vyhodnocení a škálovacích zákonů jsou generovány automaticky experimentálními skripty a ukládány do centrální složky projektu:
`results/figures/` (ve formátech PDF a PNG).

V LaTeXovém dokumentu není nutné tyto grafy duplikovat ani kopírovat do této složky. Soubor `makra.sty` definuje vyhledávací cesty:
```latex
\graphicspath{{./figures/}{../results/figures/}{./template/}}
```

V jednotlivých kapitolách proto stačí volat přímo název souboru bez relativní cesty:
```latex
\includegraphics[width=0.85\textwidth]{scaling_data_bpc.pdf}
```
LaTeX automaticky prohledá nejprve lokální `thesis/figures/` (pro vlastní schémata) a následně `results/figures/` (pro experimentální grafy).
