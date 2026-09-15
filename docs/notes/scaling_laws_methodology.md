# Metodika škálovacích zákonů pro malé jazykové modely (SLM)

## 1. Teoretický rámec (Kaplan et al., 2020 & Hoffmann et al., 2022)

Standardní škálovací zákon pro jazykové modely předpokládá mocninný vztah mezi testovací ztrátou $L$ a dvěma základními zdroji:
- Počet ne-embeddingových parametrů modelu: $N$
- Objem trénovacích dat (slov nebo tokenů): $D$

Parametrický tvar ztrátové funkce (Chinchilla / Kaplan):
$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Kde:
- $E$ je ireducibilní entropie jazyka (Bayesovský limit přirozeného jazyka toki pona).
- $\frac{A}{N^\alpha}$ reprezentuje penalizaci za nedostatečnou kapacitu modelu.
- $\frac{B}{D^\beta}$ reprezentuje penalizaci za omezený objem trénovacích dat.
- $\alpha, \beta$ jsou škálovací exponenty.

## 2. Význam normalizace na znaky (Bits-Per-Character – BPC)

Při srovnání různých tokenizérů (Char vs. BPE vs. Word) nelze použít standardní cross-entropy loss $\mathcal{L}_{\text{tok}}$, protože:
1. Znakový tokenizér generuje dlouhé sekvence malých jednotek.
2. Slovní tokenizér generuje krátké sekvence s větším slovníkem.

Pro spravedlivé informačně-teoretické srovnání přepočítáváme ztrátu na bity na znak původního textu:
$$\text{BPC} = \frac{\sum_{i} \mathcal{L}_{i}}{\sum \text{char\_len} \cdot \ln 2}$$
$$\text{PPL}_{\text{char}} = 2^{\text{BPC}}$$

Tímto převodem získáváme invariantní metriku vyjadřující skutečnou Shannonovu kompresi textu.
