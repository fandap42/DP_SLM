# Statistická analýza výsledků: SLM Toki Pona na malých datech

Tento report shrnuje empirické výsledky faktorového experimentu zkoumajícího vliv **velikosti korpusu ($D$)**, **velikosti modelu ($N$)** a **tokenizace ($T$)** na jazykové modelování minimalistického jazyka toki pona.

---

## 1. Popis experimentálního souboru
- **Celkový počet běhů:** 12
- **Faktory:**
  - Tokenizace: `char, bpe, word`
  - Modely: `micro, mini`
  - Frakce korpusu: `25%, 75%`
  - Semínka (Seeds): `42`

---

## 2. Průměrné výsledky dle konfigurace (BPC & Gramatika)

```
                                    val_bpc  grammar_acc
tokenizer model_size data_fraction                      
bpe       micro      0.25            1.6536       0.7561
                     0.75            1.4126       0.8537
          mini       0.25            1.4079       0.8293
                     0.75            1.2183       0.9268
char      micro      0.25            1.8495       0.3902
                     0.75            1.4178       0.5366
          mini       0.25            1.3867       0.5854
                     0.75            1.0728       0.8537
word      micro      0.25            1.2848       0.8537
                     0.75            1.1933       0.8780
          mini       0.25            1.1987       0.9024
                     0.75            1.1185       1.0000
```

---

## 3. Výsledky lineárního modelu / Mixed-Effects Regression
Model specifikace: `val_bpc ~ log_D + log_N + C(tokenizer) + log_D:C(tokenizer)`
Metoda odhadu: `OLS`

### Odhadnuté koeficienty a signifikance:
| Proměnná / Prediktor | Koeficient ($eta$) | p-hodnota | 95% Interval spolehlivosti |
| :--- | :--- | :--- | :--- |
| `Intercept` | 5.4515 ** | 9.8287e-03 | [1.9911, 8.9119] |
| `C(tokenizer)[T.char]` | 1.9191  | 3.4233e-01 | [-2.7848, 6.6229] |
| `C(tokenizer)[T.word]` | -1.7945  | 3.7180e-01 | [-6.4984, 2.9094] |
| `log_D` | -0.1960  | 9.9367e-02 | [-0.4454, 0.0534] |
| `log_D:C(tokenizer)[T.char]` | -0.1434  | 3.4393e-01 | [-0.4960, 0.2093] |
| `log_D:C(tokenizer)[T.word]` | 0.1178  | 4.2968e-01 | [-0.2349, 0.4705] |
| `log_N` | -0.1130 * | 1.2431e-02 | [-0.1891, -0.0369] |

*Poznámka: *** p < 0.001, ** p < 0.01, * p < 0.05*

---

## 4. Analýza rozptylu (ANOVA) a velikosti účinku ($\eta^2$)

```
                                 sum_sq   df           F    PR(>F)    eta_sq  eta_sq_percent
C(data_fraction)               0.151403  1.0  109.142192  0.001872  0.273120       27.312020
C(model_size)                  0.165370  1.0  119.210458  0.001645  0.298315       29.831528
C(tokenizer)                   0.139472  2.0   50.270778  0.004932  0.251598       25.159775
C(data_fraction):C(tokenizer)  0.041301  2.0   14.886484  0.027695  0.074505        7.450463
C(model_size):C(tokenizer)     0.052638  2.0   18.972566  0.019833  0.094955        9.495486
Residual                       0.004162  3.0         NaN       NaN  0.007507        0.750728
```

---

## 5. Odhad škálovacích zákonů (Scaling Laws)
- **Kaplan / Chinchilla rovnice:**
  `BPC = 0.555 + (77.05 / N^0.462) + (93.78 / D^0.390)`
- **Koeficient determinace ($R^2$):** `0.5714354860645129`

### Škálovací křivky dle tokenizéru ($BPC = a \cdot D^{-\beta}$):
- **BPE:** `BPC = 8.843 * D^(-0.138)` ($R^2 = 0.489$)
- **CHAR:** `BPC = 33.415 * D^(-0.238)` ($R^2 = 0.459$)
- **WORD:** `BPC = 2.853 * D^(-0.065)` ($R^2 = 0.532$)

---

## 6. Generované grafy
Grafy byly uloženy do složky `figures/`:
1. `figures/scaling_data_bpc.png` – Škálování podle velikosti korpusu
2. `figures/scaling_params_bpc.png` – Škálování podle počtu parametrů
3. `figures/interaction_effects.png` – Interakční graf Tokenizér × Data
4. `figures/grammar_vs_bpc.png` – Vztah mezi BPC a gramatickou přijatelností
