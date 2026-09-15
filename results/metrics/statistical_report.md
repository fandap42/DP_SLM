# Statistická analýza výsledků: SLM Toki Pona na malých datech

- **Celkový počet běhů:** 12
- **Tokenizace:** `char, bpe, word`
- **Modely:** `micro, mini`
- **Frakce korpusu:** `25%, 75%`

---

## 1. Průměrné výsledky dle konfigurace (BPC & Gramatika)

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

## 2. Analýza rozptylu (ANOVA) a velikosti účinku ($\eta^2$)

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

## 3. Odhad škálovacích zákonů (Scaling Laws)
- **Kaplan / Chinchilla rovnice:**
  `BPC = 0.555 + (77.05 / N^0.462) + (93.78 / D^0.390)`
- **Koeficient determinace ($R^2$):** `0.5714354860645129`
