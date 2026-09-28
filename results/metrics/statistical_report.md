# Statistická analýza výsledků: SLM Toki Pona na malých datech

- **Celkový počet běhů:** 13
- **Tokenizace:** `char, bpe, word`
- **Modely:** `micro, mini, large`
- **Frakce korpusu:** `25%, 75%, 100%`

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
word      large      1.00            1.0075       0.9512
          micro      0.25            1.2848       0.8537
                     0.75            1.1933       0.8780
          mini       0.25            1.1987       0.9024
                     0.75            1.1185       1.0000
```

---

## 2. Analýza rozptylu (ANOVA) a velikosti účinku ($\eta^2$)

```
                                     sum_sq   df             F    PR(>F)        eta_sq  eta_sq_percent
C(data_fraction)               1.356138e-01  2.0  4.888014e+01  0.005137  2.222893e-02    2.222893e+00
C(model_size)                  5.688435e+00  2.0  2.050319e+03  0.000020  9.324111e-01    9.324111e+01
C(tokenizer)                  -9.683266e-14  2.0 -3.490201e-11  1.000000 -1.587218e-14   -1.587218e-12
C(data_fraction):C(tokenizer)  7.573419e-02  4.0  1.364868e+01  0.029652  1.241385e-02    1.241385e+00
C(model_size):C(tokenizer)     1.968353e-01  4.0  3.547329e+01  0.007356  3.226395e-02    3.226395e+00
Residual                       4.161622e-03  3.0           NaN       NaN  6.821460e-04    6.821460e-02
```

---

## 3. Odhad škálovacích zákonů (Scaling Laws)
- **Kaplan / Chinchilla rovnice:**
  `BPC = 0.644 + (785.00 / N^0.682) + (108.91 / D^0.404)`
- **Koeficient determinace ($R^2$):** `0.6418825923895988`
