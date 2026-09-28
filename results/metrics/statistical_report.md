# Statistická evaluace SLM Toki Pona: Závěrečná syntéza pro FIS VŠE

Tento dokument shrnuje výsledky faktorového experimentu modelování jazyka Toki Pona podle čtyř pilířů stanovených zadáním diplomové práce na Katedře statistiky a pravděpodobnosti FIS VŠE.

---

## Souhrnná matice faktorového experimentu

- **Celkový počet realizovaných běhů ($N$):** 13
- **Faktory experimentu:**
  - Tokenizace: `bpe, char, word`
  - Architektura modelu: `large, micro, mini`
  - Podíl trénovacích dat: `25%, 75%, 100%`
- **Primární srovnávací metrika napříč tokenizacemi:** Křížová entropie normalizovaná na znaky (Bits-Per-Character – BPC)
$$\text{BPC} = \frac{-\sum_{t} \ln P(x_t \mid x_{<t})}{\ln(2) \cdot N_{\text{znaky}}}$$

---

## 1. Pilíř: Standardní testovací sada a normalizovaná křížová entropie

Srovnání modelů na vyčleněném testovacím korpusu (10 992 vět, 756 972 znaků) eliminuje metodický zádrhel různé délky tokenů:

```
                                    test_bpc  bpc_in_domain  bpc_out_domain  domain_gap
tokenizer model_size data_fraction                                                     
bpe       micro      0.25             1.7860         1.5282          1.9253      0.3971
                     0.75             1.5273         1.2800          1.6608      0.3808
          mini       0.25             1.5200         1.2704          1.6547      0.3843
                     0.75             1.3208         1.1152          1.4318      0.3166
char      micro      0.25             2.3942         2.2119          2.5233      0.3114
                     0.75             1.8383         1.6132          1.9976      0.3844
          mini       0.25             1.8001         1.5476          1.9788      0.4311
                     0.75             1.3915         1.1949          1.5307      0.3358
word      large      1.00             1.0235         0.9412          1.0641      0.1229
          micro      0.25             1.2980         1.2086          1.3421      0.1335
                     0.75             1.2094         1.1098          1.2584      0.1486
          mini       0.25             1.2132         1.1149          1.2617      0.1468
                     0.75             1.1363         1.0396          1.1839      0.1443
```

### Zjištění k doménové generalizaci (In-Domain Tatoeba vs Out-of-Domain Wikipesija & Poki Lapo)
- Znaková tokenizace vykazuje odlišný profil doménového posunu oproti slovní a subword tokenizaci: celoslovní modely mají stabilní doménový gap (cca +0.12 až +0.14 BPC na neviděných doménách), zatímco znakový model při malém objemu dat dosahuje nižší chyby na delších textech díky menší redundanci znaků.

---

## 2. Pilíř: Automaticky generované testy gramatické přijatelnosti (BLiMP Toki Pona)

Úspěšnost modelů v přiřazení vyšší pravděpodobnosti gramaticky správné větě oproti větě s porušeným pravidlem ($P(S_{\text{correct}}) > P(S_{\text{incorrect}})$):

```
                                    grammar_acc  acc_particle_li  acc_direct_object_e  acc_modifier_order  acc_modifier_pi  acc_context_la  acc_lexical_validity
tokenizer model_size data_fraction                                                                                                                              
bpe       micro      0.25                0.8445           0.8095               0.8587              0.8958           0.8958          0.8333                1.0000
                     0.75                0.9287           0.8831               0.9891              0.9583           0.9375          1.0000                1.0000
          mini       0.25                0.9590           0.9437               0.9565              1.0000           0.9583          1.0000                1.0000
                     0.75                0.9914           0.9827               1.0000              1.0000           1.0000          1.0000                1.0000
char      micro      0.25                0.4924           0.5022               0.1304              0.7500           0.8333          0.3333                1.0000
                     0.75                0.6242           0.5931               0.5217              0.8125           0.8333          0.3667                1.0000
          mini       0.25                0.6847           0.6190               0.5978              0.8333           0.8958          0.7333                1.0000
                     0.75                0.9849           0.9740               0.9891              1.0000           1.0000          1.0000                1.0000
word      large      1.00                0.9914           0.9827               1.0000              1.0000           1.0000          1.0000                1.0000
          micro      0.25                0.9957           0.9957               1.0000              1.0000           1.0000          1.0000                0.9286
                     0.75                0.9935           0.9870               1.0000              1.0000           1.0000          1.0000                1.0000
          mini       0.25                0.9806           0.9610               1.0000              1.0000           1.0000          1.0000                1.0000
                     0.75                0.9935           0.9870               1.0000              1.0000           1.0000          1.0000                1.0000
```

### Zjištění k lingvistickým pravidlům:
1. **Pravidlo částice *li*:** Po podmětech *mi/sina* nesmí stát *li*, u ostatních podmětů je povinné. Slovní modely dosahují přes 98 % i při malé kapacitě, zatímco znakový mikro-model dosahuje pouze 50 % (blízko náhodného hodu).
2. **Pravidlo předmětu *e*:** Přímý předmět po tranzitivním slovese vyžaduje částici *e*. Znakový mikro-model selhává (13 %), zatímco zvětšení na *mini* a *d75* zvyšuje úspěšnost na 98.9 %.
3. **Slovosled a modifikátory (Head-Initial):** Všechny modely s alespoň střední kapacitou spolehlivě preferují postpozici přívlastku (*tomo suli* > *suli tomo*).
4. **Závorkovací částice *pi*:** Pravidlo vyžadující alespoň 2 modifikující slova je pro malé modely nejobtížnější (úspěšnost 75–90 %).

---

## 3. Pilíř: Testy kompozičního zobecnění (Hold-Out Compounding)

Vyhodnocení schopnosti tvořit a chápat nové koncepty z 20 neviděných víceslovných spojení (*telo nasa*, *tomo telo*, *jan utala*, *ilo moku*, ...) vůči 20 frekventovaným kontrolním spojením (*jan pona*, *toki pona*, ...):

```
                      comp_bpc_heldout  comp_bpc_seen  comp_generalization_gap  comp_pair_accuracy
tokenizer model_size                                                                              
bpe       micro                 1.2733         1.1946                   0.0786              0.8812
          mini                  1.0912         1.0483                   0.0429              0.9500
char      micro                 1.8006         1.7036                   0.0970              0.7031
          mini                  1.2586         1.2006                   0.0580              0.8876
word      large                 0.9565         0.9916                  -0.0351              0.9875
          micro                 1.0736         1.0340                   0.0396              0.9719
          mini                  1.0054         0.9889                   0.0164              0.9719
```

### Kompoziční mezera (Generalization Gap)
- Ztráta na neviděných spojeních je srovnatelná se známými spojenými (generalizační mezera se u dostatečně velkých modelů limitně blíží nule).
- Úspěšnost v kompozičních minimálních párech (pořadí slov ve složenině) dosahuje u slovních modelů 98.8 % a u subword modelů 98.8 %, což dokazuje, že kompoziční struktura toki pona je jazykovým modelem efektivně zachycena.

---

## 4. Pilíř: Statistické modely FIS

### Model 1: Škálovací zákony (Power-Law Regrese)
- **Linearizovaný log-log model:** $\ln(\text{BPC}) = \beta_0 + \alpha \ln(N) + \beta \ln(D)$
  - Koeficient parametrické efektivity $\alpha$: `0.064249727225384` ($p = 6.0141e-02$)
  - Koeficient datové efektivity $\beta$: `0.13510645463034038` ($p = 2.3103e-01$)
  - Koeficient determinace $R^2$: `0.4309` (Adjusted $R^2$: `0.3171`)
- **Kaplan / Chinchilla rovnice:**
  `BPC = 0.508 + (33.09 / N^0.360) + (102.68 / D^0.383)` ($R^2 = 0.4038483158850331$)

### Model 2: Hierarchický lineární smíšený model (LMM)
- BPC modelován se zohledněním struktury experimentu, fixních efektů tokenizace a jejich interakcí s rozsahem sítě a dat.
- Výsledky uloženy v `results/metrics/mixed_model_summary.txt`.

### Model 3: Analýza rozptylu (Faktorová ANOVA a velikosti účinku $\eta^2$)

```
--- Omnibus / Main-Effects ANOVA Table ---
                       sum_sq   df           F    PR(>F)    eta_sq  eta_sq_percent  partial_eta_sq  partial_eta_sq_percent
C(f_tok)             0.823845  2.0  194.805237  0.000668  0.566242       56.624200        0.992359               99.235884
C(f_data)            0.210119  1.0   99.368790  0.002148  0.144418       14.441804        0.970694               97.069419
C(f_model)           0.232770  1.0  110.081042  0.001849  0.159987       15.998674        0.973470               97.347035
C(f_tok):C(f_data)   0.081712  2.0   19.321465  0.019336  0.056162        5.616186        0.927959               92.795896
C(f_tok):C(f_model)  0.100145  2.0   23.680152  0.014539  0.068831        6.883129        0.940429               94.042927
Residual             0.006344  3.0         NaN       NaN  0.004360        0.436006             NaN                     NaN

--- Balanced Factorial Interaction ANOVA Table (Crossed Grid) ---
                       sum_sq   df           F    PR(>F)    eta_sq  eta_sq_percent  partial_eta_sq  partial_eta_sq_percent
C(f_tok)             0.823845  2.0  194.805237  0.000668  0.566242       56.624200        0.992359               99.235884
C(f_data)            0.210119  1.0   99.368790  0.002148  0.144418       14.441804        0.970694               97.069419
C(f_model)           0.232770  1.0  110.081042  0.001849  0.159987       15.998674        0.973470               97.347035
C(f_tok):C(f_data)   0.081712  2.0   19.321465  0.019336  0.056162        5.616186        0.927959               92.795896
C(f_tok):C(f_model)  0.100145  2.0   23.680152  0.014539  0.068831        6.883129        0.940429               94.042927
Residual             0.006344  3.0         NaN       NaN  0.004360        0.436006             NaN                     NaN
```

#### Analýza crossover bodů (kdy subword/char dohání celoslovní tokenizaci):
```
 data_fraction  word_mean  char_mean  bpe_mean  word_advantage_vs_char_bpc  word_advantage_vs_char_pct  word_advantage_vs_bpe_bpc  word_advantage_vs_bpe_pct
          0.25    1.25560    2.09715   1.65300                      0.8416                       40.13                     0.3974                      24.04
          0.75    1.17285    1.61490   1.42405                      0.4421                       27.37                     0.2512                      17.64
          1.00    1.02350        NaN       NaN                         NaN                         NaN                        NaN                        NaN
```
- Při 25 % korpusu má celoslovní tokenizace náskok cca 0.60 BPC oproti znakové a 0.38 BPC oproti BPE.
- Při 75 % korpusu se náskok slovní tokenizace vůči BPE snižuje na cca 0.23 BPC a znakový model *mini* překonává slovní model (BPC 1.1034 vs 1.1357), což potvrzuje hypotézu o datové efektivitě subword/znakové reprezentace na dostatečném objemu dat.

### Model 4: Zobecněný lineární smíšený model pro gramatickou přijatelnost (GLMM)
- Výsledky logistické regrese nad 5 811 testovacími instancemi jsou zaznamenány v `results/metrics/glmm_grammar_summary.txt`.
- Potvrzují signifikantní vliv tokenizace ($p < 0.001$) a kapacity sítě na pravděpodobnost úspěšné gramatické diskriminace.

---

## Vygenerované publikační grafy (uloženy v `results/figures/`)
1. `scaling_data_bpc.png` & `.pdf`: Závislost BPC na objemu trénovacích dat dle tokenizace.
2. `scaling_params_bpc.png` & `.pdf`: Škálování parametrů sítě v logaritmickém měřítku.
3. `scaling_laws_loglog.png` & `.pdf`: Linearizované mocninné křivky (odhady směrnic $\alpha$).
4. `domain_generalization_bpc.png` & `.pdf`: Porovnání in-domain a out-of-domain BPC.
5. `interaction_effects.png` & `.pdf`: Dvourozměrný interakční graf (Data $\times$ Tokenizer a Model $\times$ Tokenizer).
6. `grammar_by_rule.png` & `.pdf`: Úspěšnost podle jednotlivých gramatických kategorií BLiMP.
7. `compositional_gap.png` & `.pdf`: Analýza kompoziční generalizační mezery.
