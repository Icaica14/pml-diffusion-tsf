---
title: "Forecasting probabilistico con modelli di diffusione"
subtitle: "Un confronto controllato su consumi elettrici · M0 → M4"
author: "Giovanni Mason (SM3800158) · Lorenzo Di Bernardo (SM3800132) · Lorenzo Karol Gobbo (SM28A00018) — PML, Prof. Bortolussi, UniTS"
date: "Giugno 2026"
lang: it
---

<!--
  DECK GENERABILE — non scrivere qui i numeri a mano.
  Sorgente narrativa: SLIDE_TEMPLATE_ELECTRICITY_IT.md (scaletta + speaker notes).
  Numeri: results/tables/comparison_electricity.md (da experiments/make_tables.py).
  Build (dalla root del repo):  bash docs/presentation/build_deck.sh
    -> deck_electricity_it.pptx  (PowerPoint/Keynote/Slides; emoji ok)
    -> deck_electricity_it.pdf   (beamer/xelatex; font Unicode, emoji rimosse, slide auto-shrink)
  Le note del relatore stanno nei blocchi ::: notes (pannello note di PowerPoint/Keynote).
  Figure embedded: solo quelle realmente esistenti (fig_cmp_*). Le 🟡 "da generare" sono omesse.
-->

# La domanda

- **Forecasting probabilistico:** prevediamo una *distribuzione* del futuro, non un singolo numero.
- Una scala di 5 modelli a complessità crescente:
  **M0** seasonal-naive → **M1** ARIMA → **M2** DeepAR → **M3** TimeGrad → **M4** TimeDiff.
- **Domanda guida:** un modello di diffusione produce *incertezza migliore* — e a che costo?

::: notes
Buongiorno. Studiamo il forecasting probabilistico: non vogliamo una sola previsione, ma una distribuzione del futuro. Confrontiamo cinque modelli su una scala di complessità crescente, dal baseline stagionale ai modelli di diffusione. La domanda non è "chi vince", ma se la diffusione *giustifica* il suo costo con un'incertezza migliore. (A, ~30s)
:::

# Perché forecasting *probabilistico*

- Prevediamo `p(futuro | passato)` = `p(x_{t+1:t+τ} | x_{t-H+1:t})`.
- Una previsione *puntuale* nasconde il rischio; una *distribuzione* lo quantifica.
- Servono **intervalli calibrati**: se dico "90%", devono coprire davvero ~90%.
- Le decisioni sotto incertezza (riserva di rete, sbilanciamento, prezzo) usano i *quantili*, non solo la media.
- `H` = contesto (passato osservato) · `τ` = orizzonte (futuro previsto).

::: notes
Forecasting probabilistico significa stimare la distribuzione del futuro condizionata al passato. La media da sola non basta: per gestire una rete elettrica mi serve sapere *quanto* posso sbagliare. Per questo valuteremo non solo l'errore puntuale, ma anche quanto sono calibrate le bande: se dico novanta percento, dentro la banda deve cadere circa il novanta percento dei casi. (A, ~45s)
:::

# Dataset e protocollo

- **Electricity**: D = 321 serie orarie di consumo, forte stagionalità giornaliera e settimanale.
- Split **70/10/20** (train/val/**test**) · contesto **H = 168** ore (1 settimana) · orizzonte **τ = 24** ore.
- **S = 100** campioni per la distribuzione predittiva · **seed = 42**.
- **Caveat di onestà (E0):** file grezzo LSTNet col *nostro* split, **non** `electricity_nips` →
  i nostri CRPS **non** sono direttamente confrontabili coi numeri pubblicati. È un confronto *interno*, controllato.

::: notes
Lavoriamo su Electricity: 321 contatori orari, con stagionalità giornaliera e settimanale molto marcata. Usiamo una settimana di storia per prevedere il giorno dopo, su uno split test tenuto da parte, con cento campioni per ricostruire la distribuzione. Un punto di onestà: usiamo il nostro split, non quello del benchmark pubblicato, quindi confrontiamo i modelli *tra loro*, non con la letteratura. (A, ~45s)
:::

# Da Exchange a Electricity — perché due dataset

- **Exchange** (D = 8, giornaliero, quasi *random walk*): sandbox per validare la pipeline.
  Qui i modelli deep **non** battono la persistenza — ed è *atteso*: **controllo negativo**.
- **Electricity** (D = 321, orario, stagionale): il test rilevante, struttura ricca, rilevanza industriale.
- Messaggio: *cambia il regime del segnale, cambia chi vince.*
- ⚠️ MASE e CRPS **non** sono confrontabili tra i due dataset (scale e denominatori diversi).

::: notes
Due dataset con ruoli diversi. Exchange, i tassi di cambio, è quasi una passeggiata aleatoria: lì la miglior previsione di domani è oggi, e nessun modello sofisticato batte questa regola — controllo negativo per validare la pipeline. Electricity è il test vero, con stagionalità forte. La morale: il vincitore dipende dal tipo di segnale. (A → B, ~45s)
:::

# La scala dei modelli: M0 → M4

- **M0 — seasonal-naive:** "domani come il ciclo precedente". Zero training, fortissimo sulle serie stagionali.
- **M1 — ARIMA:** lineare classico, un modello *per canale* (321 modelli).
- **M2 — DeepAR:** RNN autoregressiva probabilistica, addestrata su tutte le serie insieme.
- **M3 — TimeGrad:** diffusione **autoregressiva** condizionata — genera il futuro un passo alla volta.
- **M4 — TimeDiff:** diffusione **non-autoregressiva** — denoisa l'**intero blocco** futuro in una catena →
  sampling *molto* più rapido. Point forecast nitido, *ma* incertezza collassata.
- Asse: più espressività → (forse) incertezza migliore → più costo.
  **E M4 mostra che il *design* (AR vs non-AR) sposta il costo, non solo la scala.**

::: notes
Una scala di cinque modelli. M0, il seasonal-naive: nessun addestramento, ma su segnali stagionali sorprendentemente forte. M1, ARIMA, un modello per ciascuna delle 321 serie. M2, DeepAR, rete ricorrente probabilistica su tutte le serie insieme. M3, TimeGrad, primo modello di diffusione, autoregressivo: genera il futuro un passo alla volta. M4, TimeDiff, secondo, design opposto: non autoregressivo, ripulisce l'intero blocco futuro in una catena — molto più veloce. M4 dà il forecast puntuale più nitido al costo più basso, ma paga con un'incertezza degenere: il *design*, non solo la taglia, cambia il compromesso. (B, ~70s)
:::

# TimeGrad, in modo intuitivo

- **Forward:** parto dal dato vero e aggiungo rumore gaussiano, un po' alla volta, finché diventa rumore puro.
- **Reverse:** addestro una rete a **invertire** quei passi; a generazione, parto da rumore e lo ripulisco.
- **Per il forecasting:** la ripulitura è **condizionata sul passato** (stato di un RNN che riassume H ore);
  ripeto il sampling S volte → S traiettorie future = la distribuzione predittiva.
- **Il prezzo:** ogni campione richiede molti passi di denoising → sampling lento
  (`diff_steps = 100`, predict ≈ 4h 19min su L4).

::: notes
TimeGrad è un modello di diffusione. Prendo il dato vero, aggiungo rumore gradualmente finché non resta che rumore. Poi addestro una rete a fare il percorso inverso, a togliere rumore un passo alla volta. Per generare parto da puro rumore e ripulisco fino a un campione realistico. Per il forecasting condiziono sul passato — riassunto da una RNN — e ripeto cento volte: cento futuri possibili, cioè la distribuzione. Il prezzo: ogni campione richiede tanti passi, il sampling è lento, ed è il costo che quantificheremo. (B, ~60s)
:::

# Come misuriamo

- **Accuratezza puntuale:** MAE, RMSE, **MASE** (1 = come il seasonal-naive; < 1 = meglio del baseline).
- **Qualità probabilistica:** **CRPS** e pinball loss — premiano una distribuzione *intera* ben piazzata. Più basso = meglio.
- **Calibrazione:** **cov50 → 0.50** e **cov90 → 0.90** (le bande coprono quel che promettono?); width = sharpness.
- **Costo:** `fit_s` (training) e `predict_s` (sampling).
- *Un modello utile deve avere MASE < 1 e CRPS basso a un costo accettabile.*

::: notes
Tre lenti per leggere i numeri. Primo, l'accuratezza puntuale con MASE, normalizzato perché uno equivalga al seasonal-naive — sotto uno significa "meglio del baseline". Secondo, la qualità probabilistica: il CRPS, che premia una distribuzione intera ben piazzata, non solo la media. Terzo, la calibrazione: se le bande al cinquanta e novanta percento coprono davvero quella frazione. E sopra tutto, il costo. (B, ~45s)
:::

# Risultati su Electricity

- **Barra da battere: CRPS = 160.5 (M0)** — nessun deep la supera (DeepAR 253.7; TimeGrad 241.6, il più vicino; TimeDiff 287.3).
- **M4 — doppio volto:** MAE più bassa dei deep al costo minimo (~41 min vs ~4h19 di M3), *ma* cov 0.003 / 0.008 → distribuzione degenere (CRPS ≈ MAE; dettaglio in B7).

| Modello | MASE | CRPS | cov50 | cov90 |
|---|---|---|---|---|
| M0 seasonal-naive | **1.00** | **160.5** | 0.497 | 0.885 |
| M1 ARIMA | 3.66 | 867.5 | 0.592 | 0.930 |
| M2 DeepAR | 1.83 | 253.7 | 0.466 | 0.831 |
| M3 TimeGrad | 1.39 | 241.6 | 0.279 | 0.647 |
| M4 TimeDiff | 1.40 | 287.3 | **0.003** | **0.008** |

::: notes
I numeri su Electricity. Il seasonal-naive ha CRPS 160 e MASE uno — la barra da battere. ARIMA, un modello per canale su 321 serie, è fragile e va malissimo. DeepAR è 253, batte ARIMA ma non il baseline. TimeGrad scende a 241 — il più vicino al baseline, ma non lo supera, a un costo di sampling enorme. TimeDiff: come previsione puntuale è il migliore dei deep e costa pochissimo, ma la sua incertezza è collassata, le bande coprono lo zero per cento. Un segnale da manuale, in dettaglio nei backup. (C, ~70s)
:::

# Risultati — CRPS a confronto

- Barre = CRPS per modello (↓ meglio). La soglia di **M0 = 160.5** è la riga che nessun deep supera: TimeGrad le si avvicina, TimeDiff (incertezza collassata) è il più alto.

![](figures/presentation/fig_cmp_crps.png){width=56%}

::: notes
La stessa storia della tabella, ma in figura, più leggibile dal fondo dell'aula: il CRPS per modello. Il seasonal-naive fissa la barra; TimeGrad è il più vicino ma resta sopra; TimeDiff è il più alto perché la distribuzione è degenere. (C, ~15s)
:::

# Qualità vs costo — il vero trade-off

- Scatter **CRPS (qualità, ↓) vs predict_s (costo, ↓)**; angolo basso-sinistra = ideale.
- **M0** è economico *e* buono (sorpresa). **M2** paga di più per una qualità peggiore di M0.
- **M3** è il più **costoso** (~4h19) e arriva solo a 241.6: il costo non è ripagato.
- **M4** è il punto più a **sinistra** (il più economico dei deep) ma CRPS 287, il più alto: il non-AR **compra velocità, non calibrazione**.
- *Non basta vincere sul CRPS: bisogna vincere abbastanza da giustificare il costo — e la velocità non vale se l'incertezza è rotta.*

![](figures/presentation/fig_cmp_quality_cost.png){width=52%}

::: notes
Il grafico che riassume la tesi. Orizzontale il costo di campionamento, verticale la qualità probabilistica: l'ideale è l'angolo in basso a sinistra. Il seasonal-naive sta proprio lì. DeepAR costa di più e rende meno del baseline. TimeGrad è costoso — oltre quattro ore — e arriva solo a 241. TimeDiff ribalta l'asse del costo: il più economico dei deep grazie al design non-autoregressivo, ma il CRPS più alto, perché l'incertezza è collassata. La velocità da sola non basta se quello che guadagni in costo lo perdi in calibrazione. (C, ~70s)
:::

# Conclusioni e limiti

- **Take-home:** non "diffusion vince", ma il **trade-off espressività ↔ calibrazione ↔ costo**.
  Su un segnale molto stagionale un baseline semplice è una soglia seria.
- **M3:** migliora DeepAR (253.7 → 241.6) ma non il seasonal-naive (160.5). Valore **metodologico**, non una vittoria.
- **M4 — il design conta:** ~6× più rapido di M3, MAE più bassa tra i deep, ma in x0-prediction la distribuzione **collassa** (cov ≈ 0).
  L'ablazione **ε (M4ε)** lo conferma *per assurdo*: predire il rumore **non** ricalibra, **ribalta** nell'estremo opposto (cov ≈ 1, bande enormi, MASE 4.0).
  → *x0 ed ε sono i due estremi; la calibrazione vera non è uno switch di parametrizzazione.*
- **Limiti:** split non-benchmark (E0); ARIMA per-canale; sampling costoso (M3); un solo seed; no tuning esteso.
- **Lavori futuri:** calibrare oltre x0/ε (σ_θ appresa, conformal); split `electricity_nips`; CSDI; DDIM; più seed.

::: notes
In sintesi: il messaggio non è che la diffusione vince, ma che esiste un compromesso tra espressività, calibrazione e costo — e che su un segnale fortemente stagionale un baseline semplice è un avversario serio. TimeGrad batte DeepAR sul CRPS ma non il seasonal-naive: valore metodologico. Abbiamo confrontato due diffusion model con design opposto: TimeDiff, non-autoregressivo, ribalta il costo — sei volte più rapido — ma in questa parametrizzazione rompe la calibrazione. La nostra ablazione mostra che nessuno dei due estremi è calibrato: x0 azzera l'incertezza, ε la gonfia. Onesti sui limiti: split non-benchmark, ARIMA poco competitivo, sampling costoso, un solo seed. Grazie. (C, ~55s)
:::

# Backup B1a — Copertura empirica (calibrazione)

- Copertura osservata vs nominale: **cov50 → 0.50**, **cov90 → 0.90**. Sotto la diagonale = troppo sicuro; sopra = troppo incerto.
- **M4 (x0)** schiacciato sul pavimento (cov ≈ 0); **M4ε** sul soffitto (cov ≈ 1); **DeepAR** il più vicino alla diagonale.
- *Uso:* "come fai a dire che un modello è ben calibrato?"

![](figures/presentation/fig_cmp_calibration.png){width=58%}

::: notes
Backup per il Q&A sulla calibrazione: la copertura empirica contro quella nominale. Sotto la diagonale è sovra-confidenza, sopra è sotto-confidenza. Si vedono i due estremi di TimeDiff — x0 sul pavimento, ε sul soffitto — e DeepAR come il più vicino alla diagonale. (C)
:::

# Backup B1b — Sharpness (ampiezza delle bande)

- **width50/width90** = quanto sono larghe le bande. Va letta *insieme* alla copertura: stretta + scopertura = falsa sicurezza; larga + copertura = incertezza onesta.
- **M4 (x0)** bande ≈ 0 (false-sicuro); **M4ε** bande enormi (~7 700 / 11 400, le più larghe della scala).
- *Uso:* "una banda stretta è sempre meglio?" → no, conta solo accanto alla copertura.

![](figures/presentation/fig_cmp_intervals.png){width=58%}

::: notes
Sharpness: l'ampiezza delle bande, da leggere sempre accanto alla copertura. Una banda strettissima che non copre nulla è una falsa sicurezza, non un pregio — è esattamente il caso di M4 in x0. (C)
:::

# Backup B7 — I due volti della parametrizzazione (x0 vs ε) ⭐

- **Non-AR:** denoisa l'intero blocco τ×D in una catena → da qui la velocità.
- **x0 collassa:** cov 0.003/0.008, width ≈ 0, **CRPS ≈ MAE** = massa puntiforme.
  *Perché:* la rete predice un x0 quasi costante (contesto + `x_ar`), ignora il rumore `x_t`; varianza residua `1−ᾱ_0 ≈ 6·10⁻⁴` → std ≈ 0 (future-mixup aggrava).
- **La presunta cura, ε, ribalta:** M4ε **sovra-disperde** — cov50 **0.998**, cov90 **1.000**, width50 **7 686**, width90 **11 359** (le più larghe della scala), MASE **4.02**.
- **Lezione:** x0 ed ε = due estremi (sotto/sovra-confidente); x0 è il *male minore* (scelta del paper TimeDiff); la calibrazione vera richiede di più (σ_θ appresa, o conformal).
- **Bonus:** il meglio calibrato della scala è **DeepAR**, non una diffusione.

![](figures/presentation/fig_cmp_calibration.png){width=46%}

::: notes
La slide-storia del nostro risultato. Diagnosi indipendente: il rapporto RMSE/MAE resta uniforme (ε 8.97 ≈ x0 8.85 ≈ naive 8.67), quindi è sovra-dispersione uniforme, non finestre esplose — effetto reale, non bug del sampler. Meccanismo: senza ancoraggio autoregressivo il blocco non-AR sparge la varianza iniettata da ε su tutte le celle senza ricomporla. Usare per: "perché TimeDiff ha coverage zero?", "l'ablazione ε ha funzionato?", "differenza tra i due diffusion model?".
:::

# Altre slide di backup (testo, per il Q&A)

- **B2 — Tabella completa:** `results/tables/comparison_electricity.md` (tutte le colonne, vincitore per colonna in grassetto).
- **B3 — Perché ARIMA va male:** auto-ARIMA per-canale su 321 serie, ordini eterogenei, fragilità, costo.
- **B4 — Math della diffusione:** `q(x_t|x_0)`, predizione del rumore ε, loss semplificata; toy DDPM 1-D.
- **B5 — Exchange, risultati:** `results/tables/comparison_exchange.md` (controllo negativo: deep < naive).
- **B6 — DeepAR senza time-features:** effetto delle covariate calendario (se la riga `deepar_notf` è in registry).

::: notes
Backup testuali, da richiamare solo su domanda specifica. Numeri sempre dalla registry, mai a mano.
:::
