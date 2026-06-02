# Slide template — Electricity (PML, ~8–10 min, 3 relatori)

> **Come si usa questo file.** È lo scheletro slide-per-slide della presentazione.
> I **numeri** non si scrivono qui a mano: vengono da
> [`RESULTS_PLACEHOLDERS_ELECTRICITY.md`](RESULTS_PLACEHOLDERS_ELECTRICITY.md)
> (fonte di verità). M0–M4 sono **numeri reali** (da `results/registry.csv`).
> M4 TimeDiff entra nel deck *foldato* nelle slide 5/8/9/10 + backup **B7**
> (deep-dive collasso x0 → ablazione ε); promuovibile a slide piena quando arriva M4ε.
>
> **Legenda "chi parla":** A = problema/dataset/PML · B = modelli/metodo ·
> C = risultati/interpretazione/limiti. (~2.5–3 min a testa.)
>
> **Legenda "stato":** ✅ pronto · 🟡 figura ancora da generare.

---

## Slide 1 — Titolo + la domanda

- **Obiettivo:** inquadrare in una frase di cosa parla il progetto e qual è la
  domanda di ricerca, senza promettere una "vittoria".
- **Contenuto testuale:**
  - Titolo: *"Forecasting probabilistico di serie temporali con modelli di
    diffusione: un confronto controllato su consumi elettrici."*
  - Sottotitolo: *"M0 seasonal-naive → M1 ARIMA → M2 DeepAR → M3 TimeGrad → M4 TimeDiff."*
  - Domanda guida (grande, in basso): **"Un modello di diffusione produce
    incertezza migliore — e a che costo?"**
  - Nomi dei 3 relatori, corso PML, Prof. Bortolussi, UniTS.
- **Figura:** nessuna (o un piccolo fan-chart decorativo di una serie con banda di
  incertezza — `fig_teaser_fanchart.png`, opzionale).
- **Speaker notes (A, ~30s):** "Buongiorno. Studiamo il forecasting *probabilistico*:
  non vogliamo una sola previsione, ma una distribuzione del futuro. Confrontiamo
  quattro modelli su una scala di complessità crescente, dal baseline stagionale al
  modello di diffusione TimeGrad. La domanda non è 'chi vince', ma se la diffusione
  *giustifica* il suo costo con un'incertezza migliore."
- **Chi parla:** A
- **Stato:** ✅ pronto (teaser opzionale 🟡)

---

## Slide 2 — Perché forecasting *probabilistico*

- **Obiettivo:** spiegare che prevediamo una distribuzione, non un punto, e perché
  questo conta nelle decisioni reali.
- **Contenuto testuale:**
  - Frase centrale: **prevediamo `p(futuro | passato)`**, cioè
    `p(x_{t+1:t+τ} | x_{t-H+1:t})`.
  - Tre bullet:
    - una previsione *puntuale* nasconde il rischio; una *distribuzione* lo quantifica;
    - servono **intervalli calibrati** (quando dico "90%", deve essere davvero ~90%);
    - decisioni sotto incertezza (riserva di rete, sbilanciamento, prezzo) usano i
      *quantili*, non solo la media.
  - Glossario lampo: H = contesto (passato osservato), τ = orizzonte (futuro previsto).
- **Figura:** `fig_fanchart_example.png` — una finestra reale di Electricity con
  mediana + banda 50% + banda 90% (didattica).
- **Speaker notes (A, ~45s):** "Forecasting probabilistico significa stimare la
  distribuzione del futuro condizionata al passato — qui la scriviamo come `p` del
  futuro dato il passato. La media da sola non basta: per gestire una rete elettrica
  mi serve sapere *quanto* posso sbagliare. Per questo valuteremo non solo l'errore
  puntuale, ma anche **quanto sono calibrate** le bande di incertezza: se dico
  novanta percento, dentro la banda devono cadere circa il novanta percento dei casi."
- **Chi parla:** A
- **Stato:** ✅ pronto (figura 🟡 da generare)

---

## Slide 3 — Dataset e protocollo

- **Obiettivo:** dire su cosa misuriamo e con quali regole, in modo che i numeri
  dopo siano credibili e onesti.
- **Contenuto testuale:**
  - Dataset **Electricity**: D=321 serie orarie di consumo, forte stagionalità
    giornaliera e settimanale.
  - Protocollo: split **70/10/20** (train/val/**test**), contesto **H=168** ore
    (1 settimana), orizzonte **τ=24** ore (1 giorno), **S=100** campioni per la
    distribuzione predittiva, **seed=42**.
  - Caveat di onestà (E0): usiamo il file grezzo LSTNet col *nostro* split, **non**
    lo split benchmark `electricity_nips` → i nostri CRPS **non** sono direttamente
    confrontabili coi numeri pubblicati di TimeGrad/CSDI. È un confronto *interno*,
    controllato.
- **Figura:** `fig_dataset_overview.png` — qualche serie campione + heatmap
  oraria×giorno della stagionalità.
- **Speaker notes (A, ~45s):** "Lavoriamo su Electricity: 321 contatori orari, con
  una stagionalità giornaliera e settimanale molto marcata. Usiamo una settimana di
  storia per prevedere il giorno dopo, su uno split test tenuto da parte, con cento
  campioni per ricostruire la distribuzione. Un punto di onestà: usiamo il nostro
  split, non quello del benchmark pubblicato, quindi confrontiamo i modelli *tra
  loro*, non con la letteratura. È un confronto controllato, a parità di condizioni."
- **Chi parla:** A
- **Stato:** ✅ pronto (figura 🟡 da generare)

---

## Slide 4 — Da Exchange a Electricity (perché due dataset)

- **Obiettivo:** spiegare che Exchange è il banco di prova (controllo negativo) e
  Electricity è il test vero; introdurre l'idea che il baseline può essere fortissimo.
- **Contenuto testuale:**
  - **Exchange** (D=8, giornaliero, quasi *random walk*): sandbox per far girare la
    pipeline. Qui i modelli deep **non** battono la persistenza — ed è *atteso*:
    controllo negativo.
  - **Electricity** (D=321, orario, stagionale): il test rilevante, struttura ricca,
    rilevanza industriale.
  - Messaggio: *cambia il regime del segnale, cambia chi vince.* Su un random walk
    "ieri" è imbattibile; su un segnale stagionale c'è spazio per modellare.
  - Avvertenza: MASE e CRPS **non** sono confrontabili tra i due dataset (scale e
    denominatori diversi).
- **Figura:** `fig_exchange_vs_electricity.png` — due mini-serie affiancate (un
  random walk vs una stagionale) per mostrare la differenza di struttura.
- **Speaker notes (A→B, ~45s):** "Abbiamo usato due dataset con ruoli diversi.
  Exchange, i tassi di cambio, è quasi una passeggiata aleatoria: lì la miglior
  previsione di domani è oggi, e nessun modello sofisticato batte questa regola — lo
  usiamo come *controllo negativo*, per validare la pipeline. Electricity è il test
  vero, con stagionalità forte. La morale che porteremo avanti: il vincitore dipende
  dal tipo di segnale. E attenzione, le metriche non sono confrontabili tra i due."
- **Chi parla:** A → passaggio a B
- **Stato:** ✅ pronto (figura 🟡 da generare)

---

## Slide 5 — La scala dei modelli: M0 → M4

- **Obiettivo:** presentare i cinque modelli come una scala di complessità
  crescente, con una frase ciascuno; far notare che M3 e M4 sono **due** diffusion
  model con design opposto (autoregressivo vs non-autoregressivo).
- **Contenuto testuale:**
  - **M0 — seasonal-naive:** "domani come una settimana fa / il ciclo precedente".
    Zero training, fortissimo sulle serie stagionali.
  - **M1 — ARIMA:** lineare classico, un modello *per canale* (321 modelli).
  - **M2 — DeepAR:** rete ricorrente autoregressiva probabilistica, addestrata su
    tutte le serie insieme.
  - **M3 — TimeGrad:** diffusione **autoregressiva** condizionata, genera il futuro
    un passo alla volta rimuovendo rumore (CRPS 241.6: batte DeepAR, non M0).
  - **M4 — TimeDiff:** diffusione **non-autoregressiva** — denoisa l'**intero blocco**
    futuro in una sola catena → sampling *molto* più rapido (predict ~41 min vs ~4h19
    di M3). Point forecast nitido (MAE migliore tra i deep) *ma* incertezza collassata.
  - Asse implicito: più espressività → (forse) incertezza migliore → più costo —
    **e M4 mostra che il design (AR vs non-AR) sposta il costo, non solo la scala.**
- **Figura:** `fig_model_ladder.png` — la scaletta M0→…→M4 con frecce
  "espressività ↑ / costo ↑"; M3 e M4 affiancati come "diffusione AR vs non-AR".
- **Speaker notes (B, ~70s):** "Abbiamo costruito una *scala* di cinque modelli.
  M0, il seasonal-naive, dice semplicemente che il futuro assomiglia al ciclo
  precedente: nessun addestramento, ma su segnali stagionali è sorprendentemente
  forte. M1, ARIMA, è il classico lineare, un modello per ciascuna delle 321 serie.
  M2, DeepAR, è una rete ricorrente probabilistica addestrata su tutte le serie
  insieme. M3, TimeGrad, è il nostro primo modello di diffusione, *autoregressivo*:
  genera il futuro un passo alla volta. M4, TimeDiff, è il secondo, ma con un design
  opposto: *non* autoregressivo, ripulisce l'intero blocco futuro in una sola catena —
  e questo lo rende molto più veloce. Come vedremo, M4 dà il forecast puntuale più
  nitido al costo più basso, ma paga con un'incertezza degenere: un caso da manuale di
  come il *design* del modello, non solo la sua taglia, cambi il compromesso."
- **Chi parla:** B
- **Stato:** ✅ pronto (testo + righe M3, M4 reali)

---

## Slide 6 — TimeGrad, in modo intuitivo

- **Obiettivo:** spiegare la diffusione senza math pesante, con l'analogia
  rumore→segnale, e dire come la usiamo per il forecasting.
- **Contenuto testuale:**
  - **Idea (forward):** parto dal dato vero e ci aggiungo rumore gaussiano un po' alla
    volta, finché diventa rumore puro.
  - **Idea (reverse):** addestro una rete a **invertire** quei passi; a generazione,
    parto da rumore e lo ripulisco fino a un campione plausibile.
  - **Per il forecasting:** la "ripulitura" è **condizionata sul passato** (lo stato
    di un RNN che riassume H ore); ripetendo il sampling S volte ottengo S traiettorie
    future → la distribuzione predittiva.
  - **Il prezzo:** ogni campione richiede molti passi di denoising → sampling lento
    (è il costo che misureremo). diff_steps = 100, predict ≈ 4h 19min su L4.
- **Figura:** `fig_diffusion_intuition.png` — striscia rumore→segnale (forward in
  alto, reverse in basso); piccolo toy DDPM 1-D se disponibile.
- **Speaker notes (B, ~60s):** "TimeGrad è un modello di diffusione. L'idea: prendo il
  dato vero e ci aggiungo rumore gradualmente, finché non resta che rumore. Poi
  addestro una rete a fare il percorso inverso, a *togliere* rumore un passo alla
  volta. Per generare parto da puro rumore e lo ripulisco fino a ottenere un campione
  realistico. Per il forecasting condiziono questa ripulitura sul passato — riassunto
  da una rete ricorrente — e ripeto cento volte: ottengo cento futuri possibili, cioè
  la distribuzione. Il prezzo da pagare è che ogni campione richiede tanti passi: il
  sampling è lento, ed è esattamente il costo che andremo a quantificare."
- **Chi parla:** B
- **Stato:** ✅ pronto (figura 🟡; toy DDPM opzionale)

---

## Slide 7 — Come misuriamo (metriche)

- **Obiettivo:** dare al pubblico le 3 lenti per leggere i risultati: accuratezza
  puntuale, qualità probabilistica, calibrazione — più il costo.
- **Contenuto testuale:**
  - **Accuratezza puntuale:** MAE, RMSE, **MASE** (1 = come il seasonal-naive; <1 =
    meglio del baseline).
  - **Qualità probabilistica:** **CRPS** e pinball loss (premiano una distribuzione
    *intera* ben piazzata, non solo la media). Più basso = meglio.
  - **Calibrazione:** **cov50→0.50** e **cov90→0.90** (le bande coprono davvero quel
    che promettono?); width = quanto sono larghe (sharpness).
  - **Costo:** fit_s (training) e predict_s (sampling).
  - Frase chiave: *"un modello utile deve avere MASE < 1 e CRPS basso a un costo
    accettabile."*
- **Figura:** nessuna (slide-testo) oppure `fig_metrics_legend.png` (mini-schema).
- **Speaker notes (B, ~45s):** "Tre lenti per leggere i numeri. Primo, l'accuratezza
  puntuale: usiamo MASE, normalizzato in modo che uno equivalga al seasonal-naive —
  sotto uno vuol dire 'meglio del baseline'. Secondo, la qualità *probabilistica*: il
  CRPS, che premia una distribuzione intera ben piazzata, non solo la media. Terzo, la
  calibrazione: se le mie bande al cinquanta e novanta percento coprono davvero quella
  frazione di casi. E sopra tutto, il costo: quanto tempo per addestrare e per
  campionare."
- **Chi parla:** B
- **Stato:** ✅ pronto

---

## Slide 8 — Risultati su Electricity (la tabella)

- **Obiettivo:** mostrare i numeri reali M0–M4, la "barra da battere" CRPS 160.5,
  dire che nessun deep la supera, e far emergere il *doppio volto* di M4 (punto nitido,
  distribuzione collassata).
- **Contenuto testuale:**
  - Tabella ridotta (da `RESULTS_PLACEHOLDERS_ELECTRICITY.md`):
    | Modello | MASE | CRPS | cov50 | cov90 | costo |
    |---|---|---|---|---|---|
    | M0 seasonal-naive | **1.00** | **160.5** | 0.497 | 0.885 | ~59 min predict |
    | M1 ARIMA | 3.66 | 867.5 | 0.592 | 0.930 | ~2h42 |
    | M2 DeepAR | 1.83 | 253.7 | 0.466 | 0.831 | ~10 min fit + ~82 min |
    | M3 TimeGrad | 1.39 | 241.6 | 0.279 | 0.647 | ~39 min fit + ~4h19 |
    | M4 TimeDiff | 1.40 | 287.3 | **0.003** | **0.008** | **~1 min fit + ~41 min** |
  - **Fonte canonica (auto-generata, non ribattere a mano):**
    `results/tables/comparison_electricity.md` (da `experiments/make_tables.py`,
    vincitore per colonna in grassetto). In FASE B: `python -m experiments.make_tables`
    rigenera la tabella con le righe M3/M4 reali; da lì si ricopiano i numeri qui.
  - **Callout grande:** *"La barra da battere è CRPS = 160.5 (M0). Nessun modello deep
    ci riesce: DeepAR 253.7, TimeGrad 241.6 (il più vicino), TimeDiff 287.3."*
  - **Riquadro M4 (doppio volto):** *"TimeDiff dà il forecast puntuale più nitido tra
    i deep (MAE 288.5, il più basso) al costo più basso — ma la sua incertezza è rotta:
    cov50 0.003, cov90 0.008, e CRPS ≈ MAE, la firma di una distribuzione a massa
    puntiforme. Lo approfondiamo in backup (B7) con l'ablazione ε."*
  - Sotto-messaggio onesto: su Electricity **il baseline stagionale è fortissimo**;
    ARIMA per-canale è fragile su 321 serie.
- **Figura:** `fig_cmp_crps.png` (barre CRPS, M3/M4 reali) +
  `fig_cmp_mase.png` come backup.
- **Speaker notes (C, ~70s):** "Ecco i numeri su Electricity. La prima cosa che salta
  all'occhio: il seasonal-naive ha CRPS 160 e MASE uno — è la *barra da battere*.
  ARIMA, montato un modello per canale su 321 serie, è fragile e va malissimo. DeepAR
  è 253, batte ARIMA ma **non** il baseline. TimeGrad scende a 241 — il più vicino al
  baseline, ma non lo supera, e a un costo di sampling enorme. E TimeDiff? Qui c'è la
  parte interessante: come *previsione puntuale* è il migliore dei deep — l'errore MAE
  più basso — e costa pochissimo, un minuto di training e quaranta di sampling contro
  le quattro ore di TimeGrad. Ma la sua *incertezza* è collassata: le bande coprono lo
  zero per cento invece del cinquanta o novanta. È un segnale da manuale, e lo
  raccontiamo in dettaglio nelle slide di backup."
- **Chi parla:** C
- **Stato:** ✅ pronto (M0–M4 reali)

---

## Slide 9 — Qualità vs costo (il vero trade-off)

- **Obiettivo:** spostare il discorso da "chi ha il numero più basso" a "quanto paghi
  per quel numero": è la tesi della presentazione.
- **Contenuto testuale:**
  - Scatter **CRPS (qualità, ↓) vs predict_s (costo, ↓)**; ogni modello un punto;
    angolo in basso-a-sinistra = ideale.
  - Lettura: M0 è economico *e* buono (in basso-a-sinistra, sorpresa); M2 paga di più
    per una qualità peggiore di M0; M3 è il più **costoso** (sampling ~4h19) e migliora
    il CRPS solo fino a 241.6 — si sposta a sinistra di DeepAR ma resta a destra e in
    alto rispetto a M0: il costo non è ripagato. **M4 è il punto più a sinistra**
    (inferenza ~41 min, il più economico dei deep) ma con CRPS 287, il più alto dei
    deep: il design **non-AR compra velocità, non calibrazione**.
  - Frase chiave: *"Non basta vincere sul CRPS: bisogna vincere abbastanza da
    giustificare il costo di sampling — e la velocità da sola non vale se l'incertezza
    è rotta."*
- **Figura:** `fig_cmp_quality_cost.png` (scatter; M3 e M4 punti reali).
- **Speaker notes (C, ~70s):** "Questo è il grafico che riassume la nostra tesi.
  Sull'asse orizzontale il costo di campionamento, sul verticale la qualità
  probabilistica: l'ideale è l'angolo in basso a sinistra, economico e accurato. Il
  seasonal-naive sta proprio lì — ed è il risultato più istruttivo. DeepAR costa di
  più e rende meno del baseline. TimeGrad, per costruzione, è costoso: oltre quattro
  ore di sampling, e migliora il CRPS solo fino a 241, senza raggiungere M0. TimeDiff
  ribalta l'asse del costo: è il punto più a sinistra, il più economico dei deep grazie
  al design non-autoregressivo. Ma guardate l'altezza: il suo CRPS è il più alto dei
  deep, perché la sua incertezza è collassata. La morale: la velocità da sola non
  basta, se quello che guadagni in costo lo perdi in calibrazione."
- **Chi parla:** C
- **Stato:** ✅ pronto (M3, M4 reali)

---

## Slide 10 — Conclusioni e limiti

- **Obiettivo:** chiudere con la tesi del trade-off (regge in tutti gli scenari) e
  l'onestà sui limiti; lasciare un take-home memorabile.
- **Contenuto testuale:**
  - **Take-home:** la storia non è "diffusion vince", ma il **trade-off
    espressività ↔ calibrazione ↔ costo**. Su un segnale molto stagionale un baseline
    semplice è una soglia seria. E **due** diffusion model con design opposto (M3 AR,
    M4 non-AR) mostrano che a contare non è solo la taglia, ma *come* è fatto il modello.
  - **Esito M3 — Scenario A:** *"TimeGrad migliora la baseline deep probabilistica
    (DeepAR, CRPS 253.7 → 241.6) ma non supera il seasonal-naive (160.5). La
    stagionalità deterministica è già catturata benissimo dal baseline; la diffusione
    aggiunge flessibilità ma non abbastanza valore predittivo rispetto al costo."* Il
    valore è **metodologico** (confronto controllato), non una vittoria del modello.
  - **Esito M4 — il design conta:** *"TimeDiff, non-autoregressivo, è ~6× più rapido in
    sampling di TimeGrad e dà la MAE più bassa tra i deep, ma in x0-prediction la sua
    distribuzione collassa (cov ≈ 0, CRPS ≈ MAE). Non è un bug del sampler: è la
    parametrizzazione del target. L'ablazione ε (M4ε) testa se predire il rumore
    ricalibra."* Lezione: **la scelta x0 vs ε cambia la calibrazione a parità di tutto.**
  - **Limiti:** split non-benchmark (E0); ARIMA per-canale poco competitivo; sampling
    di diffusione costoso (M3); un solo seed; niente tuning esteso degli iperparametri.
  - **Lavori futuri:** ablazione **ε-prediction** (M4ε, in corso) per ricalibrare
    TimeDiff; split `electricity_nips` per confronto con la letteratura; CSDI;
    riduzione passi di sampling (DDIM-like); più seed.
- **Figura:** richiamo a `fig_cmp_quality_cost.png` (la stessa) o nessuna.
- **Speaker notes (C, ~55s):** "In sintesi: il nostro messaggio non è che la
  diffusione vince, ma che esiste un compromesso tra espressività, calibrazione e
  costo — e che su un segnale fortemente stagionale un baseline semplice è un avversario
  serio. TimeGrad batte DeepAR sul CRPS — 241 contro 253 — ma non il seasonal-naive: il
  valore è metodologico, non una vittoria del modello. E abbiamo confrontato *due*
  diffusion model con design opposto: TimeDiff, non-autoregressivo, ribalta il costo —
  sei volte più rapido — ma in questa parametrizzazione rompe la calibrazione. È la
  prova che a contare è il *design*, non solo la taglia; e la nostra ablazione sul modo
  di predire — il segnale pulito contro il rumore — serve proprio a ricalibrarlo. Siamo
  onesti sui limiti: split non-benchmark, ARIMA poco competitivo, sampling costoso, un
  solo seed. Lavori futuri: l'ablazione ε, lo split standard, CSDI. Grazie."
- **Chi parla:** C
- **Stato:** ✅ pronto (Scenario A + M4 foldato)

---

## Slide di BACKUP (non in scaletta, per il Q&A)

> Da mostrare solo se una domanda le richiama. Tutte ✅ tranne dove indicato.

- **B1 — Calibrazione in dettaglio:** `fig_cmp_calibration.png` (cov50/cov90 vs
  nominale per tutti i modelli) + `fig_cmp_intervals.png` (width50/width90, sharpness).
  *Uso:* "come fai a dire che un modello è ben calibrato?". ✅ M3 reale.
- **B2 — Tabella completa:** la tabella estesa **auto-generata**
  `results/tables/comparison_electricity.md` (tutte le colonne: pinball, width, MAE,
  RMSE, fit/predict; vincitore per colonna in grassetto). *Uso:* domande su una metrica
  specifica. ✅ M3 reale.
- **B3 — Perché ARIMA va così male:** auto-ARIMA per-canale su 321 serie, fragilità,
  ordini eterogenei, costo. *Uso:* "ARIMA non doveva essere un baseline forte?".
- **B4 — Cos'è davvero la diffusione (un filo di math):** `q(x_t|x_0)`,
  predizione del rumore ε, loss semplificata; toy DDPM 1-D. *Uso:* domanda tecnica
  sulla diffusione. 🟡 toy opzionale.
- **B5 — Exchange, risultati:** tabella Exchange **auto-generata**
  `results/tables/comparison_exchange.md` (controllo negativo: deep < naive).
  *Uso:* "e sull'altro dataset?".
- **B6 — DeepAR senza time-features (M2 ablation):** se presente in registry
  (`deepar_notf`), mostra l'effetto delle covariate calendario. *Uso:* domanda su
  ablation. 🟡 solo se la riga esiste.
- **B7 — M4 TimeDiff: il collasso di varianza e l'ablazione ε:** la slide-storia del
  doppio volto di TimeDiff. *Contenuto:* (1) cos'è il non-AR (denoisa l'intero blocco
  τ×D in una catena, da cui la velocità); (2) il collasso — cov 0.003/0.008, width ≈ 0,
  **CRPS ≈ MAE** = massa puntiforme; (3) **perché**: in x0-prediction la rete predice
  un x0 quasi costante appoggiandosi al contesto/`x_ar` e ignora il rumore `x_t`;
  l'unica varianza residua è `1−ᾱ_0 ≈ 6·10⁻⁴` → std ≈ 0 (la *future-mixup* aggrava);
  (4) **la cura**: ε-prediction (DDPM, Ho 2020) controlla esplicitamente la dispersione
  iniettata → M4ε dovrebbe ricalibrare. *Figure:* `fig_cmp_calibration.png` (M4 piatto
  sull'asse) + `fig_cmp_intervals.png` (width 2.8/6.6). *Uso:* "perché TimeDiff ha
  coverage zero?" / "qual è la differenza tra i due diffusion model?".
  ✅ x0 reale; 🟡 riga M4ε in arrivo (numeri al posto del foreshadow). Dettaglio
  numerico in `RESULTS_PLACEHOLDERS_ELECTRICITY.md` §5.

---

### Mappa figure → slide (riassunto per chi prepara il deck)

| Slide | Figura principale | Stato figura |
|---|---|---|
| 1 | `fig_teaser_fanchart.png` (opz.) | 🟡 opzionale |
| 2 | `fig_fanchart_example.png` | 🟡 da generare |
| 3 | `fig_dataset_overview.png` | 🟡 da generare |
| 4 | `fig_exchange_vs_electricity.png` | 🟡 da generare |
| 5 | `fig_model_ladder.png` | 🟡 schema |
| 6 | `fig_diffusion_intuition.png` | 🟡 schema |
| 7 | — (testo) / `fig_metrics_legend.png` | ✅ / 🟡 |
| 8 | `fig_cmp_crps.png` (+ `fig_cmp_mase.png`) | ✅ M3/M4 reali |
| 9 | `fig_cmp_quality_cost.png` | ✅ M3/M4 reali |
| 10 | richiamo S9 | ✅ |
| B1 | `fig_cmp_calibration.png`, `fig_cmp_intervals.png` | ✅ M3/M4 reali |
| B7 | `fig_cmp_calibration.png`, `fig_cmp_intervals.png` | ✅ M4 x0 · 🟡 M4ε |

> Dettaglio completo delle figure in
> [`FIGURE_PLAN_ELECTRICITY_IT.md`](FIGURE_PLAN_ELECTRICITY_IT.md).
