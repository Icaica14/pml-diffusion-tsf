# Slide template — Electricity (PML, ~8–10 min, 3 relatori)

> **Come si usa questo file.** È lo scheletro slide-per-slide della presentazione.
> I **numeri** non si scrivono qui a mano: vengono da
> [`RESULTS_PLACEHOLDERS_ELECTRICITY.md`](RESULTS_PLACEHOLDERS_ELECTRICITY.md)
> (fonte di verità). M0–M2 sono **reali**; M3 è 🔴 **PLACEHOLDER**.
>
> 🔴 **Ogni riferimento a M3 porta il marcatore**
> `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`.
>
> **Legenda "chi parla":** A = problema/dataset/PML · B = modelli/metodo ·
> C = risultati/interpretazione/limiti. (~2.5–3 min a testa.)
>
> **Legenda "stato":** ✅ pronto · 🟡 placeholder (figura/numero provvisorio) ·
> 🔵 da aggiornare dopo M3.

---

## Slide 1 — Titolo + la domanda

- **Obiettivo:** inquadrare in una frase di cosa parla il progetto e qual è la
  domanda di ricerca, senza promettere una "vittoria".
- **Contenuto testuale:**
  - Titolo: *"Forecasting probabilistico di serie temporali con modelli di
    diffusione: un confronto controllato su consumi elettrici."*
  - Sottotitolo: *"M0 seasonal-naive → M1 ARIMA → M2 DeepAR → M3 TimeGrad."*
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

## Slide 5 — La scala dei modelli: M0 → M3

- **Obiettivo:** presentare i quattro modelli come una scala di complessità
  crescente, con una frase ciascuno.
- **Contenuto testuale:**
  - **M0 — seasonal-naive:** "domani come una settimana fa / il ciclo precedente".
    Zero training, fortissimo sulle serie stagionali.
  - **M1 — ARIMA:** lineare classico, un modello *per canale* (321 modelli).
  - **M2 — DeepAR:** rete ricorrente autoregressiva probabilistica, addestrata su
    tutte le serie insieme.
  - **M3 — TimeGrad 🔴:** modello di **diffusione** condizionato, genera il futuro
    rimuovendo rumore passo dopo passo.
    `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
  - Asse implicito: più espressività → (forse) incertezza migliore → più costo.
- **Figura:** `fig_model_ladder.png` — la scaletta M0→M1→M2→M3 con frecce
  "espressività ↑ / costo ↑".
- **Speaker notes (B, ~60s):** "Abbiamo costruito una *scala* di quattro modelli.
  M0, il seasonal-naive, dice semplicemente che il futuro assomiglia al ciclo
  precedente: nessun addestramento, ma su segnali stagionali è sorprendentemente
  forte. M1, ARIMA, è il classico lineare, un modello per ciascuna delle 321 serie.
  M2, DeepAR, è una rete ricorrente probabilistica addestrata su tutte le serie
  insieme. M3, TimeGrad, è il nostro modello di diffusione: genera il futuro partendo
  da rumore e ripulendolo passo dopo passo. Salendo la scala aumenta l'espressività —
  e, come vedremo, il costo."
- **Chi parla:** B
- **Stato:** ✅ pronto (testo); 🔵 riga M3 da confermare dopo run

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
    (è il costo che misureremo). diff_steps = 100.
    `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
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

- **Obiettivo:** mostrare i numeri reali M0–M2, la "barra da battere" CRPS 160.5, e
  inserire M3 come placeholder esplicito.
- **Contenuto testuale:**
  - Tabella ridotta (da `RESULTS_PLACEHOLDERS_ELECTRICITY.md`):
    | Modello | MASE | CRPS | cov50 | cov90 | costo |
    |---|---|---|---|---|---|
    | M0 seasonal-naive | **1.00** | **160.5** | 0.497 | 0.885 | ~59 min predict |
    | M1 ARIMA | 3.66 | 867.5 | 0.592 | 0.930 | ~2h42 |
    | M2 DeepAR | 1.83 | 253.7 | 0.466 | 0.831 | ~10 min fit + ~82 min |
    | **M3 TimeGrad 🔴** | *1.35–1.70* | *185–235* | *0.46–0.52* | *0.84–0.90* | *~3h* |
  - 🔴 `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
  - **Fonte canonica (auto-generata, non ribattere a mano):**
    `results/tables/comparison_electricity.md` (da `experiments/make_tables.py`,
    vincitore per colonna in grassetto). In FASE B: `python -m experiments.make_tables`
    rigenera la tabella con la riga M3 reale; da lì si ricopiano i numeri qui.
  - **Callout grande:** *"La barra da battere è CRPS = 160.5 (M0). DeepAR non ci
    riesce (253.7). M3?"*
  - Sotto-messaggio onesto: su Electricity **il baseline stagionale è fortissimo**;
    ARIMA per-canale è fragile su 321 serie.
- **Figura:** `fig_cmp_crps.png` (barre CRPS, M3 tratteggiato/placeholder) +
  `fig_cmp_mase.png` come backup.
- **Speaker notes (C, ~60s):** "Ecco i numeri su Electricity. La prima cosa che salta
  all'occhio: il seasonal-naive ha CRPS 160 e MASE uno — è la *barra da battere*.
  ARIMA, montato un modello per canale su 321 serie, è fragile e va malissimo. DeepAR
  è il miglior modello addestrato, batte ARIMA di larga misura ma **non** raggiunge il
  baseline stagionale: 253 contro 160. La domanda diventa quindi precisa: il modello
  di diffusione riesce a scendere sotto 160 a un costo accettabile? La riga di
  TimeGrad qui è ancora un **placeholder**: il run è in corso, i numeri reali entrano
  appena finisce."
- **Chi parla:** C
- **Stato:** 🔵 **da aggiornare dopo M3** (M0–M2 ✅, M3 🔴 placeholder)

---

## Slide 9 — Qualità vs costo (il vero trade-off)

- **Obiettivo:** spostare il discorso da "chi ha il numero più basso" a "quanto paghi
  per quel numero": è la tesi della presentazione.
- **Contenuto testuale:**
  - Scatter **CRPS (qualità, ↓) vs predict_s (costo, ↓)**; ogni modello un punto;
    angolo in basso-a-sinistra = ideale.
  - Lettura: M0 è economico *e* buono (in basso-a-sinistra, sorpresa); M2 paga di più
    per una qualità peggiore di M0; M3 🔴 atteso costoso (sampling) — la domanda è se
    si sposta abbastanza a sinistra (CRPS) da giustificare lo spostamento in alto
    (costo). `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
  - Frase chiave: *"Non basta vincere sul CRPS: bisogna vincere abbastanza da
    giustificare il costo di sampling."*
- **Figura:** `fig_cmp_quality_cost.png` (scatter; M3 come punto placeholder con
  marcatore visibile).
- **Speaker notes (C, ~60s):** "Questo è il grafico che riassume la nostra tesi.
  Sull'asse orizzontale il costo di campionamento, sul verticale la qualità
  probabilistica: l'ideale è l'angolo in basso a sinistra, economico e accurato. Il
  seasonal-naive sta proprio lì — ed è il risultato più istruttivo. DeepAR costa di
  più e rende meno del baseline. TimeGrad, per costruzione, è costoso: il sampling è
  lento. Il punto non è se migliora di un'inezia il CRPS, ma se si sposta a sinistra
  *abbastanza* da ripagare il costo. Anche questo punto è un placeholder finché il run
  non finisce."
- **Chi parla:** C
- **Stato:** 🔵 **da aggiornare dopo M3** (M3 🔴 placeholder)

---

## Slide 10 — Conclusioni e limiti

- **Obiettivo:** chiudere con la tesi del trade-off (regge in tutti gli scenari) e
  l'onestà sui limiti; lasciare un take-home memorabile.
- **Contenuto testuale:**
  - **Take-home:** la storia non è "diffusion vince", ma il **trade-off
    espressività ↔ calibrazione ↔ costo**. Su un segnale molto stagionale un baseline
    semplice è una soglia seria.
  - **Esito M3** (scegliere a valle): inserire qui la riga dello **scenario A / B / C**
    da `RESULTS_PLACEHOLDERS_ELECTRICITY.md` §3.
    `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
  - **Limiti:** split non-benchmark (E0); ARIMA per-canale poco competitivo; sampling
    di diffusione costoso; un solo seed; niente tuning esteso degli iperparametri.
  - **Lavori futuri:** split `electricity_nips` per confronto con la letteratura;
    CSDI; riduzione passi di sampling (DDIM-like); più seed.
- **Figura:** richiamo a `fig_cmp_quality_cost.png` (la stessa) o nessuna.
- **Speaker notes (C, ~45s):** "In sintesi: il nostro messaggio non è che la
  diffusione vince, ma che esiste un compromesso tra espressività, calibrazione e
  costo — e che su un segnale fortemente stagionale un baseline semplice è un avversario
  serio. [Inserire la frase dello scenario A, B o C a seconda del risultato di M3.]
  Siamo onesti sui limiti: lo split non è quello del benchmark, ARIMA è poco
  competitivo per costruzione, il sampling di diffusione è costoso, e abbiamo un solo
  seed. Da qui i lavori futuri: lo split standard, CSDI, e ridurre i passi di
  campionamento. Grazie."
- **Chi parla:** C
- **Stato:** 🔵 **da aggiornare dopo M3** (scegliere scenario A/B/C)

---

## Slide di BACKUP (non in scaletta, per il Q&A)

> Da mostrare solo se una domanda le richiama. Tutte ✅ tranne dove indicato.

- **B1 — Calibrazione in dettaglio:** `fig_cmp_calibration.png` (cov50/cov90 vs
  nominale per tutti i modelli) + `fig_cmp_intervals.png` (width50/width90, sharpness).
  *Uso:* "come fai a dire che un modello è ben calibrato?". 🔵 M3 placeholder.
- **B2 — Tabella completa:** la tabella estesa **auto-generata**
  `results/tables/comparison_electricity.md` (tutte le colonne: pinball, width, MAE,
  RMSE, fit/predict; vincitore per colonna in grassetto). *Uso:* domande su una metrica
  specifica. 🔵 M3 placeholder.
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
| 8 | `fig_cmp_crps.png` (+ `fig_cmp_mase.png`) | 🔵 M3 placeholder |
| 9 | `fig_cmp_quality_cost.png` | 🔵 M3 placeholder |
| 10 | richiamo S9 | 🔵 |
| B1 | `fig_cmp_calibration.png`, `fig_cmp_intervals.png` | 🔵 |

> Dettaglio completo delle figure in
> [`FIGURE_PLAN_ELECTRICITY_IT.md`](FIGURE_PLAN_ELECTRICITY_IT.md).
