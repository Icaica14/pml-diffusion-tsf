# Piano figure — Electricity (presentazione PML)

> **Mappa di tutte le figure della presentazione.** Per ciascuna: cosa mostra,
> perché serve, da quali dati nasce, se **esiste già** o è **da generare**, e con
> quale **script**. Convenzione colori stabile in tutto il progetto:
>
> | Modello | Colore | Hex |
> |---|---|---|
> | M0 seasonal-naive | grigio | `#7f7f7f` |
> | M1 ARIMA | blu | `#1f77b4` |
> | M2 DeepAR | arancione | `#ff7f0e` |
> | M3 TimeGrad | rosso | `#d62728` |
> | M4 TimeDiff | viola | `#9467bd` |
>
> **Le figure che includono M3 e M4 ora usano i valori reali** dal registry
> (righe `electricity,timegrad` e `electricity,timediff`): nessun marcatore
> "PLACEHOLDER", tratto pieno. **M4 ha una firma visiva precisa** — sulla calibrazione
> è la linea schiacciata sull'asse (cov ≈ 0), sulla sharpness ha barre ≈ 0 etichettate
> col valore (≈ 2.8 / 6.6) per non sembrare "dato mancante".
>
> **Output attesi in:** `figures/presentation/` (PNG, ~150 dpi, sfondo bianco).

---

## Tabella sintetica

| # | File | Slide | Stato | Generata da |
|---|---|---|---|---|
| 1 | `fig_cmp_crps.png` | 8 | ✅ M3/M4 reali | `experiments/plot_presentation.py` |
| 2 | `fig_cmp_mase.png` | 8 (backup) | ✅ M3/M4 reali | `experiments/plot_presentation.py` |
| 3 | `fig_cmp_quality_cost.png` | 9, 10 | ✅ M3/M4 reali | `experiments/plot_presentation.py` |
| 4 | `fig_cmp_calibration.png` | B1, B7 | ✅ M3/M4 reali | `experiments/plot_presentation.py` |
| 5 | `fig_cmp_intervals.png` | B1, B7 | ✅ M3/M4 reali | `experiments/plot_presentation.py` |
| 6 | `fig_cmp_cost.png` | 9 (supp.) | ✅ M3/M4 reali | `experiments/plot_presentation.py` |
| 7 | `fig_fanchart_example.png` | 2 | 🟡 da generare | manuale / notebook EDA |
| 8 | `fig_dataset_overview.png` | 3 | 🟡 da generare | manuale / notebook EDA |
| 9 | `fig_exchange_vs_electricity.png` | 4 | 🟡 da generare | manuale / notebook EDA |
| 10 | `fig_model_ladder.png` | 5 | 🟡 schema | disegno/slide |
| 11 | `fig_diffusion_intuition.png` | 6 | 🟡 schema | disegno/toy DDPM |
| 12 | `fig_teaser_fanchart.png` | 1 (opz.) | 🟡 opzionale | come #7 |

> Le figure **1–6** sono **auto-generate** dallo script di presentazione (leggono la
> registry; la riga M3 è ora reale). Le **7–12** sono illustrative/didattiche e
> si possono fare a mano o riusando i notebook EDA.

---

## Dettaglio figura per figura

### 1. `fig_cmp_crps.png` — barre CRPS (la metrica principale)
- **Cosa mostra:** una barra di CRPS per modello (M0…M4), più bassa = meglio.
  Linea orizzontale tratteggiata sulla **barra da battere** = 160.5 (M0).
- **Perché:** è la slide-risultato centrale (S8). Rende visibile che DeepAR non
  raggiunge il naive e pone la domanda su M3.
- **Dati:** colonna `CRPS` da `results/registry.csv`, righe `dataset=electricity`.
- **M3:** barra rossa piena a **CRPS 241.6** (sopra la linea 160.5: non batte M0).
- **M4:** barra viola a **CRPS 287.3** — la più alta dei deep (sopra M2 253.7 e M3 241.6).
- **Stato:** ✅ generata con M3/M4 reali.

### 2. `fig_cmp_mase.png` — barre MASE (accuratezza puntuale)
- **Cosa mostra:** MASE per modello; linea a **1.0** = soglia seasonal-naive.
- **Perché:** complemento puntuale del CRPS; backup di S8.
- **Dati:** colonna `MASE`, righe electricity.
- **M3:** MASE **1.39** (barra piena; sotto DeepAR 1.83, sopra M0 1.00).
- **M4:** MASE **1.40** ≈ M3 (point accuracy competitiva; MAE 288.5 è anzi la più bassa
  tra i deep, ma MASE pesa diversamente le serie).
- **Stato:** ✅ generata con M3/M4 reali.

### 3. `fig_cmp_quality_cost.png` — scatter qualità vs costo ⭐
- **Cosa mostra:** asse X = `predict_s` (costo, log), asse Y = `CRPS` (qualità); un
  punto per modello; ideale in basso-a-sinistra. Etichette accanto ai punti.
- **Perché:** **è la tesi della presentazione** (S9, richiamo S10). Sposta il discorso
  da "chi vince" a "quanto costa".
- **Dati:** `predict_s` + `CRPS`, righe electricity.
- **M3:** punto rosso pieno a (**15 526 s**, **241.6**): in alto a destra — costoso,
  qualità sotto M0.
- **M4:** punto viola a (**2 484.7 s**, **287.3**): il **più a sinistra** (inferenza più
  economica dei deep) ma il più in alto — il non-AR compra velocità, non qualità.
- **Stato:** ✅ generata con M3/M4 reali.

### 4. `fig_cmp_calibration.png` — calibrazione (backup B1, B7) ⭐ M4 vs M4ε
- **Cosa mostra:** reliability plot — per ogni modello cov50 e cov90 vs i nominali 0.50 e
  0.90; sotto la diagonale = troppo sicuro, sopra = troppo incerto. **Il pavimento degli assi è dinamico:** se un
  modello collassa verso zero (M4), si allarga a 0 per renderlo visibile, altrimenti resta
  al framing stretto 0.4 (Exchange).
- **Perché:** risponde a "come fai a dire che è calibrato?"; è la figura-chiave di B7.
- **Dati:** `cov50`, `cov90`, righe electricity.
- **M3:** cov50 **0.279**, cov90 **0.647** — nettamente **sovra-confidente** (bande
  troppo strette).
- **M4:** cov50 **0.003**, cov90 **0.008** — la linea viola è **schiacciata sull'asse**:
  l'esempio da manuale di overconfidence (distribuzione collassata).
- **M4ε:** cov50 **0.998**, cov90 **1.000** — la seconda linea viola è **inchiodata al
  soffitto**: l'ablazione ε ribalta M4 nell'estremo opposto (sovra-dispersione). Le due
  viola agli estremi opposti del grafico *sono* la storia di B7.
- **Stato:** ✅ generata con M3/M4/M4ε reali.

### 5. `fig_cmp_intervals.png` — copertura + ampiezza bande / sharpness (backup B1, B7)
- **Cosa mostra:** due pannelli — copertura (cov50/cov90 vs nominale) e sharpness
  (width50/width90). A parità di copertura, più stretto = più informativo. **Le barre di
  ampiezza che collassano verso zero sono etichettate col valore** (così M4 ≈ 2.8/6.6 non
  sembra "dato mancante" accanto alle bande enormi di ARIMA).
- **Perché:** la calibrazione da sola non basta; mostra il compromesso copertura↔ampiezza.
- **Dati:** `cov50/cov90`, `width50`, `width90`, righe electricity.
- **M3:** width50 **237.8**, width90 **675.9** (bande strette → la sotto-copertura di
  fig. 4 è coerente).
- **M4:** width50 **≈ 2.8**, width90 **≈ 6.6** — bande **quasi nulle**: copertura ≈ 0 e
  ampiezza ≈ 0 insieme = intervalli collassati (non bande larghe mal piazzate).
- **M4ε:** width50 **≈ 7686**, width90 **≈ 11359** — **le bande più larghe della scala**
  (più larghe persino di ARIMA): copertura ≈ 1 e ampiezza enorme insieme = sovra-dispersione,
  lo specchio esatto di M4.
- **Stato:** ✅ generata con M3/M4/M4ε reali.

### 6. `fig_cmp_cost.png` — costi fit/predict (supplemento S9)
- **Cosa mostra:** barre dei tempi `fit_s` e `predict_s` per modello (scala log).
- **Perché:** quantifica il "prezzo" di cui parla C; utile se chiedono i tempi.
- **Dati:** `fit_s`, `predict_s`, righe electricity.
- **M3:** fit **2 331.9 s**, predict **15 526.0 s** (il più lento: sampling di diffusione).
- **M4:** fit **48.0 s**, predict **2 484.7 s** — il **più economico** su entrambi gli
  assi: il payoff del design non-autoregressivo (una catena inversa, non τ).
- **Stato:** ✅ generata con M3/M4 reali.

### 7. `fig_fanchart_example.png` — esempio di previsione probabilistica (S2)
- **Cosa mostra:** una finestra reale di Electricity: storia, mediana prevista, banda
  50% e 90% attorno; il vero futuro sovrapposto.
- **Perché:** rende concreto "prevediamo una distribuzione, non un punto".
- **Dati:** un campione qualunque del test (anche da M0 o M2); serve solo a illustrare.
- **M3:** non necessario.
- **Stato:** 🟡 da generare (manuale o da notebook EDA). Non bloccante.

### 8. `fig_dataset_overview.png` — panoramica Electricity (S3)
- **Cosa mostra:** qualche serie campione + heatmap ora-del-giorno × giorno-settimana
  che evidenzia la doppia stagionalità.
- **Perché:** giustifica perché il seasonal-naive è forte e perché il dataset è
  interessante.
- **Dati:** `data/processed` Electricity (non committato).
- **Stato:** 🟡 da generare. Non bloccante.

### 9. `fig_exchange_vs_electricity.png` — confronto di regime (S4)
- **Cosa mostra:** due mini-serie affiancate: un tasso di cambio (quasi random walk,
  senza stagionalità) vs un consumo elettrico (stagionale).
- **Perché:** spiega il ruolo dei due dataset e la frase "il vincitore dipende dal
  segnale".
- **Dati:** un canale Exchange + un canale Electricity.
- **Stato:** 🟡 da generare. Non bloccante.

### 10. `fig_model_ladder.png` — la scaletta dei modelli (S5)
- **Cosa mostra:** schema M0→M1→M2→M3→M4 con due frecce "espressività ↑" e "costo ↑";
  M3 e M4 affiancati come **diffusione AR (autoregressiva) vs non-AR**, per evidenziare
  che sono due design opposti, non due gradini della stessa scala.
- **Perché:** dà la mappa mentale dei cinque modelli in un colpo d'occhio.
- **Dati:** nessuno (schema).
- **Stato:** 🟡 da disegnare (anche direttamente in slide). Non bloccante.

### 11. `fig_diffusion_intuition.png` — rumore↔segnale (S6)
- **Cosa mostra:** striscia in due righe: forward (dato→rumore) in alto, reverse
  (rumore→dato) in basso; opzionale un toy DDPM 1-D che "ripulisce" una sinusoide.
- **Perché:** spiega la diffusione senza math.
- **Dati:** nessuno o un toy 1-D generato al volo.
- **Stato:** 🟡 da disegnare / toy opzionale. Non bloccante.

### 12. `fig_teaser_fanchart.png` — teaser di copertina (S1, opzionale)
- **Cosa mostra:** un fan-chart elegante, solo decorativo, per la slide titolo.
- **Stato:** 🟡 opzionale (può essere lo stesso di #7).

---

## Come si generano le figure auto (1–6)

```bash
# dalla root del repo
python3 -m experiments.plot_presentation
```

Lo script `experiments/plot_presentation.py`:
1. legge `results/registry.csv` e tiene **solo** `dataset=electricity`;
2. se la riga `model=timegrad` (Electricity) **manca**, usa i **valori-placeholder**
   da `RESULTS_PLACEHOLDERS_ELECTRICITY.md` e disegna M3 **tratteggiato** con etichetta
   "PLACEHOLDER"; se **c'è**, usa i numeri reali e toglie il marcatore;
3. salva i PNG in `figures/presentation/`.

> Dopo M3 (Fase B) basta **rieseguire lo stesso comando**: trovata la riga reale, le
> figure 1–6 si aggiornano da sole e perdono il marcatore placeholder. Vedi
> [`NEXT_PROMPT_AFTER_M3_IT.md`](NEXT_PROMPT_AFTER_M3_IT.md).
