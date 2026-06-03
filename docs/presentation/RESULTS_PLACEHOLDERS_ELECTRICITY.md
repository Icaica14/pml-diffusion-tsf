# Risultati Electricity — tabella unica (M0–M4 reali)

> **Fonte verità numerica della presentazione.** Tutti gli altri file
> (`SLIDE_TEMPLATE`, `SPEAKER_SCRIPT`, figure) leggono da qui.
> M0–M4 sono **numeri veri** (da `results/registry.csv`).
>
> **Ultimo aggiornamento:** 2026-06-03 · split test, D=321, H=168, τ=24, m=24, S=100, seed=42.
> M4 TimeDiff aggiunto coi valori reali del run Colab (riga `electricity,timediff`):
> point forecaster nitido **ma** distribuzione collassata (vedi §1.1 e §5).

---

## 1. Tabella principale (Electricity, split test)

| Modello | MASE ↓ | CRPS ↓ | pinball ↓ | cov50 (→0.50) | cov90 (→0.90) | MAE ↓ | RMSE ↓ | fit | predict | note |
|---|---|---|---|---|---|---|---|---|---|---|
| **M0** seasonal-naive | **1.0025** | **160.5** | 84.6 | 0.497 | 0.885 | 203.0 | 1 759.7 | — | ~58.7 min | baseline stagionale, *fortissimo* |
| **M1** ARIMA | 3.6638 | 867.5 | 459.0 | 0.592 | 0.930 | 1 131.9 | 12 870.6 | ~29.3 min | ~2h 42min | auto-ARIMA per-canale fragile su 321 serie |
| **M2** DeepAR | 1.8299 | 253.7 | 133.9 | 0.466 | 0.831 | 356.0 | 2 448.8 | ~10.1 min | ~82.2 min | deep probabilistico, 50 epochs |
| **M3** TimeGrad | 1.391 | 241.6 | 126.6 | 0.279 | 0.647 | 312.8 | 3 025.3 | ~38.9 min | ~4h 19min | diffusione condizionata, diff_steps 100, 50 epochs |
| **M4** TimeDiff (x0) | 1.399 | 287.3 | 143.8 | **0.003** | **0.008** | **288.5** | 2 551.5 | **~0.8 min** | **~41.4 min** | diffusione **non-AR**: il più veloce; *ma* intervalli collassati |

La riga M3 riporta i **valori reali** dal registry (run TimeGrad su Colab L4, 2026-06-01):
CRPS **241.6** — sopra la barra M0 (160.5) ma sotto DeepAR (253.7); calibrazione
**sovra-confidente** (cov50 0.279, cov90 0.647); sampling costoso (~4h 19min su L4).

### 1.1 M4 TimeDiff — il punto nitido, la distribuzione collassata (caveat onesto)

M4 è il run reale di **TimeDiff** (Shen & Kwok, ICML 2023), diffusione **non-autoregressiva**
che denoisa l'intero blocco futuro (τ×D) in una sola catena inversa. Due fatti, opposti:

- **Vince come point forecaster.** MAE **288.5** è il più basso tra i modelli deep
  (M2 356.0, M3 312.8); MASE 1.399 ≈ M3 (1.391). E lo fa **costando pochissimo**:
  fit ~48 s e predict ~41 min, contro le ~4h19 di M3 — è il **payoff del non-AR** (una
  catena inversa invece di τ passi autoregressivi).
- **Ma la distribuzione predittiva è degenerata.** cov50 **0.003** e cov90 **0.008**
  (nominali 0.50 / 0.90): gli intervalli coprono ~0. Le ampiezze sono ~0 in scala originale
  (width50 ≈ 2.8, width90 ≈ 6.6, contro le centinaia/migliaia degli altri). **Spia decisiva:**
  CRPS **287.3 ≈ MAE 288.5** — quando la predittiva è una *massa puntiforme*, il CRPS si
  riduce alla MAE. Non è un bug del sampler: è il **collasso di varianza della
  x0-prediction** (la rete predice un x0 quasi costante appoggiandosi al contesto/`x_ar`
  e ignorando il rumore `x_t`; la *future-mixup* lo aggrava insegnando a copiare il
  condizionamento deterministico). Analisi numerica in §5.

> **Lettura da slide:** *"TimeDiff ci dà il forecast puntuale più nitido al costo più
> basso, ma la sua incertezza è rotta: gli intervalli collassano a zero. È l'esempio da
> manuale di overconfidence — e la ragione per cui la parametrizzazione conta."*

**Ablazione M4ε (eps-prediction) — fatta:** abbiamo rieseguito TimeDiff predicendo il
**rumore** ε invece del segnale pulito x0 (DDPM standard, Ho 2020). Tesi iniziale: ε
controlla esplicitamente la varianza iniettata e dovrebbe **ricalibrare**. **Esito reale:
ε non ricalibra, ribalta.** TimeDiff passa dalla sotto-dispersione (cov 0.003/0.008) alla
**sovra-dispersione** estrema: cov **0.998/1.000**, width50 **7686**, width90 **11359** (le
bande più larghe della scala), e il punto degrada — MASE **4.02** (vs 1.40), CRPS **1376**
(vs 287), MAE **1045** (vs 288). Spia di consistenza: RMSE/MAE resta uniforme (ε 8.97 ≈ x0
8.85 ≈ naive 8.67) → sovra-dispersione *uniforme*, non finestre esplose: effetto reale, non
bug. Arco rivisto: *"x0 collassa → varianza residua → ε ribalta nell'eccesso opposto: la
calibrazione non è uno switch di parametrizzazione."*

### Tempi di inferenza in secondi (per lo scatter qualità/costo)

| Modello | predict_s | fit_s |
|---|---|---|
| M0 | 3 520.5 | — |
| M1 | 9 719.5 | 1 758.5 |
| M2 | 4 930.0 | 608.8 |
| M3 | 15 526.0 | 2 331.9 |
| M4 | 2 484.7 | 48.0 |

---

## 2. Le due "barre da battere"

- **CRPS = 160.5** (M0) → la soglia di **qualità probabilistica** che ogni modello più
  complesso deve superare per giustificarsi. **Nessun** modello deep ci riesce: DeepAR
  253.7, TimeGrad 241.6 (il più vicino), TimeDiff 287.3 — tutti sopra la barra.
- **MASE = 1.00** (M0) → la soglia di **accuratezza puntuale** (per costruzione il
  seasonal-naive ha MASE ≈ 1; un modello utile sta sotto 1).

> Frase-chiave della presentazione: *"Su Electricity il baseline stagionale è
> sorprendentemente forte; la domanda non è se un diffusion model è 'moderno', ma se
> il suo CRPS scende sotto 160.5 a un costo accettabile."*

---

## 3. Tre scenari interpretativi (da attivare quando arriva M3)

Quando la riga reale M3 entra nel registry, **scegliere UNO** di questi tre testi e
inserirlo nello speaker script e nelle note di slide 8–9. Gli altri due si eliminano.

> ✅ **Scelto: Scenario 🅐 A** — CRPS(M3) = **241.6** ∈ [160.5, 253.7): TimeGrad batte
> DeepAR sul CRPS ma **non** il seasonal-naive M0. (Gli scenari B e C restano qui sotto
> solo come riferimento storico.)

### 🅐 Scenario A — M3 batte DeepAR ma **non** M0
*Condizione:* `CRPS(M3) < 253.7` **e** `CRPS(M3) ≥ 160.5` (probabile *a priori*).

> "TimeGrad migliora la baseline deep probabilistica ma non supera il seasonal-naive.
> Questo suggerisce che la stagionalità deterministica è già catturata molto bene dal
> baseline, mentre la diffusione aggiunge flessibilità ma non abbastanza valore
> predittivo rispetto al costo."

**Take-home:** il valore è *metodologico* (confronto controllato), non una vittoria
del modello. Coerente con la tesi non-trionfalista.

### 🅑 Scenario B — M3 batte **anche** M0 su CRPS/calibrazione
*Condizione:* `CRPS(M3) < 160.5` (e/o coverage più vicina al nominale di M0).

> "Il modello di diffusione riesce a sfruttare struttura multivariata e incertezza non
> gaussiana, ottenendo la migliore qualità probabilistica. Il vantaggio però va letto
> insieme al costo: sampling più lento e maggiore fragilità di tooling."

**Take-home:** vittoria *condizionata* — il diffusion vince sulla qualità probabilistica
ma paga in costo; giustificato solo dove l'incertezza ha valore decisionale.

### 🅒 Scenario C — M3 **peggiora** rispetto a DeepAR
*Condizione:* `CRPS(M3) > 253.7` o instabilità evidente.

> "Il risultato negativo è informativo: maggiore espressività non implica migliore
> forecasting. Possibili cause: training insufficiente, costo del sampling,
> overfitting/underfitting, difficoltà di scalare a D=321, oppure baseline stagionale
> estremamente forte."

**Take-home:** anche questo è un esito *scientificamente valido* da presentare con
onestà; rafforza il messaggio "no free lunch".

> In **tutti e tre** i casi la presentazione regge: la tesi è il *trade-off*
> espressività ↔ calibrazione ↔ costo, non un verdetto su chi "vince".

---

## 4. Come aggiornare quando arriva M3

1. **Leggere** `results/registry.csv` e individuare la riga con
   `dataset=electricity` **e** `model=timegrad`.
2. **Estrarre** MASE, CRPS, pinball, cov50, cov90, MAE, RMSE, fit_s, predict_s,
   diff_steps, epochs.
3. **Sostituire** la riga M3 nella tabella §1 (e in §1 "tempi in secondi") con i
   valori reali; **rimuovere** i segnaposto da questo file e dagli
   altri (`SLIDE_TEMPLATE`, `SPEAKER_SCRIPT`).
4. **Rigenerare le figure**: `python3 -m experiments.plot_presentation`
   (lo script legge il registry: trovata la riga M3 reale, smette di usare il placeholder).
5. **Scegliere lo scenario** A/B/C in base alle condizioni §3 e incollare il testo
   corrispondente nelle note di slide 8–9 e nel blocco "Componente C" dello speaker script.
6. **Aggiornare il callout** di slide 8/9 se la "barra da battere" 160.5 viene superata o no.
7. Eseguire la checklist finale in `NEXT_PROMPT_AFTER_M3_IT.md`.

---

## 5. Perché M4 (x0) collassa — l'analisi di varianza

La spiegazione del caveat §1.1, in forma numerica (riproducibile, niente GPU).

**Setup.** TimeDiff è addestrato in **x0-prediction**: la rete `f_θ(x_t, t, ctx, cond)`
predice il segnale pulito `x0`, con loss `MSE(x0_hat, fut)`. Al campionamento usiamo
DDIM con η=1. La ricostruzione del rumore a ogni passo è
`eps_hat = (x_t − √ᾱ_t · x0_hat) / √(1−ᾱ_t)` e l'update porta `x_{t−1}` verso `x0_hat`.

**Il meccanismo del collasso.** Se la rete impara a predire un `x0_hat` quasi **costante**
rispetto a `x_t` — cioè si appoggia al contesto e all'inizializzazione lineare `x_ar`,
ignorando il rumore in ingresso — allora l'unica sorgente di dispersione residua nella
catena inversa è la varianza dell'ultimo passo, `1 − ᾱ_0`. Con lo schedule **cosine** a
100 step, `1 − ᾱ_0 ≈ 6·10⁻⁴`: in **spazio standardizzato** la std predittiva tende a
`√(1−ᾱ_0) ≈ 0.024`, e dopo la de-standardizzazione resta un intervallo larghissimi-mo
**ordini di grandezza più stretto** del segnale. Da qui cov ≈ 0 e width ≈ 0.

**Conferma numerica.** Simulando la ricorsione di varianza con una rete che ignora `x_t`,
la std predittiva risultante è ~0.00 (massa puntiforme); il valore osservato sui dati
(~0.01 standardizzato) è coerente. **Spia indipendente:** `CRPS 287.3 ≈ MAE 288.5` — per
una predittiva degenerata il CRPS coincide con la MAE. Entrambe puntano allo stesso fatto.

**La future-mixup aggrava.** In training `cond_future = keep·x_ar + (1−keep)·fut`: la rete
impara a *copiare* il futuro condizionante. In inferenza `cond_future = x_ar` è
**deterministico**, quindi rinforza il predire un `x0_hat` rigido — meno varianza, non più.

**La cura (ablazione M4ε).** In **eps-prediction** la rete predice il rumore `ε` e
`x0 = (x_t − √(1−ᾱ_t)·ε_hat)/√ᾱ_t`. Ora la dispersione iniettata a ogni passo è
governata esplicitamente da `ε_hat`, non residuale: è il motivo per cui DDPM standard
(Ho 2020) calibra. Tesi verificabile: M4ε deve **alzare cov50/cov90 verso il nominale** e
**avvicinare CRPS e MAE solo quanto basta** (non farli coincidere).

> Questo è il pezzo "didattico" forte del progetto: non *"il diffusion ha vinto/perso"*,
> ma *"la parametrizzazione del target cambia radicalmente la calibrazione, a parità di
> tutto il resto"* — un risultato controllato e istruttivo.
