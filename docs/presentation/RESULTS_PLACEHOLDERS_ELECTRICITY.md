# Risultati Electricity — tabella unica e placeholder M3

> **Fonte verità numerica della presentazione.** Tutti gli altri file
> (`SLIDE_TEMPLATE`, `SPEAKER_SCRIPT`, figure) leggono da qui.
> M0–M2 sono **numeri veri** (da `results/registry.csv`). M3 è **PLACEHOLDER**.
>
> 🔴 **`[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`**
>
> **Ultimo aggiornamento:** 2026-06-01 · split test, D=321, H=168, τ=24, m=24, S=100, seed=42.

---

## 1. Tabella principale (Electricity, split test)

| Modello | MASE ↓ | CRPS ↓ | pinball ↓ | cov50 (→0.50) | cov90 (→0.90) | MAE ↓ | RMSE ↓ | fit | predict | note |
|---|---|---|---|---|---|---|---|---|---|---|
| **M0** seasonal-naive | **1.0025** | **160.5** | 84.6 | 0.497 | 0.885 | 203.0 | 1 759.7 | — | ~58.7 min | baseline stagionale, *fortissimo* |
| **M1** ARIMA | 3.6638 | 867.5 | 459.0 | 0.592 | 0.930 | 1 131.9 | 12 870.6 | ~29.3 min | ~2h 42min | auto-ARIMA per-canale fragile su 321 serie |
| **M2** DeepAR | 1.8299 | 253.7 | 133.9 | 0.466 | 0.831 | 356.0 | 2 448.8 | ~10.1 min | ~82.2 min | deep probabilistico, 50 epochs |
| **M3** TimeGrad 🔴 | **1.35–1.70** | **185–235** | 95–125 | 0.46–0.52 | 0.84–0.90 | 270–330 | 2 000–2 600 | ~20–40 min | ~2–4h | **PLACEHOLDER**, diff_steps 100, 50 epochs |

🔴 **La riga M3 è `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`.**
Valori-punto usati nelle figure placeholder (centro degli intervalli):
MASE **1.53**, CRPS **210**, pinball **110**, cov50 **0.49**, cov90 **0.87**, MAE **300**,
RMSE **2300**, fit **~30 min**, predict **~3h (≈10 800 s)**.

### Tempi di inferenza in secondi (per lo scatter qualità/costo)

| Modello | predict_s | fit_s |
|---|---|---|
| M0 | 3 520.5 | — |
| M1 | 9 719.5 | 1 758.5 |
| M2 | 4 930.0 | 608.8 |
| M3 🔴 | ~10 800 (placeholder) | ~1 800 (placeholder) |

---

## 2. Le due "barre da battere"

- **CRPS = 160.5** (M0) → la soglia di **qualità probabilistica** che ogni modello più
  complesso deve superare per giustificarsi. DeepAR (253.7) **non** ci riesce.
- **MASE = 1.00** (M0) → la soglia di **accuratezza puntuale** (per costruzione il
  seasonal-naive ha MASE ≈ 1; un modello utile sta sotto 1).

> Frase-chiave della presentazione: *"Su Electricity il baseline stagionale è
> sorprendentemente forte; la domanda non è se un diffusion model è 'moderno', ma se
> il suo CRPS scende sotto 160.5 a un costo accettabile."*

---

## 3. Tre scenari interpretativi (da attivare quando arriva M3)

Quando la riga reale M3 entra nel registry, **scegliere UNO** di questi tre testi e
inserirlo nello speaker script e nelle note di slide 8–9. Gli altri due si eliminano.

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
3. **Sostituire** la riga 🔴 M3 nella tabella §1 (e in §1 "tempi in secondi") con i
   valori reali; **rimuovere** la dicitura `[PLACEHOLDER M3 …]` da questo file e dagli
   altri (`SLIDE_TEMPLATE`, `SPEAKER_SCRIPT`).
4. **Rigenerare le figure**: `python3 -m experiments.plot_presentation`
   (lo script legge il registry: trovata la riga M3 reale, smette di usare il placeholder).
5. **Scegliere lo scenario** A/B/C in base alle condizioni §3 e incollare il testo
   corrispondente nelle note di slide 8–9 e nel blocco "Componente C" dello speaker script.
6. **Aggiornare il callout** di slide 8/9 se la "barra da battere" 160.5 viene superata o no.
7. Eseguire la checklist finale in `NEXT_PROMPT_AFTER_M3_IT.md`.
