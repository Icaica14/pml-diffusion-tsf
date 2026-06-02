# Valore economico (E6) — Electricity · M0 seasonal-naive

> Generato da `experiments/run_economic.py` (dispatch batteria, plan §6.4). **Demo M0 in locale** — il confronto fra modelli (CRPS-vs-€) arriva con gli array di forecast di M1/M2/M3.

_dispatch senza export · batteria 4h @ 0.50·media, η=0.9 · tariffa TOU picco/spalla/notte = 0.3/0.15/0.08 · 6 canali (topvar) × 120 finestre · τ=24_

Bolletta più bassa = meglio. **denaro risparmiato** = bolletta(naive) − bolletta(piano); **% del max** = frazione del risparmio ottenibile dall'oracolo (previsione perfetta); **Δdist** = bolletta(deterministico) − bolletta(stocastico) = valore della distribuzione.

| Canale | naive | oracolo | det | risparmio det | % del max | sto | Δdist |
|---|---|---|---|---|---|---|---|
| 83 | 84,056.8 | 81,487.4 | 82,224.8 | 1,831.9 | 71% | 82,214.4 | 10.464 |
| 90 | 125,148.6 | 115,343.7 | 115,445.1 | 9,703.5 | 99% | 115,440.1 | 5.009 |
| 105 | 61,917.3 | 60,129.5 | 62,154.8 | -237.4 | -13% | 62,533.6 | -378.8 |
| 106 | 71,019.5 | 67,409.3 | 70,101.1 | 918.4 | 25% | 70,297.7 | -196.6 |
| 294 | 368,598.7 | 344,504.5 | 344,636.7 | 23,962.0 | 99% | 344,573.9 | 62.751 |
| 303 | 202,749.8 | 185,630.6 | 186,192.6 | 16,557.2 | 97% | 185,993.9 | 198.7 |
| **TOTALE** | **913,490.6** | **854,504.9** | **860,755.1** | **52,735.6** | **89%** | **861,053.5** | **-298.4** |
