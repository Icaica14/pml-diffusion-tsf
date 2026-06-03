# CHE_COSA_SONO — tests

> **Test automatici** (pytest). Verificano le proprietà che devono valere sempre: se una
> modifica le rompe, te ne accorgi subito invece che all'orale.

| File | Che cos'è |
|---|---|
| `test_data_contract.py` | Invarianti del contratto dati: niente leakage train→test, shape corrette di split e finestre, scaling fittato solo sul train. |
| `test_economic.py` | Correttezza dell'LP di dispatch della batteria (E6) su casi calcolabili a mano. |

> Si lanciano dalla radice del progetto con `pytest`. I test che richiedono lo stack
> pesante (GluonTS/torch) si auto-skippano se le dipendenze non sono installate in locale.
