# CHE_COSA_SONO — notebooks

> Jupyter notebook: l'**EDA** che gira in locale e i notebook **Colab** che allenano i
> modelli pesanti (M2/M3/M4) sulla GPU. M2/M3 usano lo stack pinnato (GluonTS/PyTorchTS);
> M4 TimeDiff gira sul torch "stock" di Colab, senza pin.

| File | Che cos'è |
|---|---|
| `01_eda_exchange.ipynb` | Analisi esplorativa del dataset Exchange: serie, autocorrelazione, volatilità, statistiche → figure in `figures/` e stats in `results/eda_exchange_stats.json`. |
| `colab_m2_m3.ipynb` | Notebook **Colab**: installa il combo pinnato (torch + gluonts + pytorchts) e allena **M2 DeepAR** e **M3 TimeGrad** su GPU. |
| `colab_m3.ipynb` | Variante snella per il solo **M3 TimeGrad**. |
| `colab_electricity_all.ipynb` | Notebook **Colab** per **chiudere Electricity**: Parte 1 = **M4 TimeDiff** su torch "stock" (niente pin, niente GluonTS); Parte 2 (opzionale) = sweep orizzonte E2 (M2/M3) con stack bloccato in un runtime pulito. Con cella SMOKE, BLOCCO DI RECUPERO e backup su Drive. |

> Perché Colab: lo stack pesante (GluonTS/PyTorchTS) e la GPU non sono disponibili in
> locale; M0/M1 e gli esperimenti leggeri girano invece sul portatile.
