# CHE_COSA_SONO — configs

> Una configurazione **YAML per dataset**. Principio "una config = un run": ogni numero da
> cui un modello o un esperimento dipende (contesto H, orizzonte τ, split, scaling, seed,
> periodo stagionale m) vive **qui**, mai hard-coded nel codice (vedi `CONTRIBUTING.md`).
> Cambiare dataset = cambiare il file passato con `--config`.

| File | Che cos'è |
|---|---|
| `data_exchange.yaml` | Dataset **Exchange** (8 valute vs USD, giornaliero, ~7588 passi): il dataset di "iterazione / get-it-green-first". `H=60`, `τ=30`, `m=1` (è quasi un random walk). Si fa girare qui tutta la pipeline prima di toccare il primario. |
| `data_electricity.yaml` | Dataset **Electricity** (321 serie di carico, orario, ~26304 passi): il dataset **primario**, l'headline della presentazione. `H=168` (una settimana di storia), `τ=24` (un giorno avanti), `m=24` (forte ciclo giornaliero). |

> Stesso "contratto dati" per entrambi: i campi (`source`, `split`, `window`, `scaling`,
> `eval.season_length`, `seed`) hanno la stessa struttura, quindi il codice non cambia tra
> un dataset e l'altro.
