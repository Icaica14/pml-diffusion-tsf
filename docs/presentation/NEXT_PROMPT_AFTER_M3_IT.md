# Fase B — prompt pronto da incollare quando M3 finisce

> **A cosa serve.** Quando il run TimeGrad su Electricity è finito e la riga è in
> `results/registry.csv`, **incolla il blocco qui sotto** in chat (a me, Claude). Io
> eseguo i passi e tolgo tutti i placeholder. In alternativa, se la VM Colab è morta,
> incolla il "BLOCCO DI RECUPERO" stampato dal notebook (`notebooks/colab_m3.ipynb`,
> cella ## 7) — contiene la riga CSV a piena precisione e basta a ricostruire tutto.

---

## ✂️ PROMPT DA INCOLLARE (Fase B)

```
M3 TimeGrad su Electricity è FINITO. Aggiorna la presentazione dalla Fase A alla Fase B.

CONTESTO
- Sorgente di verità numerica: docs/presentation/RESULTS_PLACEHOLDERS_ELECTRICITY.md
- I numeri reali M3 sono in results/registry.csv (riga dataset=electricity, model=timegrad).
  [Se la incollo a mano, eccola / oppure incollo il BLOCCO DI RECUPERO del notebook.]

COSA DEVI FARE (in ordine)
1. Leggi results/registry.csv e individua la riga electricity + timegrad. Estrai:
   MASE, CRPS, pinball, cov50, cov90, MAE, RMSE, width50, width90, fit_s, predict_s,
   diff_steps, epochs.
2. In docs/presentation/RESULTS_PLACEHOLDERS_ELECTRICITY.md:
   - sostituisci la riga 🔴 M3 della tabella §1 con i valori reali;
   - aggiorna la tabella "tempi in secondi" (§1) con fit_s/predict_s reali;
   - rimuovi il blocco "valori-punto placeholder" e tutte le diciture
     [PLACEHOLDER M3 ...] e i 🔴 da QUESTO file.
3. Propaga la rimozione dei placeholder e l'inserimento dei numeri reali in:
   - SLIDE_TEMPLATE_ELECTRICITY_IT.md (slide 5, 8, 9, 10 + tabella mappa);
   - SPEAKER_SCRIPT_8_MIN_IT.md (blocco C: slide 8, 9, 10);
   - FIGURE_PLAN_ELECTRICITY_IT.md (togli i 🔴 dove le figure ora hanno M3 reale);
   - QA_ORALE_ELECTRICITY_IT.md (Q16 e ogni 🔴).
4. SCEGLI LO SCENARIO in base al CRPS reale di M3 (regole in
   RESULTS_PLACEHOLDERS_ELECTRICITY.md §3):
   - CRPS(M3) < 160.5            -> Scenario B (batte anche M0)
   - 160.5 <= CRPS(M3) < 253.7   -> Scenario A (batte DeepAR ma non M0)
   - CRPS(M3) >= 253.7           -> Scenario C (peggio di DeepAR)
   Incolla SOLO la frase dello scenario scelto nello Slide 10 e nel blocco C dello
   speaker script; elimina gli altri due scenari da quei due file (lasciali pure in
   RESULTS_... come riferimento storico).
5. Aggiorna il callout "barra da battere 160.5" nelle slide 8/9: di' esplicitamente se
   M3 la supera o no.
6. Rigenera tabella e figure (entrambe leggono la registry, niente numeri a mano):
   - `python3 -m experiments.make_tables` → aggiorna
     `results/tables/comparison_electricity.md`: la riga M3 esce coi valori reali e
     senza `(parentesi)†`; da lì ricopia i numeri M3 nella tabella di Slide 8.
   - `python3 -m experiments.plot_presentation` → le figure 1–6 in
     figures/presentation/ non hanno più il marcatore "PLACEHOLDER" e il punto/barre M3
     usano i valori reali.
7. Esegui la CHECKLIST FINALE qui sotto e riportami cosa risulta.

VINCOLI
- NON modificare il codice di training. NON rifare run pesanti.
- NON committare e NON fare push senza mia autorizzazione esplicita.
- Rispondi in italiano. Nessun co-autore AI nei commit (se e quando committeremo).
- Se un numero manca o è anomalo (NaN, tempi assurdi), fermati e segnalamelo invece di
  inventare.

OUTPUT CHE VOGLIO DA TE
- la riga M3 reale (valori estratti) in chiaro;
- quale scenario hai scelto e perché (una riga);
- la lista dei file aggiornati;
- conferma che le figure sono state rigenerate senza placeholder;
- l'esito della checklist finale.
```

---

## ✅ Checklist finale "pronti per il deck"

Da spuntare prima di considerare la presentazione chiusa:

- [ ] La riga M3 in `RESULTS_PLACEHOLDERS_ELECTRICITY.md` §1 ha **numeri reali** (niente
      intervalli, niente 🔴).
- [ ] La tabella "tempi in secondi" ha `fit_s`/`predict_s` reali di M3.
- [ ] **Nessuna** occorrenza residua di `PLACEHOLDER M3` o `🔴` nei 6 file di
      `docs/presentation/` (verifica: `grep -rn "PLACEHOLDER M3\|🔴" docs/presentation/`).
- [ ] Slide 8: il callout dice chiaramente se M3 **supera o no** CRPS 160.5.
- [ ] Slide 10 + speaker C: è presente **un solo** scenario (A **o** B **o** C), coerente
      col CRPS reale.
- [ ] `python3 -m experiments.make_tables` eseguito: in `results/tables/comparison_electricity.md`
      la riga M3 è **reale** (niente `(parentesi)†`, niente nota PLACEHOLDER in fondo).
- [ ] `python3 -m experiments.plot_presentation` eseguito senza errori.
- [ ] In `figures/presentation/` le figure `fig_cmp_crps`, `fig_cmp_mase`,
      `fig_cmp_quality_cost`, `fig_cmp_calibration`, `fig_cmp_intervals`, `fig_cmp_cost`
      mostrano M3 **reale** (nessuna scritta "PLACEHOLDER", nessun tratteggio).
- [ ] Lo scatter `fig_cmp_quality_cost.png` colloca M3 in modo coerente col messaggio
      (qualità vs costo).
- [ ] Speaker script blocco C: la frase su TimeGrad usa il **numero reale**, non "run in
      corso".
- [ ] Q&A: Q16 (e ogni 🔴) aggiornata col valore reale.
- [ ] (Opzionale) decidere con l'utente se committare e/o aprire la PR
      `feature/port-ladder-exchange` → `main`.

---

## Se invece il run è fallito o si è interrotto

- Guarda a quale epoca è arrivato nel log del notebook (cella ## 7). TimeGrad **non** ha
  resume: se si è fermato a metà, va **rilanciato** da capo (stesso `colab_m3.ipynb`).
- Finché non c'è una riga `electricity,timegrad` valida nel registry, la presentazione
  **resta in Fase A** con i placeholder: è già consegnabile così (il messaggio "soglia da
  battere 160.5, run in corso" tiene).
- Non serve toccare nient'altro: i 6 file e le figure placeholder sono già coerenti.
