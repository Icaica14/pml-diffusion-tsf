# Checklist di consegna — esame PML (Prof. Bortolussi, UniTS)

**Appello: 10 giugno 2026.**
**Gruppo:** Giovanni Mason (SM3800158) · Lorenzo Di Bernardo (SM3800132) · Lorenzo Karol Gobbo (SM28A00018).
**Repo:** `github.com/Icaica14/pml-diffusion-tsf`

Questa è l'ultima cosa da controllare prima di consegnare. Spuntare ogni riga; se una riga
non si spunta, sistemarla prima di inviare.

---

## Materiale da mandare al professore

- [ ] **Slide in PDF** — `slides/deck_electricity_it.pdf`.
- [ ] **Report** — `REPORT.md` (oppure `REPORT.pdf`, se il professore preferisce un PDF).
- [ ] **Link al repository GitHub** — `Icaica14/pml-diffusion-tsf`.
- [ ] *(facoltativo)* **Sintesi di una pagina** — `SINTESI_PROGETTO.md`.
- [ ] *(facoltativo)* **Slide in PPTX** — `slides/deck_electricity_it.pptx`, se serve
      modificarle dal vivo o usare le note nel pannello note.

## Prima dell'invio

- [ ] **Appello giusto:** 10 giugno 2026.
- [ ] **Nomi e matricole** corretti su slide, report e sintesi (Giovanni Mason SM3800158,
      Lorenzo Di Bernardo SM3800132, Lorenzo Karol Gobbo SM28A00018).
- [ ] **Link GitHub** funzionante e branch corretto (`feature/port-ladder-exchange`, o `main`
      se nel frattempo è stato fatto il merge).
- [ ] **Nessun nome-segnaposto nelle slide:** aprire il PDF e controllare a occhio la slide
      titolo (ci sono i tre nomi reali con le matricole).
- [ ] **Aprire il PDF delle slide** e scorrerlo per intero (anche le backup B1–B7).
- [ ] **Aprire il PPTX** e verificare che le note del relatore ci siano nel pannello note.
- [ ] **Provare l'esposizione in 8–10 minuti** con `SCRIPT_PRESENTAZIONE.md`, cronometrando.
- [ ] **Decidere chi dice cosa:** la divisione attuale è A = Giovanni, B = Lorenzo Di
      Bernardo, C = Lorenzo Karol Gobbo (nello speaker script e nella tabella ruoli del
      README). Confermarla o scambiarla.
- [ ] **Q&A ripassato** — `DOMANDE_ORALE.md`.

## Da presentare come limiti / lavoro futuro, non come risultati

- [ ] **E0** (`electricity_nips`, split ufficiale + CRPS-sum): predisposto ma non eseguito
      (serve GPU). Il confronto con i numeri pubblicati resta lavoro futuro.
- [ ] **E2** (sweep sull'orizzonte) ed **E3** (sweep sul numero di step di diffusione): in
      corso, non finalizzati.
- [ ] **E4** (regime-shift / drift): pianificato, non eseguito.
- [ ] **Toy DDPM 1-D:** è un esempio didattico per spiegare rumore→segnale all'orale, non un
      modello dei risultati.
- [ ] **Un solo seed** (42): niente stima della varianza da inizializzazione.
- [ ] **E6** (valore economico): è una demo su M0; il confronto del valore tra modelli è
      lavoro futuro.

## Nota tecnica (export e rigenerazione, se servono)

```bash
# Report in PDF (richiede pandoc + xelatex)
pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex -V mainfont="Arial Unicode MS"

# Ricostruire le slide dopo una modifica al sorgente
bash docs/presentation/build_deck.sh

# Rigenerare tabelle e figure dal registry (i numeri non si scrivono a mano)
python3 -m experiments.make_tables
python3 -m experiments.plot_presentation
```

## Stato del repository

- Le modifiche di questa fase (matricole, sintesi di una pagina, ritocchi ai testi, cartella
  `consegna_finale/`) sono nel working tree e **non ancora committate**: rivederle con
  `git status` / `git diff` e committarle quando siete d'accordo.
