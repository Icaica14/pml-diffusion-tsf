# Forecasting probabilistico con modelli di diffusione — sintesi di una pagina

**Esame:** Probabilistic Machine Learning — Università di Trieste, Prof. Luca Bortolussi.
**Gruppo:** Giovanni Mason (SM3800158), Lorenzo Di Bernardo (SM3800132), Lorenzo Karol Gobbo (SM28A00018).
**Repo:** `github.com/Icaica14/pml-diffusion-tsf`

## Domanda
Volevamo capire se un modello di diffusione, usato per fare forecasting *probabilistico*
(stimare una distribuzione del futuro, non un solo numero), dia previsioni più accurate o
meglio calibrate rispetto a baseline classiche e deep, e a quale costo di calcolo. Abbiamo
formulato tutto come stima di `p(futuro | passato)`.

## Dati
Due dataset, stessa pipeline (cambia solo il file di configurazione):

- **Exchange** (8 serie, tassi di cambio): lo usiamo come controllo. È vicino a un random
  walk, quindi ci aspettiamo che i modelli complessi non aiutino; serve a verificare che la
  pipeline non stia "barando".
- **Electricity** (321 serie orarie): è il dataset su cui ragioniamo davvero, con forte
  stagionalità giornaliera e settimanale. Contesto `H=168` ore, orizzonte `τ=24` ore, 100
  campioni per stimare la distribuzione, seed 42.

Abbiamo usato il file grezzo LSTNet con un nostro split 70/10/20, non lo split del benchmark
`electricity_nips`. Di conseguenza i nostri CRPS sono un confronto interno tra i nostri
modelli, non un confronto diretto con i numeri pubblicati nei paper.

## Modelli (scala a complessità crescente)
- **M0** seasonal-naive — il futuro ripete il ciclo stagionale; nessun addestramento.
- **M1** ARIMA — un modello lineare per ciascuna serie.
- **M2** DeepAR — rete ricorrente probabilistica, addestrata su tutte le serie insieme.
- **M3** TimeGrad — diffusione condizionata, autoregressiva (genera un passo alla volta).
- **M4** TimeDiff — diffusione non-autoregressiva (denoisa l'intero blocco futuro in una
  volta), più una variante **M4ε** che predice il rumore invece del segnale (ablazione).

## Metriche
MASE (accuratezza puntuale, 1 ≈ seasonal-naive), **CRPS** (qualità dell'intera distribuzione
predittiva), copertura e ampiezza degli intervalli al 50% e 90% (calibrazione), e i tempi di
fit e di campionamento.

## Tre risultati
1. Su Electricity, nel nostro setting, nessun modello addestrato batte il seasonal-naive sul
   CRPS (160.5): DeepAR 253.7, TimeGrad 241.6, TimeDiff 287.3, ARIMA 867.5. Con questo
   protocollo e questo budget, la baseline stagionale è difficile da superare.
2. Il modello meglio calibrato è DeepAR, non una diffusione. TimeGrad batte DeepAR sul CRPS
   ma resta sopra il naive e costa molto in campionamento (circa 4 ore).
3. Con TimeDiff la parametrizzazione cambia molto il risultato: in forma x0 la distribuzione
   predittiva collassa (copertura ≈ 0, e il CRPS coincide quasi con la MAE); la variante ε non
   risolve il problema ma lo ribalta nell'eccesso opposto (copertura ≈ 1). Una spiegazione
   plausibile è che sul blocco non-autoregressivo né x0 né ε "nudi" producano una varianza
   ben calibrata.

## Limiti dichiarati
Un solo seed; budget di calcolo modesto (50 epoche, 100 step di diffusione, poco tuning); lo
split è nostro e non quello del benchmark; la parte sul valore economico (E6) è solo una demo
su M0. Gli esperimenti E0 (split ufficiale), E2 (orizzonte), E3 (numero di step) ed E4
(regime-shift) sono predisposti o in corso, non risultati finali.

## Cosa mandiamo al professore
Le slide in PDF, il report (`REPORT.md`, oppure `REPORT.pdf`), il link al repository GitHub e
questa sintesi.
