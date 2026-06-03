# Q&A orale — Electricity (PML, Prof. Bortolussi)

> **Domande probabili all'orale + risposte pronte.** Italiano, poche formule, tono
> onesto e non trionfalista. I numeri vengono da
> [`RESULTS_PLACEHOLDERS_ELECTRICITY.md`](RESULTS_PLACEHOLDERS_ELECTRICITY.md).
>
> **Fase B:** M3 (TimeGrad, CRPS 241.6) e M4 (TimeDiff, CRPS 287.3) sono numeri reali —
> niente più "run in corso". M4 ha un *doppio volto* (punto nitido, distribuzione
> collassata): vedi Q8, Q16-bis, Q34-bis. Analisi numerica in
> [`RESULTS_PLACEHOLDERS_ELECTRICITY.md`](RESULTS_PLACEHOLDERS_ELECTRICITY.md) §5.
>
> Categorie: **(I)** probabilistico & metodo · **(II)** modelli & diffusione ·
> **(III)** risultati & interpretazione · **(IV)** metriche & valutazione ·
> **(V)** limiti & scelte · **(VI)** trabocchetti.

---

## (I) Forecasting probabilistico & impostazione

**Q1. Cosa vuol dire "forecasting probabilistico" e perché non basta una previsione puntuale?**
Significa stimare la **distribuzione** del futuro condizionata al passato,
`p(x_{t+1:t+τ} | x_{t-H+1:t})`, non un singolo numero. Serve perché le decisioni reali
(riserva di rete, sbilanciamenti, prezzi) dipendono dal **rischio**: una media non dice
quanto posso sbagliare, una distribuzione sì, attraverso i suoi quantili.

**Q2. Cosa sono H e τ nel vostro setup, e perché quei valori?**
H=168 ore (una settimana) è il contesto osservato; τ=24 ore (un giorno) è l'orizzonte
previsto. H=168 cattura sia il ciclo giornaliero sia quello settimanale; τ=24 è scelto
per allinearci alla prediction length del benchmark `electricity_nips`, così l'orizzonte
è confrontabile con la letteratura anche se lo split è nostro.

**Q3. Perché generate 100 campioni?**
Perché i modelli probabilistici (DeepAR, TimeGrad) non danno una formula chiusa della
distribuzione predittiva: la **approssimiamo campionando**. Con S=100 traiettorie
stimiamo mediana, bande e CRPS con varianza ragionevole; di più costerebbe troppo, di
meno sarebbe rumoroso.

**Q4. Cosa significa che un intervallo è "calibrato"?**
Che la frequenza empirica di copertura corrisponde al livello nominale: se dichiaro una
banda al 90%, nel test circa il 90% dei valori veri deve caderci dentro. Lo misuriamo
con cov50→0.50 e cov90→0.90.

**Q5. Perché due dataset, Exchange ed Electricity?**
Exchange è il **controllo negativo**: è quasi un random walk, dove la persistenza è
imbattibile e i modelli complessi *devono* perdere — serve a validare che la pipeline è
corretta. Electricity è il test vero, con stagionalità forte. Mostrano insieme che il
vincitore dipende dal regime del segnale.

---

## (II) Modelli & diffusione

**Q6. Spiega TimeGrad in una frase.**
È un modello di **diffusione condizionata**: genera il futuro partendo da rumore
gaussiano e ripulendolo passo dopo passo, condizionando ogni passo sul passato riassunto
da una rete ricorrente.

**Q7. Qual è la differenza tra il processo forward e reverse nella diffusione?**
Il **forward** aggiunge rumore gaussiano al dato in molti piccoli passi finché diventa
rumore puro (è fisso, non si impara). Il **reverse** è quello che la rete impara: togliere
rumore un passo alla volta per ricostruire un campione realistico.

**Q8. Perché la rete predice il rumore e non direttamente il dato pulito?**
Perché predire il rumore (ε-prediction) aggiunto a ogni passo rende il problema più
stabile e la loss più semplice (un errore quadratico sul rumore), ed è equivalente a
stimare il gradiente della densità (score). È la formulazione standard dei DDPM (Ho 2020),
ed è quella di TimeGrad. **Attenzione — è il cuore del nostro risultato su TimeDiff:**
TimeDiff invece predice il **segnale pulito** (x0-prediction), ed è proprio questa scelta
che fa collassare la sua incertezza (vedi Q16-bis e Q34-bis). Predire ε controlla
*esplicitamente* la dispersione iniettata a ogni passo; predire x0 la lascia residuale, e
se la rete impara a predire un x0 quasi costante quella dispersione svanisce. La nostra
ablazione **M4ε** lo verifica — ma con un esito **inatteso**: predire ε **non** ricalibra
TimeDiff, lo **ribalta** nell'estremo opposto (cov 0.998/1.000, bande ~7700/11360, MASE 4.02).
Nessuna delle due parametrizzazioni naive (x0 *o* ε) è calibrata sul blocco non-autoregressivo;
il paper TimeDiff sceglie x0 come male minore. Dettaglio completo in Q34-ter.

**Q9. Come si condiziona TimeGrad sul passato?**
Una RNN comprime la finestra di contesto H in uno stato; quello stato entra come
condizione nel denoiser a ogni passo di diffusione. Così il campione generato è coerente
con la storia recente di quella serie.

**Q10. Differenza chiave tra DeepAR e TimeGrad?**
DeepAR assume una **forma parametrica** della distribuzione (es. gaussiana/Student per
canale) e la fa evolvere autoregressivamente. TimeGrad **non** assume una forma: la
costruisce per campionamento via diffusione, quindi può rappresentare incertezza non
gaussiana e correlazioni più ricche — al prezzo di un sampling molto più lento.

**Q10-bis. Differenza tra TimeGrad (M3) e TimeDiff (M4)?**
Sono **entrambi** diffusion model condizionati, ma con design opposto. **TimeGrad è
autoregressivo**: genera il futuro un passo temporale alla volta, ogni passo condizionato
sullo stato di una RNN — espressivo ma con sampling lento (τ catene di denoising in
sequenza, ~4h19). **TimeDiff è non-autoregressivo**: denoisa l'**intero blocco futuro**
(τ×D) in *una sola* catena inversa, con un backbone convoluzionale e un'inizializzazione
lineare del futuro (`x_ar`). Risultato: sampling molto più rapido (~41 min). Inoltre
TimeGrad usa ε-prediction, TimeDiff x0-prediction — differenza che spiega la calibrazione
opposta (vedi Q8, Q34-bis, Q34-ter).

**Q11. Perché ARIMA è "un modello per canale"?**
ARIMA è univariato: con 321 serie alleniamo 321 modelli auto-ARIMA indipendenti. Questo
lo rende fragile e costoso, e ignora la struttura condivisa tra serie che invece DeepAR e
TimeGrad sfruttano.

**Q12. Cos'è il seasonal-naive esattamente?**
La previsione "il futuro è uguale al ciclo stagionale precedente": per dati orari con
stagionalità giornaliera, il valore previsto per un'ora è quello della stessa ora del
ciclo precedente (season_length=24). Niente addestramento.

**Q13. Avete usato un toy DDPM?**
È previsto come artefatto **didattico** (1-D, per mostrare rumore→segnale all'orale), non
come modello di risultati. Se chiesto, lo usiamo per spiegare l'intuizione, non per i
numeri.

---

## (III) Risultati & interpretazione

**Q14. Qual è il risultato principale su Electricity?**
Che il **seasonal-naive è fortissimo**: CRPS 160.5, MASE 1.00. ARIMA va male (CRPS 867.5),
DeepAR è il miglior modello addestrato (CRPS 253.7) ma **non** batte il naive. La domanda
è se la diffusione scende sotto 160.5 a un costo accettabile.

**Q15. Perché il seasonal-naive è così difficile da battere qui?**
Perché Electricity ha una stagionalità giornaliera/settimanale molto regolare: gran parte
del segnale è "domani come il ciclo precedente". Il baseline cattura gratis proprio quella
struttura; ai modelli resta solo il margine residuo, più difficile.

**Q16. Quanto fa TimeGrad?**
CRPS **241.6**, MASE **1.39**. **Batte DeepAR** (CRPS 253.7) ma **non** il seasonal-naive
(160.5): è lo **Scenario A**. La calibrazione è sovra-confidente (cov50 0.279, cov90 0.647)
e il sampling è costoso (~4h 19min su L4). Lettura: più espressività aiuta sul
probabilistico, ma non basta a scalzare il baseline, e si paga in costo.

**Q16-bis. Quanto fa TimeDiff (M4), e perché quel risultato è interessante?**
Ha un **doppio volto**. Come *previsione puntuale* è il migliore dei deep: MAE **288.5**
(il più basso; M2 356, M3 312.8), MASE 1.40 ≈ TimeGrad, e lo ottiene al costo più basso
(fit ~48 s, predict ~41 min, contro le ~4h19 di TimeGrad — è il vantaggio del non-AR). **Ma
la sua distribuzione predittiva è collassata**: cov50 0.003, cov90 0.008 (nominali 0.50 e
0.90), ampiezze ≈ 0. Spia decisiva: **CRPS 287.3 ≈ MAE 288.5** — quando la predittiva è una
*massa puntiforme* il CRPS si riduce alla MAE. Non è un bug: è il **collasso di varianza
della x0-prediction** (la rete predice un x0 quasi costante e ignora il rumore; la
future-mixup aggrava). È interessante perché isola un fenomeno didattico pulito: *la
parametrizzazione del target (x0 vs ε) cambia la calibrazione a parità di tutto il resto.*

**Q17. E se TimeGrad non batte il seasonal-naive?**
È **esattamente il nostro caso** (Scenario A): un esito scientificamente valido, che
dimostra come più espressività non implichi miglior forecasting su un segnale già ben
spiegato dalla stagionalità. La tesi del progetto — il trade-off
espressività/calibrazione/costo — regge comunque.

**Q18. E se invece lo avesse battuto?** (controfattuale)
Sarebbe stato lo scenario B: la diffusione che sfrutta struttura e incertezza non gaussiana
per la miglior qualità probabilistica, col vantaggio sempre da pesare contro il costo di
sampling. Non è ciò che è successo: qui M3 resta sopra M0 sul CRPS (241.6 vs 160.5).

**Q19. Perché DeepAR batte ARIMA così nettamente?**
Perché impara su tutte le serie insieme (condivide struttura), usa covariate di calendario,
ed è probabilistico per costruzione; ARIMA per-canale è fragile, sensibile alla scelta
d'ordine, e non condivide informazione tra serie.

**Q20. La vostra è una storia "diffusion vince"?**
No, ed è voluto. La tesi è il **compromesso** tra espressività, calibrazione e costo. Su un
segnale stagionale un baseline semplice è una soglia seria: il valore del lavoro è il
confronto controllato, non incoronare un vincitore.

---

## (IV) Metriche & valutazione

**Q21. Perché CRPS e non solo MAE/RMSE?**
MAE/RMSE giudicano solo la previsione puntuale. Il CRPS valuta l'intera distribuzione
predittiva: premia un modello che mette massa di probabilità nel posto giusto, non solo che
azzecca la media. È la metrica giusta per il forecasting *probabilistico*.

**Q22. Cosa misura esattamente il MASE e perché è comodo?**
È l'errore assoluto normalizzato dall'errore del naive stagionale in-sample: MASE=1 ≈
seasonal-naive, <1 meglio, >1 peggio. È comodo perché adimensionale e interpretabile a
colpo d'occhio.

**Q23. Cosa sono cov50/cov90 e width50/width90?**
cov50/cov90 = frazione di valori veri dentro le bande al 50%/90% (calibrazione: vicino a
0.50/0.90 è bene). width50/width90 = ampiezza media di quelle bande (sharpness: a parità di
copertura, più stretto è più informativo). Vanno letti **insieme**.

**Q24. Un modello può essere ben calibrato ma inutile?**
Sì: bande larghissime coprono tutto (cov ottima) ma non dicono nulla (width enorme). Per
questo guardiamo copertura **e** ampiezza, e soprattutto il CRPS che le combina.

**Q25. MASE/CRPS sono confrontabili tra Exchange ed Electricity?**
No. Scale e denominatori sono diversi (valute vs kWh, stagionalità diverse). Confrontiamo i
modelli **dentro** ciascun dataset, mai i numeri tra dataset diversi.

**Q26. Avete usato la "point-adjusted F1" o metriche gonfiate?**
No. Questo è un compito di **forecasting**, valutato con CRPS/pinball/coverage/MASE: niente
aggiustamenti che gonfiano i punteggi. È un punto di onestà metodologica.

---

## (V) Limiti & scelte

**Q27. Perché lo split non è quello del benchmark?**
Per controllo e semplicità abbiamo usato il file grezzo LSTNet con split 70/10/20 nostro
(caveat E0). Conseguenza: i CRPS **non** sono direttamente confrontabili coi numeri
pubblicati di TimeGrad/CSDI. È un confronto interno, a parità di condizioni tra i nostri
modelli. Lavoro futuro: adottare `electricity_nips`.

**Q28. Un solo seed: non è poco?**
Sì, è un limite dichiarato. Con un seed non quantifichiamo la varianza da inizializzazione.
Per una pubblicazione servirebbero più seed; qui, per budget di calcolo, abbiamo fissato
seed=42 uguale per tutti, che almeno rende il confronto equo.

**Q29. Avete fatto tuning degli iperparametri?**
Tuning minimo: configurazioni ragionevoli e fisse (diff_steps=100, 50 epoche per i deep su
Electricity). Non abbiamo fatto ricerca estesa: un altro limite onesto, e una possibile
ragione se M3 non brilla.

**Q30. Perché Electricity e non un dataset con anomalie reali?**
Perché l'obiettivo del corso è il **forecasting probabilistico** con diffusione, e
Electricity è uno standard con stagionalità ricca. (Le anomalie reali sono un altro
progetto.)

**Q31. Il costo di TimeGrad è un problema pratico?**
Sì: il sampling richiede molti passi di denoising per ciascuno dei 100 campioni → predict
nell'ordine delle ore su 321 serie. È proprio il costo che mettiamo nello scatter
qualità/costo. Mitigazioni future: meno passi (DDIM-like).

---

## (VI) Trabocchetti / domande "cattive"

**Q32. "Quindi avete fatto tanto lavoro per scoprire che il naive vince?"**
Il valore è il **confronto controllato** e la quantificazione del trade-off. Scoprire che un
baseline semplice è una soglia seria su un segnale stagionale è un risultato utile e
non ovvio a priori, e mette in guardia dall'usare modelli costosi per riflesso.

**Q33. "La diffusione non è overkill per delle serie orarie?"**
Può esserlo, e parte della risposta è proprio nel grafico costo/qualità. Il punto del
progetto è misurarlo, non assumerlo: mostriamo *quando* l'espressività in più si ripaga e
quando no.

**Q34. "Come escludete che TimeGrad sia solo sotto-addestrato?"**
Non lo escludiamo del tutto: con un solo seed e tuning limitato è una causa possibile, e lo
diciamo. Mitighiamo usando le stesse 50 epoche/condizioni di DeepAR per equità; un'analisi
più estesa (più epoche/seed) è lavoro futuro.

**Q34-bis. "TimeDiff ha coverage praticamente zero: il modello è rotto o il codice è buggato?"**
Né l'uno né l'altro nel senso banale: il *point forecast* è ottimo (MAE la più bassa tra i
deep), quindi training e pipeline funzionano. È la **distribuzione** a essere degenerata, e
la causa è precisa e attesa: la **x0-prediction**. La rete impara a predire un segnale
pulito quasi costante appoggiandosi al contesto e all'init lineare, ignorando il rumore in
ingresso; così l'unica varianza residua nella catena inversa è quella dell'ultimo passo,
`1−ᾱ_0 ≈ 6·10⁻⁴`, cioè ≈ 0. La future-mixup (che in training mescola il futuro vero nel
condizionamento) lo aggrava. Lo abbiamo **verificato numericamente** (simulazione della
ricorsione di varianza) e con la spia indipendente CRPS ≈ MAE. La cura è la formulazione
DDPM standard, ε-prediction: la nostra ablazione **M4ε** la testa — e il verdetto è in
Q34-ter (ε **non** cura, ribalta). Quindi non è un bug del sampler — la matematica DDIM è
corretta — ma una proprietà della parametrizzazione.

**Q34-ter. "E l'ablazione ε ha funzionato? Ha ricalibrato TimeDiff?"**
No — ed è il risultato più interessante. Predire ε **non** ricalibra: **ribalta** TimeDiff
dall'estremo della sotto-dispersione (cov 0.003/0.008, bande ≈ 0) a quello opposto della
**sovra-dispersione** (cov **0.998/1.000**, width50 ≈ 7686, width90 ≈ 11359 — le bande più
larghe di tutta la scala), e degrada anche il punto (MASE **4.02** vs 1.40, MAE 1045 vs 288,
CRPS 1376 vs 287). Non è un bug: il rapporto RMSE/MAE resta uniforme (ε 8.97 ≈ x0 8.85 ≈
naive 8.67), quindi è sovra-dispersione *uniforme*, non qualche finestra esplosa; le formule
sono i DDPM da manuale. Il meccanismo: senza ancoraggio autoregressivo, il blocco non-AR
sparge la varianza iniettata da ε su tutte le τ×D celle senza ricomporla → varianza fuori
controllo. Lettura onesta: **nessuna parametrizzazione naive calibra** un blocco non-AR; x0
è il **male minore** (la scelta del paper TimeDiff); calibrare davvero richiede di più
(varianza appresa σ_θ, o conformal). E il meglio calibrato dell'intera scala **non** è una
diffusione: è **DeepAR** (cov 0.466/0.831).

**Q35. "Perché dovrei fidarmi della calibrazione con un solo seed e split non standard?"**
Non chiediamo fiducia assoluta: presentiamo coverage **e** width insieme, dichiariamo i
limiti (E0, un seed) e inquadriamo i numeri come confronto interno. È trasparenza, non
sovra-affermazione.

**Q36. "Qual è, in una frase, il contributo del lavoro?"**
Un confronto **onesto e controllato** lungo una scala di complessità (naive→ARIMA→DeepAR→
diffusione) su un segnale stagionale reale, che quantifica il compromesso
qualità-probabilistica ↔ costo invece di dichiarare un vincitore.

---

### Suggerimenti di consegna per il Q&A
- Numeri M3 (a memoria): CRPS **241.6**, MASE **1.39** — batte DeepAR, non il naive
  (Scenario A); sampling costoso (~4h 19min).
- Numeri M4 (a memoria): CRPS **287.3**, MASE **1.40**, MAE **288.5** (la più bassa tra i
  deep), cov **0.003 / 0.008**; il più veloce (fit ~48 s, predict ~41 min). Frase-gancio:
  *"punto nitido, distribuzione collassata; CRPS ≈ MAE = massa puntiforme; colpa della
  x0-prediction — e l'ablazione ε non la corregge: la ribalta (cov ≈ 1). x0 ed ε = i due
  estremi opposti, nessuno calibrato."*
- Tenere pronte le slide **backup B1–B7** (calibrazione, tabella completa, perché ARIMA va
  male, math della diffusione, Exchange, ablation DeepAR, **B7 = TimeDiff & ablazione ε**) →
  mappa in [`SLIDE_TEMPLATE_ELECTRICITY_IT.md`](SLIDE_TEMPLATE_ELECTRICITY_IT.md).
- Ripetere il messaggio-chiave a ogni occasione: **trade-off**, non vittoria; e per M4, *il
  design e la parametrizzazione contano quanto la taglia*.
