# Q&A orale — Electricity (PML, Prof. Bortolussi)

> **Domande probabili all'orale + risposte pronte.** Italiano, poche formule, tono
> onesto e non trionfalista. I numeri vengono da
> [`RESULTS_PLACEHOLDERS_ELECTRICITY.md`](RESULTS_PLACEHOLDERS_ELECTRICITY.md).
>
> 🔴 Dove c'è M3, in Fase A si risponde "run in corso"; in Fase B si inserisce il
> numero reale. `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
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
Perché predire il rumore aggiunto a ogni passo rende il problema più stabile e la loss
più semplice (un errore quadratico sul rumore), ed è equivalente a stimare il gradiente
della densità (score). È la formulazione standard dei DDPM.

**Q9. Come si condiziona TimeGrad sul passato?**
Una RNN comprime la finestra di contesto H in uno stato; quello stato entra come
condizione nel denoiser a ogni passo di diffusione. Così il campione generato è coerente
con la storia recente di quella serie.

**Q10. Differenza chiave tra DeepAR e TimeGrad?**
DeepAR assume una **forma parametrica** della distribuzione (es. gaussiana/Student per
canale) e la fa evolvere autoregressivamente. TimeGrad **non** assume una forma: la
costruisce per campionamento via diffusione, quindi può rappresentare incertezza non
gaussiana e correlazioni più ricche — al prezzo di un sampling molto più lento.

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

**Q16. Quanto fa TimeGrad?** 🔴
*Fase A:* "Il run è in corso su GPU; inseriamo il numero reale appena finisce. Sappiamo
già la soglia da superare: CRPS 160.5." *Fase B:* inserire CRPS reale e lo scenario A/B/C.
`[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`

**Q17. E se TimeGrad non batte il seasonal-naive?**
È un esito **scientificamente valido e atteso**: dimostra che più espressività non implica
miglior forecasting su un segnale già ben spiegato dalla stagionalità. La tesi del progetto
— il trade-off espressività/calibrazione/costo — regge comunque (scenario A o C).

**Q18. E se invece lo batte?**
Allora la diffusione sfrutta struttura e incertezza non gaussiana per la miglior qualità
probabilistica (scenario B); ma il vantaggio va pesato col costo di sampling: vale dove
l'incertezza ha valore decisionale.

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
- Se chiedono un numero M3 in Fase A: **non inventare** — "run in corso, soglia da battere
  160.5 di CRPS".
- Tenere pronte le slide **backup B1–B6** (calibrazione, tabella completa, perché ARIMA va
  male, math della diffusione, Exchange, ablation DeepAR) → mappa in
  [`SLIDE_TEMPLATE_ELECTRICITY_IT.md`](SLIDE_TEMPLATE_ELECTRICITY_IT.md).
- Ripetere il messaggio-chiave a ogni occasione: **trade-off**, non vittoria.
