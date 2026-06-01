# Speaker script — ~8–10 min, 3 relatori (Electricity)

> **Copione di parlato continuo.** Tre blocchi (A/B/C), ~2.5–3 min ciascuno.
> Da leggere "quasi a memoria": frasi brevi, niente formule lette a voce.
> I numeri vengono da [`RESULTS_PLACEHOLDERS_ELECTRICITY.md`](RESULTS_PLACEHOLDERS_ELECTRICITY.md).
>
> 🔴 **M3 = PLACEHOLDER.** Dove vedi
> `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
> il relatore C, in Fase A, dice la versione "il run è in corso"; in Fase B si
> sostituisce col numero reale **e** si sceglie lo **scenario A/B/C** (vedi blocco C).
>
> **Tempi indicativi:** A 0:00–2:45 · B 2:45–5:45 · C 5:45–9:00 (+Q&A).

---

## COMPONENTE A — problema, dataset, perché probabilistico (~2:45)

**[Slide 1 — Titolo]**
"Buongiorno a tutti. Vi presentiamo un confronto controllato tra modelli di
forecasting *probabilistico* di serie temporali, applicati ai consumi elettrici.
L'idea centrale del progetto è semplice da dire: non vogliamo prevedere un solo
numero per il futuro, ma una **distribuzione** del futuro. E vogliamo capire se un
modello moderno, basato sulla **diffusione**, produce un'incertezza migliore dei
metodi classici — e soprattutto *a che prezzo*. Quindi non è una gara a chi vince,
ma uno studio di un compromesso."

**[Slide 2 — Perché probabilistico]**
"Perché probabilistico? Perché una previsione puntuale nasconde il rischio. Se devo
gestire una rete elettrica, sapere che 'domani alle 19 il consumo sarà mille' non mi
basta: devo sapere quanto posso sbagliare, per dimensionare riserve e margini.
Formalmente stimiamo la distribuzione del futuro **condizionata** al passato: dato
quello che ho osservato, qual è la distribuzione di quello che verrà. Due parametri
chiave: H, quanta storia guardo indietro, e τ, quanto in avanti prevedo. E un
requisito che useremo come metro di giudizio: gli intervalli devono essere
**calibrati**. Se dico che un intervallo copre il novanta percento dei casi, allora
nel test deve coprirne davvero circa il novanta."

**[Slide 3 — Dataset e protocollo]**
"Misuriamo tutto su **Electricity**: 321 serie orarie di consumo, con una
stagionalità giornaliera e settimanale molto forte — c'è il ritmo del giorno e quello
dei giorni feriali contro il weekend. Il protocollo è fissato uguale per tutti i
modelli: teniamo da parte un set di test, usiamo una settimana di storia per prevedere
il giorno successivo, e generiamo cento campioni per ricostruire la distribuzione.
Un punto di onestà metodologica: usiamo il file grezzo con il *nostro* split, non lo
split del benchmark pubblicato. Quindi i nostri numeri vanno letti come confronto
*interno*, tra i nostri modelli a parità di condizioni, non come confronto con la
letteratura."

**[Slide 4 — Exchange → Electricity]**
"Una nota sul perché abbiamo due dataset. Prima di Electricity abbiamo usato Exchange,
i tassi di cambio, come banco di prova. Exchange è quasi una passeggiata aleatoria: lì
la miglior previsione per domani è semplicemente oggi, e nessun modello sofisticato
batte questa regola. Lo usiamo come **controllo negativo**: se la pipeline è corretta,
deve dirci che lì i modelli complessi *non* aiutano. Electricity, invece, ha struttura
ricca ed è il test che conta. Il messaggio che ci portiamo dietro è che **il vincitore
dipende dal tipo di segnale**. E passo la parola per i modelli."

---

## COMPONENTE B — la scala dei modelli, TimeGrad, le metriche (~3:00)

**[Slide 5 — La scala M0 → M3]**
"Grazie. Abbiamo costruito una *scala* di quattro modelli a complessità crescente.
Alla base, **M0**, il seasonal-naive: dice semplicemente che il futuro assomiglia al
ciclo precedente. Non si addestra, eppure — anticipo — su segnali stagionali è
sorprendentemente forte. Poi **M1**, ARIMA: il classico lineare, con un modello
separato per ciascuna delle 321 serie. **M2**, DeepAR: una rete ricorrente
probabilistica, addestrata su tutte le serie insieme, che impara a condividere
struttura. E in cima **M3**, TimeGrad, il nostro modello di **diffusione**. Salendo la
scala cresce l'espressività — e, come vedremo, cresce il costo."

**[Slide 6 — TimeGrad intuitivo]**
"Fermiamoci un momento su TimeGrad, perché è il cuore del progetto. L'idea della
diffusione è elegante: prendo un dato reale e ci aggiungo rumore gaussiano un po' alla
volta, finché non resta che rumore puro. Poi addestro una rete neurale a fare il
percorso **inverso**: a togliere il rumore, un passo alla volta. Una volta addestrata,
per generare un dato nuovo parto da puro rumore e lo ripulisco gradualmente, fino a
ottenere un campione realistico. Per il forecasting facciamo una cosa in più:
**condizioniamo** questa ripulitura sul passato, riassunto da una rete ricorrente.
Ripetiamo il campionamento cento volte e otteniamo cento traiettorie future possibili —
cioè la distribuzione. Il prezzo da pagare, e lo sottolineo perché tornerà, è che ogni
singolo campione richiede tanti passi di ripulitura: il sampling è **lento**."

**[Slide 7 — Le metriche]**
"Come giudichiamo i modelli? Con tre lenti, più il costo. Prima lente, accuratezza
puntuale: usiamo MASE, costruito in modo che il valore uno equivalga al seasonal-naive;
sotto uno significa 'meglio del baseline', sopra uno 'peggio'. Seconda lente, qualità
*probabilistica*: il CRPS, che a differenza dell'errore medio premia una distribuzione
intera ben posizionata — è la metrica che ci interessa di più. Terza lente, la
calibrazione: controlliamo se le bande al cinquanta e al novanta percento coprono
davvero quella frazione di casi. E infine il costo: il tempo per addestrare e,
soprattutto, il tempo per campionare. Un modello è utile se ha MASE sotto uno e CRPS
basso *a un costo accettabile*. Tre condizioni insieme. Passo ai risultati."

---

## COMPONENTE C — risultati, trade-off, interpretazione, limiti (~3:00)

**[Slide 8 — La tabella dei risultati]**
"Veniamo ai numeri su Electricity. La prima sorpresa è il baseline: il seasonal-naive
ha CRPS circa 160 e MASE uno. Questa è la **barra da battere** per tutti. ARIMA, montato
come un modello per canale su 321 serie, è fragile e ottiene risultati molto peggiori —
CRPS oltre 860. DeepAR è il miglior modello *addestrato*: batte ARIMA di larghissima
misura, scende a 253 di CRPS, ma — ed è il punto cruciale — **non** raggiunge il
seasonal-naive. 253 contro 160. Quindi la domanda per il nostro modello di diffusione
diventa chirurgica: riesce a scendere **sotto 160** di CRPS, a un costo accettabile?"

> 🔴 **FASE A (M3 in corso):** "La riga di TimeGrad, qui, è ancora un segnaposto: il
> run sta girando su GPU mentre vi parliamo, e i numeri reali li inseriamo appena
> finisce. Quello che possiamo dire fin d'ora è *quale soglia* deve superare: 160 di
> CRPS."
> `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`
>
> 🔵 **FASE B (M3 disponibile):** sostituire con il numero reale, es. "TimeGrad ottiene
> un CRPS di ___, che [supera / non supera] la barra dei 160", e proseguire con lo
> **scenario** scelto qui sotto.

**[Slide 9 — Qualità vs costo]**
"Questo grafico riassume la nostra tesi. Sull'asse orizzontale il costo di
campionamento, sul verticale la qualità probabilistica: l'angolo ideale è in basso a
sinistra — economico e accurato. Il seasonal-naive sta proprio lì, ed è il risultato
più istruttivo di tutto il lavoro: a volte la cosa semplice è anche la migliore. DeepAR
costa di più e rende meno del baseline. TimeGrad, per costruzione, vive a destra, nella
zona costosa, perché il sampling è lento. La domanda non è se migliora il CRPS di
un'inezia, ma se si sposta a sinistra **abbastanza** da ripagare lo spostamento verso
l'alto, cioè il costo."

**[Slide 10 — Conclusioni e limiti]**
"Chiudo con il messaggio e con l'onestà sui limiti. Il messaggio **non** è 'la
diffusione vince': è che esiste un compromesso fra espressività, calibrazione e costo,
e che su un segnale fortemente stagionale un baseline semplice è un avversario serio,
non un fantoccio."

> 🔴 **In Fase B, inserire QUI la frase dello scenario** (una sola) — testo completo in
> `RESULTS_PLACEHOLDERS_ELECTRICITY.md` §3:
>
> - **🅐 (M3 batte DeepAR ma non M0):** "TimeGrad migliora la baseline deep ma non
>   supera il seasonal-naive: la stagionalità è già catturata bene dal baseline, e la
>   diffusione aggiunge flessibilità ma non abbastanza valore per il suo costo. Il
>   nostro contributo è il confronto controllato, non una vittoria del modello."
> - **🅑 (M3 batte anche M0 su CRPS/calibrazione):** "TimeGrad ottiene la migliore
>   qualità probabilistica, sfruttando struttura multivariata e incertezza non
>   gaussiana — ma il vantaggio va letto insieme al costo: sampling più lento. Una
>   vittoria *condizionata*: giustificata dove l'incertezza ha valore decisionale."
> - **🅒 (M3 peggiora rispetto a DeepAR):** "Il risultato negativo è informativo: più
>   espressività non implica miglior forecasting. Possibili cause: training
>   insufficiente, costo del sampling, difficoltà a scalare a 321 serie, o un baseline
>   stagionale semplicemente troppo forte. Un esito scientificamente valido."
>
> `[PLACEHOLDER M3 — aggiornare appena disponibile results/registry.csv]`

"Sui limiti siamo trasparenti: lo split non è quello del benchmark pubblicato, quindi
non ci confrontiamo con la letteratura; ARIMA per-canale è poco competitivo per
costruzione; il sampling di diffusione è costoso; e abbiamo usato un solo seed, senza
tuning esteso. Da qui i lavori futuri: adottare lo split standard `electricity_nips`
per il confronto con i numeri pubblicati, provare CSDI, e ridurre i passi di
campionamento con uno schema tipo DDIM. Vi ringraziamo, e siamo a disposizione per le
domande."

---

### Promemoria di consegna (per chi parla)

- **A:** chiudere lo Slide 4 passando la parola con "...passo la parola per i modelli".
- **B:** non leggere formule; sul TimeGrad insistere su *rumore→segnale* e su *sampling
  lento*. Chiudere lo Slide 7 con "...passo ai risultati".
- **C:** in Fase A dire chiaramente "il run è in corso, numeri reali appena finisce";
  in Fase B leggere il numero **e** una sola frase-scenario. Tenere pronte le slide di
  **backup** (B1–B6) per il Q&A → vedi
  [`QA_ORALE_ELECTRICITY_IT.md`](QA_ORALE_ELECTRICITY_IT.md).
