# CHE_COSA_SONO — src/models

> La **"ladder" dei modelli**: dal più semplice (M0) alle due diffusioni condizionali
> (M3 autoregressiva, M4 non-autoregressiva). Ogni wrapper espone la stessa interfaccia
> (`fit` / `predict` → `(point, samples)`), così i modelli si confrontano **cella per
> cella** con le stesse metriche.

| File | Che cos'è |
|---|---|
| `naive.py` | **M0** seasonal-naive: ripete il valore di un periodo stagionale fa. L'"ancora di onestà" da battere. |
| `classical.py` | **M1** ARIMA per-canale: il baseline statistico classico (lineare-gaussiano). |
| `deepar.py` | **M2** DeepAR: RNN autoregressivo che emette una distribuzione parametrica (Student-t). Pesante (GluonTS) → Colab. |
| `timegrad.py` | **M3** TimeGrad: a ogni passo un **DDPM condizionato** sullo stato dell'RNN denoise il vettore multivariato successivo. La distribuzione predittiva è appresa e non-parametrica. Il **centro del progetto**. Pesante (PyTorchTS) → Colab. |
| `timediff.py` | **M4** TimeDiff: diffusione condizionale **non-autoregressiva** — denoise **l'intero blocco futuro** (τ×D) in una volta, predicendo x0 (non ε), con inizializzazione lineare/AR e *future-mixup* (solo in training). Backbone conv gated (DiffWave-style), **torch puro** (niente GluonTS/PyTorchTS). Campionamento più **veloce** di M3 (una sola catena inversa). Pesante → Colab. |
| `toy_ddpm.py` | DDPM condizionale **scritto da zero in puro NumPy** su dati 1-D bimodali: l'"artefatto di comprensione" per l'orale (forward `q(x_t\|x_0)`, ε-predictor con backprop a mano + Adam, sampler DDIM a passi liberi). È il motore dell'esperimento E3, gira in locale. |
| `__init__.py` | API pubblica del package models. |
