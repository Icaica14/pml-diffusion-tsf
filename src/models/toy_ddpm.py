"""A from-scratch **conditional** DDPM on 1-D toy data — the "understanding artifact".

This is the ~250-line, **pure-NumPy** denoising-diffusion model the plan asks for as the
proof that we understand the mechanism, not just the library API (plan Part 5, "toy
DDPM"; appendix E3). It implements, by hand and with no deep-learning framework:

* the **forward process** ``q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)`` (closed form), the
  exact noising of the course's §11.2.1;
* a small **MLP ε-predictor** ``ε_θ(x_t, t, c)`` with hand-derived back-propagation and a
  hand-written Adam optimiser — the network the course's §11.2.3 trains;
* the **simplified training loss** ``E‖ε − ε_θ(x_t, t, c)‖²`` (predict the noise);
* a **DDIM sampler** whose number of reverse steps is a free knob, so we can trade
  sampling steps against quality *without retraining* — this is exactly the curve
  experiment **E3** measures on the real TimeGrad, reproduced here cheaply and
  transparently.

The twist over the course (which only does *unconditional* generation) is the
**conditioning** ``c``: every network input carries the context scalar, so the model
learns a whole *conditional* distribution ``p(x | c)`` — the same "condition on the
past" idea that turns a generic diffusion model into a forecaster.

The toy target ``p(x | c)`` is deliberately **bimodal** (two branches around a
sinusoidal mean), something a Gaussian-head model *cannot* represent but a diffusion
model can — which is the whole point of bringing diffusion to forecasting.

Runs on a CPU in seconds; imports only NumPy. Nothing here touches torch or GluonTS, so
it is the most portable, most defensible piece of the project for the individual oral.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------------------
# Toy conditional target  p(x | c):  a bimodal "fork" around a sinusoidal mean.
# A unimodal/Gaussian model must average the two branches; a diffusion model keeps both.
# ---------------------------------------------------------------------------------------


def make_conditional_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` pairs ``(c, x)`` from the toy conditional distribution.

    ``c ~ U[-2, 2]`` is the conditioning scalar (the stand-in for "the past"); given
    ``c`` the target is ``x = 1.5 sin(π c) ± 0.9 + 0.12 ε`` with the ± chosen by a fair
    coin — i.e. two symmetric branches (bimodal) sharing a sinusoidal mean.
    """
    c = rng.uniform(-2.0, 2.0, size=n)
    mode = rng.integers(0, 2, size=n) * 2 - 1  # ±1
    x = 1.5 * np.sin(np.pi * c) + 0.9 * mode + 0.12 * rng.standard_normal(n)
    return c.astype(np.float64), x.astype(np.float64)


def _time_features(t_norm: np.ndarray) -> np.ndarray:
    """Fourier features of the normalised diffusion step ``t/T`` ∈ (0, 1] → (B, 5)."""
    t = t_norm[:, None]
    return np.concatenate(
        [t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t),
         np.sin(4 * np.pi * t), np.cos(4 * np.pi * t)],
        axis=1,
    )


# ---------------------------------------------------------------------------------------
# A minimal MLP (tanh hidden layers, linear output) with hand-written backprop + Adam.
# ---------------------------------------------------------------------------------------


@dataclass
class _MLP:
    sizes: tuple[int, ...]
    rng: np.random.Generator
    lr: float = 2e-3
    W: list = field(default_factory=list)
    b: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.W, self.b = [], []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            # He-style init for tanh nets keeps activations from saturating early.
            self.W.append(self.rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(bi) for bi in self.b]
        self._vb = [np.zeros_like(bi) for bi in self.b]
        self._step = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run the net, caching pre-activations/activations for ``backward``."""
        self._a = [X]
        self._z = []
        h = X
        last = len(self.W) - 1
        for i, (W, bi) in enumerate(zip(self.W, self.b)):
            z = h @ W + bi
            self._z.append(z)
            h = z if i == last else np.tanh(z)
            self._a.append(h)
        return h

    def backward_and_step(self, dout: np.ndarray) -> None:
        """Back-prop ``dout = dL/d(output)`` (already 1/B-scaled) and take an Adam step."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = dout
        for i in reversed(range(len(self.W))):
            gW[i] = self._a[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (1.0 - np.tanh(self._z[i - 1]) ** 2)
        self._adam(gW, gb)

    def _adam(self, gW, gb, b1=0.9, b2=0.999, eps=1e-8) -> None:
        self._step += 1
        bc1 = 1 - b1 ** self._step
        bc2 = 1 - b2 ** self._step
        for i in range(len(self.W)):
            for M, V, g, P in ((self._mW, self._vW, gW, self.W),
                               (self._mb, self._vb, gb, self.b)):
                M[i] = b1 * M[i] + (1 - b1) * g[i]
                V[i] = b2 * V[i] + (1 - b2) * (g[i] ** 2)
                P[i] -= self.lr * (M[i] / bc1) / (np.sqrt(V[i] / bc2) + eps)


# ---------------------------------------------------------------------------------------
# The conditional DDPM.
# ---------------------------------------------------------------------------------------


class ToyConditionalDDPM:
    """A conditional DDPM ``ε_θ(x_t, t, c)`` over scalar ``x`` conditioned on scalar ``c``.

    Parameters
    ----------
    n_steps : number of diffusion steps ``T`` used at *training* time.
    hidden  : hidden-layer widths of the ε-predictor MLP.
    beta0, beta1 : endpoints of the linear β schedule.
    lr, seed : optimiser learning rate and RNG seed.
    """

    def __init__(self, n_steps: int = 100, hidden: tuple[int, ...] = (64, 64),
                 beta0: float = 1e-4, beta1: float = 2e-2, lr: float = 2e-3,
                 seed: int = 0) -> None:
        self.n_steps = int(n_steps)
        self.rng = np.random.default_rng(seed)
        # Linear β schedule and the cumulative products ᾱ_t (the forward-process variance).
        self.betas = np.linspace(beta0, beta1, self.n_steps)
        self.alphas = 1.0 - self.betas
        self.abar = np.cumprod(self.alphas)            # ᾱ_t, length T
        # MLP input = [x_t (1)] + [time features (5)] + [c (1)] = 7.
        self.net = _MLP((7, *hidden, 1), self.rng, lr=lr)

    # -- the noise-predicting network -----------------------------------------------------
    def _eps(self, x_t: np.ndarray, t_idx: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Predict the noise ``ε`` given the noised ``x_t``, step index ``t_idx``, ``c``."""
        tn = (t_idx + 1) / self.n_steps
        feats = np.concatenate([x_t[:, None], _time_features(tn), c[:, None]], axis=1)
        return self.net.forward(feats)[:, 0]

    # -- training -------------------------------------------------------------------------
    def train(self, c: np.ndarray, x0: np.ndarray, steps: int = 4000,
              batch: int = 256, verbose: bool = False) -> list[float]:
        """Fit ``ε_θ`` by the simplified DDPM objective ``E‖ε − ε_θ(x_t, t, c)‖²``."""
        n = x0.shape[0]
        losses: list[float] = []
        for s in range(steps):
            idx = self.rng.integers(0, n, size=batch)
            xb, cb = x0[idx], c[idx]
            t = self.rng.integers(0, self.n_steps, size=batch)        # step per example
            eps = self.rng.standard_normal(batch)                      # the target noise
            a = self.abar[t]
            x_t = np.sqrt(a) * xb + np.sqrt(1.0 - a) * eps             # forward q(x_t|x_0)
            pred = self._eps(x_t, t, cb)
            resid = pred - eps                                         # dL/dpred for MSE
            self.net.backward_and_step((resid / batch)[:, None])
            if verbose and (s % max(1, steps // 10) == 0 or s == steps - 1):
                losses.append(float(np.mean(resid ** 2)))
        return losses

    # -- sampling (DDIM with a free number of steps) --------------------------------------
    def _respaced(self, n_sample_steps: int) -> np.ndarray:
        """A descending sub-sequence of the trained timesteps, length ``n_sample_steps``."""
        k = int(np.clip(n_sample_steps, 1, self.n_steps))
        seq = np.unique(np.round(np.linspace(0, self.n_steps - 1, k)).astype(int))
        return seq[::-1]

    def sample(self, c: np.ndarray, n_samples: int, n_sample_steps: int = 100,
               eta: float = 1.0) -> np.ndarray:
        """Draw ``n_samples`` trajectories per conditioning value via DDIM.

        ``n_sample_steps`` is the **knob E3 sweeps**: fewer steps = cheaper but lower
        quality. ``eta`` interpolates DDIM (0 = deterministic) ↔ ancestral DDPM (1).
        Returns an array of shape ``(len(c), n_samples)``.
        """
        seq = self._respaced(n_sample_steps)
        N = c.shape[0]
        cc = np.repeat(c, n_samples)                       # (N*S,)
        x = self.rng.standard_normal(N * n_samples)        # x_T ~ N(0, I)
        for j, i_cur in enumerate(seq):
            a_cur = self.abar[i_cur]
            a_prev = self.abar[seq[j + 1]] if j + 1 < len(seq) else 1.0
            eps = self._eps(x, np.full(N * n_samples, i_cur), cc)
            x0_pred = (x - np.sqrt(1.0 - a_cur) * eps) / np.sqrt(a_cur)
            if j + 1 == len(seq):
                x = x0_pred                                # final jump to x_0
                break
            sigma = (eta * np.sqrt((1.0 - a_prev) / (1.0 - a_cur))
                     * np.sqrt(max(1.0 - a_cur / a_prev, 0.0)))
            direction = np.sqrt(max(1.0 - a_prev - sigma ** 2, 0.0)) * eps
            z = self.rng.standard_normal(N * n_samples)
            x = np.sqrt(a_prev) * x0_pred + direction + sigma * z
        return x.reshape(N, n_samples)


# ---------------------------------------------------------------------------------------
# Sample-based CRPS (the same probabilistic-quality metric used on the real models).
# ---------------------------------------------------------------------------------------


def crps_samples(samples: np.ndarray, y: np.ndarray) -> float:
    """Mean CRPS of empirical forecasts ``samples`` (N, S) against observations ``y`` (N,).

    Energy form: ``CRPS = E|X − y| − ½ E|X − X'|``, the second term via the sorted-sample
    estimator ``E|X−X'| = (2/S²) Σ_i (2i − S − 1) x_(i)``.
    """
    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    S = samples.shape[1]
    term1 = np.abs(samples - y[:, None]).mean(axis=1)
    xs = np.sort(samples, axis=1)
    coef = 2.0 * np.arange(1, S + 1) - S - 1
    ediff = (2.0 / S ** 2) * (xs * coef).sum(axis=1)
    return float(np.mean(term1 - 0.5 * ediff))
