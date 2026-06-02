"""M4 — TimeDiff, a *non-autoregressive* conditional-diffusion forecaster.

Where M3 (TimeGrad, Rasul et al. 2021) is **autoregressive** — an RNN walks the
horizon one step at a time and a DDPM denoises the next D-vector conditioned on the
RNN state — **TimeDiff** (Shen & Kwok, ICML 2023, *"Non-autoregressive Conditional
Diffusion Models for Time Series Prediction"*) denoises the **whole future block**
``x0 ∈ R^{tau×D}`` in one shot. It trades TimeGrad's step-by-step RNN for a
convolutional denoiser plus two conditioning tricks, and the headline practical payoff
is **fast sampling**: one reverse chain over the entire horizon instead of ``tau``
autoregressive sampling rounds. On Electricity that is the difference between TimeGrad's
multi-hour predict and a TimeDiff predict measured in minutes — exactly the
"quality-vs-cost" contrast this project is built to expose, now *inside* the diffusion
family rather than only diffusion-vs-classical.

The three moving parts (one sentence each)
------------------------------------------
* **x0-prediction DDPM.** The forward process noises the future block over ``T`` steps
  (``q(x_t | x_0)`` with a cosine/linear ``beta`` schedule); the network is trained to
  predict the *clean* block ``x0`` (not the noise ``epsilon``) — the parameterization the
  TimeDiff paper found more stable for forecasting — under a plain MSE loss.
* **Autoregressive (linear) initialization.** A shared ``Linear(H -> tau)`` maps each
  channel's lookback to a coarse future ``x_ar``. It is a cheap, strong prior that the
  denoiser conditions on, so the diffusion only has to model the *residual* structure
  around a sane linear forecast rather than the signal from scratch.
* **Future mixup (training only).** During training the conditioning future is a random
  per-element blend ``m ⊙ x_ar + (1 - m) ⊙ x0`` (Bernoulli mask ``m``): occasionally
  revealing the true future eases learning. At inference ``m = 1`` everywhere, so the
  conditioning is the past-derived ``x_ar`` alone — **no future leakage**.

Backbone
--------
A DiffWave-style stack of gated, (optionally) dilated 1-D conv residual blocks over the
``tau`` time axis with ``D`` channels. Each block is conditioned on (a) a pooled encoding
of the lookback, (b) a projection of the conditioning future ``cond_future``, and a
learned positional embedding over the ``tau`` positions; the diffusion step ``t`` enters
through a sinusoidal embedding added to the block input. Output is ``x0_hat ∈ R^{tau×D}``.

Contract & environment (identical discipline to M2/M3 — leakage-free, like-for-like)
-----------------------------------------------------------------------------------
This wrapper exposes the same surface the rest of the ladder consumes:
``fit(train_series)`` then ``predict(contexts) -> (point, samples)`` with
``point`` shape ``(N, tau, D)`` and ``samples`` shape ``(N, S, tau, D)`` — the tensor
:mod:`src.eval.metrics` already scores, so M0..M4 are compared cell-for-cell.

Unlike M3 this model is **self-contained PyTorch** — it does *not* go through
GluonTS/PyTorchTS, so it sidesteps that stack's fragile version pinning. It trains by
**random-cropping** ``(H + tau)`` windows from the single multivariate train series
(``ds.raw_splits["train"]``) and standardizes per channel internally (z-score, stats from
the train series only — the same leakage-free discipline as the data contract); samples
are de-standardized back to the original scale before they leave :meth:`predict`. ``torch``
is imported **lazily inside the methods** so the light/local env can still import this
module (the local torch is ABI-broken against NumPy 2.x); real fit/predict run on the
Colab GPU. A fast ``--smoke`` path in ``experiments/run_timediff.py`` validates the whole
fit→predict→score→registry plumbing in seconds before spending GPU hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Noise schedules (pure NumPy — unit-testable without torch)
# ---------------------------------------------------------------------------
def linear_beta_schedule(n_steps: int, beta0: float = 1e-4, beta_end: float = 0.1) -> np.ndarray:
    """The classic linearly-spaced ``beta_t`` schedule (DDPM, Ho et al. 2020)."""
    return np.linspace(beta0, beta_end, n_steps, dtype=np.float64)


def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> np.ndarray:
    """Nichol & Dhariwal (2021) cosine schedule, returned as ``beta_t`` in ``[0, 0.999]``.

    Defined through the cumulative ``abar_t`` (a shifted-cosine), then differenced into
    ``beta_t = 1 - abar_t / abar_{t-1}`` and clipped — gentler noising near ``t=0`` than
    the linear schedule, which often helps the high-frequency hourly load structure.
    """
    steps = n_steps + 1
    x = np.linspace(0, n_steps, steps, dtype=np.float64)
    abar = np.cos(((x / n_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    abar = abar / abar[0]
    betas = 1.0 - (abar[1:] / abar[:-1])
    return np.clip(betas, 1e-8, 0.999)


def _respaced_steps(n_diffusion: int, n_sample_steps: int) -> list[int]:
    """Descending list of timestep indices to visit at sampling time.

    ``n_sample_steps >= n_diffusion`` walks the full chain ``[T-1, ..., 0]``. A smaller
    value evenly subsamples the grid (still including the last, cleanest step) for a
    DDIM-style speed/quality trade — the same respacing the project's toy DDPM uses, so
    the two diffusion samplers behave consistently.
    """
    if n_sample_steps >= n_diffusion:
        return list(range(n_diffusion - 1, -1, -1))
    idx = np.linspace(0, n_diffusion - 1, n_sample_steps).round().astype(int)
    return sorted(set(int(i) for i in idx), reverse=True)


# ---------------------------------------------------------------------------
# Network factory (torch imported lazily so the light env can import this module)
# ---------------------------------------------------------------------------
def _build_network(
    target_dim: int,
    horizon: int,
    context_length: int,
    hidden: int,
    n_res_blocks: int,
    kernel: int,
    dilation_cycle: int,
):
    """Construct and return the TimeDiff denoiser ``nn.Module`` (x0-predictor).

    All ``torch`` symbols live inside this factory, so importing
    :mod:`src.models.timediff` never imports torch — only building a model does. The
    returned module exposes ``ar_init(ctx)`` (the linear AR prior) and
    ``forward(x_t, t, ctx, cond_future) -> x0_hat``.
    """
    import math

    import torch
    import torch.nn as nn

    class _StepEmbedding(nn.Module):
        """Sinusoidal embedding of the diffusion step ``t`` -> an MLP -> (B, hidden)."""

        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        def forward(self, t):  # t: (B,) long
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
            )
            args = t.float()[:, None] * freqs[None, :]
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if emb.shape[-1] < self.dim:  # odd dim padding
                emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
            return self.mlp(emb)

    class _ResBlock(nn.Module):
        """Gated dilated-conv residual block (DiffWave-style) with conditioning."""

        def __init__(self, ch: int, k: int, dilation: int):
            super().__init__()
            pad = dilation * (k - 1) // 2
            self.conv = nn.Conv1d(ch, 2 * ch, k, padding=pad, dilation=dilation)
            self.cond_conv = nn.Conv1d(ch, 2 * ch, 1)
            self.res_conv = nn.Conv1d(ch, ch, 1)
            self.skip_conv = nn.Conv1d(ch, ch, 1)

        def forward(self, h, cond):
            y = self.conv(h) + self.cond_conv(cond)
            a, b = y.chunk(2, dim=1)
            y = torch.tanh(a) * torch.sigmoid(b)
            res = self.res_conv(y)
            skip = self.skip_conv(y)
            return (h + res) / math.sqrt(2.0), skip

    class TimeDiffNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.target_dim = target_dim
            self.horizon = horizon
            self.context_length = context_length
            self.hidden = hidden

            # Autoregressive (linear) initialization: per-channel H -> tau, shared weights.
            self.ar = nn.Linear(context_length, horizon)

            # Lookback encoder -> a pooled (B, hidden) summary broadcast over tau.
            self.ctx_conv = nn.Sequential(
                nn.Conv1d(target_dim, hidden, kernel, padding=kernel // 2),
                nn.SiLU(),
                nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
            )
            self.input_proj = nn.Conv1d(target_dim, hidden, 1)
            self.cond_proj = nn.Conv1d(target_dim, hidden, 1)
            self.pos = nn.Parameter(torch.randn(1, hidden, horizon) * 0.02)
            self.step_emb = _StepEmbedding(hidden)
            self.blocks = nn.ModuleList(
                [_ResBlock(hidden, kernel, 2 ** (i % dilation_cycle)) for i in range(n_res_blocks)]
            )
            self.out = nn.Sequential(
                nn.Conv1d(hidden, hidden, 1), nn.SiLU(), nn.Conv1d(hidden, target_dim, 1)
            )

        def ar_init(self, ctx):
            # ctx: (B, H, D) -> (B, D, H) -> Linear over H -> (B, D, tau) -> (B, tau, D)
            return self.ar(ctx.transpose(1, 2)).transpose(1, 2)

        def forward(self, x_t, t, ctx, cond_future):
            # x_t, cond_future: (B, tau, D); ctx: (B, H, D); t: (B,)
            h = self.input_proj(x_t.transpose(1, 2))                       # (B, hidden, tau)
            h = h + self.step_emb(t).unsqueeze(-1)
            ctx_feat = self.ctx_conv(ctx.transpose(1, 2)).mean(dim=-1, keepdim=True)  # (B, hidden, 1)
            cond = ctx_feat + self.cond_proj(cond_future.transpose(1, 2)) + self.pos  # (B, hidden, tau)
            skip_total = 0.0
            for blk in self.blocks:
                h, skip = blk(h, cond)
                skip_total = skip_total + skip
            out = self.out(skip_total / math.sqrt(len(self.blocks)))       # (B, D, tau)
            return out.transpose(1, 2)                                     # (B, tau, D)

    return TimeDiffNet()


# ---------------------------------------------------------------------------
# The contract-aware wrapper
# ---------------------------------------------------------------------------
@dataclass
class TimeDiffForecaster:
    """A self-contained PyTorch TimeDiff (M4) with the ladder's ``fit``/``predict`` surface.

    Parameters
    ----------
    target_dim : number ``D`` of channels denoised **jointly** in each future block.
    horizon : forecast length ``tau`` (the whole block is denoised at once).
    context_length : ``H`` steps of lookback the conditioning reads.
    freq, start : carried only for the registry row's provenance (the math ignores them).
    diff_steps : number of diffusion steps ``T``.
    beta_schedule : ``"cosine"`` (default) or ``"linear"``.
    beta_end : end value for the linear schedule (ignored by cosine).
    hidden, n_res_blocks, kernel, dilation_cycle : denoiser width / depth / conv shape.
    mixup_prob : per-element probability of keeping ``x_ar`` (vs revealing the true future)
        in the training future-mixup. ``1.0`` disables mixup; inference always uses
        ``x_ar`` alone.
    max_epochs, num_batches_per_epoch, batch_size, lr : the training budget. Each batch is
        ``batch_size`` windows random-cropped from the train series.
    n_samples : number ``S`` of trajectories sampled per window at predict time.
    n_sample_steps : reverse steps actually walked (``>= diff_steps`` = full chain; smaller
        = DDIM-style respaced acceleration).
    eta : DDIM stochasticity. ``1.0`` ≈ ancestral (restores predictive spread — better
        calibrated on multi-modal targets); ``0.0`` deterministic.
    predict_batch_size : windows per GPU batch at sampling time (memory cap; never changes
        the samples). Each window expands to ``n_samples`` parallel chains.
    clip_x0 : clamp the predicted ``x0`` to ``±clip_x0`` (in standardized space) each step
        for sampling stability; ``None`` disables.
    device : ``"cuda"`` on Colab, ``"cpu"`` for the smoke path.
    seed : best-effort determinism seed.

    Attributes set by :meth:`fit`
    -----------------------------
    net_ : the trained denoiser (``None`` until fit).
    mean_, std_ : per-channel standardization stats (from the train series only).
    train_loss_ : last epoch's mean training loss (a convergence sanity check).
    """

    target_dim: int
    horizon: int
    context_length: int
    freq: str | None = None
    start: str | None = None
    diff_steps: int = 100
    beta_schedule: str = "cosine"
    beta_end: float = 0.1
    hidden: int = 64
    n_res_blocks: int = 4
    kernel: int = 3
    dilation_cycle: int = 2
    mixup_prob: float = 0.5
    max_epochs: int = 50
    num_batches_per_epoch: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    n_samples: int = 100
    n_sample_steps: int = 100
    eta: float = 1.0
    predict_batch_size: int = 16
    clip_x0: float | None = 5.0
    device: str = "cuda"
    seed: int = 0
    net_: object = field(default=None, repr=False)
    mean_: object = field(default=None, repr=False)
    std_: object = field(default=None, repr=False)
    train_loss_: float | None = field(default=None, repr=False)

    # -- schedule -------------------------------------------------------------
    def _betas(self) -> np.ndarray:
        if self.beta_schedule == "cosine":
            return cosine_beta_schedule(self.diff_steps)
        if self.beta_schedule == "linear":
            return linear_beta_schedule(self.diff_steps, beta_end=self.beta_end)
        raise ValueError(f"Unknown beta_schedule {self.beta_schedule!r} (use 'cosine'|'linear').")

    def _diffusion_tensors(self, torch, device):
        """Precompute the DDPM/DDIM coefficient tensors on ``device``."""
        betas = self._betas()
        alphas = 1.0 - betas
        abar = np.cumprod(alphas)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        return {
            "abar": t(abar),
            "sqrt_abar": t(np.sqrt(abar)),
            "sqrt_one_minus_abar": t(np.sqrt(1.0 - abar)),
        }

    # -- fit ------------------------------------------------------------------
    def fit(self, train_series: np.ndarray) -> "TimeDiffForecaster":
        """Train one global TimeDiff on the raw multivariate train series ``(L, D)``.

        Standardizes per channel (train-only stats), then for ``max_epochs ×
        num_batches_per_epoch`` steps random-crops ``batch_size`` windows of ``H + tau``,
        noises the future block, and regresses the clean block under MSE with future-mixup
        conditioning. ``train_series`` is ``ds.raw_splits["train"]`` (original scale).
        """
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device(self.device)

        series = np.asarray(train_series, dtype=np.float64)
        if series.ndim != 2:
            raise ValueError(f"Expected (L, D) train series, got shape {series.shape}.")
        L, D = series.shape
        if D != self.target_dim:
            raise ValueError(f"train series has D={D} but target_dim={self.target_dim}.")
        span = self.context_length + self.horizon
        if L < span:
            raise ValueError(f"train series length {L} < H+tau={span}.")

        self.mean_ = series.mean(axis=0)
        self.std_ = series.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0  # guard near-constant channels (zeros early on)
        z = (series - self.mean_) / self.std_
        z_t = torch.tensor(z, dtype=torch.float32, device=device)

        sch = self._diffusion_tensors(torch, device)
        net = _build_network(
            self.target_dim, self.horizon, self.context_length,
            self.hidden, self.n_res_blocks, self.kernel, self.dilation_cycle,
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        n_starts = L - span + 1
        net.train()
        last = float("nan")
        for epoch in range(self.max_epochs):
            running = 0.0
            for _ in range(self.num_batches_per_epoch):
                starts = torch.randint(0, n_starts, (self.batch_size,), device=device)
                offs = torch.arange(span, device=device)
                idx = starts[:, None] + offs[None, :]           # (B, span)
                win = z_t[idx]                                  # (B, span, D)
                ctx = win[:, : self.context_length]
                fut = win[:, self.context_length :]
                t = torch.randint(0, self.diff_steps, (self.batch_size,), device=device)
                noise = torch.randn_like(fut)
                sa = sch["sqrt_abar"][t][:, None, None]
                soma = sch["sqrt_one_minus_abar"][t][:, None, None]
                x_t = sa * fut + soma * noise
                x_ar = net.ar_init(ctx)
                keep = (torch.rand_like(fut) < self.mixup_prob).float()
                cond_future = keep * x_ar + (1.0 - keep) * fut
                x0_hat = net(x_t, t, ctx, cond_future)
                loss = torch.mean((x0_hat - fut) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += float(loss.detach())
            last = running / max(self.num_batches_per_epoch, 1)
            print(f"  [timediff] epoch {epoch + 1:>3d}/{self.max_epochs}  loss={last:.5f}")

        self.net_ = net
        self.train_loss_ = last
        self._free_cuda_cache()
        return self

    @staticmethod
    def _free_cuda_cache() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch absent / CPU-only
            pass

    # -- sampling -------------------------------------------------------------
    def _sample_block(self, torch, net, sch, ctx, x_ar):
        """DDIM (x0-parameterized) reverse chain for a batch of conditioning rows.

        ``ctx``/``x_ar`` are ``(M, H, D)`` / ``(M, tau, D)`` already expanded to one row
        per (window, sample). Returns ``(M, tau, D)`` standardized samples.
        """
        device = ctx.device
        M = ctx.shape[0]
        x = torch.randn(M, self.horizon, self.target_dim, device=device)
        steps = _respaced_steps(self.diff_steps, self.n_sample_steps)
        abar = sch["abar"]
        for i, t_cur in enumerate(steps):
            t_prev = steps[i + 1] if i + 1 < len(steps) else -1
            ab_t = abar[t_cur]
            ab_prev = abar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            t_b = torch.full((M,), t_cur, device=device, dtype=torch.long)
            x0 = net(x, t_b, ctx, x_ar)
            if self.clip_x0 is not None:
                x0 = torch.clamp(x0, -self.clip_x0, self.clip_x0)
            eps = (x - torch.sqrt(ab_t) * x0) / torch.sqrt(1.0 - ab_t)
            sigma = self.eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * torch.sqrt(1.0 - ab_t / ab_prev)
            coef = torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0)
            noise = torch.randn_like(x) if t_prev >= 0 else torch.zeros_like(x)
            x = torch.sqrt(ab_prev) * x0 + torch.sqrt(coef) * eps + sigma * noise
        return x

    # -- predict --------------------------------------------------------------
    def predict(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(point, samples)`` for a batch of context windows.

        ``contexts`` : ``(N, H, D)`` on the **original** scale. We standardize with the
        train stats, draw ``n_samples`` DDIM trajectories per window over the whole
        horizon at once, then de-standardize.

        Returns
        -------
        point   : ``(N, tau, D)`` — the per-position sample mean.
        samples : ``(N, S, tau, D)`` — the sampled predictive trajectories.
        """
        if self.net_ is None:
            raise RuntimeError("predict() called before fit().")
        import torch

        contexts = np.asarray(contexts, dtype=np.float64)
        if contexts.ndim != 3:
            raise ValueError(f"Expected (N, H, D) contexts, got shape {contexts.shape}.")
        N, H, D = contexts.shape
        if D != self.target_dim:
            raise ValueError(f"contexts have D={D} channels but target_dim={self.target_dim}.")
        if H != self.context_length:
            raise ValueError(f"contexts have H={H} but context_length={self.context_length}.")

        device = torch.device(self.device)
        sch = self._diffusion_tensors(torch, device)
        net = self.net_
        net.eval()
        S, tau = self.n_samples, self.horizon
        mean = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        std = torch.tensor(self.std_, dtype=torch.float32, device=device)

        samples = np.empty((N, S, tau, D), dtype=np.float64)
        point = np.empty((N, tau, D), dtype=np.float64)
        wb = max(int(self.predict_batch_size), 1)
        with torch.no_grad():
            for s0 in range(0, N, wb):
                s1 = min(s0 + wb, N)
                ctx_raw = torch.tensor(contexts[s0:s1], dtype=torch.float32, device=device)
                ctx_z = (ctx_raw - mean) / std                       # (b, H, D)
                x_ar = net.ar_init(ctx_z)                            # (b, tau, D)
                # expand to one row per (window, sample)
                ctx_rep = ctx_z.repeat_interleave(S, dim=0)          # (b*S, H, D)
                x_ar_rep = x_ar.repeat_interleave(S, dim=0)          # (b*S, tau, D)
                z = self._sample_block(torch, net, sch, ctx_rep, x_ar_rep)  # (b*S, tau, D)
                z = z * std + mean                                  # de-standardize
                b = s1 - s0
                z = z.reshape(b, S, tau, D)
                arr = z.double().cpu().numpy()
                samples[s0:s1] = arr
                point[s0:s1] = arr.mean(axis=1)
        return point, samples
