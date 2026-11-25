"""Bounded, episode-local temporal ensembling in normalized action coordinates."""
from collections import deque
from dataclasses import dataclass
import math
import torch
from .math_utils import spatial_indices


@dataclass(frozen=True)
class EnsembleResult:
    action: torch.Tensor
    weights: torch.Tensor
    ages: torch.Tensor
    effective_horizon: torch.Tensor
    divergence_l2: torch.Tensor
    uncertainty: torch.Tensor | None
    fresh_uncertainty: torch.Tensor | None


class TemporalEnsembler:
    """One instance per environment. Call update once at EVERY control timestep.

    Age i means t-generation_time, with i=0 for the fresh chunk. Static weights
    are exp(-m*i), as requested. Some original ACT rollout scripts weight by
    oldest-first array index; that ordering is intentionally not reproduced.

    Per-dimension Laplace scales b are reduced with an arithmetic mean for the
    scalar uncertainty gate. Both gates use only the first six spatial
    coordinates for 7D LIBERO actions; gripper is excluded. All seven
    coordinates remain in the returned action. Other action spaces can
    supply spatial_dims explicitly. Diagnostic divergence is its L2 root.
    Storage is O(K^2*A), bounded independently of episode length.
    """
    def __init__(self, chunk_size=50, *, use_dute=False, m=0.01,
                 lambda_1=0., lambda_2=0., spatial_dims=None):
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        if any(not math.isfinite(x) or x < 0 for x in (m, lambda_1, lambda_2)):
            raise ValueError('Gate coefficients must be finite and nonnegative')
        self.chunk_size, self.use_dute = chunk_size, use_dute
        self.m, self.lambda_1, self.lambda_2 = m, lambda_1, lambda_2
        self.spatial_dims = spatial_dims
        self.reset()

    def reset(self):
        self._chunks = deque(maxlen=self.chunk_size)
        self._last_t = None

    @torch.no_grad()
    def update(self, t: int, mean: torch.Tensor,
               scale: torch.Tensor | None = None) -> EnsembleResult:
        """Consume [K,A] mean and optional positive [K,A] b, return [A] action."""
        if t < 0 or (self._last_t is not None and t != self._last_t + 1):
            raise ValueError('Timesteps must be consecutive; reset between episodes')
        if mean.ndim != 2 or mean.shape[0] != self.chunk_size or mean.shape[1] < 1:
            raise ValueError('mean must have shape [chunk_size, action_dim]')
        selected = spatial_indices(mean.shape[1], self.spatial_dims)
        if not torch.isfinite(mean).all():
            raise ValueError('Non-finite action chunk')
        if scale is not None:
            if scale.shape != mean.shape or scale.device != mean.device:
                raise ValueError('scale must match mean shape and device')
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('Laplace scales must be finite and positive')
        if self.use_dute and self.lambda_1 > 0 and scale is None:
            raise ValueError('Uncertainty gating requires a trained uncertainty head')
        if self._chunks:
            previous = self._chunks[-1][1]
            if previous.shape != mean.shape or previous.device != mean.device:
                raise ValueError('Action shape/device changed within an episode')
            if (self._chunks[-1][2] is None) != (scale is None):
                raise ValueError('Uncertainty availability changed within an episode')
        # Own detached copies: caller mutation must not affect stored chunks.
        self._chunks.append((t, mean.detach().float().clone(),
                             None if scale is None else scale.detach().float().clone()))
        self._last_t = t
        while self._chunks and t - self._chunks[0][0] >= self.chunk_size:
            self._chunks.popleft()
        ages = torch.tensor([t - origin for origin, _, _ in self._chunks],
                            device=mean.device, dtype=torch.long)
        candidates = torch.stack([chunk[t-origin] for origin, chunk, _ in self._chunks])
        uncertainty = None if scale is None else torch.stack(
            [b[t-origin, list(selected)].mean() for origin, _, b in self._chunks])
        # Float64 gating avoids overflow even for large, finite float32 chunks.
        spatial = candidates[:, list(selected)].double()
        squared = (spatial - spatial[-1]).square().sum(-1)
        logits = -self.m * ages.double()
        if self.use_dute:
            if self.lambda_1:
                logits = logits - self.lambda_1 * uncertainty.double()
            if self.lambda_2:
                logits = logits - self.lambda_2 * squared
        weights = torch.softmax(logits, dim=0)
        action = (weights[:, None] * candidates.double()).sum(0).to(mean.dtype)
        return EnsembleResult(action, weights, ages,
                              weights.square().sum().reciprocal(), squared.sqrt(),
                              uncertainty, None if uncertainty is None else uncertainty[-1])
