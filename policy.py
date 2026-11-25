"""ACT / PA-ACT policy wrapper: input normalisation, loss and inference."""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import nn


@dataclass(frozen=True)
class ActionPrediction:
    mean: torch.Tensor
    log_scale: torch.Tensor | None = None

    @property
    def scale(self):
        return None if self.log_scale is None else self.log_scale.float().exp()


def reconstruction_loss(mean, target, is_pad, log_scale=None, *,
                        legacy_l1_reduction=True, samplewise=False):
    """L1 for vanilla ACT, Laplace NLL (|a - mu| / b + log b) for PA-ACT.

    legacy_l1_reduction keeps ACT's original behaviour of dividing by B*K*A
    (padding included). log 2 is dropped from the NLL since it is constant.
    """
    valid = (~is_pad)[..., None].expand_as(mean)
    # index first so NaNs in padded targets never reach the gradient
    error = (target.float()[valid] - mean.float()[valid]).abs()
    if log_scale is None:
        return error.sum() / (mean.numel() if legacy_l1_reduction else valid.sum())
    s = log_scale.float()[valid].clamp(-6., 2.)
    nll = error * torch.exp(-s) + s
    if not samplewise:
        return nll.mean()
    # average per example first, then over the batch
    counts = valid.reshape(len(mean), -1).sum(1)
    ids = torch.arange(len(mean), device=mean.device)[:, None, None].expand_as(mean)[valid]
    sums = nll.new_zeros(len(mean)).scatter_add(0, ids, nll)
    return (sums / counts).mean()


def kl_divergence(mu, logvar):
    mu, logvar = mu.float().flatten(1), logvar.float().flatten(1)
    return (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(1).mean()


class ACTPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from detr.models.detr_vae import build
        args = dict(hidden_dim=256, dropout=0.1, nheads=8, dim_feedforward=2048,
                    enc_layers=4, dec_layers=7, pre_norm=False, backbone='resnet18',
                    dilation=False, position_embedding='sine', masks=False,
                    lr_backbone=1e-5, num_queries=50,
                    camera_names=['agentview', 'robot0_eye_in_hand'],
                    state_dim=14, action_dim=None)
        args.update(cfg)
        device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build(SimpleNamespace(**args)).to(device)

        backbone = [p for n, p in self.model.named_parameters() if p.requires_grad and 'backbone' in n]
        rest = [p for n, p in self.model.named_parameters() if p.requires_grad and 'backbone' not in n]
        self.optimizer = torch.optim.AdamW(
            [{'params': rest}, {'params': backbone, 'lr': args['lr_backbone']}],
            lr=cfg.get('lr', 1e-4), weight_decay=cfg.get('weight_decay', 1e-4), foreach=False)

        self.kl_weight = float(cfg.get('kl_weight', 10.))
        self.legacy_l1_reduction = cfg.get('legacy_l1_reduction', True)
        self.samplewise_nll = cfg.get('samplewise_nll', False)
        self.gripper_qpos_indices = tuple(cfg.get('gripper_qpos_indices', (7, 8)))
        self.gripper_open_width_m = float(cfg.get('gripper_open_width_m', .06))
        # qpos stats are only needed to read the real gripper width for the
        # phase-aware variant; they come from stats.npz, not the checkpoint
        self.register_buffer('_qpos_mean', None, persistent=False)
        self.register_buffer('_qpos_std', None, persistent=False)

    def set_observation_normalization(self, mean, std):
        device = next(self.model.parameters()).device
        self._qpos_mean = torch.as_tensor(mean, device=device, dtype=torch.float32).clone()
        self._qpos_std = torch.as_tensor(std, device=device, dtype=torch.float32).clone()

    def gripper_open_from_qpos(self, qpos):
        """True while finger width |q7|+|q8| >= gripper_open_width_m (60 mm)."""
        if self._qpos_mean is None:
            raise ValueError('call set_observation_normalization first')
        raw = qpos * self._qpos_std + self._qpos_mean
        width = raw[:, list(self.gripper_qpos_indices)].abs().sum(-1)
        return width >= self.gripper_open_width_m

    def forward(self, qpos, image, actions=None, is_pad=None, *, return_uncertainty=False,
                gripper_open=None):
        mean_rgb = image.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
        std_rgb = image.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
        image = (image - mean_rgb) / std_rgb
        if actions is not None:
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

        kwargs = {}
        if self.model.phase_aware_wrist_attention:
            kwargs['gripper_open'] = self.gripper_open_from_qpos(qpos) if gripper_open is None else gripper_open
        raw, _, (mu, logvar) = self.model(qpos, image, None, actions, is_pad, **kwargs)
        if self.model.use_uncertainty:
            mean, s = raw.chunk(2, dim=-1)
        else:
            mean, s = raw, None

        if actions is None:
            return ActionPrediction(mean, s) if return_uncertainty else mean
        recon = reconstruction_loss(mean, actions, is_pad, s,
                                    legacy_l1_reduction=self.legacy_l1_reduction,
                                    samplewise=self.samplewise_nll)
        kl = kl_divergence(mu, logvar)
        name = 'l1' if s is None else 'nll'
        return {name: recon, 'kl': kl, 'loss': recon + self.kl_weight * kl}

    def configure_optimizers(self):
        return self.optimizer
