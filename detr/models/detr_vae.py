"""ACT model with the two PA-ACT additions.

1. Laplace uncertainty head: the action head outputs 2*A values per query
   (mean and log b) instead of A.
2. Wrist attention bias: the first `short_horizon` decoder queries get an
   additive log(global_attention_scale) on every agentview memory token.

With both switched off this is exactly upstream ACT (same parameter names),
so vanilla checkpoints load without any key changes.
"""
import math
import torch
from torch import nn
from .detr_vae_standard import (DETRVAE as StandardDETRVAE, build_encoder,
                                CNNMLP, build_cnnmlp, mlp,
                                get_sinusoid_encoding_table, reparametrize)
from .transformer import build_transformer, wrist_attention_bias


class DETRVAE(StandardDETRVAE):
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries,
                 camera_names, *, action_dim=None, use_uncertainty=False,
                 use_wrist_attention=False, short_horizon=8,
                 wrist_camera='robot0_eye_in_hand', global_attention_scale=0.1,
                 phase_aware_wrist_attention=False, use_qpos_dropout=False,
                 qpos_dropout_prob=0.3):
        super().__init__(backbones, transformer, encoder, state_dim,
                         num_queries, camera_names)
        self.action_dim = state_dim if action_dim is None else action_dim
        self.state_dim = state_dim
        self.use_uncertainty = use_uncertainty
        self.use_wrist_attention = use_wrist_attention
        self.short_horizon = short_horizon
        self.global_attention_scale = global_attention_scale
        self.phase_aware_wrist_attention = phase_aware_wrist_attention
        self.use_qpos_dropout = use_qpos_dropout
        self.qpos_dropout_prob = float(qpos_dropout_prob)
        if not 0 <= self.qpos_dropout_prob <= 1:
            raise ValueError('qpos_dropout_prob must be in [0, 1]')
        if phase_aware_wrist_attention and not use_wrist_attention:
            raise ValueError('phase_aware_wrist_attention needs use_wrist_attention')
        self.wrist_index = camera_names.index(wrist_camera) if use_wrist_attention else None
        if use_wrist_attention and not 0 < short_horizon <= num_queries:
            raise ValueError('short_horizon must be in [1, num_queries]')

        hidden = transformer.d_model
        # Only replace upstream layers when the size actually differs, so the
        # vanilla model keeps the same init order and checkpoint shapes.
        output_dim = self.action_dim * (2 if use_uncertainty else 1)
        if self.action_head.out_features != output_dim:
            self.action_head = nn.Linear(hidden, output_dim)
        if use_uncertainty:
            # log b starts at 0, i.e. b = 1 for every action dimension
            with torch.no_grad():
                self.action_head.weight[self.action_dim:].zero_()
                self.action_head.bias[self.action_dim:].zero_()
        for name, dim in [('input_proj_robot_state', state_dim),
                          ('encoder_joint_proj', state_dim),
                          ('encoder_action_proj', self.action_dim)]:
            if getattr(self, name).in_features != dim:
                setattr(self, name, nn.Linear(dim, hidden))

    def forward(self, qpos, image, env_state=None, actions=None, is_pad=None, *, gripper_open=None):
        """Returns (prediction, is_pad_logits, [mu, logvar]).

        prediction is [B, K, A] for vanilla ACT and [B, K, 2A] = [mean, log b]
        when the uncertainty head is on.
        """
        batch = len(qpos)
        if self.phase_aware_wrist_attention and gripper_open is None:
            raise ValueError('phase-aware attention needs gripper_open')
        if self.training and self.use_qpos_dropout and self.qpos_dropout_prob > 0:
            # drop the whole proprio vector for some samples (same mask for
            # the CVAE encoder and the decoder)
            keep = torch.rand((batch, 1), device=qpos.device) >= self.qpos_dropout_prob
            qpos = qpos * keep.to(qpos.dtype)

        if actions is not None:
            if is_pad is None:
                is_pad = torch.zeros(actions.shape[:2], device=actions.device, dtype=torch.bool)
            # padded steps can hold garbage, zero them before the encoder
            clean_actions = actions.masked_fill(is_pad[..., None], 0)
            tokens = torch.cat((self.cls_embed.weight[None].expand(batch, -1, -1),
                                self.encoder_joint_proj(qpos)[:, None],
                                self.encoder_action_proj(clean_actions)), dim=1)
            padding = torch.cat((is_pad.new_zeros((batch, 2)), is_pad), dim=1)
            cls = self.encoder(tokens.transpose(0, 1),
                               pos=self.pos_table.transpose(0, 1),
                               src_key_padding_mask=padding)[0]
            latent_mu, logvar = self.latent_proj(cls).chunk(2, dim=-1)
            sample = latent_mu + (0.5 * logvar).exp() * torch.randn_like(latent_mu)
        else:
            # inference: z = 0 (mean of the prior)
            latent_mu = logvar = None
            sample = qpos.new_zeros(batch, self.latent_dim)
        latent = self.latent_out_proj(sample)

        features, positions, widths = [], [], []
        for index in range(len(self.camera_names)):
            maps, pos = self.backbones[0](image[:, index])
            feature = self.input_proj(maps[0])
            features.append(feature)
            positions.append(pos[0])
            widths.append(feature.shape[-1])
        src, pos = torch.cat(features, dim=3), torch.cat(positions, dim=3)

        kwargs = {}
        if self.use_wrist_attention:
            kwargs['memory_attention_bias'] = wrist_attention_bias(
                self.num_queries, self.short_horizon, widths, src.shape[-2],
                self.wrist_index, self.global_attention_scale,
                device=src.device, dtype=src.dtype,
                gripper_open=gripper_open if self.phase_aware_wrist_attention else None,
                num_heads=self.transformer.nhead)
        hidden = self.transformer(src, None, self.query_embed.weight, pos,
                                  latent, self.input_proj_robot_state(qpos),
                                  self.additional_pos_embed.weight, **kwargs)[0]
        prediction = self.action_head(hidden)
        if self.use_uncertainty:
            mean, log_scale = prediction.chunk(2, dim=-1)
            prediction = torch.cat((mean, log_scale.clamp(-6., 2.)), dim=-1)
        return prediction, self.is_pad_head(hidden), [latent_mu, logvar]


def build(args):
    from .backbone import build_backbone
    return DETRVAE([build_backbone(args)], build_transformer(args), build_encoder(args),
                   getattr(args, 'state_dim', 14), args.num_queries, args.camera_names,
                   action_dim=getattr(args, 'action_dim', None),
                   use_uncertainty=getattr(args, 'use_uncertainty', False),
                   use_wrist_attention=getattr(args, 'use_wrist_attention', False),
                   short_horizon=getattr(args, 'short_horizon', 8),
                   wrist_camera=getattr(args, 'wrist_camera', 'robot0_eye_in_hand'),
                   global_attention_scale=getattr(args, 'global_attention_scale', 0.1),
                   phase_aware_wrist_attention=getattr(args, 'phase_aware_wrist_attention', False),
                   use_qpos_dropout=getattr(args, 'use_qpos_dropout', False),
                   qpos_dropout_prob=getattr(args, 'qpos_dropout_prob', 0.3))
