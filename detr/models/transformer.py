"""ACT transformer plus an optional additive bias on decoder cross-attention.

Upstream layers and parameter names are untouched (see transformer_standard.py).
"""
import torch
from torch import Tensor
from pa_act.math_utils import precision_memory_bias
from .transformer_standard import (
    Transformer as StandardTransformer,
    TransformerEncoder, TransformerEncoderLayer,
    TransformerDecoder, TransformerDecoderLayer,
)


def wrist_attention_bias(num_queries: int, short_horizon: int,
                         camera_widths: list[int], height: int,
                         wrist_index: int, global_scale: float,
                         *, device, dtype, gripper_open=None, num_heads=None) -> Tensor:
    """Bias of shape [K, 2 + H*sum(W)] added to the decoder attention logits.

    Memory layout follows ACT: [latent, qpos, flatten(H, concat_W(cameras))].
    Rows 0..short_horizon-1 get log(global_scale) on non-wrist image tokens,
    every other row is zero.

    If gripper_open (bool [B]) is given, examples with a closed gripper get no
    bias at all and the result is [B*heads, K, L].
    """
    if not 0 < short_horizon <= num_queries:
        raise ValueError('short_horizon must be in [1, num_queries]')
    row = precision_memory_bias(camera_widths, height, wrist_index, global_scale)
    bias = torch.zeros(num_queries, len(row), device=device, dtype=dtype)
    bias[:short_horizon] = torch.tensor(row, device=device, dtype=dtype)
    if gripper_open is not None:
        # torch.where instead of multiply: 0 * -inf would give NaN
        bias = torch.where(gripper_open[:, None, None], bias[None], torch.zeros_like(bias)[None])
        bias = bias.repeat_interleave(num_heads, dim=0)
    return bias


class Transformer(StandardTransformer):
    def forward(self, src, mask, query_embed, pos_embed, latent_input=None,
                proprio_input=None, additional_pos_embed=None, *,
                memory_attention_bias=None):
        if memory_attention_bias is None:
            return super().forward(src, mask, query_embed, pos_embed, latent_input,
                                   proprio_input, additional_pos_embed)
        # same as upstream forward, but passes memory_mask to the decoder
        batch = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        pos = pos_embed.flatten(2).permute(2, 0, 1)
        if pos.shape[1] == 1:
            pos = pos.expand(-1, batch, -1)
        pos = torch.cat((additional_pos_embed[:, None].expand(-1, batch, -1), pos))
        src = torch.cat((torch.stack((latent_input, proprio_input)), src))
        queries = query_embed[:, None].expand(-1, batch, -1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos)
        decoded = self.decoder(torch.zeros_like(queries), memory,
                               memory_mask=memory_attention_bias.to(src.dtype),
                               memory_key_padding_mask=mask, pos=pos, query_pos=queries)
        return decoded.transpose(1, 2)


def build_transformer(args):
    return Transformer(d_model=args.hidden_dim, dropout=args.dropout,
                       nhead=args.nheads, dim_feedforward=args.dim_feedforward,
                       num_encoder_layers=args.enc_layers,
                       num_decoder_layers=args.dec_layers,
                       normalize_before=args.pre_norm,
                       return_intermediate_dec=False)
