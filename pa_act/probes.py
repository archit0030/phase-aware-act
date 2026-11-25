"""Extra per-step logging during evaluation (does not change the actions).

- the executed (ensembled) action, so an episode can be replayed later
- how much decoder cross-attention goes to the wrist vs. agentview camera
"""
import torch


def attention_masses(weights, camera_widths, height, wrist_index, short_horizon=8):
    """Split attention [B, K, L] into latent / proprio / wrist / external mass.

    Image tokens follow ACT's layout: cameras are concatenated along width
    before flattening, so each row of H holds all cameras side by side.
    """
    if weights.ndim == 4:  # per-head weights -> average heads
        weights = weights.mean(1)
    spatial = weights[..., 2:].reshape(*weights.shape[:-1], height, sum(camera_widths))
    start = sum(camera_widths[:wrist_index])
    stop = start + camera_widths[wrist_index]
    wrist = spatial[..., start:stop].sum((-1, -2))
    groups = dict(latent=weights[..., 0], proprio=weights[..., 1], wrist=wrist,
                  external=spatial.sum((-1, -2)) - wrist)
    out = {}
    for name, rows in [('short', slice(0, short_horizon)), ('long', slice(short_horizon, None))]:
        if weights[:, rows].shape[1]:
            out[name] = {k: float(v[:, rows].mean()) for k, v in groups.items()}
    return out


class AttentionProbe:
    """Forward hooks that read the decoder attention weights."""

    def __init__(self, policy):
        self.shapes, self.weights, self.handles = [], [], []
        self.wrist_index = policy.model.camera_names.index('robot0_eye_in_hand')
        self.handles.append(policy.register_forward_pre_hook(self._reset))
        self.handles.append(policy.model.input_proj.register_forward_hook(self._feature))
        for layer in policy.model.transformer.decoder.layers:
            self.handles.append(layer.multihead_attn.register_forward_hook(self._attention))

    def _reset(self, module, inputs):
        self.shapes, self.weights = [], []

    def _feature(self, module, inputs, output):
        self.shapes.append(tuple(output.shape[-2:]))

    def _attention(self, module, inputs, output):
        if output[1] is not None:
            self.weights.append(output[1].detach())

    def summarize(self):
        if not self.shapes or not self.weights:
            return None
        height = self.shapes[0][0]
        widths = [s[1] for s in self.shapes]
        return [attention_masses(w, widths, height, self.wrist_index) for w in self.weights]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class RolloutLogger:
    def __init__(self, policy, attention_every=10):
        self.probe = AttentionProbe(policy)
        self.every = attention_every

    @torch.no_grad()
    def observe(self, t, prediction, result):
        fresh = prediction.mean[0, 0].float()
        return {
            'selected_action_normalized': result.action.cpu().tolist(),
            'fresh_action_normalized': fresh.cpu().tolist(),
            'fresh_weight': float(result.weights[-1]),
            'decoder_attention_by_layer': self.probe.summarize() if t % self.every == 0 else None,
        }

    def close(self):
        self.probe.close()
