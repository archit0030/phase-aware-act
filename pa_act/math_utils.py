"""Shared, dependency-free layout contracts used by tensor code and CPU checks."""
import math


def spatial_indices(action_dim, dimensions=None):
    """LIBERO OSC_POSE: first six are spatial, final coordinate is gripper.

    Other action dimensions retain the original all-coordinate convention unless
    an explicit subset is supplied. A 7D selection must never contain gripper 6.
    """
    if not isinstance(action_dim, int) or action_dim < 1:
        raise ValueError('action_dim must be a positive integer')
    indices = tuple(range(6 if action_dim == 7 else action_dim)) if dimensions is None else tuple(dimensions)
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or i < 0 or i >= action_dim for i in indices)
            or (action_dim == 7 and 6 in indices)):
        raise ValueError('Spatial indices must be unique valid coordinates and exclude LIBERO gripper 6')
    return indices


def precision_memory_bias(camera_widths, height, wrist_index, global_scale):
    """One decoder logit-bias row in ACT's [latent,qpos,flatten(H,W)] order."""
    if (not camera_widths or any(type(w) is not int or w <= 0 for w in camera_widths)
            or type(height) is not int or height <= 0):
        raise ValueError('Camera feature sizes must be positive integers')
    if not 0 <= wrist_index < len(camera_widths):
        raise ValueError('Missing wrist camera')
    if not math.isfinite(global_scale) or not 0 <= global_scale <= 1:
        raise ValueError('global_scale must lie in [0, 1]')
    penalty = math.log(global_scale) if global_scale > 0 else -math.inf
    row = [value for index, width in enumerate(camera_widths)
           for value in [0. if index == wrist_index else penalty] * width]
    return [0., 0.] + row * height


def validate_camera_contract(config):
    """Prevent an image-key permutation silently reversing the precision bias."""
    names=config['camera_names']
    if config['image_obs_keys'] != [name+'_image' for name in names]:
        raise ValueError('image_obs_keys must follow camera_names exactly (camera + _image)')
    dataset_names={'agentview':'agentview_rgb','robot0_eye_in_hand':'eye_in_hand_rgb'}
    if 'dataset_image_keys' in config:
        if any(name not in dataset_names for name in names):
            raise ValueError('Unsupported camera in the standard LIBERO dataset adapter')
        if config['dataset_image_keys'] != [dataset_names[name] for name in names]:
            raise ValueError('dataset_image_keys must follow camera_names in the same camera order')
