"""LIBERO HDF5 chunks: episode boundaries, normalization and RGB orientation."""
from bisect import bisect_right
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def read_qpos(demo, config):
    return np.concatenate([np.asarray(demo['obs'][key], dtype=np.float64).reshape(len(demo['actions']), -1)
                           for key in config['dataset_qpos_keys']], axis=-1)


def inspect_demonstrations(path, config, required=50):
    offset=config.get('dataset_action_offset',0)
    if type(offset) is not int or offset<0:raise ValueError('Invalid dataset action offset')
    with h5py.File(path,'r') as handle:
        keys = sorted((key for key in handle['data'] if key.startswith('demo_')),
                      key=lambda key: int(key.rsplit('_',1)[1]))
        if len(keys) < required:
            raise ValueError(f'{path}: need {required} Standard demonstrations, got {len(keys)}')
        keys = keys[:required]
        for key in keys:
            demo = handle['data'][key]
            actions = np.asarray(demo['actions'])
            qpos = read_qpos(demo, config)
            if actions.ndim != 2 or actions.shape[1] != config['action_dim'] or len(actions) <= offset:
                raise ValueError(f'Invalid action shape in {key}')
            if qpos.shape != (len(actions),config['state_dim']):
                raise ValueError(f'Invalid joint/gripper state shape in {key}')
            if not np.isfinite(actions).all() or not np.isfinite(qpos).all():
                raise ValueError(f'Non-finite demonstration values in {key}')
            for camera in config['dataset_image_keys']:
                images = demo['obs'][camera]
                expected = (len(actions),config.get('image_height',128), config.get('image_width',128),3)
                if images.shape != expected or images.dtype != np.uint8:
                    raise ValueError(f'Expected uint8 {expected} images at {key}/{camera}, got {images.shape}')
    return keys


def normalization_stats(path, keys, config, floor=.01):
    if floor <= 0 or not np.isfinite(floor):
        raise ValueError('Normalization std floor must be finite and positive')
    totals = {}
    offset=config.get('dataset_action_offset',0)
    with h5py.File(path,'r') as handle:
        for key in keys:
            demo=handle['data'][key]
            qpos=read_qpos(demo,config)
            actions=np.asarray(demo['actions'],dtype=np.float64)
            if offset:qpos=qpos[:-offset]
            actions=actions[offset:]
            for name, values in [('qpos',qpos),('action',actions)]:
                count, total, squared = totals.get(name,(0,0.,0.))
                totals[name]=(count+len(values),total+values.sum(0),squared+(values**2).sum(0))
    stats={}
    for name,(count,total,squared) in totals.items():
        mean=total/count
        std=np.sqrt(np.maximum(0.,squared/count-mean**2))
        stats[name+'_mean']=mean.astype(np.float32)
        stats[name+'_std']=np.maximum(std,floor).astype(np.float32)
    return stats


class LiberoChunkDataset(Dataset):
    def __init__(self,path,keys,config,stats):
        self.path,self.keys,self.config,self.stats=str(path),list(keys),config,stats
        self.action_offset=config.get('dataset_action_offset',0)
        if type(self.action_offset) is not int or self.action_offset<0:raise ValueError('Invalid dataset action offset')
        self.offsets=[0]
        with h5py.File(path,'r') as handle:
            for key in keys:
                length=len(handle['data'][key]['actions'])-self.action_offset
                if length<1:raise ValueError('Trajectory has no future action targets')
                self.offsets.append(self.offsets[-1]+length)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self,index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode=bisect_right(self.offsets,index)-1
        t=index-self.offsets[episode]
        k=self.config['num_queries']
        # Per-sample context avoids sharing h5py handles across DataLoader workers.
        with h5py.File(self.path,'r') as handle:
            demo=handle['data'][self.keys[episode]]
            q=np.concatenate([np.asarray(demo['obs'][key][t]).reshape(-1)
                              for key in self.config['dataset_qpos_keys']]).astype(np.float32)
            image=np.stack([np.asarray(demo['obs'][key][t]) for key in self.config['dataset_image_keys']])
            if self.config.get('dataset_flip_images_vertical',False):
                image=image[:,::-1]
            start=t+self.action_offset
            real=np.asarray(demo['actions'][start:start+k],dtype=np.float32)
        action=np.zeros((k,self.config['action_dim']),dtype=np.float32)
        action[:len(real)]=(real-self.stats['action_mean'])/self.stats['action_std']
        pad=np.arange(k)>=len(real)
        q=(q-self.stats['qpos_mean'])/self.stats['qpos_std']
        return (torch.from_numpy(q),torch.from_numpy(image.copy()).permute(0,3,1,2).float()/255.,
                torch.from_numpy(action),torch.from_numpy(pad))
