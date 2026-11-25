"""Gradient accumulation and resumable training state."""
import torch


class AccumulatingOptimizer:
    def __init__(self,model,optimizer,scaler=None,max_norm=1.):
        self.model,self.optimizer,self.scaler,self.max_norm=model,optimizer,scaler,max_norm
        self.samples=0
        optimizer.zero_grad(set_to_none=True)

    def backward(self,mean_loss,batch_size):
        if batch_size<1:
            raise ValueError('Empty microbatch')
        total=mean_loss*batch_size
        if self.scaler is None:
            total.backward()
        else:
            self.scaler.scale(total).backward()
        self.samples+=batch_size

    def step(self):
        if self.samples==0:
            raise ValueError('No accumulated samples')
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self.samples)
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.max_norm,error_if_nonfinite=True)
        if self.scaler is None:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.samples=0


def save_training_state(path,model,optimizer,scaler,generator,metadata):
    """Save model, optimizer and RNG state so training can resume mid-run."""
    cuda=next(model.parameters()).is_cuda
    payload=dict(metadata=metadata,model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                 optimizer=optimizer.state_dict(),scaler=None if scaler is None else scaler.state_dict(),
                 generator=generator.get_state(),cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if cuda else None)
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def restore_training_state(path,model,optimizer,scaler,generator):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and state['scaler'] is not None:scaler.load_state_dict(state['scaler'])
    generator.set_state(state['generator'])
    torch.set_rng_state(state['cpu_rng'])
    if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['metadata']
