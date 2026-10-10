"""One explicit history read; every cached entry is independently encoded.

The Python training forward replays a legal prefix. Its autograd workspace is
not claimed as persistent inference state. Recent tokens, exact KV and the
matrix are the only history-bearing quantities used by the prediction.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .memory import VARIANTS,Writer,Reader,matched_fifo_frames


@dataclass
class ActorBatch:
    features: torch.Tensor
    times: torch.Tensor
    shutters: torch.Tensor
    valid: torch.Tensor
    query_time: torch.Tensor


def batch(rows,device):
    n=max(len(r["features"]) for r in rows)
    b,s,f=len(rows),rows[0]["features"].shape[1],rows[0]["features"].shape[2]
    x=torch.zeros(b,n,s,f); t=torch.zeros(b,n); a=torch.zeros(b,n,2)
    valid=torch.zeros(b,n,dtype=torch.bool)
    for i,r in enumerate(rows):
        length=len(r["features"])
        x[i,-length:]=r["features"].float(); t[i,-length:]=r["times"]
        a[i,-length:]=r["shutters"]; valid[i,-length:]=True
    actor=ActorBatch(x.to(device),t.to(device),a.to(device),valid.to(device),t[:,-1].to(device))
    target=torch.stack([r["targets"].float() for r in rows]).to(device)
    masks=torch.stack([r["target_masks"] for r in rows]).to(device)
    return actor,target,masks


class Predictor(nn.Module):
    def __init__(self,feature_dim=1280,tokens=16,dim=128,heads=4,local_layers=3):
        super().__init__()
        self.config=dict(feature_dim=feature_dim,tokens=tokens,dim=dim,heads=heads,local_layers=local_layers)
        self.dim,self.heads,self.width,self.tokens=dim,heads,dim//heads,tokens
        self.register_buffer("mean",torch.zeros(feature_dim))
        self.register_buffer("std",torch.ones(feature_dim))
        self.input=nn.Linear(feature_dim,dim)
        self.space=nn.Parameter(torch.randn(tokens,dim)*.02)
        self.action=nn.Linear(2,dim)
        layer=nn.TransformerEncoderLayer(dim,heads,4*dim,dropout=0,batch_first=True,norm_first=True)
        self.local=nn.TransformerEncoder(layer,local_layers,enable_nested_tensor=False)
        self.local_read=nn.MultiheadAttention(dim,heads,dropout=0,batch_first=True)
        self.q=nn.Linear(dim,dim);self.k=nn.Linear(dim,dim);self.v=nn.Linear(dim,dim)
        self.history_out=nn.Linear(dim,dim)
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim*4),nn.GELU(),nn.Linear(dim*4,feature_dim))

    def norm_target(self,y):
        return (y-self.mean)/self.std

    def time_embedding(self,t):
        f=torch.exp(torch.arange(self.dim//2,device=t.device)*(-9.21034/(self.dim//2)))
        z=t[...,None]*f
        return torch.cat([z.sin(),z.cos()],-1)

    def split(self,x):
        b,n,_=x.shape
        return x.reshape(b,n,self.heads,self.width).transpose(1,2)

    def prepare(self,a):
        if a.features.shape[1]<4 or not a.valid[:,-4:].all():
            raise ValueError("Need four real recent observations")
        x=self.input(self.norm_target(a.features))+self.space
        x=x+self.time_embedding(a.times)[:,:,None,:]+self.action(a.shutters)[:,:,None,:]
        local=self.local(x[:,-4:].flatten(1,2))
        horizons=a.query_time.new_tensor([1,4,8])
        when=a.query_time[:,None]+horizons[None,:]
        queries=x[:,-1:, :, :].expand(-1,3,-1,-1)+self.time_embedding(when)[:,:,None,:]
        queries=queries+self.action(torch.ones(len(x),3,2,device=x.device))[:,:,None,:]
        queries=queries.flatten(1,2)
        queries=queries+self.local_read(queries,local,local,need_weights=False)[0]
        historical=x[:,:-4].flatten(1,2)
        q,k,v=self.split(self.q(queries)),self.split(self.k(historical)),self.split(self.v(historical))
        mask=a.valid[:,:-4].repeat_interleave(self.tokens,dim=1)
        # All exact-history arms use the same BF16 cache representation.
        return queries,q,k.to(torch.bfloat16).float(),v.to(torch.bfloat16).float(),mask

    def attend(self,q,k,v,mask):
        if k.shape[2]==0:
            return torch.zeros_like(q)
        scores=q@k.transpose(-1,-2)/(self.width**.5)
        scores=scores.masked_fill(~mask[:,None,None,:],-1e9)
        weights=scores.softmax(-1)*mask[:,None,None,:]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        return weights@v

    def finish(self,queries,history):
        history=history.transpose(1,2).reshape(len(queries),-1,self.dim)
        return self.head(queries+self.history_out(history)).reshape(len(queries),3,self.tokens,-1)

    def forward(self,a,mode="full",drop_history=False):
        z,q,k,v,mask=self.prepare(a)
        if mode=="recent":mask=torch.zeros_like(mask)
        if drop_history:
            mask=mask & (torch.rand(mask.shape[0],mask.shape[1]//self.tokens,device=mask.device)>.25).repeat_interleave(self.tokens,1)
        return self.finish(z,self.attend(q,k,v,mask))


class Candidate(nn.Module):
    def __init__(self,backbone,variant):
        super().__init__()
        self.backbone=backbone.requires_grad_(False)
        self.variant=variant
        if variant in VARIANTS:
            rule,calibrated=VARIANTS[variant]
            self.writer=Writer(rule,backbone.heads,backbone.width)
            self.reader=Reader(backbone.width,calibrated)
            self.memory_scale=nn.Parameter(torch.full((backbone.heads,),.1))
        elif variant not in ("full","recent","fifo"):
            raise ValueError(variant)
        self.calibration_gain=nn.Parameter(torch.ones(backbone.heads,backbone.width))
        self.calibration_bias=nn.Parameter(torch.zeros(backbone.heads,backbone.width))

    def forward(self,a,zero_memory=False,swap_index=None,fixed_state=None,disable_adapter=False,return_state=False):
        b=self.backbone
        with torch.no_grad():z,q,k,v,mask=b.prepare(a)
        history_mask=mask.clone()
        state=None
        if self.variant in VARIANTS:
            cut=max(0,k.shape[2]-8*b.tokens)
            state=self.writer.replay(k[:,:,:cut],v[:,:,:cut],mask[:,:cut]) if fixed_state is None else fixed_state
            history_mask[:,:cut]=False
            if swap_index is not None:state=state[swap_index]
        elif self.variant=="recent":history_mask[:]=False
        elif self.variant=="fifo":
            frames=matched_fifo_frames(b.heads,b.width,b.tokens)
            history_mask[:,:max(0,k.shape[2]-frames*b.tokens)]=False
        read=b.attend(q,k,v,history_mask)
        if state is not None and not zero_memory:
            memory=self.reader.read(q,state,disable_adapter=disable_adapter)
            memory=memory/(memory.square().mean(-1,keepdim=True)+1e-6).sqrt()
            read=read+self.memory_scale[None,:,None,None]*memory
        read=read*self.calibration_gain[None,:,None,:]+self.calibration_bias[None,:,None,:]
        pred=b.finish(z,read)
        return (pred,state) if return_state else pred

    def train(self,mode=True):
        super().train(mode);self.backbone.eval();return self


@torch.no_grad()
def set_normalization(model,train):
    # Per-channel train-only statistics; no Dev/Test fitting.
    total=0;sm=torch.zeros_like(model.mean,dtype=torch.float64);ss=sm.clone()
    for r in train:
        x=r["features"].double().flatten(0,1)
        sm+=x.sum(0);ss+=x.square().sum(0);total+=len(x)
    mean=sm/total;std=(ss/total-mean.square()).clamp_min(.01).sqrt()
    model.mean.copy_(mean.float());model.std.copy_(std.float())
