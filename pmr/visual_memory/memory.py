"""Explicit persistent state; state cost is measured, never estimated by tokens."""
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = {"A0": ("add",False), "A1": ("add",True),
            "B0": ("gated_add",False), "B1": ("gated_add",True),
            "C0": ("delta",False), "C1": ("delta",True)}


class Writer(nn.Module):
    def __init__(self, rule, heads=4, width=32):
        super().__init__()
        self.rule, self.heads, self.width = rule, heads, width
        if rule not in ("add","gated_add","delta"):
            raise ValueError(rule)
        if rule != "add":
            self.gate = nn.Linear(2*width,2)
            nn.init.zeros_(self.gate.weight)
            # alpha in [0.95,1], beta in [0,1]; matched initial gate networks.
            nn.init.constant_(self.gate.bias, 0.)

    def reset(self, batch, device):
        return torch.zeros(batch,self.heads,self.width,self.width,device=device,dtype=torch.float32)

    def write(self, state, k, v, valid=None):
        k, v = F.normalize(k.float(), dim=-1, eps=1e-6), v.float()
        if self.rule == "add":
            out = state + k.unsqueeze(-1) * v.unsqueeze(-2)
        else:
            gate = self.gate(torch.cat([k,v],-1)).sigmoid()
            a = .95 + .05*gate[...,0,None,None]
            b = gate[...,1,None,None]
            decayed = a * state
            update = v
            if self.rule == "delta":
                update = v - torch.einsum("bhd,bhde->bhe",k,decayed)
            out = decayed + b*k.unsqueeze(-1)*update.unsqueeze(-2)
        if valid is not None:
            out = torch.where(valid[:,None,None,None],out,state)
        return out

    def replay(self, keys, values, valid):
        # keys [B,H,T*S,D], fixed chronological frame then raster patch order.
        state = self.reset(keys.shape[0],keys.device)
        for i in range(keys.shape[2]):
            state = self.write(state,keys[:,:,i],values[:,:,i],valid[:,i])
        return state


class Reader(nn.Module):
    def __init__(self, width=32, calibrated=False, rank=8):
        super().__init__()
        self.calibrated = calibrated
        if calibrated:
            self.down = nn.Linear(width,rank,bias=False)
            self.up = nn.Linear(rank,width,bias=False)
            nn.init.zeros_(self.up.weight)

    def read(self, query, state, disable_adapter=False):
        if self.calibrated and not disable_adapter:
            query = query + self.up(self.down(query))
        return torch.einsum("bhnd,bhde->bhne",F.normalize(query.float(),dim=-1,eps=1e-6),state)


def tensor_bytes(*tensors):
    return sum(x.numel()*x.element_size() for x in tensors if x is not None)


def persistent_cost(heads=4, width=32, tokens=16, recent=4, exact=8, kv_bytes=2,
                    memory=True):
    # Recent independent encoded tokens are FP32; exact history is BF16 K/V.
    # Public time/action are already encoded; four int64 controls are included.
    recent_bytes = recent*tokens*heads*width*4
    kv = exact*2*tokens*heads*width*kv_bytes
    state = heads*width*width*4 if memory else 0
    return dict(recent_bytes=recent_bytes,exact_kv_bytes=kv, matrix_bytes=state, control_bytes=32,
                total_bytes=recent_bytes+kv+state+32, recent_frames=recent, exact_frames=exact)


def matched_fifo_frames(heads=4,width=32,tokens=16,exact=8,kv_bytes=2):
    return exact + heads*width*width*4//(2*tokens*heads*width*kv_bytes)


class MemoryBank:
    """Bounded inference reference. One new independent observation per append.

    Project must produce [B,H,S,D] K/V from the departing recent frame.
    Entries are stored once; FIFO moves are always chronological.
    """
    def __init__(self,writer,batch_size,device,recent_frames=4,exact_frames=8):
        self.writer,self.batch_size,self.device=writer,batch_size,device
        self.recent_frames,self.exact_frames=recent_frames,exact_frames
        self.reset()

    def reset(self):
        self.recent=[];self.exact=[]
        self.state=self.writer.reset(self.batch_size,self.device)
        self.control=torch.zeros(4,dtype=torch.int64,device=self.device)

    def append(self,frame,project):
        self.control[0]+=1
        self.recent.append(frame.float())
        if len(self.recent)>self.recent_frames:
            old=self.recent.pop(0);k,v=project(old)
            self.exact.append((k.to(torch.bfloat16),v.to(torch.bfloat16)))
            if len(self.exact)>self.exact_frames:
                k,v=self.exact.pop(0)
                for i in range(k.shape[2]):
                    self.state=self.writer.write(self.state,k[:,:,i],v[:,:,i])
                self.control[1]+=1
        self.control[2]=len(self.recent);self.control[3]=len(self.exact)

    def read(self,query,reader):
        return reader.read(query,self.state)

    def bytes(self):
        return tensor_bytes(self.state,self.control,*self.recent,*[x for kv in self.exact for x in kv])
