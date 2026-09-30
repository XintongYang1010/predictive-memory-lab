from pathlib import Path
import torch
from torch import nn
from .adapter import ACTIONS, FEATURES
from .common import seed_all, write_json, clock


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean",torch.zeros(len(FEATURES)))
        self.register_buffer("std",torch.ones(len(FEATURES)))
        self.net=nn.Sequential(nn.Linear(len(FEATURES),64),nn.GELU(),nn.Linear(64,64),nn.GELU(),nn.Linear(64,len(ACTIONS)))

    def forward(self,x):
        return self.net(((x-self.mean)/self.std).clamp(-20,20))

    @torch.no_grad()
    def choose(self,feature):
        return ACTIONS[int(self(feature).argmax(-1))]


def train_controller(train_rows,dev_rows,out,device="cpu",steps=500,batch_size=64,seed=0):
    seed_all(seed)
    if not train_rows or not dev_rows:
        raise ValueError("Nonempty, scene-disjoint train and dev labels required")
    if {r["scene_id"] for r in train_rows}&{r["scene_id"] for r in dev_rows}:
        raise ValueError("Scene leakage between controller train/dev")
    model=Controller().to(device)
    x=torch.tensor([r["features"] for r in train_rows],dtype=torch.float32,device=device)
    y=torch.tensor([r["label"] for r in train_rows],device=device)
    weight=torch.tensor([r["weight"] for r in train_rows],device=device)
    xd=torch.tensor([r["features"] for r in dev_rows],dtype=torch.float32,device=device)
    yd=torch.tensor([r["label"] for r in dev_rows],device=device)
    model.mean.copy_(x.mean(0))
    model.std.copy_(x.std(0,unbiased=False).clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=float("inf")
    best_weights=None
    history=[]
    start=clock(device)
    for step in range(1,steps+1):
        model.train()
        indices=torch.randint(len(x),(min(batch_size,len(x)),),device=device)
        loss=(nn.functional.cross_entropy(model(x[indices]),y[indices],reduction="none")*weight[indices]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step==1 or step%50==0 or step==steps:
            model.eval()
            with torch.no_grad():
                dev=float(nn.functional.cross_entropy(model(xd),yd))
            history.append({"step":step,"train_loss":float(loss.detach()),"dev_loss":dev})
            print(f"controller step {step}/{steps}: train={float(loss.detach()):.4f} dev={dev:.4f}",flush=True)
            if dev<best:
                best=dev
                best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_weights is None:
        raise RuntimeError("No finite controller checkpoint")
    elapsed=clock(device)-start
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":best_weights,"actions":list(ACTIONS),"features":list(FEATURES),
                "seed":seed,"steps":steps,"train_scenes":sorted({r["scene_id"] for r in train_rows}),
                "dev_scenes":sorted({r["scene_id"] for r in dev_rows})},out)
    write_json(out.with_suffix(".training.json"),{"history":history,"seconds":elapsed,
               "parameters":sum(p.numel() for p in model.parameters()),"train_rows":len(train_rows),"dev_rows":len(dev_rows)})
    model.load_state_dict(best_weights)
    return model.eval()


def load_controller(path,device="cpu"):
    obj=torch.load(path,map_location=device,weights_only=True)
    if obj["features"]!=list(FEATURES) or obj["actions"]!=list(ACTIONS):
        raise ValueError("Controller schema mismatch")
    model=Controller().to(device)
    model.load_state_dict(obj["state_dict"],strict=True)
    return model.eval()
