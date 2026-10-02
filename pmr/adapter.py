"""Explicit state and interventions around an unmodified RealMotion forward.

The intervention unit is the previous update's six trajectory modes. Scene
context is preserved, except for the separately named reset_all diagnostic.
"""
import math
import torch
from .common import clone, sha256
from .upstream import activate

ACTIONS = ("keep", "reweight", "rebuild", "reset")
DIAGNOSTICS = ACTIONS + ("drop_worst", "random_drop", "decay", "reset_all")
FEATURES = ("min_rmse", "max_rmse", "mean_rmse", "std_rmse", "weighted_rmse",
            "prior_entropy", "age_seconds", "speed", "abs_heading_change",
            "observed_fraction", "endpoint_spread", "has_evidence",
            "update_time_seconds", "rebuild_available")
FEATURE_SCHEMA = 2
MODEL_KWARGS = dict(embed_dim=128, future_steps=60, encoder_depth=4,
                    num_heads=8, mlp_ratio=4., qkv_bias=False, drop_path=.2)


def observed_only(frame):
    return {k:clone(v) for k,v in frame.items() if k not in ("target","target_mask","memory_dict")}


def rotation(theta):
    c,s=theta.cos(),theta.sin()
    return torch.stack((torch.stack((c,-s),-1),torch.stack((s,c),-1)),-2)


def evidence(frame, state):
    """Fresh observed overlap in world coordinates, before upstream recentering.

    glo_y_hat in this pinned upstream is a world-oriented displacement,
    despite its name: add its saved origin to obtain absolute coordinates.
    """
    mem=state["memory"]
    previous_step=round(float(mem["timestamp"][0])*10)
    current_step=round(float(frame["timestamp"][0])*10)
    history=frame["x_positions"][0,0]
    hist_start=current_step-len(history)
    lo=max(previous_step,hist_start)
    hi=min(current_step,previous_step+mem["glo_y_hat"].shape[2])
    prior=mem["_prob"][0]
    age=float(frame["timestamp"][0]-mem["timestamp"][0])
    mse=prior.new_zeros(prior.shape)
    valid_count=0
    if hi>lo:
        mask=frame["x_valid_mask"][0,0,lo-hist_start:hi-hist_start]
        valid_count=int(mask.sum())
        if valid_count:
            actual=history[lo-hist_start:hi-hist_start] @ rotation(frame["theta"][0]).T + frame["origin"][0]
            predicted=mem["glo_y_hat"][0,:,lo-previous_step:hi-previous_step]+mem["origin"][0]
            mse=(predicted[:,mask]-actual[mask]).square().sum(-1).mean(-1)
    rmse=mse.sqrt()
    endpoints=mem["glo_y_hat"][0,:,-1]
    spread=(endpoints-endpoints.mean(0)).norm(dim=-1).mean()
    angle=(frame["theta"][0]-mem["theta"][0]+math.pi)%(2*math.pi)-math.pi
    feat=torch.stack((rmse.min(),rmse.max(),rmse.mean(),rmse.std(unbiased=False),
                      (prior*rmse).sum(),-(prior*prior.clamp_min(1e-12).log()).sum(),
                      rmse.new_tensor(age),frame["x_velocity"][0,0,-1],angle.abs(),
                      rmse.new_tensor(valid_count/max(hi-lo,1)),spread,
                      rmse.new_tensor(float(valid_count>0)),
                      rmse.new_tensor(current_step/10.),
                      rmse.new_tensor(float(state["has_prior_context"])) ))
    return {"features":feat,"mse":mse,"prior":prior,"age":age,
            "valid_count":valid_count,"from_step":lo,"until_step":hi}


class RealMotionAdapter:
    def __init__(self, device="cpu", checkpoint=None, trust_checkpoint=False, random_init=False):
        module,_=activate()
        self.device=torch.device(device)
        self.model=module.RealMotion(**MODEL_KWARGS).to(self.device)
        if checkpoint is None and not random_init:
            raise ValueError("Real experiments require --checkpoint; random weights are smoke-only")
        self.checkpoint_hash="RANDOM_INIT_SMOKE_ONLY"
        if checkpoint:
            obj=torch.load(checkpoint,map_location="cpu",weights_only=not trust_checkpoint)
            weights=obj.get("state_dict",obj)
            # Released remotion.pth uses model.*, while the published loader
            # expects net.*. Normalize known wrappers, then require every key.
            for prefix in ("module.","net.","model."):
                if weights and all(k.startswith(prefix) for k in weights):
                    weights={k[len(prefix):]:v for k,v in weights.items()}
            # Historical NATTEN versions sometimes saved optional relative bias.
            modules=dict(self.model.named_modules())
            for key,value in weights.items():
                if key.endswith(".rpb") and key[:-4] in modules:
                    layer=modules[key[:-4]]
                    if getattr(layer,"rpb",None) is None:
                        layer.rpb=torch.nn.Parameter(torch.zeros_like(value,device=self.device))
            self.model.load_state_dict(weights,strict=True)
            self.checkpoint_hash=sha256(checkpoint)
        self.model.eval().requires_grad_(False)
        self._edit="keep"
        self._mask=None
        self._scale=1.
        self.forward_calls=0
        self.model.traj_interact.register_forward_pre_hook(self._before_interaction,with_kwargs=True)
        self.model.traj_interact.register_forward_hook(self._after_interaction)
        self.model.stream_loc.register_forward_hook(self._after_correction)

    def _before_interaction(self,module,args,kwargs):
        if self._mask is not None:
            kwargs=dict(kwargs)
            kwargs["mask"]=self._mask.expand(args[0].shape[1],-1)
        return args,kwargs

    def _after_interaction(self,module,args,output):
        return args[0] if self._edit=="reset" else output

    def _after_correction(self,module,args,output):
        return torch.zeros_like(output) if self._edit=="reset" else output*self._scale

    @torch.no_grad()
    def _forward(self,frame,memory,edit="keep",mask=None,scale=1.):
        self._edit,self._mask,self._scale=edit,mask,scale
        data=observed_only(frame)
        data["memory_dict"]=clone(memory)
        try:
            self.forward_calls+=1
            out=self.model(data)
        finally:
            self._edit,self._mask,self._scale="keep",None,1.
        for k in ("y_hat","pi"):
            if not bool(torch.isfinite(out[k]).all()):
                raise FloatingPointError(f"Nonfinite {k}")
        out["memory_dict"]["_prob"]=out["pi"].softmax(-1)
        return out

    @torch.no_grad()
    def step(self,frame,state=None,edit="keep",sigma=2.,drop_count=2,seed=0,tau=1.):
        if edit not in DIAGNOSTICS:
            raise ValueError(edit)
        memory=clone(state["memory"]) if state else None
        mask=None
        scale=1.
        if state:
            ev=evidence(frame,state)
            if ev["age"]<=0 or ev["age"]>6:
                raise ValueError("Memory must precede this observation by (0,6] seconds")
            if edit=="reset_all":
                memory=None
            elif edit=="rebuild":
                # Re-encode the same retained previous observation without its
                # historical context. Replace only trajectory-bank fields.
                fresh=self._forward(state["previous_observation"],None)["memory_dict"]
                for k in ("x_mode","glo_y_hat","_prob"):
                    memory[k]=clone(fresh[k])
            elif edit=="reweight" and ev["valid_count"]:
                if sigma<=0:
                    raise ValueError("sigma must be positive")
                logw=ev["prior"].clamp_min(1e-12).log()-ev["mse"]/(2*sigma*sigma)
                mask=logw-logw.max()
            elif edit in ("drop_worst","random_drop") and ev["valid_count"]:
                k=len(ev["mse"])
                if not 0<drop_count<k:
                    raise ValueError("drop_count must leave at least one mode")
                if edit=="drop_worst":
                    indices=ev["mse"].argsort(descending=True)[:drop_count]
                else:
                    gen=torch.Generator(device="cpu").manual_seed(seed)
                    indices=torch.randperm(k,generator=gen)[:drop_count].to(self.device)
                mask=ev["mse"].new_zeros(k)
                mask[indices]=-torch.inf
            elif edit=="decay":
                if tau<=0:
                    raise ValueError("tau must be positive")
                scale=math.exp(-ev["age"]/tau)
        # Ensure exact integer frame lag in upstream's floor-based index.
        # Native schedule is 30/40/50 and yields exactly one second here.
        out=self._forward(frame,memory,edit,mask,scale)
        return out,{"memory":clone(out["memory_dict"]),"previous_observation":observed_only(frame),
                    "has_prior_context":memory is not None}
