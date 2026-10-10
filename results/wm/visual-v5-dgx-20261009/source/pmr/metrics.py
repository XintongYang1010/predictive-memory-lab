"""Per-scene metrics with identical valid timestamps across paired branches."""
import torch


def scores(pred, prob, target, valid):
    # Single scene/focal actor: [K,H,2], [K], [H,2], [H].
    if not bool(valid.any()):
        raise ValueError("No valid evaluation timestamps")
    p=prob / prob.sum()
    pred,target=pred[:,valid],target[valid]
    d=(pred-target).norm(dim=-1)
    fde=d[:,-1]
    best_fde=int(fde.argmin())
    z=pred.flatten(1)/(pred.shape[1]**0.5)
    y=target.flatten()/(pred.shape[1]**0.5)
    es=(p*(z-y).norm(dim=-1)).sum()-0.5*(p[:,None]*p[None,:]*torch.cdist(z,z)).sum()
    top=int(p.argmax())
    return {"energy_score":float(es),"minADE6":float(d.mean(-1).min()),
            "minFDE6":float(fde.min()),"minADE1":float(d[top].mean()),
            "minFDE1":float(fde[top]),"miss_rate":float(fde.min()>2),
            "b_minFDE6":float(fde[best_fde]+(1-p[best_fde])**2),
            "b_minFDE6_upstream":float((fde+(1-p)**2).min()),
            "mode_entropy":float(-(p*p.clamp_min(1e-12).log()).sum()),
            "valid_steps":int(valid.sum())}


def frame_scores(out, frame, common=False):
    pred=out["y_hat"][0]
    target=frame["target"][0,0]
    valid=frame["target_mask"][0,0]
    if common:
        # Updates at steps 40 and 50 both evaluate absolute [50,100).
        step=round(float(frame["timestamp"][0])*10)
        lo,hi=50-step,100-step
        if lo<0 or hi>pred.shape[1]:
            raise ValueError("Common target interval outside forecast")
        pred,target,valid=pred[:,lo:hi],target[lo:hi],valid[lo:hi]
    return scores(pred,out["pi"][0].softmax(-1),target,valid)
