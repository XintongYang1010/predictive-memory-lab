#!/usr/bin/env python3
"""Gated V5.1 full-cache/P0/adaptation runner; original V5 gate stays failed."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import v5_confirmation_reader as reader
from v5_temporal_confirmation import SCHEMA as CONFIRM_SCHEMA, score_quartets
from pmr.visual_memory.common import Budget, atomic_json, digest, runtime, source_identity
from pmr.visual_memory.features import Encoder, IJEPA_COMMIT, cache_split, load_records
from pmr.visual_memory.feature_workers import FeatureWorkers
from pmr.visual_memory.memory import VARIANTS
from pmr.visual_memory.slurm_backend import NYU_ACCOUNT, validate_runtime
from pmr.visual_memory.train import train_p0, train_candidate, reader_only
from pmr.visual_memory.report import report as original_report

SCHEMA = "v5.1-temporal-confirmed-pilot-1"
CHECKPOINT_SHA256 = "0382013c481743e9ccea89f970bc6c6aa126aa19a62127500d6e672a641aae22"
ENCODER_IDENTITY = dict(encoder="official-IJEPA-ViT-H14-224-target_encoder",
    code_commit=IJEPA_COMMIT, checkpoint_sha256=CHECKPOINT_SHA256, pretrained=True, grid=4)
EXPECTED_COUNTS = {"confirm": {"blocks": 256, "views": 1024},
                   "bridge": {"paired_worlds": 256, "episodes": 512}}


def read_json(path):
    return json.loads(Path(path).read_text())


def array_identity(array):
    array = np.ascontiguousarray(array)
    return dict(sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                dtype=str(array.dtype), shape=list(array.shape))


def validate_prerequisites(model_folder, confirmation_folder, bridge_folder):
    """Reopen serialized evidence and rescore primary predictions, without GPU.

    Reports are not authorization on their own. The reader verifies model and
    source hashes; this boundary verifies both result receipts, exact counts,
    prediction hashes and all registered primary conditions independently.
    """
    model_folder = Path(model_folder)
    arrays, selections, frozen = reader.load_frozen(model_folder)
    identity = frozen["cache_identity"]
    if identity.get("plan_sha256") != digest(reader.PLAN):
        raise ValueError("Frozen reader uses a different registered plan")
    plan_encoder = read_json(reader.PLAN)["encoder"]
    cache_encoder = {k: plan_encoder[k] for k in ("source_commit", "checkpoint_sha256", "model")}
    if identity.get("encoder_identity") != cache_encoder:
        raise ValueError("Prerequisite encoder differs from the pinned pooled encoder")
    if identity.get("pooling") != "4x4_flattened" or identity.get("feature_dtype") != "float32":
        raise ValueError("Prerequisite representation differs from pooled BF16-derived features")
    sources = identity.get("source_identity", {})
    if not sources or any((ROOT / name).resolve().is_relative_to(ROOT) is False or
                          not (ROOT / name).is_file() or digest(ROOT / name) != value
                          for name, value in sources.items()):
        raise ValueError("Prerequisite extraction source changed after FIT freeze")
    if (frozen.get("predictor_input_audit", {}).get("features_only") is not True or
            frozen.get("predictor_input_audit", {}).get("labels_or_group_ids_used_at_prediction") is not False):
        raise ValueError("Frozen predictor input audit is missing or unsafe")
    evidence = {}
    for role, folder in (("confirm", confirmation_folder), ("bridge", bridge_folder)):
        folder = Path(folder)
        report_path, cache_path, predictions_path = (folder / name for name in
            ("report.json", "cache_receipt.json", "predictions.npz"))
        report, cache = read_json(report_path), read_json(cache_path)
        if (report.get("schema") != CONFIRM_SCHEMA or report.get("role") != role or
                report.get("counts") != EXPECTED_COUNTS[role] or report.get("test_access") is not False):
            raise ValueError("Prerequisite role/count/Test declaration mismatch")
        if (report.get("primary_reader") != "concat" or
                report.get("primary_metric_conditions_met") is not True or
                report.get("original_v5_gate") != "failed_unchanged"):
            raise ValueError("A registered primary prerequisite failed or old gate changed")
        if (report.get("sources") != reader.current_sources() or
                report.get("frozen_reader_sha256") != frozen["receipt_sha256"] or
                report.get("cache_identity") != identity):
            raise ValueError("Prerequisite source/model/cache identity mismatch")
        if (report.get("cache_receipt_sha256") != digest(cache_path) or
                report.get("predictions_sha256") != digest(predictions_path)):
            raise ValueError("Prerequisite artifact SHA256 mismatch")
        if (cache.get("schema") != CONFIRM_SCHEMA or cache.get("role") != role or
                reader.identity(cache) != identity or
                cache.get("cache_sha256") != report.get("cache_sha256") or
                cache.get("frozen_reader_sha256") != frozen["receipt_sha256"] or
                cache.get("frozen_before_generation") is not True or
                cache.get("unseen_before_release") is not True or
                cache.get("test_opened") is not False or cache.get("groups") != 256 or
                cache.get("examples") != (1024 if role == "confirm" else 512) or
                not cache.get("entry_hashes")):
            raise ValueError("Prerequisite extraction/freshness receipt mismatch")
        if role == "confirm" and cache.get("unique_image_rule") != "encode_once_reuse":
            raise ValueError("Confirmation lacks exact reused image anchors")
        with np.load(predictions_path, allow_pickle=False) as z:
            pred, target = z["concat"], z["targets"]
            expected = (256, 4, 8) if role == "confirm" else (256, 2, 8)
            if pred.shape != expected or target.shape != expected:
                raise ValueError("Prerequisite prediction count/shape mismatch")
            ids_key = "block_ids" if role == "confirm" else "pair_ids"
            ids = np.arange(256, dtype=np.int64) if role == "confirm" else np.arange(128, 384, dtype=np.int64)
            entries = cache["entry_hashes"]
            if entries.get(ids_key) != array_identity(ids) or entries.get("targets") != array_identity(target):
                raise ValueError("Prediction targets or exact unique group IDs differ from cache receipt")
            if role == "confirm":
                primary = score_quartets(pred, target, z["concat_repeat"],
                                         arrays["concat__target_mean"][:4])
            else:
                primary = reader.score_bridge(pred, target, arrays["concat__target_mean"])
        if primary.get("metric_conditions_met") is not True or not all(primary["checks"].values()):
            raise ValueError("Recomputed primary prerequisite did not pass")
        if primary != report.get("scores", {}).get("concat"):
            raise ValueError("Primary report differs from independently recomputed predictions")
        evidence[role] = dict(report_sha256=digest(report_path),
            cache_receipt_sha256=digest(cache_path), cache_sha256=report["cache_sha256"],
            predictions_sha256=digest(predictions_path), counts=report["counts"],
            checks=primary["checks"], primary_metric_conditions_met=True)
    return dict(schema=SCHEMA, prerequisite_schema=CONFIRM_SCHEMA,
        status="verified_pass", p0_authorized=True, test_access=False,
        original_v5_gate="failed_unchanged", plan_sha256=digest(reader.PLAN),
        frozen_reader_sha256=frozen["receipt_sha256"],
        model_sha256=digest(model_folder / "frozen_model.npz"),
        selection_sha256=digest(model_folder / "train_selection.json"),
        selected_primary_reader=selections["concat"]["selected"],
        cache_identity=identity, sources=reader.current_sources(), evidence=evidence)


def validate_profile(path):
    path = Path(path)
    profile, status = read_json(path), read_json(path.with_name("profile_status.json"))
    if (profile.get("status") != "complete" or profile.get("throughput_gate_passed") is not True or
            status.get("status") != "complete" or profile.get("encoder") != ENCODER_IDENTITY):
        raise ValueError("The exact pinned encoder requires a successful prior profile")
    return dict(profile_sha256=digest(path), status_sha256=digest(path.with_name("profile_status.json")),
                encoder=profile["encoder"], original_run_status="feature_gate_failed")


def validate_cumulative_budget(prior_profile, budget_root):
    old = read_json(Path(prior_profile).with_name("budget.json"))
    current = read_json(Path(budget_root) / "budget.json")
    prior_leases = {lease["lease_id"]: lease for lease in old["leases"]}
    new_leases = {lease["lease_id"]: lease for lease in current["leases"]}
    if (current["charged_gpuh"] + 1e-8 < old["charged_gpuh"] or
            any(new_leases.get(key) != value for key, value in prior_leases.items())):
        raise ValueError("Shared budget ledger does not preserve all original charged leases")


def execution_identity():
    """Include new runners/reader/protocols not covered by the original glob."""
    result = source_identity()
    files = dict(result["files"])
    extra = list((ROOT / "scripts").glob("*v5*")) + [ROOT / "docs" / "V5_1_EXECUTION_PROTOCOL.md"]
    for path in extra:
        if path.is_file():
            files[str(path.relative_to(ROOT))] = digest(path)
    result.update(files=dict(sorted(files.items())),
                  source_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    return result


def require_stage(out, stage):
    if read_json(Path(out) / f"{stage}_status.json").get("status") != "complete":
        raise ValueError(f"Successful {stage} stage required")


def write_report(out):
    """Reuse table/plot generation, then apply actual V5.1 stage requirements."""
    out = Path(out)
    original_report(out)
    result = read_json(out / "report_status.json")
    required = {"features", "p0", "replay", *("adapt_" + v for v in [*VARIANTS, "full", "recent", "fifo"])}
    incomplete = sorted(s for s in required if not (out / f"{s}_status.json").exists() or
                        read_json(out / f"{s}_status.json").get("status") != "complete")
    result.update(schema=SCHEMA, prerequisite="temporal_prerequisite.json", incomplete_stages=incomplete,
                  status="partial" if result["missing_variants"] or incomplete else "pilot_complete")
    atomic_json(out / "report_status.json", result)
    manifest = read_json(out / "run_manifest.json")
    manifest["status"] = "complete" if result["status"] == "pilot_complete" else "partial"
    atomic_json(out / "run_manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("features", "p0", "adapt", "replay", "report"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--budget-root", type=Path, required=True)
    p.add_argument("--reader-model", type=Path, required=True)
    p.add_argument("--confirmation", type=Path, required=True)
    p.add_argument("--bridge", type=Path, required=True)
    p.add_argument("--prior-profile", type=Path, required=True)
    p.add_argument("--ijepa-code", type=Path)
    p.add_argument("--ijepa-checkpoint", type=Path)
    p.add_argument("--backend", choices=("nyu", "dgx"), default="dgx")
    p.add_argument("--account")
    p.add_argument("--partition")
    p.add_argument("--qos")
    p.add_argument("--gpu-model", choices=("A100", "H800"), default="H800")
    p.add_argument("--allocation-job")
    p.add_argument("--shared-allocation-gpus", type=int)
    p.add_argument("--gpu-index", type=int)
    p.add_argument("--feature-workers", type=int, choices=(1, 8), default=8)
    p.add_argument("--allow-gpu-sharing", action="store_true")
    p.add_argument("--min-free-gpu-gib", type=float, default=40.)
    p.add_argument("--lease-hours", type=float, default=.5)
    p.add_argument("--variant", choices=[*VARIANTS, "full", "recent", "fifo"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--frame-batch", type=int, default=32)
    a = p.parse_args()
    a.out, a.budget_root = a.out.resolve(), a.budget_root.resolve()
    if a.batch_size != 32 or a.frame_batch < 1:
        raise ValueError("Registered downstream global batch is 32; frame batch must be positive")
    if (a.phase == "adapt") != (a.variant is not None):
        raise ValueError("Exactly adaptation requires a registered --variant")
    if a.out == a.prior_profile.resolve().parent or (a.out / "feature_gate.json").exists():
        raise ValueError("V5.1 must use a fresh separate directory without the old feature gate")
    if not (a.budget_root / "budget.json").is_file():
        raise ValueError("Shared cumulative budget ledger must already include the original run")
    validate_cumulative_budget(a.prior_profile, a.budget_root)
    prerequisite = validate_prerequisites(a.reader_model, a.confirmation, a.bridge)
    profile = validate_profile(a.prior_profile)
    config = dict(schema=SCHEMA, train=2048, dev=256, test=512, seed=42, test_access=False,
        round_limit_gpuh=60, source=execution_identity(), prior_profile=profile,
        prerequisite=prerequisite, budget_root=str(a.budget_root),
        execution=dict(backend=a.backend, account=a.account, partition=a.partition,
                       gpu_model=a.gpu_model, feature_workers=a.feature_workers,
                       batch_size=a.batch_size, frame_batch=a.frame_batch))
    manifest_path = a.out / "run_manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path).get("configuration") != config:
            raise ValueError("V5.1 run source, prerequisite or configuration changed")
        if read_json(a.out / "temporal_prerequisite.json") != prerequisite:
            raise ValueError("Stored V5.1 prerequisite changed")
    elif a.phase == "features":
        a.out.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, dict(configuration=config, status="partial"))
        atomic_json(a.out / "temporal_prerequisite.json", prerequisite)
    else:
        raise ValueError("Full feature extraction must initialize the new V5.1 run")
    if a.phase == "report":
        write_report(a.out)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Scientific downstream runner requires CUDA")
    workers = a.feature_workers if a.phase == "features" else 1
    account = a.account or (NYU_ACCOUNT if a.backend == "nyu" else None)
    if a.backend == "nyu" and (a.gpu_model != "A100" or a.allocation_job):
        raise ValueError("NYU requires its dedicated A100 allocation")
    slurm = validate_runtime(a.backend, account, a.partition, a.gpu_model,
        torch.cuda.device_count(), torch.cuda.get_device_name(), allocation_job=a.allocation_job,
        qos=a.qos, shared_allocation_gpus=a.shared_allocation_gpus,
        gpu_index=a.gpu_index, expected_worker_gpus=workers,
        allow_gpu_sharing=a.allow_gpu_sharing, min_free_gpu_gib=a.min_free_gpu_gib)
    stage = "adapt_" + a.variant if a.phase == "adapt" else a.phase
    limit = {"features": 8., "p0": 12., "replay": 4.}.get(a.phase, 4. if a.variant in VARIANTS else 2.)
    budget = Budget(a.budget_root, stage, limit, a.lease_hours, gpu_count=workers)
    status = "partial"
    receipt = dict(schema=SCHEMA, stage=stage, status="running", start_unix=time.time(),
        job_id=slurm["job_id"], step_id=slurm["step_id"], allocation=slurm,
        lease_hours=a.lease_hours, worker_gpus=workers,
        prerequisite_sha256=digest(a.out / "temporal_prerequisite.json"))
    receipt_path = a.out / "steps" / f"{stage}_{slurm['job_id']}_{slurm['step_id'] or 'batch'}.json"
    atomic_json(receipt_path, receipt)
    try:
        cache = a.out / "features"
        if a.phase == "features":
            if not a.ijepa_code or not a.ijepa_checkpoint:
                raise ValueError("Full features need pinned official encoder code/checkpoint paths")
            context = (FeatureWorkers(a.ijepa_code, a.ijepa_checkpoint, workers, budget,
                       a.out / "workers", ENCODER_IDENTITY) if workers > 1 else
                       nullcontext(Encoder(a.ijepa_code, a.ijepa_checkpoint)))
            with context as encoder:
                if encoder.identity != ENCODER_IDENTITY:
                    raise ValueError("Full-cache encoder differs from confirmed representation")
                for split, count in (("train", 2048), ("dev", 256)):
                    if workers > 1:
                        encoder.cache_split(cache, split, count, a.frame_batch)
                    else:
                        cache_split(encoder, cache, split, count, budget, a.frame_batch)
                atomic_json(a.out / "data_manifest.json", dict(schema=SCHEMA,
                    train=2048, dev=256, test=512, test_status="reserved_unopened",
                    encoder=ENCODER_IDENTITY, paired_variants_stay_together=True,
                    prerequisite_sha256=receipt["prerequisite_sha256"]))
        else:
            require_stage(a.out, "features")
            for split, count in (("train", 2048), ("dev", 256)):
                cached = read_json(cache / split / "manifest.json")
                if (cached.get("configuration", {}).get("encoder") != ENCODER_IDENTITY or
                        cached.get("configuration", {}).get("count") != count or
                        not cached.get("files")):
                    raise ValueError("Full cache identity/count/hash inventory differs from confirmed encoder")
            train, dev = load_records(cache, "train"), load_records(cache, "dev")
            if len(train) != 2048 or len(dev) != 256:
                raise ValueError("Original downstream Train/Dev counts changed")
            if a.phase == "p0":
                train_p0(train, dev, a.out, "cuda", budget, batch_size=32)
            elif a.phase == "adapt":
                require_stage(a.out, "p0")
                train_candidate(a.variant, train, dev, a.out, "cuda", budget, batch_size=32)
            else:
                require_stage(a.out, "p0")
                require_stage(a.out, "adapt_C0")
                reader_only(train, dev, a.out, "cuda", budget)
        budget.check()
        status = "complete"
    except BaseException as exc:
        receipt["reason"] = str(exc)
        raise
    finally:
        budget.close(status)
        receipt.update(status=status, end_unix=time.time())
        atomic_json(receipt_path, receipt)
        atomic_json(a.out / f"{stage}_status.json", dict(schema=SCHEMA, stage=stage,
            status=status, runtime=runtime(), slurm=slurm, reason=receipt.get("reason"),
            prerequisite_sha256=receipt["prerequisite_sha256"]))
    if a.phase == "replay":
        write_report(a.out)


if __name__ == "__main__":
    main()
