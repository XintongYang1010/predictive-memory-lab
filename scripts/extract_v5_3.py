#!/usr/bin/env python3
"""Eight-GPU extraction of V5.3 matched nuisance FIT, anchor, and fresh bridge.

No scores or reader fitting occur here. Anchor/bridge require a verified frozen reader, GitHub readback receipt,
and explicit provenance evidence that their reserved data were unseen. Existing scientific source, caches, and gates are not modified.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import re
import signal
import subprocess
import sys
import time
import traceback

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[variable] = "1"

import numpy as np
import PIL
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"))
from pmr.visual_memory.common import Budget, BudgetExceeded, atomic_json, digest
from pmr.visual_memory.data import episode
from pmr.visual_memory.feature_workers import WorkerBudget
from pmr.visual_memory.features import Encoder as FrozenEncoder
from pmr.visual_memory.slurm_backend import validate_runtime
from v5_3_data import (SCHEMA, make_block, VIEW_INDICES, STATIC_VIEW_INDICES,
                       NEGATIVE_INDICES, UNIQUE_ORDER, BRIDGE_PAIR_IDS)

PLAN = ROOT/"scripts/v5_3_plan.json"
ORIGINAL_DATA_SHA256 = "ce5082f3be9006dcceb46fbaffadc27349f977a77abcf8992eda3e80f8775368"
V52_BASELINE_GPUH = 2.2082544136047373
V53_NEW_LIMIT_GPUH = 3.75
CUMULATIVE_LIMIT_GPUH = min(6., V52_BASELINE_GPUH+V53_NEW_LIMIT_GPUH)
IDENTITY_KEYS = ("plan_sha256", "source_commit", "encoder_identity", "source_identity", "pooling", "feature_dtype")
SOURCE_FILES = tuple(sorted({"scripts/extract_v5_3.py", "scripts/v5_3_data.py",
                "scripts/v5_3_reader.py", "scripts/v5_3_plan.json", "scripts/launch_v5_3.py",
                "scripts/v5_temporal_confirmation.py", "scripts/extract_v5_confirmation.py",
                "scripts/launch_v5_1.py", "scripts/launch_v5_2.py", "docs/V5_3_EXECUTION_PROTOCOL.md"}
                | {path.relative_to(ROOT).as_posix() for path in (ROOT/"pmr/visual_memory").glob("*.py")}))


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_identity(array):
    array = np.asarray(array)
    return dict(sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
                dtype=str(array.dtype), shape=list(array.shape))


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cache_identity(source_commit, checkpoint):
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("Provide the frozen 40-character source commit")
    if (ROOT/".git").exists():
        actual = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        if actual != source_commit:
            raise ValueError("Deployment Git HEAD differs from the declared frozen source commit")
    sources = {name: digest(ROOT/name) for name in SOURCE_FILES}
    if sources["pmr/visual_memory/data.py"] != ORIGINAL_DATA_SHA256:
        raise ValueError("Original V5 generator differs from the frozen d0b4fef source")
    plan = json.loads(PLAN.read_text())
    if plan.get("schema") != SCHEMA:
        raise ValueError("Plan schema differs from the V5.3 data schema")
    if digest(checkpoint) != plan["encoder"]["checkpoint_sha256"]:
        raise ValueError("Official checkpoint differs from the registered SHA256")
    return dict(plan_sha256=digest(PLAN), source_commit=source_commit,
                encoder_identity=dict(source_commit=plan["encoder"]["source_commit"],
                                      checkpoint_sha256=plan["encoder"]["checkpoint_sha256"],
                                      model=plan["encoder"]["model"]),
                source_identity=sources, pooling="4x4_flattened", feature_dtype="float32")


def read_receipt(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Provenance receipts must be existing regular files")
    return json.loads(path.read_text())


def check_release(args, identity):
    """All production roles require fresh-data evidence before generation."""
    freshness=read_receipt(args.freshness_audit)
    if (freshness.get("schema")!="v5.3-freshness-audit-1" or freshness.get("verified") is not True
            or freshness.get("test_opened") is not False
            or freshness.get("unseen_namespaces")!=[SCHEMA+"/fit",SCHEMA+"/anchor"]
            or freshness.get("unseen_original_dev_pair_ids")!=list(BRIDGE_PAIR_IDS)
            or freshness.get("source_commit")!=identity["source_commit"]
            or freshness.get("plan_sha256")!=identity["plan_sha256"]
            or not 0<float(freshness.get("verified_unix",0))<=time.time()):
        raise ValueError("Freshness audit does not authorize the exact registered source and unseen ranges")
    result=dict(freshness_audit_sha256=digest(args.freshness_audit),
                frozen_reader_sha256=None,freeze_publication_sha256=None)
    if args.role == "fit":
        if args.freeze_receipt is not None or args.freeze_publication is not None:
            raise ValueError("FIT must precede reader freeze and evaluation release")
        return result
    if args.role not in ("anchor","bridge") or args.freeze_receipt is None or args.freeze_publication is None:
        raise ValueError("Anchor/bridge require frozen reader and verified GitHub readback before generation")
    frozen=read_receipt(args.freeze_receipt)
    if (frozen.get("schema")!="v5.3-reader-freeze-1" or frozen.get("status")!="frozen_before_evaluation"
            or frozen.get("fit_blocks")!=256 or frozen.get("fit_views")!=4096
            or frozen.get("views_per_fit_block")!=16 or frozen.get("arm_views")!={"S":2048,"B":4096}
            or frozen.get("cache_identity")!=identity
            or frozen.get("freshness_audit_sha256")!=result["freshness_audit_sha256"]):
        raise ValueError("Frozen S/B reader counts, source/cache identity, or freshness receipt mismatch")
    expected_sources={name:digest(ROOT/name) for name in
                      ("scripts/v5_3_reader.py","scripts/v5_3_data.py","scripts/v5_3_plan.json")}
    if frozen.get("sources")!=expected_sources:
        raise ValueError("Reader, generator, or plan changed after freeze")
    for name in ("frozen_model.npz","training_config.json"):
        path=args.freeze_receipt.parent/name
        if path.is_symlink() or not path.is_file() or frozen.get("files",{}).get(name)!=digest(path):
            raise ValueError("Frozen reader file hash mismatch: "+name)
    publication=read_receipt(args.freeze_publication)
    frozen_sha=digest(args.freeze_receipt)
    if (publication.get("verified_readback") is not True
            or publication.get("frozen_reader_sha256")!=frozen_sha
            or publication.get("source_commit")!=identity["source_commit"]
            or publication.get("plan_sha256")!=identity["plan_sha256"]
            or not 0<float(frozen.get("created_unix",0))<=float(publication.get("verified_unix",0))<=time.time()):
        raise ValueError("Frozen-reader GitHub readback is missing, stale, or precedes model freeze")
    result.update(frozen_reader_sha256=frozen_sha,freeze_publication_sha256=digest(args.freeze_publication))
    return result


def encode_images(encoder, rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (224, 224, 3):
        raise ValueError("Expected registered 224x224 RGB inputs")
    # Each unique image enters exactly once. All views and negative queries
    # are reconstructed from these exact tensors by the frozen reader.
    features = encoder(rgb)
    if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[:2] != (len(rgb), 16):
        raise ValueError("Encoder must return one 4x4 pooled spatial tensor per image")
    result = features.float().cpu().numpy().reshape(len(rgb), -1)
    if not np.isfinite(result).all():
        raise ValueError("Encoder produced nonfinite pooled features")
    return result


def encode_block(block, encoder):
    if block.unique_rgb.shape != (6,224,224,3) or block.targets.shape != (16,8) or block.negative_targets.shape != (2,8):
        raise ValueError("A matched block has six unique images, sixteen views, and two reused center controls")
    features = encode_images(encoder, block.unique_rgb)
    inputs = dict(unique_rgb=[array_identity(image) for image in block.unique_rgb],
                  targets=array_identity(block.targets), negative_targets=array_identity(block.negative_targets),
                  metadata=block.metadata, cue_times=[6,7], shutters="both_open", unique_images_encoded=6)
    return features, np.asarray(block.targets,dtype=np.float32), inputs, np.asarray(block.negative_targets,dtype=np.float32)


def encode_bridge(episodes, encoder):
    if len(episodes) != 2:
        raise ValueError("Bridge preserves both variants of each original paired world")
    cues, targets, metadata = [], [], []
    for ep in episodes:
        indices = []
        for time_value in (6., 7.):
            candidates = np.flatnonzero(ep.times == time_value)
            if len(candidates) != 1:
                raise ValueError("Original bridge must contain exactly one cue at each time 6 and 7")
            indices.append(int(candidates[0]))
        if not np.all(ep.shutters[indices] == 1):
            raise ValueError("Original bridge cue shutters must be open")
        cues.append(ep.rgb[indices]); targets.append(ep.probe_state); metadata.append(ep.metadata)
    rgb = np.concatenate(cues)
    targets = np.stack(targets).astype(np.float32)
    features = encode_images(encoder, rgb).reshape(2, 2, -1)
    inputs = dict(cue_rgb=[array_identity(image) for image in rgb], targets=array_identity(targets),
                  metadata=metadata, cue_times=[6, 7], images_encoded=4,
                  original_generator_sha256=ORIGINAL_DATA_SHA256)
    return features, targets, inputs


def unit_paths(out, unit):
    return Path(out)/"shards"/f"{unit:06d}.npz", Path(out)/"shards"/f"{unit:06d}.json"


def read_unit(out, unit, configuration_sha256, role):
    path, receipt_path = unit_paths(out, unit)
    if not path.exists() and not receipt_path.exists():
        return None
    if path.is_symlink() or receipt_path.is_symlink() or not path.is_file() or not receipt_path.is_file():
        raise ValueError(f"Incomplete or linked shard {unit}; preserve it and inspect before resuming")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("unit") != unit or receipt.get("role") != role
            or receipt.get("configuration_sha256") != configuration_sha256
            or receipt.get("bytes") != path.stat().st_size or receipt.get("sha256") != digest(path)):
        raise ValueError(f"Shard provenance or checksum mismatch: {unit}")
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    expected_x, expected_y = ((2, 2), (2, 8)) if role == "bridge" else ((6,), (16, 8))
    x, y = arrays.get("features"), arrays.get("targets")
    if (set(arrays) != ({"features", "targets"} if role == "bridge" else {"features", "targets", "negative_targets"}) or not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
            or x.dtype != np.float32 or y.dtype != np.float32
            or x.shape[:-1] != expected_x or y.shape != expected_y
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or receipt.get("entry_hashes") != {name: array_identity(array) for name, array in arrays.items()}):
        raise ValueError(f"Shard tensor shape, dtype or array checksum mismatch: {unit}")
    if role != "bridge":
        negative=arrays["negative_targets"]
        if (negative.shape!=(2,8) or negative.dtype!=np.float32 or not np.isfinite(negative).all()
                or not np.array_equal(negative[:,4:],np.zeros((2,4),dtype=np.float32))
                or not np.array_equal(negative[:,:4],np.tile(y[0,:4],(2,1)))):
            raise ValueError("Invalid repeated-center negative targets")
    return arrays, receipt


def save_unit(out, unit, role, configuration_sha256, features, targets, inputs, negative_targets=None):
    path, receipt_path = unit_paths(out, unit)
    if path.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to replace an existing shard: " + str(path))
    arrays = dict(features=features, targets=targets)
    if role != "bridge":
        arrays["negative_targets"]=negative_targets
    atomic_npz(path, **arrays)
    atomic_json(receipt_path, dict(role=role, unit=unit, configuration_sha256=configuration_sha256,
        bytes=path.stat().st_size, sha256=digest(path),
        entry_hashes={name: array_identity(array) for name, array in arrays.items()}, inputs=inputs))


def worker(index, role, units, out, code, checkpoint, config_sha, identity,
           events, cancelled, begin, deadline):
    torch.set_num_threads(1)
    budget = WorkerBudget(deadline, cancelled)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: cancelled.set())
    key = f"worker_{index}_{os.getpid()}"
    receipt_path = Path(out)/"workers"/(key+".json")
    receipt = dict(worker=index, pid=os.getpid(), role=role, assigned_units=units,
                   completed_units=[], status="starting", start_unix=time.time(),
                   job_id=os.environ.get("SLURM_JOB_ID"), step_id=os.environ.get("SLURM_STEP_ID"))
    atomic_json(receipt_path, receipt)
    with (Path(out)/"workers"/(key+".log")).open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            budget.check(); torch.cuda.set_device(index)
            encoder = FrozenEncoder(code, checkpoint, device=f"cuda:{index}", grid=4)
            actual = dict(source_commit=encoder.identity["code_commit"],
                          checkpoint_sha256=encoder.identity["checkpoint_sha256"],
                          model="official I-JEPA ViT-H/14 target_encoder")
            if actual != identity["encoder_identity"] or encoder.grid != 4:
                raise ValueError("Worker encoder differs from registered encoder identity")
            receipt.update(status="ready", encoder=encoder.identity,
                           gpu=torch.cuda.get_device_name(index),
                           visible_uuid=os.environ["CUDA_VISIBLE_DEVICES"].split(",")[index])
            atomic_json(receipt_path, receipt); events.put(dict(event="ready", worker=index))
            while not begin.wait(.2):
                budget.check()
            for unit in units:
                budget.check()
                if role == "bridge":
                    value = encode_bridge([episode("dev", unit, variant) for variant in (0, 1)], encoder)
                else:
                    value = encode_block(make_block(role, unit), encoder)
                budget.check()
                save_unit(out, unit, role, config_sha, *value)
                receipt["completed_units"].append(unit)
                receipt.update(status="running")
                atomic_json(receipt_path, receipt)
                print(json.dumps(dict(role=role, unit=unit, status="complete")), flush=True)
            receipt["status"] = "complete"
        except BaseException as exc:
            receipt.update(status="partial", reason=str(exc))
            traceback.print_exc(); events.put(dict(event="error", worker=index, reason=str(exc)))
        finally:
            receipt["end_unix"] = time.time()
            atomic_json(receipt_path, receipt)
        if receipt["status"] == "complete":
            events.put(dict(event="done", worker=index))


def extract_parallel(args, units, configuration_sha256, identity, budget):
    context = mp.get_context("spawn")
    events, cancelled, begin = context.Queue(), context.Event(), context.Event()
    processes = []
    deadline = budget.start + budget.charge/budget.gpus*3600 - 60
    def wait_for(event, pending):
        pending = set(pending)
        while pending:
            budget.check()
            try:
                message = events.get(timeout=.25)
            except queue.Empty:
                failed = [p.pid for i, p in enumerate(processes) if i in pending and p.exitcode is not None]
                if failed:
                    raise RuntimeError("Extraction worker exited unexpectedly: " + str(failed))
                continue
            if message["event"] == "error":
                raise RuntimeError(f"Extraction worker {message['worker']} failed: {message['reason']}")
            if message["event"] != event or message["worker"] not in pending:
                raise RuntimeError("Unexpected worker receipt: " + str(message))
            pending.remove(message["worker"])
    success = False
    try:
        for index in range(args.workers):
            assigned = [unit for unit in units if unit % args.workers == index]
            process = context.Process(target=worker,
                args=(index, args.role, assigned, str(args.out), str(args.ijepa), str(args.checkpoint),
                      configuration_sha256, identity, events, cancelled, begin, deadline), daemon=True)
            processes.append(process); process.start()
            # Bound transient checkpoint deserialization RAM; all workers start
            # encoding together only after all eight official encoders are ready.
            wait_for("ready", {index})
        begin.set(); wait_for("done", range(args.workers)); success = True
    finally:
        if not success:
            cancelled.set(); begin.set()
        join_deadline = time.monotonic()+5
        for process in processes:
            process.join(timeout=max(0, join_deadline-time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill(); process.join(timeout=2)
        events.close()


def file_inventory(out):
    files = {}
    for path in sorted(Path(out).rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in scientific cache outputs")
        if path.is_file() and path != Path(out)/"manifest.json":
            files[path.relative_to(out).as_posix()] = dict(bytes=path.stat().st_size, sha256=digest(path))
    return files


def merge_cache(args, identity, configuration, slurm, budget):
    units = list(BRIDGE_PAIR_IDS) if args.role == "bridge" else list(range(256))
    rows, inputs = [], {}
    for unit in units:
        budget.check()
        value = read_unit(args.out, unit, json_sha(configuration), args.role)
        if value is None:
            raise ValueError("Missing completed shard " + str(unit))
        arrays, receipt = value
        if arrays["features"].shape[-1] != 16*1280:
            raise ValueError("Official ViT-H pooled feature width must be 16x1280")
        rows.append(arrays); inputs[str(unit)] = receipt["inputs"]
    expected_shards = {f"{unit:06d}{suffix}" for unit in units for suffix in (".npz", ".json")}
    if {path.name for path in (args.out/"shards").iterdir()} != expected_shards:
        raise ValueError("Shard inventory differs from the exact registered 256 groups")
    arrays = dict(features=np.stack([row["features"] for row in rows]),
                  targets=np.stack([row["targets"] for row in rows]), role=np.array(args.role))
    arrays["pair_ids" if args.role == "bridge" else "block_ids"] = np.asarray(units, dtype=np.int64)
    if args.role != "bridge":
        arrays["negative_targets"] = np.stack([row["negative_targets"] for row in rows])
    cache = args.out/"cache.npz"
    if cache.exists() or (args.out/"receipt.json").exists():
        receipt_path = args.out/"receipt.json"
        if not cache.is_file() or not receipt_path.is_file() or cache.is_symlink() or receipt_path.is_symlink():
            raise ValueError("Incomplete aggregate output; preserve and inspect before resuming")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("configuration_sha256") != json_sha(configuration)
                or receipt.get("cache_sha256") != digest(cache)
                or receipt.get("entry_hashes") != {name: array_identity(array) for name, array in arrays.items()}
                or {key: receipt.get(key) for key in IDENTITY_KEYS} != identity):
            raise ValueError("Existing aggregate differs from validated shards or configuration")
        return receipt
    atomic_npz(cache, **arrays)
    receipt = dict(schema=SCHEMA, role=args.role, **identity, cache_sha256=digest(cache),
                   cache_bytes=cache.stat().st_size,
                   entry_hashes={name: array_identity(array) for name, array in arrays.items()},
                   unique_image_rule="encode_once_reuse" if args.role != "bridge" else "original_variant_cues",
                   groups=256, examples=512 if args.role == "bridge" else 4096,
                   images_encoded=1024 if args.role == "bridge" else 1536,
                   view_indices=VIEW_INDICES.tolist(), static_view_indices=STATIC_VIEW_INDICES.tolist(),
                   negative_indices=NEGATIVE_INDICES.tolist(), unique_order=list(UNIQUE_ORDER),
                   freshness_audit_sha256=configuration["freshness_audit_sha256"],
                   configuration_sha256=json_sha(configuration), input_records_sha256=json_sha(inputs),
                   runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                                torch=torch.__version__, pillow=PIL.__version__, cuda=torch.version.cuda,
                                platform=platform.platform()), slurm=slurm,
                   test_opened=False, original_v5_gate="failed_unchanged", p0_authorized=False)
    if args.role != "bridge":
        receipt["coincident_nuisance_pixel_blocks"]=sum(record["metadata"]["nuisance_coincident_pixels"]
                                                        for record in inputs.values())
    if args.role != "fit":
        receipt.update(frozen_reader_sha256=configuration["frozen_reader_sha256"],
                       freeze_publication_sha256=configuration["freeze_publication_sha256"],
                       frozen_before_generation=True, unseen_before_release=True)
    atomic_json(args.out/"receipt.json", receipt)
    return receipt


def run_reserved(args, budget):
    if args.out.exists() and any(path.is_symlink() for path in args.out.rglob("*")):
        raise ValueError("Symlinks are not allowed inside a role cache")
    identity = cache_identity(args.source_commit, args.checkpoint)
    budget.check()
    release = check_release(args, identity)
    configuration = dict(schema=SCHEMA, role=args.role, identity=identity, workers=8,
                         units=list(BRIDGE_PAIR_IDS) if args.role == "bridge" else list(range(256)),
                         **release, unseen_before_release=True)
    manifest_path = args.out/"manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("configuration") != configuration:
            raise ValueError("Resume configuration differs from the original role/source/encoder/freeze")
        if manifest.get("status") == "complete":
            if manifest.get("files") != file_inventory(args.out):
                raise ValueError("Completed role cache inventory or checksums changed")
            return json.loads((args.out/"receipt.json").read_text()), None
    elif args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh role directory or an intact resumable manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    for folder in ("shards", "workers"):
        (args.out/folder).mkdir(exist_ok=True)
    atomic_json(manifest_path, dict(status="partial", configuration=configuration))
    slurm = validate_runtime("dgx", args.account, args.partition, args.gpu_model,
        torch.cuda.device_count(), torch.cuda.get_device_name(0), allocation_job=args.allocation_job,
        qos=args.qos, shared_allocation_gpus=8, expected_worker_gpus=8,
        allow_gpu_sharing=args.allow_gpu_sharing, min_free_gpu_gib=args.min_free_gpu_gib)
    try:
        pending = [unit for unit in configuration["units"]
                   if read_unit(args.out, unit, json_sha(configuration), args.role) is None]
        if pending:
            extract_parallel(args, pending, json_sha(configuration), identity, budget)
        budget.check()
        if {name: digest(ROOT/name) for name in SOURCE_FILES} != identity["source_identity"]:
            raise ValueError("Extraction source changed during execution")
        if check_release(args, identity) != release:
            raise ValueError("Release or freshness receipt changed during extraction")
        result = merge_cache(args, identity, configuration, slurm, budget)
        budget.check()
    except BaseException as exc:
        atomic_json(manifest_path, dict(status="partial", configuration=configuration,
                                       reason=str(exc), files=file_inventory(args.out)))
        raise
    return result, configuration


def check_budget_headroom(charged_gpuh, lease_hours):
    """Apply the fixed registered round cap even without the controller."""
    if (not np.isfinite(charged_gpuh) or not np.isfinite(lease_hours)
            or charged_gpuh < 0 or not 0 < lease_hours <= .09):
        raise ValueError("Finite nonnegative accounting and a positive at-most .09h lease required")
    if charged_gpuh+lease_hours*8 > CUMULATIVE_LIMIT_GPUH+1e-8:
        raise BudgetExceeded("V5.3 lease exceeds fixed baseline +3.75GPUh or cumulative6GPUh cap")


def run(args):
    if args.workers != 8 or args.role not in ("fit", "anchor", "bridge") or not 0 < args.lease_hours <= .09:
        raise ValueError("Registered extraction requires eight workers and one scientific role")
    if not (args.budget_root/"budget.json").is_file():
        raise ValueError("Existing inherited budget.json required; extractor never initializes a fresh ledger")
    if args.out.is_symlink():
        raise ValueError("Role output directory cannot be a symlink")
    args.out = args.out.resolve()
    if args.min_free_gpu_gib < 40:
        raise ValueError("At least40GiB free per shared GPU is required")
    prior_path=args.prior_budget.resolve()
    inherited=read_receipt(args.prior_budget)
    if inherited.get("charged_gpuh",0)<V52_BASELINE_GPUH-1e-10:
        raise ValueError("The complete V5.2 cumulative cost must be inherited")
    prior_scene=prior_path.parent.parent
    for candidate in (args.out,args.budget_root.resolve()):
        if candidate==prior_scene or prior_scene in candidate.parents:
            raise ValueError("Prior scene and its ledger are read-only")
    with open(args.budget_root/"budget.json") as stream:
        prior_budget = json.load(stream)
    if (prior_budget["charged_gpuh"]<inherited["charged_gpuh"]
            or not {row["lease_id"] for row in inherited["leases"]}.issubset(
                {row["lease_id"] for row in prior_budget["leases"]})):
        raise ValueError("New ledger must preserve all inherited V5.2 leases and cost")
    # The immutable pre-V5.3 baseline prevents later ledger inheritance from
    # resetting this round's3.75GPUh allowance. At least54GPUh remains reserved.
    check_budget_headroom(prior_budget["charged_gpuh"],args.lease_hours)
    budget = Budget(args.budget_root, "v53_"+args.role, 1.2, args.lease_hours, gpu_count=8)
    status = "partial"
    try:
        result, configuration = run_reserved(args, budget)
        status = "complete"
    finally:
        budget.close(status)
    if configuration is not None:
        atomic_json(args.out/"manifest.json", dict(status="complete", configuration=configuration,
                    budget_lease=budget.lease, files=file_inventory(args.out)))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", required=True, choices=("fit", "anchor", "bridge"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ijepa", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--budget-root", required=True, type=Path)
    p.add_argument("--workers", type=int, choices=(8,), default=8)
    p.add_argument("--lease-hours", type=float, default=.09)
    p.add_argument("--freeze-receipt", type=Path)
    p.add_argument("--freeze-publication", type=Path)
    p.add_argument("--freshness-audit", required=True, type=Path)
    p.add_argument("--prior-budget", required=True, type=Path)
    p.add_argument("--allocation-job", required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--qos", required=True)
    p.add_argument("--gpu-model", default="H800")
    p.add_argument("--allow-gpu-sharing", action="store_true")
    p.add_argument("--min-free-gpu-gib", type=float, default=40.)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(run(args), indent=2, allow_nan=False))
