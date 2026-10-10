#!/usr/bin/env python3
"""Eight-GPU extraction of registered FIT, confirmation, or original-task bridge.

No scores or reader fitting occur here. Confirm/bridge require a verified frozen
reader and an explicit attestation that their reserved data were unseen before
initial release. Existing scientific source, caches, and gates are not modified.
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
from v5_temporal_confirmation import SCHEMA, make_quartet

PLAN = ROOT/"scripts/v5_temporal_confirmation_plan.json"
ORIGINAL_DATA_SHA256 = "ce5082f3be9006dcceb46fbaffadc27349f977a77abcf8992eda3e80f8775368"
IDENTITY_KEYS = ("plan_sha256", "source_commit", "encoder_identity", "source_identity", "pooling", "feature_dtype")
SOURCE_FILES = tuple(sorted({"scripts/extract_v5_confirmation.py", "scripts/v5_temporal_confirmation.py",
                "scripts/v5_confirmation_reader.py", "scripts/v5_temporal_confirmation_plan.json",
                "scripts/launch_v5_1.py", "scripts/run_v5_1_downstream.py", "docs/V5_1_EXECUTION_PROTOCOL.md"}
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
    if digest(checkpoint) != plan["encoder"]["checkpoint_sha256"]:
        raise ValueError("Official checkpoint differs from the registered SHA256")
    return dict(plan_sha256=digest(PLAN), source_commit=source_commit,
                encoder_identity=dict(source_commit=plan["encoder"]["source_commit"],
                                      checkpoint_sha256=plan["encoder"]["checkpoint_sha256"],
                                      model=plan["encoder"]["model"]),
                source_identity=sources, pooling="4x4_flattened", feature_dtype="float32")


def check_freeze(role, path, identity, attest_unseen):
    if role == "fit":
        if path is not None or attest_unseen:
            raise ValueError("FIT precedes reader freeze; do not supply evaluation attestations")
        return None
    if role not in ("confirm", "bridge") or path is None or not attest_unseen:
        raise ValueError("Confirm/bridge require --freeze-receipt and --attest-unseen before generation")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Freeze receipt must be a regular existing file")
    receipt = json.loads(path.read_text())
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "frozen_before_confirmation"
            or receipt.get("fit_blocks") != 256 or receipt.get("fit_views") != 1024
            or receipt.get("cache_identity") != identity):
        raise ValueError("Reader freeze schema, completion, counts, or cache identity mismatch")
    sources = {name: digest(ROOT/"scripts"/name) for name in
               ("v5_confirmation_reader.py", "v5_temporal_confirmation.py", "v5_temporal_confirmation_plan.json")}
    if receipt.get("sources") != sources:
        raise ValueError("Reader, generator, or plan changed after freeze")
    for name in ("frozen_model.npz", "train_selection.json"):
        file = path.parent/name
        if file.is_symlink() or not file.is_file() or receipt.get("files", {}).get(name) != digest(file):
            raise ValueError("Frozen reader artifact hash mismatch: " + name)
    return digest(path)


def encode_images(encoder, rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (224, 224, 3):
        raise ValueError("Expected registered 224x224 RGB inputs")
    # Every unique quartet image enters this one call once. Repeated anchors and
    # four ordered views are reconstructed from these exact tensors by readers.
    features = encoder(rgb)
    if not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[:2] != (len(rgb), 16):
        raise ValueError("Encoder must return one 4x4 pooled spatial tensor per image")
    result = features.float().cpu().numpy().reshape(len(rgb), -1)
    if not np.isfinite(result).all():
        raise ValueError("Encoder produced nonfinite pooled features")
    return result


def encode_quartet(quartet, encoder):
    if quartet.unique_rgb.shape[0] != 3 or quartet.targets.shape != (4, 8):
        raise ValueError("A complete quartet has three unique images and four targets")
    features = encode_images(encoder, quartet.unique_rgb)
    inputs = dict(unique_rgb=[array_identity(image) for image in quartet.unique_rgb],
                  targets=array_identity(quartet.targets), metadata=quartet.metadata,
                  cue_times=[6, 7], shutters="both_open", unique_images_encoded=3)
    return features, np.asarray(quartet.targets, dtype=np.float32), inputs


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
    expected_x, expected_y = ((2, 2), (2, 8)) if role == "bridge" else ((3,), (4, 8))
    x, y = arrays.get("features"), arrays.get("targets")
    if (set(arrays) != {"features", "targets"} or not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray)
            or x.dtype != np.float32 or y.dtype != np.float32
            or x.shape[:-1] != expected_x or y.shape != expected_y
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or receipt.get("entry_hashes") != {name: array_identity(array) for name, array in arrays.items()}):
        raise ValueError(f"Shard tensor shape, dtype or array checksum mismatch: {unit}")
    return arrays, receipt


def save_unit(out, unit, role, configuration_sha256, features, targets, inputs):
    path, receipt_path = unit_paths(out, unit)
    if path.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to replace an existing shard: " + str(path))
    arrays = dict(features=features, targets=targets)
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
                    value = encode_quartet(make_quartet(role, unit), encoder)
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
    units = list(range(128, 384)) if args.role == "bridge" else list(range(256))
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
                   groups=256, examples=512 if args.role == "bridge" else 1024,
                   images_encoded=1024 if args.role == "bridge" else 768,
                   configuration_sha256=json_sha(configuration), input_records_sha256=json_sha(inputs),
                   runtime=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                                torch=torch.__version__, pillow=PIL.__version__, cuda=torch.version.cuda,
                                platform=platform.platform()), slurm=slurm,
                   test_opened=False, original_v5_gate="failed_unchanged")
    if args.role != "fit":
        receipt.update(frozen_reader_sha256=configuration["frozen_reader_sha256"],
                       frozen_before_generation=True, unseen_before_release=True)
    atomic_json(args.out/"receipt.json", receipt)
    return receipt


def run_reserved(args, budget):
    if args.out.exists() and any(path.is_symlink() for path in args.out.rglob("*")):
        raise ValueError("Symlinks are not allowed inside a role cache")
    identity = cache_identity(args.source_commit, args.checkpoint)
    budget.check()
    frozen_sha = check_freeze(args.role, args.freeze_receipt, identity, args.attest_unseen)
    configuration = dict(schema=SCHEMA, role=args.role, identity=identity, workers=8,
                         units=list(range(128, 384)) if args.role == "bridge" else list(range(256)),
                         frozen_reader_sha256=frozen_sha,
                         unseen_before_release=bool(args.attest_unseen))
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
        if check_freeze(args.role, args.freeze_receipt, identity, args.attest_unseen) != frozen_sha:
            raise ValueError("Freeze receipt changed during extraction")
        result = merge_cache(args, identity, configuration, slurm, budget)
        budget.check()
    except BaseException as exc:
        atomic_json(manifest_path, dict(status="partial", configuration=configuration,
                                       reason=str(exc), files=file_inventory(args.out)))
        raise
    return result, configuration


def run(args):
    if args.workers != 8 or args.role not in ("fit", "confirm", "bridge"):
        raise ValueError("Registered extraction requires eight workers and one scientific role")
    if not (args.budget_root/"budget.json").is_file():
        raise ValueError("Existing inherited budget.json required; extractor never initializes a fresh ledger")
    if args.out.is_symlink():
        raise ValueError("Role output directory cannot be a symlink")
    args.out = args.out.resolve()
    with open(args.budget_root/"budget.json") as stream:
        prior_budget = json.load(stream)
    # Preserve at least54GPUh for a later, separately authorized full pilot. The
    # inherited .3916GPUh is retained in this ledger, never silently reset.
    if prior_budget["charged_gpuh"]+args.lease_hours*8 > 6+1e-8:
        raise BudgetExceeded("Extraction lease would consume the 54GPUh reserved for a possible later pilot")
    budget = Budget(args.budget_root, "temporal_"+args.role, 2., args.lease_hours, gpu_count=8)
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
    p.add_argument("--role", required=True, choices=("fit", "confirm", "bridge"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ijepa", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--budget-root", required=True, type=Path)
    p.add_argument("--workers", type=int, choices=(8,), default=8)
    p.add_argument("--lease-hours", type=float, default=.20)
    p.add_argument("--freeze-receipt", type=Path)
    p.add_argument("--attest-unseen", action="store_true",
                   help="Attest that reserved confirmation/bridge inputs were unseen before initial release")
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
