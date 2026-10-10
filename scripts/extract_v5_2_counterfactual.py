#!/usr/bin/env python3
"""Extract four paired nuisance interventions on the already inspected bridge.

Only yellow rectangle locations change. No scores, fitting, new seeds, Test
access, or changes to old source/scenes occur. The old reader stays frozen.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import time
import traceback

if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import extract_v5_confirmation as reusable
from pmr.visual_memory.common import Budget, BudgetExceeded, atomic_json, digest
from pmr.visual_memory.data import episode, seed_for, reflect
from pmr.visual_memory.feature_workers import WorkerBudget
from pmr.visual_memory.features import Encoder
from pmr.visual_memory.slurm_backend import validate_runtime
from v5_confirmation_reader import load_cache

SCHEMA = "v5_2-yellow-counterfactual-1"
CONDITIONS = ("original", "static_t6", "static_t7", "swap_nuisance")
CONDITION_INDICES = np.array(((0, 3), (0, 2), (1, 3), (1, 2)), dtype=np.int64)
UNIQUE_ORDER = ("t6@x6", "t6@x7", "t7@x6", "t7@x7")
PROTOCOL = ROOT/"docs/V5_2_CONTROLLED_DIAGNOSTIC_DRAFT.md"
SOURCE_FILES = tuple(sorted(set(reusable.SOURCE_FILES) | {str(PROTOCOL.relative_to(ROOT))}
                           | {p.relative_to(ROOT).as_posix() for p in (ROOT/"scripts").glob("*v5_2*.py")}))
array_identity, json_sha, atomic_npz = reusable.array_identity, reusable.json_sha, reusable.atomic_npz


def rectangle_mask(center, size=224):
    mask = np.zeros((size, size), dtype=bool)
    mask[int(.48*size):int(.52*size)+1, max(0, center-5):min(size, center+6)] = True
    return mask


def draw_nuisance(frame, center):
    """Restore the exact original background/grid, then draw the yellow glyph."""
    if frame.shape != (224, 224, 3) or frame.dtype != np.uint8:
        raise ValueError("Expected original 224x224 uint8 RGB")
    if not isinstance(center, (int, np.integer)) or not 0 <= center < 224:
        raise ValueError("Nuisance center must be an in-frame integer pixel")
    result = frame.copy()
    lo, hi = int(.48*224), int(.52*224)+1
    result[lo:hi] = (30, 35, 42)
    result[lo:hi, ::14] = (42, 47, 54)
    result[rectangle_mask(center)] = (240, 220, 50)
    return result


def intervene(frame, old_center, new_center):
    reconstructed = draw_nuisance(frame, old_center)
    if not np.array_equal(reconstructed, frame):
        raise ValueError("Original RGB reconstruction is not bitwise identical")
    result = draw_nuisance(frame, new_center)
    allowed = rectangle_mask(old_center) | rectangle_mask(new_center)
    if not np.array_equal(result[~allowed], frame[~allowed]):
        raise ValueError("Intervention changed a pixel outside the old/new rectangle union")
    return result


def cue_target_masks(pair, variant):
    """Reconstruct actual PIL target masks at6/7, before the first update.

    These are analysis-only guards. The two RNG draws and reflection bounds
    are the pinned original generator's cue-state calculation, never inputs to
    the encoder. Original future target_masks are deliberately not used.
    """
    rng=np.random.default_rng(seed_for("dev",pair))
    position=rng.uniform([.2,.23],[.8,.32],(2,2));position[1,1]+=.42
    velocity=rng.uniform(.008,.018,(2,2))*rng.choice([-1,1],(2,2))
    velocity*=1 if variant==0 else -1
    states=[];masks=[]
    for time_value in (6.,7.):
        state=position+velocity*time_value
        state[:,0]=reflect(state[:,0],.10,.90)
        state[0,1]=reflect(state[0,1],.15,.40);state[1,1]=reflect(state[1,1],.60,.85)
        states.append(state.reshape(-1));frame_masks=[]
        for point in state:
            x,y=point*224;radius=224*.06
            image=Image.new("L",(224,224),0)
            ImageDraw.Draw(image).ellipse((int(x-radius),int(y-radius),int(x+radius),int(y+radius)),fill=1)
            frame_masks.append(np.asarray(image).copy())
        masks.append(np.stack(frame_masks))
    target=np.concatenate([states[1],states[1]-states[0]]).astype(np.float32)
    return np.stack(masks),target


def variant_images(ep, pair):
    times = []
    for value in (6., 7.):
        found = np.flatnonzero(ep.times == value)
        if len(found) != 1:
            raise ValueError("Expected exactly one original cue at each time6,7")
        times.append(int(found[0]))
    if not np.all(ep.shutters[times] == 1):
        raise ValueError("Cue targets must remain visible")
    centers = [int((.5+.4*np.sin(t*2.3+pair))*224) for t in (6., 7.)]
    masks,target=cue_target_masks(pair,ep.metadata["variant"])
    if not np.array_equal(target,ep.probe_state):
        raise ValueError("Reconstructed cue target masks disagree with original physical labels")
    allowed=rectangle_mask(centers[0])|rectangle_mask(centers[1])
    if np.any(masks[:,:,allowed]):
        raise ValueError("Yellow rectangle intervention overlaps an actual PIL target mask")
    a, b = ep.rgb[times]
    images = np.stack((intervene(a, centers[0], centers[0]),
                       intervene(a, centers[0], centers[1]),
                       intervene(b, centers[1], centers[0]),
                       intervene(b, centers[1], centers[1])))
    if not np.array_equal(images[[0, 3]], ep.rgb[times]):
        raise ValueError("Original condition pixels differ from original episode")
    for slot,time_index in enumerate((0,0,1,1)):
        target_pixels=np.any(masks[time_index],axis=0)
        if not np.array_equal(images[slot][target_pixels],ep.rgb[times[time_index]][target_pixels]):
            raise ValueError("Intervention changed a target pixel")
    return images, dict(metadata=ep.metadata, cue_times=[6, 7], nuisance_centers=centers,
                       original_pixel_parity=True, outside_rectangle_union_unchanged=True,
                       target_masks_disjoint=True,target_pixels_unchanged=True,
                       cue_target_masks=[array_identity(mask) for mask in masks],
                       cue_target_pixels=[array_identity(ep.rgb[times[i]][np.any(mask,axis=0)]) for i,mask in enumerate(masks)],
                       rectangle_union_mask=array_identity(allowed),
                       unique_rgb=[array_identity(image) for image in images],
                       target_identity=array_identity(ep.probe_state))


def encode_pair(episodes, pair, encoder, old_targets, expected_inputs):
    if len(episodes) != 2 or np.asarray(old_targets).shape != (2, 8):
        raise ValueError("Each pair requires both original variants and their archived targets")
    targets = np.stack([ep.probe_state for ep in episodes]).astype(np.float32)
    if not np.array_equal(targets, old_targets):
        raise ValueError("Regenerated targets differ from the frozen bridge cache")
    images, inputs = [], []
    for ep in episodes:
        unique, record = variant_images(ep, pair)
        images.append(unique); inputs.append(record)
    original_rgb=[record["unique_rgb"][slot] for record in inputs for slot in (0,3)]
    if original_rgb!=expected_inputs["cue_rgb"] or array_identity(targets)!=expected_inputs["targets"]:
        raise ValueError("Original RGB/target hashes differ from the archived bridge inputs")
    # Match the old bridge's exact original batch composition and ordering.
    original_batch=np.concatenate([unique[[0,3]] for unique in images])
    edited_batch=np.concatenate([unique[[1,2]] for unique in images])
    original_features=reusable.encode_images(encoder,original_batch).reshape(2,2,-1)
    edited_features=reusable.encode_images(encoder,edited_batch).reshape(2,2,-1)
    features=np.empty((2,4,original_features.shape[-1]),dtype=np.float32)
    features[:,[0,3],:]=original_features;features[:,[1,2],:]=edited_features
    return features, targets, inputs


def original_inputs(scene):
    scene = Path(scene).resolve()
    frozen_path = scene/"reader/freeze_receipt.json"
    frozen = json.loads(frozen_path.read_text())
    if (frozen.get("schema") != "v5-temporal-confirm-1" or frozen.get("status") != "frozen_before_confirmation"
            or frozen.get("fit_blocks") != 256 or frozen.get("fit_views") != 1024):
        raise ValueError("Expected the original completed frozen reader")
    for name in ("frozen_model.npz", "train_selection.json"):
        path = scene/"reader"/name
        if path.is_symlink() or frozen["files"][name] != digest(path):
            raise ValueError("Original frozen reader artifact hash mismatch")
    for name, expected in frozen["sources"].items():
        if digest(ROOT/"scripts"/name) != expected or digest(scene/"code/scripts"/name) != expected:
            raise ValueError("Current/deployed original reader sources differ from freeze")
    frozen["receipt_sha256"] = digest(frozen_path)
    cache = scene/"bridge_cache/cache.npz"; receipt_path = scene/"bridge_cache/receipt.json"
    data = load_cache(cache, receipt_path, "bridge", frozen)
    receipt = data["receipt"]
    manifest=json.loads((scene/"bridge_cache/manifest.json").read_text())
    if manifest.get("status")!="complete" or manifest.get("files")!=reusable.file_inventory(scene/"bridge_cache"):
        raise ValueError("Original bridge manifest/inventory/checksums differ")
    archived_inputs={}
    for pair in range(128,384):
        record=json.loads((scene/"bridge_cache/shards"/f"{pair:06d}.json").read_text())
        if record["role"]!="bridge" or record["unit"]!=pair or record["configuration_sha256"]!=json_sha(manifest["configuration"]):
            raise ValueError("Original bridge shard provenance mismatch")
        archived_inputs[str(pair)]=record["inputs"]
    if json_sha(archived_inputs)!=receipt["input_records_sha256"]:
        raise ValueError("Original bridge input records aggregate hash differs")
    data["archived_inputs"]=archived_inputs
    if digest(ROOT/"pmr/visual_memory/data.py") != reusable.ORIGINAL_DATA_SHA256:
        raise ValueError("Original generator differs from its frozen source")
    for name in ("pmr/visual_memory/data.py", "pmr/visual_memory/features.py"):
        if digest(ROOT/name) != receipt["source_identity"][name]:
            raise ValueError("Original generator/encoder implementation changed")
    reference = dict(frozen_reader_sha256=frozen["receipt_sha256"],
        original_bridge_cache_sha256=digest(cache), original_bridge_receipt_sha256=digest(receipt_path),
        original_cache_identity=frozen["cache_identity"], original_scene=str(scene))
    return data, reference


def shard_paths(out, pair):
    path = Path(out)/"shards"/f"{pair:06d}.npz"
    return path, path.with_suffix(".json")


def read_shard(out, pair, config_sha):
    path, receipt_path = shard_paths(out, pair)
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.is_file() or not receipt_path.is_file() or path.is_symlink() or receipt_path.is_symlink():
        raise ValueError("Incomplete/linked shard preserved; inspect before resume")
    record = json.loads(receipt_path.read_text())
    if (record.get("pair_id") != pair or record.get("configuration_sha256") != config_sha
            or record.get("sha256") != digest(path) or record.get("bytes") != path.stat().st_size):
        raise ValueError("Shard provenance/hash mismatch")
    with np.load(path, allow_pickle=False) as file:
        arrays = {key: file[key] for key in file.files}
    if (set(arrays) != {"features", "targets"} or arrays["features"].shape[:2] != (2, 4)
            or arrays["features"].ndim != 3 or arrays["targets"].shape != (2, 8)
            or any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values())
            or record["entry_hashes"] != {key: array_identity(a) for key, a in arrays.items()}):
        raise ValueError("Shard arrays or per-entry checksums invalid")
    return arrays, record


def save_shard(out, pair, config_sha, features, targets, inputs):
    path, receipt = shard_paths(out, pair)
    if path.exists() or receipt.exists():
        raise FileExistsError("Do not overwrite an existing shard")
    arrays = dict(features=features, targets=targets)
    atomic_npz(path, **arrays)
    atomic_json(receipt, dict(pair_id=pair, configuration_sha256=config_sha, bytes=path.stat().st_size,
        sha256=digest(path), entry_hashes={key: array_identity(a) for key, a in arrays.items()}, inputs=inputs))


def worker(index, pairs, args, config_sha, reference, expected_targets, expected_inputs, events, cancel, begin, deadline):
    torch.set_num_threads(1)
    budget = WorkerBudget(deadline, cancel)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: cancel.set())
    key = f"worker_{index}_{os.getpid()}"
    receipt_path = args.out/"workers"/(key+".json")
    receipt = dict(worker=index, pid=os.getpid(), assigned_pairs=pairs, completed_pairs=[],
                   status="starting", start_unix=time.time(), job_id=os.environ.get("SLURM_JOB_ID"),
                   step_id=os.environ.get("SLURM_STEP_ID"))
    atomic_json(receipt_path, receipt)
    with (args.out/"workers"/(key+".log")).open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            budget.check(); torch.cuda.set_device(index)
            encoder = Encoder(args.ijepa, args.checkpoint, device=f"cuda:{index}", grid=4)
            expected = reference["original_cache_identity"]["encoder_identity"]
            if (encoder.identity["code_commit"] != expected["source_commit"]
                    or encoder.identity["checkpoint_sha256"] != expected["checkpoint_sha256"]):
                raise ValueError("Encoder differs from the original frozen reader's encoder")
            receipt.update(status="ready", encoder=encoder.identity, gpu=torch.cuda.get_device_name(index),
                           visible_uuid=os.environ["CUDA_VISIBLE_DEVICES"].split(",")[index])
            atomic_json(receipt_path, receipt); events.put(dict(event="ready", worker=index))
            while not begin.wait(.2):
                budget.check()
            for pair in pairs:
                budget.check()
                value = encode_pair([episode("dev", pair, variant) for variant in (0, 1)], pair,
                                    encoder, expected_targets[pair-128],expected_inputs[str(pair)])
                budget.check(); save_shard(args.out, pair, config_sha, *value)
                receipt["completed_pairs"].append(pair); receipt["status"] = "running"
                atomic_json(receipt_path, receipt)
                print(json.dumps(dict(pair=pair, status="complete", images_encoded=8)), flush=True)
            receipt["status"] = "complete"
        except BaseException as exc:
            receipt.update(status="partial", reason=str(exc)); traceback.print_exc()
            events.put(dict(event="error", worker=index, reason=str(exc)))
        finally:
            receipt["end_unix"] = time.time(); atomic_json(receipt_path, receipt)
        if receipt["status"] == "complete":
            events.put(dict(event="done", worker=index))


def extract_parallel(args, pairs, config_sha, reference, old_targets, old_inputs, budget):
    context = mp.get_context("spawn")
    events, cancel, begin = context.Queue(), context.Event(), context.Event()
    processes = []
    deadline = budget.start+budget.charge/budget.gpus*3600-60
    def wait(event, pending):
        pending = set(pending)
        while pending:
            budget.check()
            try:
                message = events.get(timeout=.25)
            except queue.Empty:
                if any(processes[i].exitcode is not None for i in pending):
                    raise RuntimeError("Counterfactual worker exited before receipt")
                continue
            if message["event"] == "error":
                raise RuntimeError("Counterfactual worker failed: "+message["reason"])
            if message["event"] != event or message["worker"] not in pending:
                raise RuntimeError("Unexpected counterfactual worker receipt")
            pending.remove(message["worker"])
    success = False
    try:
        for index in range(8):
            process = context.Process(target=worker, args=(index, [p for p in pairs if p % 8 == index], args,
                config_sha, reference, old_targets, old_inputs, events, cancel, begin, deadline), daemon=True)
            processes.append(process); process.start(); wait("ready", {index})
        begin.set(); wait("done", range(8)); success = True
    finally:
        if not success:
            cancel.set(); begin.set()
        end = time.monotonic()+5
        for process in processes:
            process.join(timeout=max(0, end-time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill(); process.join(timeout=2)
        events.close()


def feature_parity(new, old):
    new, old = np.asarray(new, dtype=np.float64), np.asarray(old, dtype=np.float64)
    difference = new-old
    return dict(bitwise_equal=bool(np.array_equal(new, old)), max_abs=float(np.abs(difference).max()),
                rmse=float(np.sqrt(np.mean(difference**2))),
                relative_l2=float(np.linalg.norm(difference.ravel())/max(np.linalg.norm(old.ravel()), 1e-12)),
                original_feature_rms=float(np.sqrt(np.mean(old**2))))


def merge(args, config, reference, original, slurm, budget):
    records, inputs = [], {}
    for pair in range(128, 384):
        budget.check()
        row = read_shard(args.out, pair, json_sha(config))
        if row is None:
            raise ValueError("Missing fixed pair shard")
        arrays, receipt = row
        if arrays["features"].shape != (2, 4, 20480):
            raise ValueError("Official pooled feature width differs from20480")
        records.append(arrays); inputs[str(pair)] = receipt["inputs"]
    expected = {f"{p:06d}{suffix}" for p in range(128,384) for suffix in (".npz", ".json")}
    if {p.name for p in (args.out/"shards").iterdir()} != expected:
        raise ValueError("Shard inventory is not the exact256 inspected pairs")
    arrays = dict(features=np.stack([r["features"] for r in records]),
        targets=np.stack([r["targets"] for r in records]), pair_ids=np.arange(128,384,dtype=np.int64),
        condition_indices=CONDITION_INDICES, condition_names=np.array(CONDITIONS),
        unique_image_order=np.array(UNIQUE_ORDER), role=np.array("counterfactual"))
    if not np.array_equal(arrays["targets"], original["targets"]):
        raise ValueError("Merged targets differ from the original bridge labels")
    original_features=arrays["features"][:,:,[0,3],:]
    parity=feature_parity(original_features, original["features"])
    cache=args.out/"cache.npz"; receipt_path=args.out/"receipt.json"
    entry_hashes={key:array_identity(value) for key,value in arrays.items()}
    if cache.exists() or receipt_path.exists():
        if not cache.is_file() or not receipt_path.is_file():
            raise ValueError("Incomplete aggregate preserved; inspect before resume")
        receipt=json.loads(receipt_path.read_text())
        if receipt.get("cache_sha256")!=digest(cache) or receipt.get("entry_hashes")!=entry_hashes or receipt.get("configuration_sha256")!=json_sha(config):
            raise ValueError("Existing aggregate does not match validated shards")
        return receipt
    atomic_npz(cache,**arrays)
    receipt=dict(schema=SCHEMA,role="counterfactual",cache_sha256=digest(cache),cache_bytes=cache.stat().st_size,
        entry_hashes=entry_hashes,configuration_sha256=json_sha(config),**reference,
        source_commit=args.source_commit,source_identity=config["source_identity"],protocol_sha256=config["protocol_sha256"],
        encoder_identity=reference["original_cache_identity"]["encoder_identity"],pooling="4x4_flattened",feature_dtype="float32",
        conditions=list(CONDITIONS),condition_indices=CONDITION_INDICES.tolist(),unique_image_order=list(UNIQUE_ORDER),
        groups=256,variants_per_pair=2,unique_images_per_variant=4,images_encoded=2048,batch_size=4,
        batch_layout=[["v0:u00","v0:u11","v1:u00","v1:u11"],["v0:u01","v0:u10","v1:u01","v1:u10"]],
        unique_image_rule="encode_four_once_per_variant_reuse_across_conditions",
        original_pixel_parity=True,outside_rectangle_union_unchanged=True,targets_bitwise_equal_original_bridge=True,
        original_rgb_hashes_equal_archived_bridge=True,target_masks_disjoint=True,target_pixels_unchanged=True,
        input_records_sha256=json_sha(inputs),original_feature_parity=parity,slurm=slurm,
        runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda),
        test_opened=False,new_reader_fit=False,old_gate_mutation=False,p0_authorized=False,
        interpretation="Exploratory paired intervention on the previously inspected original bridge, not a fresh confirmation")
    atomic_json(receipt_path,receipt)
    return receipt


def run_reserved(args,budget):
    if args.out.exists() and any(p.is_symlink() for p in args.out.rglob("*")):
        raise ValueError("Symlinks are not allowed in a scientific output cache")
    original,reference=original_inputs(args.prior_scene);budget.check()
    if re.fullmatch(r"[0-9a-f]{40}",args.source_commit) is None:
        raise ValueError("Frozen source commit required")
    if (ROOT/".git").exists() and subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()!=args.source_commit:
        raise ValueError("Source commit differs from deployment HEAD")
    if digest(args.checkpoint)!=reference["original_cache_identity"]["encoder_identity"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from original frozen encoder")
    sources={name:digest(ROOT/name) for name in SOURCE_FILES}
    config=dict(schema=SCHEMA,source_commit=args.source_commit,source_identity=sources,
        protocol_sha256=digest(PROTOCOL),reference=reference,pair_ids=list(range(128,384)),workers=8,batch_size=4,
        conditions=list(CONDITIONS),condition_indices=CONDITION_INDICES.tolist(),
        batch_layout="original4_then_edited4_exact_prior_original_batch_order")
    manifest_path=args.out/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest.get("configuration")!=config:
            raise ValueError("Resume identity differs from the frozen experiment")
        if manifest.get("status")=="complete":
            if manifest.get("files")!=reusable.file_inventory(args.out):
                raise ValueError("Completed output checksum/inventory mismatch")
            return json.loads((args.out/"receipt.json").read_text()),None
    elif args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh output directory or intact partial manifest")
    for subdir in ("shards","workers"):
        (args.out/subdir).mkdir(parents=True,exist_ok=True)
    atomic_json(manifest_path,dict(status="partial",configuration=config))
    slurm=validate_runtime("dgx",args.account,args.partition,"H800",torch.cuda.device_count(),torch.cuda.get_device_name(0),
        allocation_job=args.allocation_job,qos=args.qos,shared_allocation_gpus=8,expected_worker_gpus=8,
        allow_gpu_sharing=args.allow_gpu_sharing,min_free_gpu_gib=args.min_free_gpu_gib)
    try:
        pending=[p for p in range(128,384) if read_shard(args.out,p,json_sha(config)) is None]
        if pending:
            extract_parallel(args,pending,json_sha(config),reference,original["targets"],original["archived_inputs"],budget)
        budget.check()
        if sources!={name:digest(ROOT/name) for name in SOURCE_FILES}:
            raise ValueError("Source changed during extraction")
        if digest(args.prior_scene/"reader/freeze_receipt.json")!=reference["frozen_reader_sha256"] or digest(args.prior_scene/"bridge_cache/cache.npz")!=reference["original_bridge_cache_sha256"]:
            raise ValueError("Original frozen inputs changed during extraction")
        result=merge(args,config,reference,original,slurm,budget);budget.check()
        return result,config
    except BaseException as exc:
        atomic_json(manifest_path,dict(status="partial",configuration=config,reason=str(exc),files=reusable.file_inventory(args.out)))
        raise


def run(args):
    if args.workers!=8 or not 0<args.lease_hours<=.20:
        raise ValueError("Use eight workers and at most a12minute1.6GPUh lease")
    if args.out.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    args.out=args.out.resolve();args.prior_scene=args.prior_scene.resolve()
    if args.out==args.prior_scene or args.prior_scene in args.out.parents:
        raise ValueError("The prior scene is read-only; choose a separate new run")
    args.budget_root=args.budget_root.resolve()
    if args.budget_root==args.prior_scene or args.prior_scene in args.budget_root.parents:
        raise ValueError("The prior ledger is read-only; inherit it in the new run directory")
    if not (args.budget_root/"budget.json").is_file():
        raise ValueError("An inherited cumulative ledger is required")
    inherited=json.loads((args.prior_scene/"ledger/budget.json").read_text())
    ledger=json.loads((args.budget_root/"budget.json").read_text())
    if ledger["charged_gpuh"]+1e-8<inherited["charged_gpuh"] or not {x["lease_id"] for x in inherited["leases"]}<={x["lease_id"] for x in ledger["leases"]}:
        raise ValueError("Previous chargedGPUh and leases must be preserved")
    if ledger["charged_gpuh"]+args.lease_hours*8>6+1e-8:
        raise BudgetExceeded("Diagnostic envelope would exceed6GPUh; preserve54GPUh for later authorized work")
    budget=Budget(args.budget_root,"counterfactual_v5_2",2.,args.lease_hours,gpu_count=8)
    status="partial"
    try:
        result,config=run_reserved(args,budget);status="complete"
    finally:
        budget.close(status)
    if config is not None:
        atomic_json(args.out/"manifest.json",dict(status="complete",configuration=config,budget_lease=budget.lease,files=reusable.file_inventory(args.out)))
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("out","ijepa","checkpoint","budget-root"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--prior-scene","--original-scene",dest="prior_scene",type=Path,required=True)
    for name in ("source-commit","allocation-job","account","partition","qos"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--workers",type=int,choices=(8,),default=8)
    p.add_argument("--lease-hours",type=float,default=.20)
    p.add_argument("--allow-gpu-sharing",action="store_true")
    p.add_argument("--min-free-gpu-gib",type=float,default=40.)
    return p


if __name__=="__main__":
    print(json.dumps(run(parser().parse_args()),indent=2,allow_nan=False))
