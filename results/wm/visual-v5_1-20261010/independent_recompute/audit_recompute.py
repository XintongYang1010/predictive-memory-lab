#!/usr/bin/env python3
"""Read-only archive audit and CPU prediction replay; never fit or generate data."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "4"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import time
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_identity(array):
    return dict(sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
                dtype=str(array.dtype), shape=list(array.shape))


def json_read(path):
    return json.loads(Path(path).read_text())


def verify_inventory(root, manifest):
    declared = manifest["files"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root/"archive_manifest.json" and p != root/"manifest.json"}
    if actual != set(declared):
        raise ValueError("Inventory mismatch at " + str(root))
    total = 0
    for name, record in declared.items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or parts.as_posix() != name:
            raise ValueError("Unsafe manifest path")
        file = root/name
        if any(p.is_symlink() for p in (file, *file.parents) if p != root.parent):
            raise ValueError("Symlink in archived input")
        if file.stat().st_size != record["bytes"] or sha(file) != record["sha256"]:
            raise ValueError("Archived file checksum mismatch: " + str(file))
        total += record["bytes"]
    return dict(files=len(declared), bytes=total)


def compare_json(original, replay, path="", differences=None):
    differences = [] if differences is None else differences
    if isinstance(original, dict):
        if not isinstance(replay, dict) or set(original) != set(replay):
            raise ValueError("Report keys differ: " + path)
        for key in original:
            if key != "predictions_sha256":
                compare_json(original[key], replay[key], path+"/"+key, differences)
    elif isinstance(original, list):
        if not isinstance(replay, list) or len(original) != len(replay):
            raise ValueError("Report list differs: " + path)
        for i, (a, b) in enumerate(zip(original, replay)):
            compare_json(a, b, path+"/"+str(i), differences)
    elif isinstance(original, (float, int)) and not isinstance(original, bool):
        difference = abs(original-replay)
        if not math.isfinite(difference) or difference > 1e-8:
            raise ValueError("Numerical report mismatch: " + path)
        differences.append(dict(path=path, original=original, replay=replay, absolute_difference=difference))
    elif original != replay:
        raise ValueError("Report value differs: " + path)
    return differences


def audit(scene, repo, out):
    started = time.monotonic()
    sys.path.insert(0, str(repo/"scripts"))
    import v5_confirmation_reader as reader
    import v5_temporal_confirmation as protocol
    archive = json_read(scene/"archive_manifest.json")
    input_inventory = verify_inventory(scene, archive)
    frozen_metadata = json_read(scene/"reader/freeze_receipt.json")
    for name, expected in frozen_metadata["sources"].items():
        if sha(repo/"scripts"/name) != expected or sha(scene/"code/scripts"/name) != expected:
            raise ValueError("Local/deployed reader, helper, or plan differs from the freeze")

    def forbidden(*args, **kwargs):
        raise RuntimeError("Fitting, selection, or data generation is forbidden in independent replay")
    for name in ("fit", "fit_kernel", "train_models", "choose_reader"):
        setattr(reader, name, forbidden)
    for name in ("make_quartet", "seed_for", "render", "choose_reader", "kernel_predict"):
        setattr(protocol, name, forbidden)
    original_rng = np.random.default_rng
    def registered_rng(seed=None):
        if seed != 7709:
            raise RuntimeError("No new seed allowed; only registered bootstrap7709 may be replayed")
        return original_rng(seed)
    np.random.default_rng = registered_rng

    arrays, selections, frozen = reader.load_frozen(scene/"reader")
    role_audits, rgb_hash_sets = {}, {}
    for role in ("fit", "confirm", "bridge"):
        folder = scene/(role+"_cache")
        manifest = json_read(folder/"manifest.json")
        if manifest["status"] != "complete":
            raise ValueError("Role cache incomplete: " + role)
        verified = verify_inventory(folder, manifest)
        data = reader.load_cache(folder/"cache.npz", folder/"receipt.json", role,
                                 None if role == "fit" else frozen)
        receipt = data["receipt"]
        if receipt["test_opened"] is not False or receipt["groups"] != 256:
            raise ValueError("Role receipt count/Test condition violated")
        for name, expected in receipt["source_identity"].items():
            if sha(scene/"code"/name) != expected:
                raise ValueError("Deployed runtime source differs from role receipt: " + name)
        config = manifest["configuration"]
        if receipt["configuration_sha256"] != json_sha(config):
            raise ValueError("Configuration hash mismatch")
        units = np.arange(128, 384) if role == "bridge" else np.arange(256)
        if config["role"] != role or config["units"] != units.tolist():
            raise ValueError("Role/group configuration mismatch")
        inputs, rgb_hashes = {}, []
        for index, unit in enumerate(units.tolist()):
            path = folder/"shards"/f"{unit:06d}.npz"
            record = json_read(path.with_suffix(".json"))
            if (record["unit"] != unit or record["role"] != role
                    or record["sha256"] != sha(path) or record["bytes"] != path.stat().st_size
                    or record["configuration_sha256"] != json_sha(config)):
                raise ValueError("Shard identity/checksum mismatch")
            with np.load(path, allow_pickle=False) as shard:
                for key in ("features", "targets"):
                    if not np.array_equal(shard[key], data[key][index]):
                        raise ValueError("Aggregate is not an exact ordered merge of its shards")
                    if record["entry_hashes"][key] != array_identity(shard[key]):
                        raise ValueError("Shard array hash mismatch")
            inputs[str(unit)] = record["inputs"]
            labels = inputs[str(unit)]["targets"]
            if labels != array_identity(data["targets"][index]):
                raise ValueError("Input target receipt differs from cached target")
            metadata = inputs[str(unit)]["metadata"]
            if role == "bridge":
                if [m["variant"] for m in metadata] != [0, 1] or any(m["pair_id"] != f"dev-{unit:06d}" or m["split"] != "dev" for m in metadata):
                    raise ValueError("Bridge metadata mismatch")
                rgb_hashes.extend(x["sha256"] for x in inputs[str(unit)]["cue_rgb"])
            else:
                if metadata["role"] != role or metadata["block"] != unit or metadata["group_id"] != f"{protocol.SCHEMA}/{role}/{unit:06d}":
                    raise ValueError("Quartet metadata mismatch")
                if inputs[str(unit)]["unique_images_encoded"] != 3:
                    raise ValueError("Quartet unique-image count mismatch")
                rgb_hashes.extend(x["sha256"] for x in inputs[str(unit)]["unique_rgb"])
        if receipt["input_records_sha256"] != json_sha(inputs):
            raise ValueError("Input record aggregate hash mismatch")
        rgb_hash_sets[role] = set(rgb_hashes)
        workers = [json_read(p) for p in sorted((folder/"workers").glob("*.json"))]
        if len(workers) != 8 or {w["worker"] for w in workers} != set(range(8)):
            raise ValueError("Expected eight independent worker receipts")
        if any(w["status"] != "complete" or w["assigned_units"] != w["completed_units"] for w in workers):
            raise ValueError("Worker assignment incomplete")
        if sorted(sum((w["completed_units"] for w in workers), [])) != units.tolist():
            raise ValueError("Worker groups are missing, duplicated, or unexpected")
        if len({w["visible_uuid"] for w in workers}) != 8:
            raise ValueError("Workers did not select eight distinct GPUs")
        if role == "fit":
            if (frozen["fit_cache_sha256"] != sha(folder/"cache.npz") or
                    frozen["fit_cache_receipt_sha256"] != sha(folder/"receipt.json") or
                    not np.array_equal(arrays["fit_features"], data["features"]) or
                    not np.array_equal(arrays["fit_block_ids"], data["block_ids"])):
                raise ValueError("Frozen model basis differs from FIT cache")
            fit_mean = data["targets"].astype(np.float64).reshape(-1, 8).mean(0)
            for name in reader.INPUTS:
                if not np.array_equal(arrays[name+"__target_mean"], fit_mean):
                    raise ValueError("Frozen target mean differs from complete FIT labels")
        role_audits[role] = dict(inventory=verified, feature_shape=list(data["features"].shape),
            target_shape=list(data["targets"].shape), groups=256, examples=receipt["examples"],
            canonical_first_id=int(units[0]), canonical_last_id=int(units[-1]),
            aggregate_exactly_matches_shards=True, input_records_hash_verified=True,
            eight_complete_distinct_gpu_workers=True, gpu_steps=sorted({str(w["job_id"])+"."+str(w["step_id"]) for w in workers}),
            cache_sha256=receipt["cache_sha256"], receipt_sha256=data["receipt_sha256"],
            input_rgb_hash_count=len(rgb_hashes), distinct_input_rgb_hash_count=len(set(rgb_hashes)))
        del data
    if rgb_hash_sets["fit"] & rgb_hash_sets["confirm"]:
        raise ValueError("Identical recorded RGB hash overlaps FIT and confirmation")
    comparisons = {}
    for role in ("confirm", "bridge"):
        generated = out/(role+"_replay")
        report = reader.evaluate(scene/"reader", scene/(role+"_cache/cache.npz"),
            scene/(role+"_cache/receipt.json"), role, generated)
        original = json_read(scene/(role+"_report/report.json"))
        if original["predictions_sha256"] != sha(scene/(role+"_report/predictions.npz")):
            raise ValueError("Original prediction archive hash mismatch")
        if (scene/(role+"_report/cache_receipt.json")).read_bytes() != (scene/(role+"_cache/receipt.json")).read_bytes():
            raise ValueError("Original report receipt copy mismatch")
        prediction_diffs = {}
        with np.load(scene/(role+"_report/predictions.npz"), allow_pickle=False) as old, np.load(generated/"predictions.npz", allow_pickle=False) as new:
            if set(old.files) != set(new.files):
                raise ValueError("Prediction entry set mismatch")
            for key in old.files:
                delta = float(np.max(np.abs(old[key]-new[key])))
                if old[key].shape != new[key].shape or not np.isfinite(delta) or delta > 1e-9:
                    raise ValueError("Prediction replay mismatch: " + role+"/"+key)
                prediction_diffs[key] = dict(shape=list(old[key].shape), max_absolute_difference=delta,
                                            bitwise_equal=bool(np.array_equal(old[key], new[key])))
        differences = compare_json(original, report)
        comparisons[role] = dict(predictions=prediction_diffs,
            metadata_and_decisions_exact_match=True,
            all_report_numbers_max_absolute_difference=max(x["absolute_difference"] for x in differences),
            report_number_comparisons=differences,
            original_npz_sha256=original["predictions_sha256"], replay_npz_sha256=report["predictions_sha256"],
            primary_passed=report["primary_metric_conditions_met"], primary_scores=report["scores"]["concat"])
    final_inventory = verify_inventory(scene, archive)
    if input_inventory != final_inventory:
        raise ValueError("Original scene changed during read-only audit")
    summary = dict(status="verified_independent_frozen_prediction_replay", script_sha256=sha(__file__),
        archive_manifest_sha256=sha(scene/"archive_manifest.json"), original_scene_unchanged=True,
        original_scene_inventory=input_inventory, local_sources_match_freeze=True,
        reader_frozen_receipt_sha256=frozen["receipt_sha256"], cache_roles=role_audits,
        fit_confirm_recorded_rgb_hash_overlap=0, comparisons=comparisons,
        safeguards=dict(fitting_disabled=True, data_generation_disabled=True, no_gpu=True,
                        no_remote_access=True, no_test_access=True, only_registered_bootstrap_seed=7709),
        interpretation="Confirmation passes; original-distribution bridge fails. P0 remains unauthorized, original V5 gate remains failed.",
        limitations=["This is numerical replay of the same frozen artifacts, not a new independent scientific sample.",
            "Input RGB hashes and freshness attestations are audited as archived provenance; no inputs were regenerated.",
            "Cross-platform BLAS can differ in last floating-point bits; prediction tolerance1e-9 and report tolerance1e-8 were fixed before replay.",
            "Bootstrap intervals reuse the registered seed and remain conditional on the frozen reader and registered sample."],
        runtime=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, platform=platform.platform(),
                     wall_seconds=time.monotonic()-started))
    (out/"audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    lines = ["# Independent frozen-reader replay", "", summary["interpretation"], "",
        f"Verified the original {input_inventory['files']} files ({input_inventory['bytes']} bytes) before and after replay; the scene remained unchanged.",
        "All three role caches contain exactly256 canonical groups; their arrays exactly match every checked shard. The frozen basis equals FIT cache tensors and IDs. Eight distinct complete GPU-worker receipts were verified per role.", "",
        "| Role | Primary decision | Max prediction absolute difference | Max report-number absolute difference |",
        "|---|---|---:|---:|"]
    for role, row in comparisons.items():
        lines.append(f"| {role} | {'passes' if row['primary_passed'] else 'fails'} | {max(v['max_absolute_difference'] for v in row['predictions'].values()):.3g} | {row['all_report_numbers_max_absolute_difference']:.3g} |")
    lines += ["", "Prediction/score recomputation used the saved coefficients and moments only. Fit, selection and data-generation functions were disabled; only the original bootstrap seed7709 was permitted. No Test or GPU access occurred.", ""]
    lines += ["- "+text for text in summary["limitations"]]
    (out/"README.md").write_text("\n".join(lines)+"\n")
    return {key: summary[key] for key in ("status", "original_scene_inventory", "interpretation", "runtime")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out/"audit.json").exists() or (a.out/"confirm_replay").exists():
        raise FileExistsError("Do not overwrite a previous independent replay")
    print(json.dumps(audit(a.scene.resolve(), a.repo.resolve(), a.out.resolve()), indent=2))
