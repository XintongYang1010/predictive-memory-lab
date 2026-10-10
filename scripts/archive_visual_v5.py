#!/usr/bin/env python3
"""Package a downloaded V5 workspace; never upload or delete the original run."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


VARIANTS = ("A0", "A1", "B0", "B1", "C0", "C1", "full", "recent", "fifo")
STAGES = ("profile", "features", "p0", *("adapt_" + v for v in VARIANTS), "replay")
SKIP_PARTS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache"}
CACHE_PARTS = {"features", "probe_features"}
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def metadata(path):
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def safe_files(root):
    """Never follow symlinks or silently include an external environment."""
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in archive input: {path}")
        if path.is_file():
            yield path


def source_files(code):
    required = ("START_HERE.md", "docs/CODEX_DGX_HANDOFF.md", "docs/VISUAL_MEMORY_V5.md",
                "pmr/visual_memory/__main__.py", "scripts/launch_visual_v5.py")
    for name in required:
        if not (code / name).is_file():
            raise FileNotFoundError(f"Necessary source is missing: {code / name}")
    roots = {"pmr", "scripts", "tests", "docs", ".github"}
    for path in safe_files(code):
        relative = path.relative_to(code)
        if relative.parts[0] in roots or (len(relative.parts) == 1 and
                (path.suffix in {".md", ".toml", ".txt"} or path.name == ".gitignore")):
            yield path


def assess(run):
    states = {stage: read_json(run / f"{stage}_status.json") for stage in STAGES}
    missing = [f"{stage}_status.json: complete" for stage, value in states.items()
               if not value or value.get("status") != "complete"]
    required = ["run_manifest.json", "profile.json", "data_manifest.json", "budget.json",
                "feature_gate.json", "overfit_gate.json", "history_gate.json", "p0.pt",
                "p0_dev.jsonl", "report_status.json", "metrics_by_family.csv",
                "state_costs.csv", "pressure_curves.png", "failure_ledger.jsonl",
                "reader_only_C0/per_episode.jsonl", "reader_only_C0/best.pt"]
    for variant in VARIANTS:
        required.extend(f"{variant}/{name}" for name in
                        ("model.pt", "per_episode.jsonl", "status.json", "lr_search.json", "fit/learning_curves.jsonl"))
    missing.extend(name for name in required if not (run / name).is_file())
    for gate in ("feature_gate", "overfit_gate", "history_gate"):
        value = read_json(run / f"{gate}.json")
        if not value or value.get("passed") is not True:
            missing.append(f"{gate}.json: passed")
    report = read_json(run / "report_status.json")
    if not report or report.get("status") != "pilot_complete":
        missing.append("report_status.json: pilot_complete")
    return {"experiment_status": "complete" if not missing else "partial",
            "stages": states, "missing_completion_requirements": missing}


def archive(work_root, out, run_name="run1", checkpoint_archive=None):
    work_root, out = Path(work_root).resolve(), Path(out).resolve()
    if out == work_root or work_root in out.parents:
        raise ValueError("Output must be outside the downloaded workspace")
    if out.exists():
        raise FileExistsError(f"Use a fresh output directory: {out}")
    if Path(run_name).is_absolute() or ".." in Path(run_name).parts:
        raise ValueError("Run name must be relative to the workspace")
    code, run = work_root / "code", work_root / run_name
    if not run.is_dir() or not (work_root / "creation_record.json").is_file():
        raise FileNotFoundError("Need the run directory and original creation_record.json")
    source = list(source_files(code))
    assessment = assess(run)
    archive_path = Path(checkpoint_archive).resolve() if checkpoint_archive else out.with_name(out.name + "-checkpoints.tar.gz")
    if archive_path.exists():
        raise FileExistsError(archive_path)
    if archive_path == work_root or work_root in archive_path.parents or archive_path == out or out in archive_path.parents:
        raise ValueError("Checkpoint archive must be outside both workspace and result directory")

    files, checkpoints, exclusions = [], [], {}
    def exclude(category, path):
        entry = exclusions.setdefault(category, {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += path.stat().st_size
    for path in source:
        files.append((path, Path("source") / path.relative_to(code)))
    for path in safe_files(run):
        relative = path.relative_to(run)
        if relative.parts[0] in CACHE_PARTS:
            if path.name == "manifest.json":
                files.append((path, Path("run") / relative))
            else:
                exclude("reproducible_feature_tensors", path)
        elif path.suffix in CHECKPOINT_SUFFIXES:
            checkpoints.append((path, (Path("run") / relative).as_posix()))
        elif path.name == "active.lock" or path.name.endswith(".tmp"):
            exclude("transient_or_incomplete_files", path)
        else:
            files.append((path, Path("run") / relative))
    for folder in ("logs", "previews"):
        if (work_root / folder).exists():
            files.extend((path, Path(folder) / path.relative_to(work_root / folder))
                         for path in safe_files(work_root / folder))
    for path in sorted(work_root.iterdir()):
        if path.is_symlink():
            raise ValueError(f"Refusing workspace symlink: {path}")
        if path.is_file() and path.suffix in {".json", ".txt", ".log", ".csv", ".yaml", ".yml", ".md"}:
            files.append((path, Path("workspace") / path.name))
    if len({name.as_posix() for _, name in files}) != len(files):
        raise ValueError("Duplicate archive paths")
    out.mkdir(parents=True)
    for path, name in files:
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    checkpoint_files = {name: metadata(path) for path, name in checkpoints}
    checkpoint_info = None
    if checkpoints:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "x:gz") as bundle:
            for path, name in checkpoints:
                bundle.add(path, arcname=name, recursive=False)
            content = (json.dumps(checkpoint_files, indent=2) + "\n").encode()
            member = tarfile.TarInfo("CHECKPOINTS.json"); member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
        checkpoint_info = {"filename": archive_path.name, **metadata(archive_path), "files": checkpoint_files}

    try:
        git_commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"],
                                            text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_commit = None
    profile = read_json(run / "profile.json") or {}
    source_manifest = read_json(run / "run_manifest.json") or {}
    reconstruction = {
        "source_commit_if_available": git_commit,
        "recorded_configuration": source_manifest.get("configuration"),
        "recorded_encoder": profile.get("encoder"),
        "excluded": {**exclusions,
                     "official_assets": "Not copied. Recreate pinned I-JEPA source and weights using source/scripts/prepare_visual_v5.sh; compare recorded encoder code/checkpoint hashes.",
                     "python_environments": "Not copied. Recreate isolated environment using archived preparation script and environment logs.",
                     "git_metadata": "Not copied. Actual source snapshot and per-file SHA256 hashes are included."},
        "feature_reconstruction": "Use the archived source/configuration, encoder identity, and run/features or run/probe_features manifests. Test remains unopened. A failed feature gate must be investigated; do not bypass it.",
        "checkpoint_restore": "Verify checkpoint asset first, then extract its run/ tree over the archived run/ tree. CHECKPOINTS.json lists every training checkpoint.",
        "publication": "Upload this result directory to GitHub and checkpoint archive as a release asset; reread/download and run this helper with --verify before considering cleanup.",
        "cleanup": "This helper does not authorize cleanup. Incomplete/failed experiments must retain the remote workspace."}
    write_json(out / "reconstruction.json", reconstruction)
    write_json(out / "experiment_status.json", assessment)
    rows = ["# V5 experiment archive", "", f"Experiment status: **{assessment['experiment_status']}**.", "",
            "This records actual available artifacts. Packaging success is not scientific completion.", "",
            "| Stage | State | Job ID |", "|---|---|---|"]
    for stage, value in assessment["stages"].items():
        value = value or {}
        rows.append(f"| {stage} | {value.get('status', 'not run / unavailable')} | {(value.get('runtime') or {}).get('job_id', '')} |")
    rows.extend(["", "GPU runtime: `" + json.dumps(profile.get("runtime", source_manifest.get("latest_runtime", {})), sort_keys=True) + "`.", "",
                 "## Scientific gates", ""])
    for name in ("feature_gate", "overfit_gate", "history_gate"):
        value = read_json(run / f"{name}.json")
        rows.extend([f"### {name}", "", "```json", json.dumps(value, indent=2), "```", ""])
    if (out / "run/pressure_curves.png").exists():
        rows.extend(["![Dev pressure curves](run/pressure_curves.png)", ""])
    rows.extend(["## Files and interpretation", "",
                 "- [Completion assessment](experiment_status.json) lists missing requirements and actual stage errors.",
                 "- [Reconstruction and exclusions](reconstruction.json) records source/configuration/encoder identity.",
                 "- [SHA256 inventory](archive_manifest.json) covers every packaged file and separate checkpoint asset.",
                 "- `run/` contains available metrics, curves, gate results, budget, and submission receipts; `logs/` and `previews/` preserve available evidence.",
                 "- Controlled synthetic RGB pilot; Dev-only findings. Test is reserved and must remain unopened.",
                 "- Do not interpret a failed prerequisite as a comparison between memory architectures. Missing downstream outputs are expected after a failed gate.", "",
                 "Remote cleanup has **not** been performed or authorized by this packaging helper.", ""])
    (out / "README.md").write_text("\n".join(rows))
    inventory = {path.relative_to(out).as_posix(): metadata(path) for path in safe_files(out)}
    manifest = {"schema": "v5-archive-1", **assessment, "files": inventory, "checkpoint_archive": checkpoint_info}
    write_json(out / "archive_manifest.json", manifest)
    (out / "SHA256SUMS").write_text("".join(f"{entry['sha256']}  {name}\n" for name, entry in sorted(inventory.items()))
                                    + f"{sha256(out / 'archive_manifest.json')}  archive_manifest.json\n")
    verify(out, archive_path if checkpoints else None)
    return manifest


def verify(out, checkpoint_archive=None):
    out = Path(out).resolve()
    manifest = read_json(out / "archive_manifest.json")
    if not manifest or manifest.get("schema") != "v5-archive-1":
        raise ValueError("Missing or unsupported archive manifest")
    checksums = {}
    for line in (out / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        if name in checksums:
            raise ValueError("Duplicate checksum entry")
        checksums[name] = checksum
    expected_sums = {name: entry["sha256"] for name, entry in manifest["files"].items()}
    expected_sums["archive_manifest.json"] = sha256(out / "archive_manifest.json")
    if checksums != expected_sums:
        raise ValueError("SHA256SUMS does not match archive manifest")
    for name, entry in manifest["files"].items():
        path = out / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or out not in path.resolve().parents or metadata(path) != entry):
            raise ValueError(f"Archive file hash/size/path mismatch: {name}")
    info = manifest.get("checkpoint_archive")
    if info:
        path = Path(checkpoint_archive) if checkpoint_archive else out.parent / info["filename"]
        if metadata(path) != {key: info[key] for key in ("bytes", "sha256")}:
            raise ValueError("Checkpoint archive hash/size mismatch")
        with tarfile.open(path, "r:gz") as bundle:
            members = bundle.getmembers()
            if len(members) != len(info["files"]) + 1 or {m.name for m in members} != {*info["files"], "CHECKPOINTS.json"}:
                raise ValueError("Checkpoint archive member inventory mismatch")
            for member in members:
                if not member.isfile():
                    raise ValueError("Checkpoint archive contains a non-regular member")
                stream = bundle.extractfile(member)
                if member.name == "CHECKPOINTS.json":
                    if json.load(stream) != info["files"]:
                        raise ValueError("Embedded checkpoint inventory mismatch")
                else:
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                    if {"bytes": member.size, "sha256": digest.hexdigest()} != info["files"][member.name]:
                        raise ValueError(f"Checkpoint hash mismatch: {member.name}")
    return {"verified": True, "experiment_status": manifest["experiment_status"],
            "files": len(manifest["files"]), "checkpoints": len((info or {}).get("files", {}))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-name", default="run1")
    parser.add_argument("--checkpoint-archive", type=Path)
    parser.add_argument("--verify", type=Path, metavar="DOWNLOADED_RESULT_DIR")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify, args.checkpoint_archive)
    else:
        if not args.work_root or not args.out:
            parser.error("packaging requires --work-root and --out")
        manifest = archive(args.work_root, args.out, args.run_name, args.checkpoint_archive)
        result = {"out": str(args.out), "experiment_status": manifest["experiment_status"],
                  "files": len(manifest["files"]),
                  "checkpoint_archive": (manifest["checkpoint_archive"] or {}).get("filename")}
    print(json.dumps(result, indent=2))
