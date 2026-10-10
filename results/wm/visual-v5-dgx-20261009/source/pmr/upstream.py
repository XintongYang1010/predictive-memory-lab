"""Fetch a pinned upstream and prepare an ignored, auditable local overlay."""
from pathlib import Path
import hashlib
import importlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/fudan-zvg/RealMotion.git"
UPSTREAM_COMMIT = "e292609f03ccd415c4f2ac6dd7fd907c098508e8"


def prepare(source=None):
    source = Path(source or ROOT / "external/RealMotion")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", UPSTREAM_URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", UPSTREAM_COMMIT], check=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected RealMotion {UPSTREAM_COMMIT}; got {head}")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError("Upstream checkout has tracked changes; use a clean pinned checkout")
    overlay = ROOT / "external/portable"
    target = overlay / "realmotion"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source / "realmotion", target)
    replacements = {
        "from natten import NeighborhoodAttention1D": "from pmr.compat import NeighborhoodAttention1D",
        "from timm.models.layers import DropPath": "from pmr.compat import DropPath",
        # Upstream's -1 reshape is ambiguous when no non-focal actors survive.
        ".view(B, x_others.size(1), -1, 2)": ".view(B, x_others.size(1), self.dense_predictor[-1].out_features // 2, 2)",
    }
    changes = []
    for path in sorted(target.rglob("*.py")):
        original = path.read_text()
        updated = original
        for before, after in replacements.items():
            updated = updated.replace(before, after)
        if updated != original:
            path.write_text(updated)
            changes.append({"path":str(path.relative_to(overlay)),
                            "original_sha256":hashlib.sha256(original.encode()).hexdigest(),
                            "portable_sha256":hashlib.sha256(updated.encode()).hexdigest()})
    (overlay / "patch_manifest.json").write_text(json.dumps({"commit":head,"changes":changes}, indent=2))
    return overlay


def activate():
    overlay = ROOT / "external/portable"
    if not (overlay / "patch_manifest.json").exists():
        raise RuntimeError("Run bash scripts/setup.sh (or python -m pmr setup) first")
    manifest = json.loads((overlay / "patch_manifest.json").read_text())
    if manifest["commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("Stale upstream overlay; run setup again")
    sys.path.insert(0, str(overlay))
    return importlib.import_module("realmotion.model.realmotion"), importlib.import_module("realmotion.datamodules.av2_dataset")
