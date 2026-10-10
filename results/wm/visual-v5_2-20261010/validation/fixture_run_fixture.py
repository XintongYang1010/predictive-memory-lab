#!/usr/bin/env python3
"""Software-only scorer integration fixture built from already-saved features.

No nuisance intervention is performed. Four slot names exercise the production
scoring API, but slots intentionally duplicate old t6/t7 cached features. This
fixture is not experimental evidence and must not enter a scientific cache.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "4"

from argparse import Namespace
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo/"scripts"))
import extract_v5_2_counterfactual as extractor
import score_v5_2_counterfactual as scorer
from v5_confirmation_reader import sha256


def write(path, value):
    with open(path, "x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    started = time.monotonic()
    out = Path(__file__).resolve().parent
    scene = repo/"execution/visual-v5_1-20261010/continuation-20261010"
    paths = [scene/"bridge_cache/cache.npz", scene/"bridge_report/predictions.npz",
             scene/"reader/freeze_receipt.json", scene/"reader/frozen_model.npz"]
    before = {str(path.relative_to(repo)): sha256(path) for path in paths}
    frozen = json.loads(paths[2].read_text())
    original_receipt = scene/"bridge_cache/receipt.json"
    with np.load(paths[0], allow_pickle=False) as z:
        old = z["features"]
        targets, pairs = z["targets"], z["pair_ids"]
    arrays = dict(features=old[:, :, [0, 0, 1, 1], :], targets=targets, pair_ids=pairs,
                  condition_indices=extractor.CONDITION_INDICES.copy(),
                  condition_names=np.array(extractor.CONDITIONS),
                  unique_image_order=np.array(extractor.UNIQUE_ORDER), role=np.array("counterfactual"))
    assert arrays["features"].shape == (256, 2, 4, 20480)
    np.testing.assert_array_equal(arrays["features"][:, :, [0, 3]], old)
    np.testing.assert_array_equal(pairs, np.arange(128, 384))
    cache, receipt_path = out/"software_only_features.npz", out/"software_only_receipt.json"
    with open(cache, "xb") as stream:
        np.savez_compressed(stream, **arrays)
    sources = {name: sha256(repo/name) for name in extractor.SOURCE_FILES}
    reference = dict(frozen_reader_sha256=before[str(paths[2].relative_to(repo))],
        original_bridge_cache_sha256=before[str(paths[0].relative_to(repo))],
        original_bridge_receipt_sha256=sha256(original_receipt),
        original_cache_identity=frozen["cache_identity"], original_scene=str(scene))
    receipt = dict(schema=extractor.SCHEMA, role="counterfactual", software_only=True,
        fixture="NO INTERVENTION: slots [old_t6,old_t6,old_t7,old_t7]; names are API labels only",
        scientific_evidence=False, cache_sha256=sha256(cache), cache_bytes=cache.stat().st_size,
        entry_hashes={name: extractor.array_identity(array) for name, array in arrays.items()}, **reference,
        source_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        source_identity=sources, protocol_sha256=sha256(extractor.PROTOCOL),
        encoder_identity=frozen["cache_identity"]["encoder_identity"], pooling="4x4_flattened", feature_dtype="float32",
        conditions=list(extractor.CONDITIONS), condition_indices=extractor.CONDITION_INDICES.tolist(),
        unique_image_order=list(extractor.UNIQUE_ORDER), groups=256, variants_per_pair=2,
        slot_count_per_variant=4, images_encoded=0, new_images_generated=0,
        targets_bitwise_equal_original_bridge=True, test_opened=False, new_reader_fit=False,
        old_gate_mutation=False, p0_authorized=False,
        interpretation="Software integration test only: no intervention or new encoding; all four conditions must be identical")
    write(receipt_path, receipt)
    args = Namespace(cache=cache, receipt=receipt_path, model=scene/"reader",
                     prior_predictions=paths[1], out=out/"software_only_report")
    with open(out/"test.log", "x", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        print("SOFTWARE ONLY: no intervention, no scientific-data generation, no GPU")
        report = scorer.evaluate(args)
        with np.load(args.out/"predictions.npz", allow_pickle=False) as z:
            identical = {}
            for method in ("concat", "t6", "t7"):
                p = z[method]
                max_difference = max(float(np.max(np.abs(p[:, :, c]-p[:, :, 0]))) for c in range(4))
                np.testing.assert_array_equal(p, np.repeat(p[:, :, :1], 4, axis=2))
                identical[method] = max_difference
        for method, score in report["scores"].items():
            assert score["primary_dx_mse_reduction"] == 0
            assert score["secondary_balanced_dx_mse_reduction"] == 0
            assert score["secondary_static_average_motion_nmse_reduction"] == 0
            assert score["primary_dx_paired_95ci"] == [0., 0.]
            assert score["secondary_balanced_dx_paired_95ci"] == [0., 0.]
            assert report["original_prediction_parity"][method]["passed"] is True
        assert report["reused_single_frame_inputs_passed"] is True
        assert all(value == 0 for value in report["reused_single_frame_input_max_differences"].values())
        assert before == {str(path.relative_to(repo)): sha256(path) for path in paths}
        assert sources == {name: sha256(repo/name) for name in sources}
        print("PASS: all condition arrays bitwise identical; all effects/CIs exactly zero; parity/reuse passed; original files unchanged")
    summary = dict(software_only=True, scientific_evidence=False, status="passed",
        dimensions=list(arrays["features"].shape), targets_and_pair_ids_unchanged=True,
        all_condition_predictions_max_abs_differences=identical,
        all_primary_and_balanced_effects_exactly_zero=True, all_primary_and_balanced_CIs_exactly_zero=True,
        original_prediction_parity=report["original_prediction_parity"],
        single_frame_reuse=report["reused_single_frame_input_max_differences"],
        original_files_unchanged=before, sources_unchanged=sources,
        no_gpu=True, no_new_images=True, no_fitting=True, test_access=False,
        python=sys.executable, numpy=np.__version__, elapsed_seconds=time.monotonic()-started,
        fixture_script_sha256=sha256(__file__), report_sha256=sha256(args.out/"report.json"))
    write(out/"summary.json", summary)
    (out/"README.md").write_text("# Software-only no-intervention scorer fixture\n\n"
        "Existing bridge t6/t7 features were duplicated into four slots `[t6,t6,t7,t7]`. "
        "The production scorer was called with its strict256×2×4×20480 cache schema and the unchanged frozen reader. "
        "Condition names here are software API labels; no scientific intervention, new image, encoder, model fitting, GPU, or Test access occurred.\n\n"
        "All three readers produced bitwise-identical predictions across four conditions. Primary and balanced effects and confidence bounds are exactly zero; "
        "original prediction parity and exact single-frame reuse passed. Original cache/model/prediction hashes were unchanged. "
        "See summary.json and test.log. Do not mix this fixture into scientific results.\n")
    print(json.dumps({k: summary[k] for k in ("software_only", "status", "dimensions", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
