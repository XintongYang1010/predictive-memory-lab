#!/usr/bin/env python3
"""BR-0.1 CPU development run. No downloads, GPU jobs, or original gate edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CV_ROOT = ROOT.parent / "CV_project"
sys.path.insert(0, str(HERE))
import core


def imports(cv_root=CV_ROOT):
    sys.path.insert(0, str(cv_root))
    import torch
    from experiments.phase2.memory import MemoryEntry, StreamingMemory
    from experiments.phase2 import literature_baselines as lib
    torch.set_num_threads(2)
    return torch, MemoryEntry, StreamingMemory, lib


def student_inputs(lib, candidates, context):
    return lib.make_student_inputs(np.stack([e.representation for e in candidates]),
                                   np.stack([e.representation for e in context.recent]),
                                   np.array([e.timestamp for e in candidates], np.int64),
                                   np.array([e.timestamp for e in context.recent], np.int64),
                                   context.current_time)


def parent_after_arrival(snapshot, new_entry):
    next_recent = snapshot.recent + (new_entry,)
    if len(next_recent) <= core.RECENT:
        return (), next_recent
    return snapshot.retained + (next_recent[0],), next_recent[1:]


def future_attention_teacher(lib, candidates, recent, query_keys, query_times, decision_time):
    """Offline only; receives query KEYS, never target residuals/deletion losses.

    Attention is conditioned on historical candidates. Zero-mass queries are
    omitted, not replaced with invented uniform attention. All legal probes are
    genuinely after the retention decision, so only the future group exists.
    """
    query_times = np.asarray(query_times, np.int64)
    if not len(query_times) or np.any(query_times <= decision_time):
        raise ValueError("teacher requires genuinely future queries after the decision")
    if len(query_keys) != len(query_times):
        raise ValueError("query keys and times differ")
    rows, times, masses = [], [], []
    for key, qt in zip(query_keys, query_times):
        _, weights = core.read(tuple(candidates) + tuple(recent), key, int(qt))
        a = weights[:len(candidates)].mean(axis=1)
        mass = float(a.sum())
        if mass > 0:
            rows.append(a / mass); times.append(int(qt)); masses.append(mass)
    if not rows:
        return None, {"available_queries": len(query_times), "valid_queries": 0,
                      "reason": "no future query places attention on a historical candidate"}
    prob = lib.balanced_attention_targets(np.asarray(rows, np.float64),
                                         np.array(times, np.int64),
                                         np.array([e.timestamp for e in candidates], np.int64),
                                         decision_time=decision_time)
    return prob, {"available_queries": len(query_times), "valid_queries": len(rows),
                  "query_times": times, "candidate_times": [e.timestamp for e in candidates],
                  "decision_time": int(decision_time), "future_group_only": True,
                  "conditional_history_attention": np.asarray(rows).tolist(),
                  "history_attention_mass": masses,
                  "target": prob.tolist(),
                  "nonuniform": bool(np.ptp(prob) > 1e-6)}


def make_entries(data, i, MemoryEntry):
    """Valid-observation timestamps exclude invalid splice transition W=0."""
    r, o, w, keys = data
    ordinal = 0
    for t in range(len(w)):
        if not w[t]:
            yield t, ordinal, None
            continue
        entry = MemoryEntry(f"e{i:03d}", ordinal, core.representation(r[t], o[t], keys[t]))
        yield t, ordinal, entry
        ordinal += 1


def data_for(arrays, condition, index):
    return tuple(arrays[f"{condition}__{name}"][index] for name in ("R", "O", "W", "key"))


def future_indices(writes, source_t):
    return [t for t in range(source_t + 1, min(len(writes), source_t + core.HORIZON + 1)) if writes[t]]


def collect_training(arrays, manifest, split, capacity, api):
    _, MemoryEntry, StreamingMemory, lib = api
    from experiments.phase2.memory import PolicyContext
    records, traces, attempted = [], [], 0
    for condition in core.CONDITIONS:
        for i in split["train_indices"]:
            data = data_for(arrays, condition, i)
            r, o, w, keys = data
            stream = StreamingMemory(core.RECENT, capacity, "fifo", seed=core.SEED)
            ordinal_by_t = np.cumsum(w).astype(np.int64) - 1
            for t, ordinal, entry in make_entries(data, i, MemoryEntry):
                if entry is None:
                    continue
                candidates, recent = parent_after_arrival(stream.snapshot, entry)
                if t in core.ANCHORS and len(candidates) == capacity + 1:
                    attempted += 1
                    future = future_indices(w, t)
                    target, trace = future_attention_teacher(lib, candidates, recent, keys[future],
                                                             ordinal_by_t[future], ordinal)
                    trace.update(condition=condition, source_index=i, original_t=t, capacity=capacity)
                    traces.append(trace)
                    if target is not None:
                        context = PolicyContext(ordinal, recent)
                        inputs = student_inputs(lib, candidates, context)
                        records.append(lib.SalienceTrainingRecord(inputs, target,
                                       manifest["ordered_a_ids"][i], "train", "future_attention"))
                stream.push(entry)
    return records, traces, {"attempted_parents": attempted, "records": len(records),
                            "skipped_no_history_attention": attempted - len(records),
                            "nonuniform_records": sum(t.get("nonuniform", False) for t in traces),
                            "causal_student_only": True, "teacher_future_only_degeneracy": True}


def frozen_parent_oracle(candidates, recent, data, source_t, chosen_deleted_id):
    """Exact local deletion enumeration; no intervention reaches online policy."""
    r, o, w, keys = data
    future = future_indices(w, source_t)
    query_ordinals = np.cumsum(w).astype(np.int64) - 1
    losses = []
    denominator = float(o[future].sum())
    for dropped in range(len(candidates)):
        retained = tuple(e for j, e in enumerate(candidates) if j != dropped)
        num = 0.
        for t in future:
            prediction, _ = core.read(retained + tuple(recent), keys[t], int(query_ordinals[t]))
            num += float(((prediction - r[t]) ** 2 * o[t]).sum())
        losses.append(num / denominator if denominator else 0.)
    chosen = next(j for j, entry in enumerate(candidates) if entry.entry_id == chosen_deleted_id)
    best = int(np.argmin(losses))
    return {"chosen_drop": chosen, "oracle_drop": best, "candidate_losses": losses,
            "chosen_loss": losses[chosen], "oracle_loss": losses[best],
            "regret": max(0., losses[chosen] - losses[best]), "future_den": denominator,
            "candidate_times": [e.timestamp for e in candidates],
            "candidate_keys": [core.item_key(e) for e in candidates],
            "recent_times": [e.timestamp for e in recent],
            "future_original_times": future, "future_keys": keys[future].tolist(),
            "oracle_scope": "same-parent one deletion; fixed recent; no intervening writes"}


def run_stream(data, index, capacity, method, model, api, audit=True):
    _, MemoryEntry, StreamingMemory, lib = api
    tracker = None
    if method == "WR":
        tracker = lib.CausalSurpriseTracker()
        callback = tracker.score
    elif method == "PaFu":
        callback = lambda candidates, context: lib.student_probabilities(model, student_inputs(lib, candidates, context))
    elif method == "FIFO":
        callback = None
    else:
        raise ValueError(method)
    stream = StreamingMemory(core.RECENT, capacity, "attention" if callback else "fifo",
                             seed=core.SEED, score_callback=callback)
    r, o, w, keys = data
    predictions = np.zeros_like(r, np.float64)
    parents, peak_actual, peak_controller, peak_items = [], 0, 0, 0
    for t, ordinal, entry in make_entries(data, index, MemoryEntry):
        snapshot = stream.snapshot
        predictions[t], _ = core.read(snapshot.retained + snapshot.recent, keys[t], ordinal)
        if entry is None:
            continue
        candidates, recent = parent_after_arrival(snapshot, entry)
        if tracker:
            tracker.observe(entry, past_window=snapshot.recent)
        pushed = stream.push(entry)
        if tracker:
            tracker.retain_only(pushed.recent + pushed.retained)
        if audit and t in core.ANCHORS and len(candidates) == capacity + 1:
            parent = frozen_parent_oracle(candidates, recent, data, t, pushed.transition.deleted_id)
            parent.update(original_t=t, decision_time=ordinal)
            parents.append(parent)
        account = stream.accounting()
        aux = tracker.accounting()["auxiliary_logical_bytes"] if tracker else 0
        items = sum(e.payload_bytes for e in pushed.retained + pushed.recent)
        actual = account["logical_payload_bytes"] + aux
        controller = actual - items
        peak_actual = max(peak_actual, actual)
        peak_items = max(peak_items, items)
        peak_controller = max(peak_controller, controller)
        if controller > core.CONTROLLER_RESERVATION:
            raise ValueError("actual controller exceeds frozen common reserved envelope")
    weights = lib.student_accounting(model) if method == "PaFu" else {"student_parameter_bytes": 0,
               "student_buffer_bytes": 0, "student_logical_bytes": 0}
    accounting = {"capacity": capacity, "recent_length": core.RECENT,
        "reserved_item_bytes": (capacity + core.RECENT) * 288,
        "reserved_controller_bytes": core.CONTROLLER_RESERVATION,
        "reserved_dynamic_total_bytes": (capacity + core.RECENT) * 288 + core.CONTROLLER_RESERVATION,
        "peak_actual_logical_dynamic_bytes": peak_actual, "peak_actual_item_bytes": peak_items,
        "peak_actual_controller_bytes": peak_controller, "unused_reservation_is_not_actual_allocation": True,
        "shared_lookup_bytes": 0, "hidden_feature_archive_bytes": 0,
        "includes_python_allocator_or_temporary_workspace": False, **weights}
    return predictions, parents, accounting


def all_observations_mean(data):
    r, o, w, keys = data
    sums = np.zeros((core.DK, core.D), np.float64)
    counts = np.zeros_like(sums)
    pred = np.zeros_like(r, np.float64)
    for t, key in enumerate(keys):
        np.divide(sums[key], counts[key], out=pred[t], where=counts[key] > 0)
        if w[t]:
            sums[key] += r[t] * o[t]
            counts[key] += o[t]
    return pred


def oracle_summary(parents):
    result = {}
    for method in sorted({p["method"] for p in parents}):
        for condition in core.CONDITIONS:
            p = [x for x in parents if x["method"] == method and x["condition"] == condition]
            if not p:
                continue
            den = sum(x["future_den"] for x in p)
            result[f"{method}/{condition}"] = {
                "parents": len(p), "positive_regret_parents": sum(x["regret"] > 1e-10 for x in p),
                "weighted_chosen_loss": sum(x["chosen_loss"] * x["future_den"] for x in p) / den,
                "weighted_oracle_loss": sum(x["oracle_loss"] * x["future_den"] for x in p) / den,
                "weighted_regret": sum(x["regret"] * x["future_den"] for x in p) / den,
                "max_parent_regret": max(x["regret"] for x in p)}
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def frozen_sources(api, cv_root):
    lib = api[-1]
    memory_module = sys.modules[api[1].__module__]
    sources = {"protocol": HERE / "PROTOCOL.md", "core": HERE / "core.py", "runner": Path(__file__),
               "shared_mechanisms": Path(lib.__file__), "shared_memory": Path(memory_module.__file__)}
    if sources["shared_mechanisms"].resolve() != (cv_root / "experiments/phase2/literature_baselines.py").resolve():
        raise ValueError("shared mechanism module resolved outside the requested CV repository")
    if sources["shared_memory"].resolve() != (cv_root / "experiments/phase2/memory.py").resolve():
        raise ValueError("shared memory module resolved outside the requested CV repository")
    return {name: {"path": str(path.resolve()), "sha256": core.sha256(path)} for name, path in sources.items()}


def verify_sources(sources):
    for name, record in sources.items():
        if core.sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"source changed during development run: {name}")


def report(summary, destination):
    lines = ["# BR-0.1 frozen-residual bank development results", "",
        "This is an L0-derived frozen-residual diagnostic, not complete L0/visual evaluation or paper reproduction.",
        "Source-disjoint train/dev: 69 train streams, 24 dev streams, 3 boundary-purged streams. Reused exploratory dev; seed 17 only.",
        "No original V6 gate changed, no FINAL/test opened, no GPU or data restoration.", "",
        "## Held-out development MPE (lower is better)", "",
        "| Method | EXACT POST | NEWPATH POST | CHANGE POST | CHANGE first20 |", "|---|---:|---:|---:|---:|"]
    metrics = summary["metrics"]
    for method in sorted(metrics["EXACT/POST"]["mpe"]):
        values = [metrics[k]["mpe"][method] for k in ("EXACT/POST", "NEWPATH/POST", "CHANGE/POST", "CHANGE/DET")]
        lines.append("| " + method + " | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines += ["", "## Same-parent frozen-bank oracle", "",
        "The oracle enumerates every deletion in each policy's own K+1 parent, with the same R=4 recent entries and six future queries. No intervening writes. It is a local privileged diagnostic, not a globally optimal online policy.", "",
        "| Policy/condition | Parents | Positive-regret parents | Chosen MPE | Oracle MPE | Regret |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in summary["oracle"].items():
        lines.append(f"| {name} | {value['parents']} | {value['positive_regret_parents']} | {value['weighted_chosen_loss']:.6f} | {value['weighted_oracle_loss']:.6f} | {value['weighted_regret']:.6f} |")
    lines += ["", "## Limits", "",
        "- WR is a causal Gaussian/MAD/peak/NMS adapter on residual/mask/key features, not the visual paper system.",
        "- PaFu is trained on the fixed reader's real conditional long-memory attention; only future groups exist here. No deletion utility is used as teacher. Teacher coverage and nonuniformity are in summary.json.",
        "- Within K, item count and reserved controller envelope match. Actual dynamic bytes and extra student weights differ and are itemized; unused reservation is not allocated memory.",
        "- A1/A2/A6 are 90,720-byte matrix-state references, not equal-byte bank competitors. The all-observations mean uses 352,512 bytes of float64 sufficient statistics.",
        "- A2 replays frozen gates originally produced from full Z, which is absent here; it is not a deployable input-parity comparison with these bank policies.",
        "- The residual/mask/key embedding includes a scalar key/647. Its ordinal geometry is an adaptation limitation; failures do not isolate the original papers' memory mechanisms.",
        "- Parent oracles across policies can have different past candidate sets. Compare paired regret within a parent; do not infer global optimality or a common-history advantage from cross-parent oracle scores.",
        "- Only eight source A clusters are evaluated; adjacent CHANGE pairs still share sources within the dev partition. Paired intervals in summary.json are exploratory, not independent confirmation.",
        "- No RGB, original actions, continuous pose, or Z is available; top bad cases are timestamp/key/loss traces, not visual explanations.", ""]
    destination.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    base = ROOT / "deployment/v6-h800-20261015/a2-gate-attribution-prepared/frozen"
    parser.add_argument("--input", type=Path, default=base / "a2_frozen_inputs.npz")
    parser.add_argument("--manifest", type=Path, default=base / "manifest.json")
    parser.add_argument("--references", type=Path, default=ROOT / "deployment/v6-h800-20261015/oracle-s2-terminal/evidence/runs/main/S2/eval/T0_s0.5/ns0/rows.jsonl.gz")
    parser.add_argument("--cv-root", type=Path, default=CV_ROOT)
    parser.add_argument("--out", type=Path, default=HERE / "results/br-0.1-dev-seed17")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    arrays, manifest, inventory = core.load_inputs(args.input, args.manifest)
    split = core.source_split(manifest)
    if args.inventory_only:
        print(json.dumps({"inventory": inventory, "split": split}, indent=2)); return
    if args.out.exists():
        raise ValueError("output already exists; refusing to overwrite prior experiment")
    args.out.mkdir(parents=True)
    started = time.time()
    api = imports(args.cv_root)
    torch, _, _, lib = api
    sources = frozen_sources(api, args.cv_root)
    write_json(args.out / "frozen-protocol.json", {"version": "BR-0.1", "protocol_sha256": core.sha256(HERE / "PROTOCOL.md"),
        "core_sha256": core.sha256(HERE / "core.py"), "runner_sha256": core.sha256(__file__),
        "seed": core.SEED, "capacities": core.CAPACITIES, "recent": core.RECENT,
        "anchors": core.ANCHORS, "future_horizon": core.HORIZON, "split": split, "inventory": inventory,
        "resolved_cv_root": str(args.cv_root.resolve()), "frozen_sources_before_fit": sources})
    refs, reproduction = core.audit_references(arrays, manifest, args.references)
    write_json(args.out / "reference-reproduction.json", reproduction)
    print(json.dumps({"stage": "references", **reproduction}), flush=True)
    rows, all_parents, budgets, training = [], [], {}, {}
    for condition in core.CONDITIONS:
        for i in split["dev_indices"]:
            data = data_for(arrays, condition, i)
            for arm in ("A1", "A2", "A6", "all_observations_mean", "no_memory"):
                pred = refs[(condition, arm)][i] if arm in ("A1", "A2", "A6") else all_observations_mean(data) if arm == "all_observations_mean" else np.zeros_like(data[0])
                rows.append({"condition": condition, "source_index": i, "method": "REF/" + arm,
                             **core.window_sums(pred, *data[:3])})
    for capacity in core.CAPACITIES:
        records, traces, info = collect_training(arrays, manifest, split, capacity, api)
        if not records:
            raise ValueError(f"K={capacity}: no nonempty attention teacher records")
        allowed = [manifest["ordered_a_ids"][i] for i in split["train_indices"]]
        model, curve = lib.train_salience_student(records, seed=core.SEED, epochs=100, hidden=32,
                                                  lr=.001, train_group_ids=allowed)
        verify_sources(sources)
        torch.save({"state_dict": model.state_dict(), "input_dim": model.input_dim, "hidden": model.hidden,
                    "seed": core.SEED, "protocol": "BR-0.1", "train_source_ids": allowed}, args.out / f"student-k{capacity}.pt")
        training[str(capacity)] = {**info, "loss_curve": curve, "final_epoch": 100,
                                  "shared_module_sha256": core.sha256(Path(lib.__file__)),
                                  "student_checkpoint_sha256": core.sha256(args.out / f"student-k{capacity}.pt")}
        write_json(args.out / f"teacher-k{capacity}.json", traces)
        print(json.dumps({"stage": "trained", "capacity": capacity, **info, "initial_loss": curve[0], "final_loss": curve[-1]}), flush=True)
        for method in ("FIFO", "WR", "PaFu"):
            method_name = f"K{capacity}/{method}"
            method_budgets = []
            for condition in core.CONDITIONS:
                for i in split["dev_indices"]:
                    data = data_for(arrays, condition, i)
                    pred, parents, accounting = run_stream(data, i, capacity, method, model, api)
                    rows.append({"condition": condition, "source_index": i, "method": method_name,
                                 **core.window_sums(pred, *data[:3])})
                    for parent in parents:
                        parent.update(condition=condition, source_index=i, method=method_name)
                    all_parents.extend(parents)
                    method_budgets.append(accounting)
            budgets[method_name] = {key: max(x[key] for x in method_budgets) if isinstance(method_budgets[0][key], (int, float)) and not isinstance(method_budgets[0][key], bool) else method_budgets[0][key]
                                    for key in method_budgets[0]}
            print(json.dumps({"stage": "evaluated", "method": method_name, "streams": 24}), flush=True)
    comparisons = [(f"K{k}/FIFO", f"K{k}/{m}") for k in core.CAPACITIES for m in ("WR", "PaFu")]
    verify_sources(sources)
    summary = {"protocol": "BR-0.1", "claim": inventory["claim"], "inventory": inventory, "split": split,
               "reference_reproduction": reproduction, "training": training, "budgets": budgets,
               "metrics": core.paired_summary(rows, comparisons), "oracle": oracle_summary(all_parents),
               "runtime_seconds": time.time() - started, "gpu_used": False, "raw_data_restored": False,
               "shared_module_sha256": core.sha256(Path(lib.__file__)),
               "frozen_sources_before_fit": sources, "sources_unchanged_after_run": True,
               "original_prereg_edited": False, "final_test_opened": False}
    write_json(args.out / "rows.json", rows)
    write_json(args.out / "parents.json", all_parents)
    write_json(args.out / "bad-cases.json", sorted(all_parents, key=lambda x: x["regret"], reverse=True)[:30])
    write_json(args.out / "summary.json", summary)
    report(summary, args.out / "REPORT.md")
    print(json.dumps({"stage": "complete", "out": str(args.out), "seconds": summary["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
