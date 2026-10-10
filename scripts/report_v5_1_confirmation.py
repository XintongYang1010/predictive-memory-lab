#!/usr/bin/env python3
"""Posthoc V5.1 tables, plots and Chinese summary from a fetched run directory.

This standalone renderer never generates scientific data, fits a model, changes
a gate, contacts a cluster, or authorizes P0. Keep it outside the frozen deployed
scientific source. Missing reports mean missing evidence, not scientific failure.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

INPUTS = ("concat", "t6", "t7")
LABELS = {"concat": "Two frames", "t6": "t6 only", "t7": "t7 only"}
DIMS = ("A_dx", "A_dy", "B_dx", "B_dy")
SCHEMA = "v5-temporal-confirm-1"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


class Inputs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inventory = {}

    def register(self, path):
        path = Path(path)
        self.inventory[str(path.relative_to(self.root))] = dict(bytes=path.stat().st_size, sha256=sha(path))

    def json(self, relative, optional=False):
        path = self.root / relative
        if optional and not path.exists():
            return None
        self.register(path)
        return json.loads(path.read_text())


def load_report(inputs, role, frozen):
    folder = inputs.root / (role+"_report")
    report = inputs.json(role+"_report/report.json", optional=True)
    if report is None:
        return None
    counts = {"blocks": 256, "views": 1024} if role == "confirm" else {"paired_worlds": 256, "episodes": 512}
    if (report.get("schema") != SCHEMA or report.get("role") != role or
            report.get("counts") != counts or report.get("test_access") is not False or
            report.get("primary_reader") != "concat"):
        raise ValueError(f"{role}: invalid report schema/count/Test declaration")
    copied = inputs.json(role+"_report/cache_receipt.json")
    if report.get("cache_receipt_sha256") != sha(folder / "cache_receipt.json"):
        raise ValueError(f"{role}: copied cache receipt hash mismatch")
    if copied.get("cache_sha256") != report.get("cache_sha256"):
        raise ValueError(f"{role}: cache hash references disagree")
    if frozen is not None and report.get("frozen_reader_sha256") != sha(inputs.root / "reader/freeze_receipt.json"):
        raise ValueError(f"{role}: frozen reader receipt mismatch")
    pred_path = folder / "predictions.npz"
    inputs.register(pred_path)
    if report.get("predictions_sha256") != sha(pred_path):
        raise ValueError(f"{role}: prediction hash mismatch")
    expected = (256, 4, 8) if role == "confirm" else (256, 2, 8)
    with np.load(pred_path, allow_pickle=False) as arrays:
        if arrays["targets"].shape != expected or not np.isfinite(arrays["targets"]).all():
            raise ValueError(f"{role}: target shape/count/finiteness mismatch")
        for name in INPUTS:
            if arrays[name].shape != expected or not np.isfinite(arrays[name]).all():
                raise ValueError(f"{role}: {name} prediction shape/count/finiteness mismatch")
            score = report["scores"][name]
            if role == "confirm":
                values = [score["absolute_displacement_skill"], score["position_skill"],
                          *score["primary_lower_bounds"],
                          *score["contrast_skills"]["end_anchor"], *score["contrast_skills"]["start_anchor"]]
                if len(score["primary_lower_bounds"]) != 9:
                    raise ValueError("Confirmation must report the registered nine lower bounds")
            else:
                values = [score["motion_skill"], score["position_skill"],
                          score["motion_mean_lower_bound"], *score["per_dimension_skill"]]
                if len(score["per_dimension_skill"]) != 8:
                    raise ValueError("Bridge must report all eight target coordinates")
            if not np.isfinite(values).all():
                raise ValueError(f"{role}: nonfinite reported metrics")
            if score["metric_conditions_met"] != all(score["checks"].values()):
                raise ValueError(f"{role}: internally inconsistent check flags")
        if report["primary_metric_conditions_met"] != report["scores"]["concat"]["metric_conditions_met"]:
            raise ValueError(f"{role}: primary flag differs from primary score")
    # Hash/count checks validate the rendered evidence. Scientific re-scoring is
    # the independent downstream verifier's responsibility, not this renderer's.
    return report


def metric_rows(report):
    rows = []
    if report is None:
        return rows
    role = report["role"]
    for name in INPUTS:
        score = report["scores"][name]
        if role == "confirm":
            for offset, anchor in ((0, "end_anchor"), (4, "start_anchor")):
                for i, dim in enumerate(DIMS):
                    rows.append(dict(reader=name, metric=anchor+"_"+dim,
                        point=score["contrast_skills"][anchor][i],
                        lower_bound=score["primary_lower_bounds"][offset+i], threshold=0,
                        bound_kind="one-sided percentile, Bonferroni .05/9"))
            rows.append(dict(reader=name, metric="absolute_motion_mean", point=score["absolute_displacement_skill"],
                lower_bound=score["primary_lower_bounds"][8], threshold=0,
                bound_kind="one-sided percentile, Bonferroni .05/9"))
        else:
            for i, dim in enumerate(DIMS):
                rows.append(dict(reader=name, metric="motion_"+dim,
                    point=score["per_dimension_skill"][4+i], lower_bound=None, threshold=0, bound_kind="point criterion"))
            rows.append(dict(reader=name, metric="motion_mean", point=score["motion_skill"],
                lower_bound=score["motion_mean_lower_bound"], threshold=0, bound_kind="one-sided 95%, paired world"))
        rows.append(dict(reader=name, metric="position_mean", point=score["position_skill"],
                         lower_bound=None, threshold=.1, bound_kind="point criterion"))
    return rows


def cv_rows(selections):
    result = []
    for name in INPUTS if selections else ():
        selection = selections[name]
        for trial in selection["trials"]:
            result.append(dict(reader=name, kernel=trial["kernel"], mean_loss_alpha=trial["mean_loss_alpha"],
                mean_cv_normalized_mse=trial["mean_loss"], selected=trial == selection["selected"],
                **{f"fold_{i}": value for i, value in enumerate(trial["fold_losses"], 1)}))
    return result


def collect_jobs(inputs, controllers):
    rows = []
    for controller, data in controllers.items():
        for job in data.get("jobs", []):
            stage = job.get("stage", "unknown")
            logs = job.get("logs", []) or [p.name for p in (inputs.root / "logs").glob(stage+"_*.log")]
            step_ids = sorted(set(m.group(1) for name in logs
                              if (m := re.search(r"_(\d+\.\d+)\.log$", str(name)))))
            for actual in step_ids or [""]:
                rows.append(dict(stage=stage, slurm_step=actual,
                    allocation_job=job.get("allocation_job", ""), selected_gpus=job.get("selected_gpus"),
                    status=job.get("status", "unknown"), returncode=job.get("returncode"),
                    controller=controller))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot_all(out, confirm, bridge, selections):
    if not any((confirm, bridge, selections)):
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    colors = ("#2166ac", "#d95f02", "#6a3d9a")
    if confirm:
        score = confirm["scores"]["concat"]
        labels = [anchor+" / "+dim for anchor in ("End anchor", "Start anchor") for dim in DIMS]
        points = np.array(score["contrast_skills"]["end_anchor"]+score["contrast_skills"]["start_anchor"])
        lower = np.array(score["primary_lower_bounds"][:8])
        y = np.arange(8)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.hlines(y, np.minimum(points, lower), np.maximum(points, lower), color="#9ca3af", linewidth=2)
        ax.scatter(points, y, label="Point skill", color=colors[0], s=42, zorder=3)
        ax.scatter(lower, y, label="Adjusted one-sided lower bound", color=colors[1], marker="|", s=180, zorder=4)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y, labels); ax.invert_yaxis()
        ax.set_xlabel("Contrast skill (zero = structural single-frame null)")
        ax.set_title("Primary pooled two-frame reader: all eight contrast coordinates")
        ax.legend(loc="best", fontsize=8); ax.grid(axis="x", alpha=.15)
        fig.tight_layout(); fig.savefig(out / "confirmation_contrasts.png", dpi=180); plt.close(fig)
        files.append("confirmation_contrasts.png")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        for i, name in enumerate(INPUTS):
            s = confirm["scores"][name]
            axes[0].scatter(i, s["absolute_displacement_skill"], color=colors[i], s=50)
            axes[0].plot([i, i], [s["primary_lower_bounds"][8], s["absolute_displacement_skill"]], color=colors[i])
            axes[0].scatter(i, s["primary_lower_bounds"][8], marker="_", color=colors[i], s=100)
            axes[1].bar(i, s["position_skill"], color=colors[i], width=.55)
        for ax in axes:
            ax.set_xticks(range(3), [LABELS[n] for n in INPUTS]); ax.axhline(0, color="black", lw=1)
            ax.grid(axis="y", alpha=.15)
        axes[0].set_ylabel("Mean displacement skill"); axes[0].set_title("Absolute motion: point and adjusted lower bound")
        axes[1].axhline(.1, color="black", linestyle="--", lw=1)
        axes[1].set_ylabel("Position skill vs frozen FIT mean"); axes[1].set_title("Position (dashed threshold = 0.1)")
        fig.tight_layout(); fig.savefig(out / "confirmation_controls.png", dpi=180); plt.close(fig)
        files.append("confirmation_controls.png")
    if bridge:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        x = np.arange(4)
        for i, name in enumerate(INPUTS):
            s = bridge["scores"][name]
            axes[0].bar(x+(i-1)*.24, s["per_dimension_skill"][4:], width=.23, color=colors[i], label=LABELS[name])
            axes[1].scatter(i, s["motion_skill"], color=colors[i], s=50)
            axes[1].plot([i, i], [s["motion_mean_lower_bound"], s["motion_skill"]], color=colors[i])
            axes[1].scatter(i, s["motion_mean_lower_bound"], marker="_", color=colors[i], s=100)
        axes[0].set_xticks(x, DIMS); axes[0].legend(fontsize=8)
        axes[0].set_title("Original-distribution bridge: coordinate skills")
        axes[1].set_xticks(range(3), [LABELS[n] for n in INPUTS]); axes[1].set_title("Mean motion: point and one-sided 95% lower bound")
        for ax in axes:
            ax.axhline(0, color="black", linestyle="--", lw=1); ax.set_ylabel("Skill vs frozen FIT mean")
            ax.grid(axis="y", alpha=.15)
        fig.tight_layout(); fig.savefig(out / "bridge_controls.png", dpi=180); plt.close(fig)
        files.append("bridge_controls.png")
    if selections:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
        for ax, name in zip(axes, INPUTS):
            for kernel in ("linear", "quadratic"):
                trials = sorted((t for t in selections[name]["trials"] if t["kernel"] == kernel), key=lambda t:t["mean_loss_alpha"])
                ax.plot([t["mean_loss_alpha"] for t in trials], [t["mean_loss"] for t in trials], marker="o", label=kernel)
            selected = selections[name]["selected"]
            ax.scatter(selected["mean_loss_alpha"], selected["mean_loss"], marker="*", s=140, color="black", zorder=4)
            ax.set_xscale("log"); ax.set_xlabel("Mean-loss alpha"); ax.set_title(LABELS[name]); ax.grid(alpha=.15)
        axes[0].set_ylabel("FIT-only grouped CV normalized MSE"); axes[-1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "fit_reader_selection.png", dpi=180); plt.close(fig)
        files.append("fit_reader_selection.png")
    return files


def conclusion(confirm, bridge, controllers):
    if confirm is None:
        failed = any(v.get("status") == "execution_failed" for v in controllers.values())
        return ("归档记录有执行失败；尚无完整的时间确认报告，科学门槛结论未知。" if failed else
                "尚未提供时间确认报告；目前不能判断新前置门槛是否通过。")
    if not confirm["primary_metric_conditions_met"]:
        return "新时间确认门槛未通过；按登记协议停止其后的桥接/P0链。失败不证明表征完全没有时间信息。"
    if bridge is None:
        return "时间确认报告的主指标通过；尚未提供原分布桥接报告，不能视为完整前置条件通过。"
    if not bridge["primary_metric_conditions_met"]:
        return "时间确认报告通过，但原分布桥接门槛未通过；尚无进入P0的依据。"
    return "时间确认与原分布桥接报告均通过。这支持登记任务中的两帧证据可用；P0与记忆比较仍需各自的训练、门槛和结果。"


def render(run_root, out):
    inputs = Inputs(run_root)
    if not inputs.root.is_dir():
        raise ValueError("Fetched run root must exist")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; existing evidence is never overwritten")
    frozen = inputs.json("reader/freeze_receipt.json", optional=True)
    selections = inputs.json("reader/train_selection.json", optional=True)
    if frozen is not None and selections is not None:
        if frozen.get("files", {}).get("train_selection.json") != sha(inputs.root / "reader/train_selection.json"):
            raise ValueError("FIT selection hash differs from frozen reader receipt")
    confirm = load_report(inputs, "confirm", frozen)
    bridge = load_report(inputs, "bridge", frozen)
    source = inputs.json("source_manifest.json", optional=True)
    budget = inputs.json("ledger/budget.json", optional=True)
    publication = inputs.json("freeze_publication.json", optional=True)
    controllers = {p.name: inputs.json(p.name) for p in sorted(inputs.root.glob("*_controller.json"))}
    downstream = inputs.json("p0_run/report_status.json", optional=True)
    source_commits = {d["source_commit"] for d in controllers.values() if d.get("source_commit")}
    if source and source.get("commit"):
        source_commits.add(source["commit"])
    if frozen and frozen.get("cache_identity", {}).get("source_commit"):
        source_commits.add(frozen["cache_identity"]["source_commit"])
    if len(source_commits) > 1:
        raise ValueError("Source commit declarations disagree")
    jobs = collect_jobs(inputs, controllers)
    sentence = conclusion(confirm, bridge, controllers)
    summary = dict(schema="v5.1-posthoc-report-1", conclusion_zh=sentence,
        scientific_gate_mutated=False, p0_authorized_by_this_renderer=False,
        original_v5_gate="failed_unchanged", source_commit=next(iter(source_commits), None),
        confirmation_present=confirm is not None, bridge_present=bridge is not None,
        confirmation_report_passed=confirm["primary_metric_conditions_met"] if confirm else None,
        bridge_report_passed=bridge["primary_metric_conditions_met"] if bridge else None,
        controllers={k:{field:v.get(field) for field in ("action", "status", "failed_role", "reason")} for k,v in controllers.items()},
        cumulative_charged_gpuh=budget.get("charged_gpuh") if budget else None,
        stage_charged_gpuh=budget.get("stages") if budget else None,
        active_budget_leases=[v.get("lease_id") for v in budget.get("leases", []) if v.get("status") == "reserved_or_interrupted"] if budget else [],
        active_controller_envelopes=[v.get("stage") for v in budget.get("controller_envelopes", []) if v.get("status") == "active"] if budget else [],
        freeze_publication_record=publication, downstream_report_status=downstream,
        jobs=jobs, verification_scope="Serialized report/count/flag and prediction/cache-receipt hashes; not a new scientific re-fit or gate authorization")
    out.mkdir(parents=True)
    files = plot_all(out, confirm, bridge, selections)
    for name, rows in (("confirmation_metrics.csv", metric_rows(confirm)), ("bridge_metrics.csv", metric_rows(bridge)),
                       ("train_cv.csv", cv_rows(selections)), ("jobs.csv", jobs)):
        write_csv(out / name, rows)
    lines = ["# V5.1 时间证据确认 / Temporal confirmation", "", sentence, "",
        "原 V5 的池化位移门槛失败保持不变。本报告是独立的事后展示，不更改门槛、不授予P0权限，也不将时间证据确认称为记忆机制有效。", "",
        f"科学源码：`{summary['source_commit'] or '归档未提供'}`。", ""]
    if frozen:
        lines += [f"冻结读出器收据 SHA256：`{sha(inputs.root / 'reader/freeze_receipt.json')}`。", ""]
    lines += ["## 已归档阶段", "", "| 阶段 | 记录状态 |", "|---|---|"]
    for name, data in controllers.items():
        lines.append(f"| {name} | {data.get('status', 'unknown')} |")
    if not controllers:
        lines.append("| 控制器 | 未提供；不能据此推断已完成或失败 |")
    lines += ["", "`running`/队列状态不代表科学完成；缺少报告也不等于科学门槛失败。", ""]
    for role, report, title in (("confirm", confirm, "时间确认"), ("bridge", bridge, "原分布桥接")):
        if not report:
            continue
        lines += ["## "+title, "", "| 读出器 | 位移均值 skill | 相应单侧下界 | 位置 skill | 主检查全满足 |", "|---|---:|---:|---:|---|"]
        for name in INPUTS:
            s = report["scores"][name]
            motion = s["absolute_displacement_skill"] if role == "confirm" else s["motion_skill"]
            lower = s["primary_lower_bounds"][8] if role == "confirm" else s["motion_mean_lower_bound"]
            lines.append(f"| {LABELS[name]} | {motion:.6f} | {lower:.6f} | {s['position_skill']:.6f} | {s['metric_conditions_met']} |")
        failed = [k for k,v in report["scores"]["concat"]["checks"].items() if not v]
        lines += ["", "主读出器是冻结的双帧 concat；单帧结果全部展示，不用于事后更换赢家。", "",
                  "主检查未满足项："+("、".join(failed) if failed else "无（以本报告覆盖的指标检查为限）")+"。", ""]
        if role == "confirm":
            lines += ["确认集为256个完整四-view block。八个逐坐标对比下界与绝对位移均值下界都必须>0，位置skill必须>0.1；零对比是结构性的单帧零假设。单侧bootstrap下界按九项作Bonferroni调整（.05/9），条件于冻结读出器。反对称模型下两类anchor对比会相同，不能称作八项独立证据。", ""]
        else:
            lines += ["桥接为原生成器Dev pair128–383、每对保留两个variant。下界按完整world重采样，为单侧95%；另要求四个位移坐标点skill全部>0。它检验回到原分布的迁移，不独立证明时间利用。", ""]
    if selections:
        lines += ["## FIT内选择（无确认集选参）", "", "| 输入 | 选中kernel | mean-loss alpha | FIT CV normalized MSE |", "|---|---|---:|---:|"]
        for name in INPUTS:
            selected = selections[name]["selected"]
            lines.append(f"| {LABELS[name]} | {selected['kernel']} | {selected['mean_loss_alpha']:.9g} | {selected['mean_loss']:.6f} |")
        lines += ["", "全部候选与四fold分数见 `train_cv.csv`；惩罚形式为 `K+n_fit*alpha*I`。", ""]
    lines += ["## 作业与累计预算", "", "| 阶段 | 实际Slurm step | 父allocation | 选中GPU数 | 状态 |", "|---|---|---|---:|---|"]
    for job in jobs:
        lines.append(f"| {job['stage']} | {job['slurm_step'] or '未提供step证据'} | {job['allocation_job']} | {job['selected_gpus']} | {job['status']} |")
    if not jobs:
        lines.append("| 未提供 | 无 | 无 | — | 不推断 |")
    if budget:
        lines += ["", f"归档总账累计收费 **{budget['charged_gpuh']:.6f} GPUh / 60 GPUh**，包含继承的旧轮次记录；以总账是否含未闭合lease区分保留额度与已结算用量。", "",
                  "未闭合lease："+("、".join(summary["active_budget_leases"]) if summary["active_budget_leases"] else "无（按已归档总账）")+"。",
                  "尚在活动的控制器预算包络："+("、".join(summary["active_controller_envelopes"]) if summary["active_controller_envelopes"] else "无（按已归档总账）")+"；存在时不能视为最终结算。", ""]
    else:
        lines += ["", "未提供累计预算总账；不推算剩余额度。", ""]
    if downstream:
        lines += ["下游报告记录状态：`"+str(downstream.get("status"))+"`。训练完成本身不等于某一记忆方案优于对照；需另读所有架构的完整比较。", ""]
    lines += ["## 图表与复现", ""]
    for filename in files:
        lines += [f"![{filename}]({filename})", ""]
    lines += ["```bash", "python3 scripts/report_v5_1_confirmation.py --run-root FETCHED_RUN_ROOT --out NEW_REPORT_DIR", "```", "",
              "输入文件与SHA256见 `input_inventory.json`，机器摘要见 `summary.json`。脚本校验展示所用报告/缓存收据/预测文件的关联哈希、数量和状态一致性；它不代替独立的门槛复算、完整归档验证或GitHub回读。", "",
              "Test与完整特征缓存不被本脚本打开。原失败现场及本轮清理状态应以执行/归档收据单独确认。", ""]
    (out / "README.md").write_text("\n".join(lines))
    dump(out / "summary.json", summary)
    dump(out / "input_inventory.json", inputs.inventory)
    dump(out / "artifact_inventory.json", {p.name:dict(bytes=p.stat().st_size, sha256=sha(p))
                                            for p in sorted(out.iterdir()) if p.is_file()})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = render(args.run_root, args.out)
    print(json.dumps({k:summary[k] for k in ("conclusion_zh", "confirmation_report_passed", "bridge_report_passed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
