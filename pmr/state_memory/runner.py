"""Train and compare causal memory mechanisms on paired occluded trajectories."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from . import SCHEMA_VERSION
from .data import TaskConfig, make_batch, corrupt_history
from .metrics import evaluate_predictions, summarize, bootstrap_paired
from .models import make_model


class BudgetExceeded(RuntimeError):
    pass


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExceeded('Run wall-time budget reached; partial outputs retained.')


def forecast_loss(prediction, targets, mask, horizons=(0, 1, 4)):
    h = prediction.new_tensor(horizons).view(1, 1, 1, -1, 1)
    forecast = prediction[..., None, :2] + prediction[..., None, 2:] * h
    valid = mask[..., None].expand_as(forecast)
    if not bool(valid.any()):
        raise ValueError('No valid supervised forecasts in batch')
    return (forecast - targets).square()[valid].mean()


def predict(model, observations, device, batch_size, deadline):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for begin in range(0, len(observations), batch_size):
            check_budget(deadline)
            x = torch.as_tensor(observations[begin:begin + batch_size], device=device)
            y = model(x)
            if not bool(torch.isfinite(y).all()):
                raise FloatingPointError('Non-finite state prediction')
            outputs.append(y.cpu().numpy())
    return np.concatenate(outputs)


def numpy_loss(prediction, batch, horizons):
    h = np.asarray(horizons)[None, None, None, :, None]
    f = prediction[..., None, :2] + prediction[..., None, 2:] * h
    return float(np.mean(((f - batch['targets']) ** 2)[batch['target_mask']]))


def fingerprint():
    root = Path(__file__).resolve().parents[2]
    result = {}
    for folder in ('pmr', 'scripts', 'tests'):
        for path in sorted((root / folder).rglob('*')):
            if path.suffix in ('.py', '.sh'):
                result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def git_info():
    def git(*arguments):
        try:
            return subprocess.check_output(['git', *arguments], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {'commit': git('rev-parse', 'HEAD'), 'working_tree_status': git('status', '--porcelain')}


def batch_manifest(batch, config):
    arrays = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in batch.items()
              if isinstance(v, np.ndarray)}
    return {'config': asdict(config), 'episode_ids': list(batch['episode_id']),
            'array_sha256': arrays, 'generator_manifest': batch.get('manifest', {})}


def train_model(model, train, dev, args, seed, model_name, out, deadline):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        return {'selected_step': 0, 'selection': 'parameter-free causal analytic baseline',
                'seconds': 0., 'training_steps': 0}
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    random = np.random.default_rng(seed)
    tensors = {k: torch.as_tensor(train[k], device=args.device)
               for k in ('observation', 'targets', 'target_mask')}
    best_loss = float('inf')
    selected_step = 0
    best_state = None
    start = time.monotonic()
    checkpoint_path = out / 'checkpoints' / f'{model_name}_seed{seed}.pt'
    checkpoint_path.parent.mkdir(exist_ok=True)
    interval = max(1, args.steps // 10)
    with (out / 'training.jsonl').open('a') as log:
        for step in range(1, args.steps + 1):
            check_budget(deadline)
            model.train()
            indices = torch.as_tensor(random.integers(len(train['observation']), size=args.batch_size), device=args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tensors['observation'][indices])
            loss = forecast_loss(prediction, tensors['targets'][indices], tensors['target_mask'][indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % interval == 0 or step == args.steps:
                dev_prediction = predict(model, dev['observation'], args.device, args.batch_size, deadline)
                dev_loss = numpy_loss(dev_prediction, dev, (0, 1, 4))
                record = {'model': model_name, 'seed': seed, 'step': step,
                          'train_batch_mse': float(loss.detach()), 'dev_future_mse': dev_loss,
                          'gradient_norm_before_clip': float(gradient), 'seconds': time.monotonic() - start}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                log.flush()
                if dev_loss < best_loss:
                    best_loss, selected_step = dev_loss, step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save({'state_dict': best_state, 'model': model_name, 'seed': seed,
                                'args': vars(args), 'selected_step': step, 'dev_future_mse': dev_loss,
                                'selection_split': 'dev', 'schema': SCHEMA_VERSION}, checkpoint_path)
                print(json.dumps(record), flush=True)
    model.load_state_dict(best_state)
    return {'selected_step': selected_step, 'dev_future_mse': best_loss,
            'selection': 'minimum clean Dev multi-horizon MSE; no Eval selection or Keep fallback',
            'seconds': time.monotonic() - start, 'training_steps': args.steps}


def save_bad_cases(out, model_name, seed, length, batch, prediction, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = out / 'bad_cases'
    directory.mkdir(exist_ok=True)
    worst = sorted(range(len(rows)), key=lambda i: rows[i]['future_mse'], reverse=True)[:3]
    cases = []
    for rank, i in enumerate(worst):
        changed, preserved = int(batch['changed_object'][i]), int(batch['preserved_object'][i])
        reveal = int(batch['reveal_step'][i])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, obj, title in zip(axes, (changed, preserved), ('Corrected object', 'Unobserved object')):
            truth = batch['state'][i, :, obj, :2]
            pred = prediction[i, :, obj, :2]
            axis.plot(truth[:, 0], truth[:, 1], '-o', ms=2, label='true state')
            axis.plot(pred[:, 0], pred[:, 1], '--', label='predicted state')
            visible = batch['observation'][i, :, obj, 4] > .5
            measured = batch['observation'][i, visible, obj, :2]
            axis.scatter(measured[:, 0], measured[:, 1], marker='x', label='visible evidence')
            axis.scatter(*truth[reveal], marker='*', s=80, label='reveal time')
            axis.set_title(f'{title} (ID {obj})')
            axis.set_aspect('equal', adjustable='datalim')
            axis.legend(fontsize=7)
            axis.set_xlabel('x (synthetic units)')
            axis.set_ylabel('y (synthetic units)')
        name = f'{model_name}_seed{seed}_T{length}_worst{rank}.png'
        fig.suptitle(f'{model_name}: seed {seed}, T={length}, episode {i}')
        fig.tight_layout()
        fig.savefig(directory / name, dpi=120)
        plt.close(fig)
        cases.append({'rank': rank, 'episode_id': batch['episode_id'][i], 'metrics': rows[i],
                      'plot': name, 'selection': 'largest clean Eval future MSE; descriptive only'})
    write_json(directory / f'{model_name}_seed{seed}_T{length}.json', cases)


def save_training_curves(out):
    path = out / 'training.jsonl'
    if not path.exists():
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, seed in sorted({(r['model'], r['seed']) for r in records}):
        rows = [r for r in records if (r['model'], r['seed']) == (name, seed)]
        for axis, key in zip(axes, ('train_batch_mse', 'dev_future_mse')):
            axis.plot([r['step'] for r in rows], [r[key] for r in rows], label=f'{name}/{seed}')
            axis.set_xlabel('Optimizer step')
            axis.set_ylabel(key + ' (h=0,1,4)')
            axis.set_yscale('log')
    axes[-1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'training_curves.png', dpi=120)
    plt.close(fig)


def paired_comparisons(rows_by_key):
    comparisons = []
    for (seed, name, length, condition), rows in rows_by_key.items():
        if condition != 'clean':
            continue
        metrics_by_reference = {'window': ('future_mse',)}
        if name == 'slot':
            metrics_by_reference['gru'] = ('hidden_position_rmse', 'changed_post_position_rmse',
                'preserved_post_position_rmse', 'history_pair_difference_rmse')
        for reference, metrics in metrics_by_reference.items():
            reference_rows = rows_by_key.get((seed, reference, length, condition))
            if reference_rows is None:
                continue
            for metric in metrics:
                comparisons.append({'reference': reference, 'model': name, 'seed': seed,
                    'length': length, 'condition': condition,
                    'primary': reference == 'gru' and length == 48 and metric == 'hidden_position_rmse',
                    **bootstrap_paired(reference_rows, rows, metric=metric)})
    return comparisons


def report(out, summaries, comparisons, args, status):
    lines = ['# V4 State Memory Mechanism Experiment', '',
             f'Status: **{status}**. Smoke only: **{args.smoke}**.', '',
             'Supervised synthetic 2D motion with stable object IDs, measured velocities and occlusion. '
             'Not an AV2 result, visual world model, JEPA reproduction, or novelty claim.', '',
             'Learned arms see identical observations and train on multi-horizon position loss. '
             'Only clean Dev selects checkpoints. History corruption is an evaluation stress test. '
             'Seeds share data; uncertainty intervals are per-seed paired episode bootstraps, not independent seed evidence. '
             'Future MSE includes reconstruction at h=0 alongside h=1,4. Hidden RMSE uses the same natural '
             'occlusion mask across history-corruption conditions, excluding the deleted initial frames.', '',
             '| Model | Seed | Length | MSE (h=0,1,4) | Hidden RMSE | Changed RMSE | Preserved RMSE | Never recovered |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in summaries:
        if item['condition'] == 'clean':
            m = item['metrics']
            values = [m.get(k) for k in ('future_mse', 'hidden_position_rmse',
                'changed_post_position_rmse', 'preserved_post_position_rmse', 'never_recovered_rate')]
            cells = ' | '.join('NA' if value is None else f'{value:.6f}' for value in values)
            lines.append(f"| {item['model']} | {item['seed']} | {item['length']} | {cells} |")
    lines += ['', 'Primary mechanism comparison: clean T=48, GRU minus slot hidden-position RMSE. '
              'Positive gain favors slot. Each interval resamples paired episodes within one model seed.', '',
              '| Seed | Gain | 95% CI | Paired episodes |', '|---:|---:|---|---:|']
    for item in comparisons:
        if item['primary']:
            lines.append(f"| {item['seed']} | {item['mean_gain']:.6f} | {item['ci95']} | {item['paired_episodes']} |")
    lines += ['', 'See `summary.json` for all conditions, recovery and preservation metrics, '
              '`episode_metrics.jsonl` for exact paired records, `training_curves.png` and `training.jsonl` for training/Dev curves, '
              '`trajectories/*.npz` for observations/targets/predictions, and `bad_cases/` for failures.', '',
              'Interpretation: the analytic tracker uses the known constant-velocity law and is a causal '
              'solvability control. A low collateral-change score must be read alongside state accuracy; '
              'a model ignoring all memory could otherwise appear stable. Object slots have an identity '
              'and conditional-independence prior. Report parameter counts and memory scalars; these '
              'arms are not parameter- or FLOP-matched. Larger T also changes the state range, so length '
              'generalization alone does not isolate retention from extrapolation.', '',
              'Do not select architectures or thresholds on these Eval results and reuse them as '
              'confirmation. Use fresh data namespaces for a subsequent confirmatory run.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    with (out / 'metrics.csv').open('w', newline='') as f:
        metric_names = sorted({k for item in summaries for k, v in item['metrics'].items() if not isinstance(v, dict)})
        writer = csv.DictWriter(f, fieldnames=['model', 'seed', 'length', 'condition', *metric_names])
        writer.writeheader()
        for item in summaries:
            writer.writerow({**{k: item[k] for k in ('model', 'seed', 'length', 'condition')},
                             **{k: item['metrics'].get(k) for k in metric_names}})
    write_json(out / 'summary.json', {'schema': SCHEMA_VERSION, 'status': status, 'smoke_only': args.smoke,
               'results': summaries, 'paired_comparisons': comparisons,
               'paired_vs_window': [c for c in comparisons if c['reference'] == 'window'],
               'scope': 'synthetic supervised mechanism experiment; no real-world or visual claims'})


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    deadline = start + args.max_seconds
    summaries, comparisons, completed = [], [], []
    stats = {'status': 'running', 'smoke_only': args.smoke, 'schema': SCHEMA_VERSION,
             'completed_models': completed, 'expected_models': len(args.seeds) * len(args.models)}
    write_json(out / 'run_stats.json', stats)
    write_json(out / 'settings.json', vars(args))
    try:
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no automatic CPU fallback')
        torch.set_num_threads(args.cpu_threads)
        torch.use_deterministic_algorithms(True)
        write_json(out / 'environment.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
                   'numpy': np.__version__, 'cuda_runtime': torch.version.cuda, 'device': args.device,
                   'gpu': torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
                   'git': git_info()})
        write_json(out / 'source_manifest.json', fingerprint())
        config = TaskConfig(length=args.train_length, reveal_step=args.train_length // 2)
        namespace = 'smoke_' if args.smoke else ''
        train = make_batch(config, args.train_episodes, args.data_seed, split=namespace + 'train')
        dev = make_batch(config, args.dev_episodes, args.data_seed, split=namespace + 'dev')
        evaluation = {}
        manifests = {'train': batch_manifest(train, config), 'dev': batch_manifest(dev, config)}
        for length in args.eval_lengths:
            cfg = TaskConfig(length=length, reveal_step=length // 2)
            evaluation[length] = make_batch(cfg, args.eval_episodes, args.data_seed, split=namespace + f'eval_T{length}')
            manifests[f'eval_T{length}'] = batch_manifest(evaluation[length], cfg)
        all_ids = [identifier for value in manifests.values() for identifier in value['episode_ids']]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError('Overlapping dataset episode identities')
        write_json(out / 'datasets_manifest.json', manifests)
        (out / 'trajectories').mkdir()
        for length, batch in evaluation.items():
            np.savez_compressed(out / 'trajectories' / f'data_T{length}.npz',
                                **{k: v for k, v in batch.items() if isinstance(v, np.ndarray)})
        rows_by_key = {}
        for seed in args.seeds:
            for name in args.models:
                check_budget(deadline)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                model = make_model(name, n_objects=config.n_objects, hidden_dim=args.hidden_dim,
                                   window=args.window, memory_slots=args.memory_slots).to(args.device)
                selected = train_model(model, train, dev, args, seed, name, out, deadline)
                metadata = model.metadata() if hasattr(model, 'metadata') else {}
                metadata['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                for length, batch in evaluation.items():
                    observations = batch['observation']
                    begin = time.monotonic()
                    clean = predict(model, observations, args.device, args.batch_size, deadline)
                    seconds = time.monotonic() - begin
                    paired = predict(model, batch['paired_observation'], args.device, args.batch_size, deadline)
                    history = predict(model, batch['history_pair_observation'], args.device, args.batch_size, deadline)
                    outputs = {'clean': clean, 'no_change_pair': paired, 'history_pair': history}
                    for condition in ('delete', 'shuffle', 'stale'):
                        outputs[condition] = predict(model, corrupt_history(observations, condition, args.data_seed),
                                                     args.device, args.batch_size, deadline)
                    np.savez_compressed(out / 'trajectories' / f'{name}_seed{seed}_T{length}.npz', **outputs)
                    for condition, predictions in outputs.items():
                        scoring = batch
                        if condition == 'no_change_pair':
                            scoring = {**batch, 'observation': batch['paired_observation'],
                                       'state': batch['paired_state'], 'targets': batch['paired_targets'],
                                       'target_mask': batch.get('paired_target_mask', batch['target_mask'])}
                        elif condition == 'history_pair':
                            scoring = {**batch, 'observation': batch['history_pair_observation'],
                                       'state': batch['history_pair_state'], 'targets': batch['history_pair_targets'],
                                       'target_mask': batch['history_pair_target_mask']}
                        rows = evaluate_predictions(predictions, scoring,
                                  recovery_threshold=args.recovery_threshold,
                                  paired_prediction=paired if condition == 'clean' else None,
                                  history_prediction=history if condition == 'clean' else None)
                        rows_by_key[(seed, name, length, condition)] = rows
                        item = {'model': name, 'seed': seed, 'length': length, 'condition': condition,
                                'metrics': summarize(rows), 'model_metadata': metadata, 'training': selected}
                        if condition == 'clean':
                            item['inference_seconds_per_episode'] = seconds / len(observations)
                            save_bad_cases(out, name, seed, length, batch, predictions, rows)
                        summaries.append(item)
                        with (out / 'episode_metrics.jsonl').open('a') as f:
                            for row in rows:
                                f.write(json.dumps({'model': name, 'seed': seed, 'length': length,
                                                   'condition': condition, **row}, allow_nan=False) + '\n')
                completed.append({'model': name, 'seed': seed})
                stats['seconds'] = time.monotonic() - start
                write_json(out / 'run_stats.json', stats)
                report(out, summaries, comparisons, args, 'running')
        check_budget(deadline)
        comparisons = paired_comparisons(rows_by_key)
        save_training_curves(out)
        stats['status'] = 'complete'
        stats['peak_cuda_bytes'] = torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None
        report(out, summaries, comparisons, args, 'complete')
    except Exception as error:
        stats['status'] = 'budget_exceeded' if isinstance(error, BudgetExceeded) else 'failed'
        stats['error'] = f'{type(error).__name__}: {error}'
        report(out, summaries, comparisons, args, stats['status'])
        raise
    finally:
        stats['seconds'] = time.monotonic() - start
        write_json(out / 'run_stats.json', stats)
    print(f'STATE_MEMORY_V4_COMPLETE: {out}', flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--train-episodes', type=int, default=4096)
    parser.add_argument('--dev-episodes', type=int, default=512)
    parser.add_argument('--eval-episodes', type=int, default=512)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--data-seed', type=int, default=20261010)
    parser.add_argument('--train-length', type=int, default=24)
    parser.add_argument('--eval-lengths', default='24,48,96')
    parser.add_argument('--models', default='window,retrieval,gru,slot,analytic')
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--window', type=int, default=4)
    parser.add_argument('--memory-slots', type=int, default=8)
    parser.add_argument('--max-seconds', type=float, default=3600.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--recovery-threshold', type=float, default=.15)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    try:
        args.seeds = [int(x) for x in args.seeds.split(',')]
        args.eval_lengths = [int(x) for x in args.eval_lengths.split(',')]
        args.models = args.models.split(',')
        for name in ('seeds', 'eval_lengths', 'models'):
            values = getattr(args, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f'{name} must be a nonempty unique list')
        if not set(args.models) <= {'window', 'retrieval', 'gru', 'slot', 'analytic'}:
            raise ValueError('Unknown model name')
        for name in ('steps', 'train_episodes', 'dev_episodes', 'eval_episodes', 'hidden_dim',
                     'batch_size', 'window', 'memory_slots', 'max_seconds', 'lr', 'cpu_threads', 'recovery_threshold'):
            if getattr(args, name) <= 0 or not np.isfinite(getattr(args, name)):
                raise ValueError(f'{name} must be positive and finite')
        if min([args.train_length, *args.eval_lengths]) < max(12, args.window + 3):
            raise ValueError('Lengths must support initial cues, reveal and a hidden final window')
        torch.device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    try:
        run(parse_args(argv))
    except BudgetExceeded as error:
        raise SystemExit(str(error)) from error
