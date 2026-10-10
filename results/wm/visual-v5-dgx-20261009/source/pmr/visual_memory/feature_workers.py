"""Persistent spawn workers: one frozen encoder per GPU, disjoint paired records."""
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import time
import traceback

from .common import BudgetExceeded, atomic_json
from .data import specs
from .features import Encoder, begin_cache, cache_records, finish_cache


def paired_shards(split, count, workers):
    if workers < 1:
        raise ValueError("Feature worker count must be positive")
    return [[spec for spec in specs(split, count) if spec[1] % workers == index]
            for index in range(workers)]


class WorkerBudget:
    def __init__(self, deadline, cancelled):
        self.deadline, self.cancelled = deadline, cancelled

    def check(self):
        if self.cancelled.is_set() or time.monotonic() >= self.deadline:
            raise BudgetExceeded("Feature worker cancelled or stage wall lease nearly exhausted")


def _worker(index, code, checkpoint, commands, results, cancelled, deadline, logs, expected_identity):
    import torch
    torch.set_num_threads(1)
    budget = WorkerBudget(deadline, cancelled)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda *_: cancelled.set())
    logs = Path(logs); logs.mkdir(parents=True, exist_ok=True)
    key = f"feature_worker_{index}_{os.getpid()}"
    receipt = dict(worker=index, pid=os.getpid(), status="starting", start_unix=time.time(), tasks=[])
    atomic_json(logs / f"{key}.json", receipt)
    with open(logs / f"{key}.log", "a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            budget.check()
            torch.cuda.set_device(index)
            encoder = Encoder(code, checkpoint, device=f"cuda:{index}")
            if encoder.identity != expected_identity:
                raise ValueError("Worker encoder differs from the passed official profile")
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            receipt.update(status="ready", encoder=encoder.identity,
                           gpu=torch.cuda.get_device_name(index),
                           visible_device=visible[index] if index < len(visible) else None,
                           job_id=os.environ.get("SLURM_JOB_ID"), step_id=os.environ.get("SLURM_STEP_ID"))
            atomic_json(logs / f"{key}.json", receipt)
            results.put(dict(event="ready", worker=index, identity=encoder.identity))
            while True:
                budget.check()
                try:
                    task = commands.get(timeout=.5)
                except queue.Empty:
                    continue
                if task is None:
                    break
                receipt.update(status="running", current_task=task["task_id"])
                atomic_json(logs / f"{key}.json", receipt)
                cache_records(encoder, task["folder"], task["specs"], budget, task["batch_frames"])
                done = dict(task_id=task["task_id"], folder=task["folder"], records=len(task["specs"]), status="complete")
                receipt["tasks"].append(done)
                receipt.update(status="ready", current_task=None)
                atomic_json(logs / f"{key}.json", receipt)
                results.put(dict(event="done", worker=index, **done))
            receipt["status"] = "complete"
        except BaseException as exc:
            receipt.update(status="partial", reason=str(exc))
            traceback.print_exc()
            results.put(dict(event="error", worker=index, reason=str(exc)))
        finally:
            receipt["end_unix"] = time.time()
            atomic_json(logs / f"{key}.json", receipt)


class FeatureWorkers:
    """Parent holds the only stage lease and is the only cache manifest writer."""
    def __init__(self, code, checkpoint, workers, budget, logs, encoder_identity):
        self.budget, self.identity = budget, encoder_identity
        self.processes, self.commands = [], []
        self.next_task = 0
        context = mp.get_context("spawn")
        self.cancelled, self.results = context.Event(), context.Queue()
        # Match Budget.check's one-minute checkpoint margin; monotonic clock is
        # shared across local processes. The parent still checks its real lease.
        deadline = budget.start + budget.charge / budget.gpus * 3600 - 60
        try:
            for index in range(workers):
                commands = context.Queue()
                process = context.Process(target=_worker,
                    args=(index, str(code), str(checkpoint), commands, self.results,
                          self.cancelled, deadline, str(logs), self.identity), daemon=True)
                self.commands.append(commands); self.processes.append(process)
                process.start()
                # Sequential loading bounds transient host memory for the 9GB
                # official checkpoint. Encoding work runs on all GPUs together.
                self._wait("ready", {index})
        except BaseException:
            self.close()
            raise

    def _wait(self, event, pending, task_id=None):
        pending = set(pending)
        while pending:
            self.budget.check()
            try:
                message = self.results.get(timeout=.25)
            except queue.Empty:
                failed = [p.pid for p in self.processes if p.pid is not None and not p.is_alive()]
                if failed:
                    raise RuntimeError(f"Feature workers exited before completion: {failed}")
                continue
            if message["event"] == "error":
                raise RuntimeError(f"Feature worker {message['worker']} failed: {message['reason']}")
            if (message["event"] != event or message["worker"] not in pending
                    or (task_id is not None and message.get("task_id") != task_id)):
                raise RuntimeError(f"Unexpected feature worker receipt: {message}")
            pending.remove(message["worker"])

    def cache_split(self, root, split, count, batch_frames=32):
        self.budget.check()
        folder = begin_cache(root, split, count, self.identity)
        task_id = self.next_task; self.next_task += 1
        for commands, shard in zip(self.commands, paired_shards(split, count, len(self.processes))):
            commands.put(dict(task_id=task_id, folder=str(folder), specs=shard, batch_frames=batch_frames))
        self._wait("done", range(len(self.processes)), task_id)
        finish_cache(root, split, count, self.identity, self.budget)

    def close(self):
        # A clean shutdown lets receipts finish. Errors/signals request a stop;
        # bounded joins followed by kill prevent a wedged CUDA worker hanging us.
        for commands in self.commands:
            commands.put(None)
        deadline = time.monotonic() + 5
        for process in self.processes:
            if process.pid is not None:
                process.join(timeout=max(0, deadline-time.monotonic()))
        self.cancelled.set()
        for process in self.processes:
            if process.pid is not None and process.is_alive():
                process.kill()
                process.join(timeout=2)
        for commands in self.commands:
            commands.close()
        self.results.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback_value):
        if kind is not None:
            self.cancelled.set()
        self.close()
