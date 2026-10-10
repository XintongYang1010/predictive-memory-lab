"""Process-level tests for parallel stage reservations and refunds."""
import json
import multiprocessing
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from pmr.visual_memory.common import Budget, BudgetExceeded


def reserve_worker(root, stage, cap, hours, ready, release, results, used=.125):
    ready.wait(30)
    try:
        budget = Budget(root, stage, cap, hours)
    except (BudgetExceeded, BlockingIOError) as error:
        results.put(("rejected", stage, type(error).__name__))
        return
    results.put(("reserved", stage, budget.lease["lease_id"]))
    if not release.wait(30):
        raise RuntimeError("Test did not release worker")
    budget.elapsed = lambda: used
    budget.close("complete")
    results.put(("closed", stage, budget.lease["charged_gpuh"]))


def interrupted_worker(root, results):
    Budget(root, "interrupted", 4, 4)
    results.put("reserved")
    while True:
        signal.pause()


class ParallelBudget(unittest.TestCase):
    def setUp(self):
        self.context = multiprocessing.get_context("spawn")
        self.processes = []
        self.addCleanup(self.stop_workers)

    def stop_workers(self):
        for process in self.processes:
            if process.is_alive():
                process.kill()
            process.join(10)

    def start_workers(self, root, requests):
        ready, release = self.context.Event(), self.context.Event()
        results = self.context.Queue()
        for stage, cap, hours in requests:
            process = self.context.Process(target=reserve_worker,
                args=(root, stage, cap, hours, ready, release, results))
            process.start()
            self.processes.append(process)
        ready.set()
        reservations = [results.get(timeout=30) for _ in requests]
        return release, results, reservations

    def finish_workers(self, release, results, count):
        release.set()
        closed = [results.get(timeout=30) for _ in range(count)]
        self.assertTrue(all(result[0] == "closed" for result in closed), closed)
        for process in self.processes:
            process.join(30)
            self.assertEqual(process.exitcode, 0)

    def test_eight_distinct_stages_reserve_and_refund_without_lost_updates(self):
        with tempfile.TemporaryDirectory() as root:
            release, results, reservations = self.start_workers(root,
                [(f"candidate_{i}", 4, .5) for i in range(8)])
            self.assertTrue(all(row[0] == "reserved" for row in reservations), reservations)
            self.assertEqual(len({row[2] for row in reservations}), 8)
            data = json.loads((Path(root)/"budget.json").read_text())
            self.assertEqual(data["charged_gpuh"], 4)
            self.assertEqual(len(data["leases"]), 8)
            self.finish_workers(release, results, 8)
            data = json.loads((Path(root)/"budget.json").read_text())
            self.assertEqual(data["charged_gpuh"], 1)
            self.assertEqual(sum(data["stages"].values()), 1)
            self.assertEqual(sum(row["charged_gpuh"] for row in data["leases"]), 1)
            self.assertTrue(all(row["status"] == "complete" for row in data["leases"]))

    def test_competing_reservations_cannot_cross_round_cap(self):
        with tempfile.TemporaryDirectory() as root:
            # A legacy ledger without lease IDs remains readable and charged.
            (Path(root)/"budget.json").write_text(json.dumps(dict(charged_gpuh=52.,
                stages={"previous": 52.}, leases=[dict(stage="previous",
                charged_gpuh=52., status="complete")])))
            release, results, reservations = self.start_workers(root,
                [(f"candidate_{i}", 4, 4) for i in range(4)])
            self.assertEqual(sum(row[0] == "reserved" for row in reservations), 2)
            self.assertEqual(sum(row[0] == "rejected" and row[2] == "BudgetExceeded"
                                 for row in reservations), 2)
            data = json.loads((Path(root)/"budget.json").read_text())
            self.assertEqual(data["charged_gpuh"], 60)
            self.finish_workers(release, results, 2)
            data = json.loads((Path(root)/"budget.json").read_text())
            self.assertEqual(data["charged_gpuh"], 52.25)
            self.assertEqual(data["stages"]["previous"], 52)
            self.assertEqual(data["leases"][0]["charged_gpuh"], 52)

    def test_same_stage_has_one_active_owner(self):
        with tempfile.TemporaryDirectory() as root:
            release, results, reservations = self.start_workers(root,
                [("same_stage", 4, .5), ("same_stage", 4, .5)])
            self.assertEqual(sum(row[0] == "reserved" for row in reservations), 1)
            self.assertEqual(sum(row[0] == "rejected" and row[2] == "BlockingIOError"
                                 for row in reservations), 1)
            self.finish_workers(release, results, 1)
            # Closing releases the stage lock; previous actual use still counts.
            with self.assertRaises(BudgetExceeded):
                Budget(root, "same_stage", .5, .5)
            resumed = Budget(root, "same_stage", .5, .25)
            resumed.close("complete")

    def test_hard_kill_keeps_full_reservation_and_releases_stage_lock(self):
        with tempfile.TemporaryDirectory() as root:
            results = self.context.Queue()
            process = self.context.Process(target=interrupted_worker,args=(root, results))
            process.start(); self.processes.append(process)
            self.assertEqual(results.get(timeout=30), "reserved")
            process.kill(); process.join(10)
            data = json.loads((Path(root)/"budget.json").read_text())
            self.assertEqual(data["charged_gpuh"], 4)
            self.assertEqual(data["leases"][0]["status"], "reserved_or_interrupted")
            with self.assertRaises(BudgetExceeded):
                Budget(root, "interrupted", 4, .1)
            other = Budget(root, "other", 4, .5)
            other.elapsed = lambda: .125
            other.close("complete")
            data = json.loads((Path(root)/"budget.json").read_text())
            self.assertEqual(data["charged_gpuh"], 4.125)
            self.assertEqual(data["stages"]["interrupted"], 4)

    def test_eight_gpu_lease_accounts_wall_time_and_signal(self):
        with tempfile.TemporaryDirectory() as root:
            original_handler = signal.getsignal(signal.SIGUSR1)
            with patch("pmr.visual_memory.common.time.monotonic", return_value=10000):
                budget = Budget(root, "features", 8, 1, gpu_count=8)
            self.assertEqual(budget.charge, 8)
            self.assertEqual(budget.lease["gpu_count"], 8)
            with patch("pmr.visual_memory.common.time.monotonic", return_value=10450):
                self.assertEqual(budget.elapsed(), 1)
                budget.check()
                os.kill(os.getpid(), signal.SIGUSR1)
                with self.assertRaises(BudgetExceeded):
                    budget.check()
                budget.close("partial")
                budget.close("complete")  # A repeated close cannot refund twice.
            self.assertEqual(signal.getsignal(signal.SIGUSR1), original_handler)
            data = json.loads((Path(root)/"budget.json").read_text())
            self.assertEqual(data["charged_gpuh"], 1)
            self.assertEqual(data["stages"]["features"], 1)
            self.assertEqual(data["leases"][0]["status"], "partial")


if __name__ == "__main__":
    unittest.main()
