import unittest
from unittest.mock import Mock
from pmr.visual_memory.profiling import measure_encoder


class Profiling(unittest.TestCase):
    def test_cold_start_is_separate_from_representative_throughput(self):
        events = []
        encoder = lambda batch: events.append(("encode", len(batch)))
        reset = lambda: events.append(("reset", 0))
        episodes = [(writes, repeat, [0] * (16 + writes)) for writes in (0, 8, 32) for repeat in range(3)]
        # Cold kernels take 100s, each warmed representative episode takes 2s.
        ticks = iter([0, 100] + [x for i in range(9) for x in (100 + i * 2, 102 + i * 2)])
        budget = Mock()
        result = measure_encoder(encoder, budget, [0] * 32, episodes, 32,
                                 lambda: None, reset, lambda: 1234, lambda: next(ticks))
        self.assertEqual(result["cold_start_seconds"], 100)
        self.assertEqual(result["seconds"], 18)
        self.assertEqual(result["frames"], 3 * (16 + 24 + 48))
        self.assertEqual(result["seconds_per_frame"], 18 / 264)
        self.assertEqual(result["warmup_frames"], 32)
        self.assertEqual(result["measurement_repeats_per_write_count"], 3)
        self.assertEqual(result["max_gpu_memory_bytes"], 1234)
        self.assertEqual(events[:2], [("encode", 32), ("reset", 0)])
        self.assertEqual(budget.check.call_count, 13)

    def test_budget_stops_profile_before_another_encoder_call(self):
        encoder = Mock()
        budget = Mock()
        budget.check.side_effect = [None, RuntimeError("lease exhausted")]
        ticks = iter([0, 1, 1])
        with self.assertRaisesRegex(RuntimeError, "lease exhausted"):
            measure_encoder(encoder, budget, [0], [(0, 0, [0])], 1,
                            lambda: None, lambda: None, lambda: 0, lambda: next(ticks))
        self.assertEqual(encoder.call_count, 1)


if __name__ == "__main__":
    unittest.main()
