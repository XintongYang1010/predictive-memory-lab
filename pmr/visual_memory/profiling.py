"""Bounded encoder timing with cold initialization kept outside throughput."""
import time


def measure_encoder(encoder, budget, warmup_rgb, measurement_episodes, frame_batch,
                    synchronize, reset_peak_memory, peak_memory, clock=time.monotonic):
    if frame_batch < 1 or not len(warmup_rgb) or not measurement_episodes:
        raise ValueError("Profiling needs positive batching and nonempty RGB inputs")
    budget.check()
    start = clock()
    encoder(warmup_rgb)
    synchronize()
    cold_seconds = clock() - start
    reset_peak_memory()
    measurements = []
    for writes, repeat, rgb in measurement_episodes:
        # RGB construction is outside this timer, as in the original profile.
        synchronize()
        start = clock()
        frames = 0
        for offset in range(0, len(rgb), frame_batch):
            budget.check()
            batch = rgb[offset:offset + frame_batch]
            encoder(batch)
            frames += len(batch)
        synchronize()
        measurements.append(dict(writes=writes, repeat=repeat, frames=frames,
                                 seconds=clock() - start))
    frames = sum(row["frames"] for row in measurements)
    seconds = sum(row["seconds"] for row in measurements)
    return dict(seconds=seconds, frames=frames, seconds_per_frame=seconds / frames,
                cold_start_seconds=cold_seconds, warmup_frames=len(warmup_rgb),
                frame_batch=frame_batch, measurement_episode_count=len(measurements),
                measurement_repeats_per_write_count=len(measurements) // 3,
                measurements=measurements, max_gpu_memory_bytes=peak_memory(),
                timing_scope="encoder calls and batch dispatch; RGB generation excluded")
