import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pmr.predictive_memory_v6 import worlds_l1, worlds_real
from pmr.predictive_memory_v6.data import make_world, geometry_of
from pmr.predictive_memory_v6.geometry import fold
from pmr.predictive_memory_v6.metrics import score_episodes, summarize
from pmr.predictive_memory_v6.references import run_reference


class TestL1(unittest.TestCase):
    def test_visibility_rules(self):
        pa = np.array([[[3.0, 0.0], [-2.0, 0.0], [3.0, 3.0], [6.0, 0.05], [2.0, 0.6]]])
        vis = worlds_l1.visibility(pa)[0]
        self.assertTrue(vis[0])           # straight ahead
        self.assertFalse(vis[1])          # behind
        self.assertFalse(vis[2])          # 45 deg off axis > 30 deg half FoV
        self.assertFalse(vis[3])          # occluded by block 0
        self.assertTrue(vis[4])           # nearer, off the ray to block 0

    def test_tokens_and_fold_dynamics(self):
        T = 60
        rng = np.random.default_rng(0)
        starts = rng.integers(1, 16, (4, 2)).astype(float)
        vel = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]], float)
        unf = starts[None] + vel[None] * np.arange(T)[:, None, None]
        pos = fold(unf, 1.0, 15.0)
        pos3 = np.stack([pos[..., 0], np.zeros_like(pos[..., 0]), pos[..., 1]], -1)
        agent = np.tile(np.array([8.0, 0.0, 8.0]), (T, 1))
        pa = pos - agent[:, None, [0, 2]]           # heading +x
        info = {"block_pos_world": pos3, "agent_pos_world": agent, "block_pos_agent": pa,
                "agent_dir": np.zeros(T)}
        tok = worlds_l1.tokens_from_block_info(info)
        self.assertEqual(tok["vis"].shape, (T, worlds_l1.N_MAX))
        self.assertFalse(tok["exists"][:, 4:].any())
        with tempfile.TemporaryDirectory() as d:
            dst = Path(d) / "l1" / "tokens"
            dst.mkdir(parents=True)
            arr = {k: tok[k][None] for k in ("pos", "vis", "exists", "obs", "ego", "u", "agent_dir")}
            arr["app"], arr["n_blocks"] = tok["app"][None], np.array([4])
            np.savez(dst / "dev.npz", **arr)
            chk = worlds_l1.check_dynamics(d, "dev")
            self.assertTrue(chk["inside_box"])
            self.assertEqual(chk["fold_consistent_fraction"], 1.0)
            data = make_world("l1_dev", 3, 40, 0, cache=d)
            self.assertEqual(data["vis"].shape, (3, 40, worlds_l1.N_MAX))
            g = geometry_of(data)
            p = run_reference("analytic_kf", data, g)
            s = summarize(score_episodes(data, p, g))
            self.assertIsNotNone(s["hidden_rmse"])

    def test_command_matches_flowm_recipe(self):
        cmd = worlds_l1.command("x", 1, 7, "biased_walk_v2")
        for flag in ("--store-block-info", "--grid-cardinal-only", "--box-allow-overlap", "--room-size"):
            self.assertIn(flag, cmd)
        self.assertNotIn("--box-and-ball", cmd)
        self.assertEqual(cmd[cmd.index("--forward-prob") + 1], "0.90")
        self.assertEqual(cmd[cmd.index("--num-blocks") + 1], "7")


def _write(root, name, rows):
    (root / f"{name}.json").write_text(json.dumps(rows))


class TestReal(unittest.TestCase):
    def test_nuscenes_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "meta"
            root.mkdir()
            T = 40
            samples = [{"token": f"s{t}", "timestamp": 1_000_000 + 500_000 * t, "scene_token": "sc",
                        "prev": f"s{t-1}" if t else "", "next": f"s{t+1}" if t < T - 1 else ""} for t in range(T)]
            _write(root, "sample", samples)
            _write(root, "scene", [{"token": "sc", "name": "scene-0001", "first_sample_token": "s0", "log_token": "l"}])
            _write(root, "category", [{"token": "c", "name": "vehicle.car"}])
            _write(root, "instance", [{"token": "i0", "category_token": "c"}, {"token": "i1", "category_token": "c"}])
            _write(root, "visibility", [{"token": str(k), "level": lv} for k, lv in
                                        zip(range(1, 5), ("v0-40", "v40-60", "v60-80", "v80-100"))])
            _write(root, "ego_pose", [{"token": f"e{t}", "translation": [100.0 + t, 50.0, 0], "rotation": [1, 0, 0, 0]}
                                      for t in range(T)])
            _write(root, "sample_data", [{"token": f"sd{t}", "sample_token": f"s{t}", "ego_pose_token": f"e{t}",
                                          "is_key_frame": True, "filename": f"samples/LIDAR_TOP/x{t}.bin"} for t in range(T)])
            anns = []
            for t in range(T):
                for inst, v in (("i0", 2.0), ("i1", -1.0)):
                    hidden = inst == "i0" and 10 <= t < 16
                    anns.append({"token": f"a{inst}{t}", "sample_token": f"s{t}", "instance_token": inst,
                                 "visibility_token": "1" if hidden else "4",
                                 "translation": [100.0 + v * t, 60.0, 0], "size": [2, 4, 1.5],
                                 "num_lidar_pts": 0 if hidden else 30, "num_radar_pts": 0 if hidden else 2})
            _write(root, "sample_annotation", anns)
            cache = Path(d) / "cache"
            st = worlds_real.prepare_nuscenes(root, cache)
            self.assertEqual(st["scenes"], 1)
            self.assertEqual(st["event_tracks"], 1)
            self.assertEqual(st["hidden_run_histogram"], {6: 1})
            split = worlds_real._hash_split("scene-0001")
            data = make_world(f"real_nuscenes_{split}", -1, 38, 0, cache=cache)
            g = geometry_of(data)
            self.assertEqual(g.kind, "plane")
            p = run_reference("cv_kf", data, g)
            s = summarize(score_episodes(data, p, g))
            self.assertLess(s["hidden_rmse"], 1e-2)         # linear motion through the occlusion
            pz = run_reference("pause", data, g)
            self.assertGreater(summarize(score_episodes(data, pz, g))["hidden_rmse"], 1.0)

    def test_mot_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            seq = Path(d) / "MOT17" / "train" / "MOT17-02-FRCNN"
            (seq / "gt").mkdir(parents=True)
            (seq / "seqinfo.ini").write_text("[Sequence]\nname=MOT17-02-FRCNN\nframeRate=25\nimWidth=1920\n"
                                            "imHeight=1080\nseqLength=400\n")
            lines = []
            for f in range(1, 401):
                for tid, x0 in ((1, 100), (2, 900)):
                    vis = 0.05 if (tid == 1 and 100 <= f < 160) else 0.9
                    lines.append(f"{f},{tid},{x0 + f},{300},{50},{120},1,1,{vis}")
                lines.append(f"{f},9,10,10,5,5,0,7,1.0")   # ignored row (flag 0, class 7)
            (seq / "gt" / "gt.txt").write_text("\n".join(lines))
            cache = Path(d) / "cache"
            st = worlds_real.prepare_mot([seq.parent], cache, target_fps=5.0, T=64, stride=32)
            self.assertIn("MOT17-02", st["sequences"])
            self.assertEqual(st["sequences"]["MOT17-02"]["stride_frames"], 5)
            arr = np.load(cache / "real" / "mot" / "MOT17-02.npz")
            self.assertLessEqual(arr["exists"].sum(-1).max(), 2)   # ignored track never enters

    def test_has_event_and_pseudo(self):
        ex = np.ones(10, bool)
        hid = np.zeros(10, bool)
        hid[4:6] = True
        self.assertTrue(worlds_real.has_event(ex, hid))
        hid2 = np.zeros(10, bool)
        hid2[0:3] = True
        self.assertFalse(worlds_real.has_event(ex, hid2))
        ep = {"vis": np.ones((30, 2), bool)}
        out = worlds_real.pseudo_occlusion(ep, np.random.default_rng(0))
        self.assertTrue((~out["vis"]).sum() >= 6)
        self.assertTrue(out["vis"][:2].all())


if __name__ == "__main__":
    unittest.main()
