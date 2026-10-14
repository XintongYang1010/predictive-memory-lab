"""Memory Maze geometry, token cache, conditions and L1 perturbations (numpy)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pmr.predictive_memory_v6 import data as Dm
from pmr.predictive_memory_v6 import fake_maze, prepare, prereg
from pmr.predictive_memory_v6 import maze as M


def dda_los(layout, p, q):
    """Exact grid traversal reference (Amanatides-Woo)."""
    x, y = p
    dx, dy = q[0] - p[0], q[1] - p[1]
    cx, cy = int(np.floor(x)), int(np.floor(y))
    qx, qy = int(np.floor(q[0])), int(np.floor(q[1]))
    sx, sy = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    tmx = ((cx + (sx > 0)) - x) / dx if dx != 0 else np.inf
    tmy = ((cy + (sy > 0)) - y) / dy if dy != 0 else np.inf
    tdx = abs(1 / dx) if dx != 0 else np.inf
    tdy = abs(1 / dy) if dy != 0 else np.inf
    while (cx, cy) != (qx, qy):
        if M.wall_at(layout, np.array(cx), np.array(cy)):
            return False
        if tmx < tmy:
            if tmx > 1:
                break
            tmx += tdx
            cx += sx
        else:
            if tmy > 1:
                break
            tmy += tdy
            cy += sy
    return True


class TestGeometry(unittest.TestCase):
    def test_los_matches_exact_dda(self):
        rng = np.random.default_rng(0)
        agree, n = 0, 0
        for s in range(5):
            lay = fake_maze.random_maze(9, np.random.default_rng(s))
            free = np.argwhere(lay == 1)
            for _ in range(200):
                a = free[rng.integers(len(free))][::-1] + rng.uniform(0.1, 0.9, 2)
                b = free[rng.integers(len(free))][::-1] + 0.5
                agree += bool(M.line_of_sight(lay, a, b)) == dda_los(lay, a, b)
                n += 1
        self.assertGreater(agree / n, 0.98)

    def test_egocentric_and_axis_fit(self):
        ep = fake_maze.make_episode(3, images=False)
        conv, errs = M.fit_axis_convention([ep])
        self.assertEqual(conv, "right_fwd")
        self.assertLess(errs[conv], 1e-9)
        d = np.array([[0.0, 1.0]])
        v = M.egocentric(np.array([[1.0, 0.0]]), d, "right_fwd")
        np.testing.assert_allclose(v, [[1.0, 0.0]])          # +x is to the right when facing +y

    def test_local_grid_and_keys(self):
        ep = fake_maze.make_episode(4, images=False)
        tok = M.episode_tokens(ep, "right_fwd")
        self.assertEqual(tok["y"].shape, (1001, M.value_dim(3)))
        own = 2                                               # f = 0, lateral 0 in the 5x5 grid
        self.assertTrue(tok["o"][:, own].all())               # own cell always observed
        self.assertTrue((tok["y"][:, own] == 1).all())        # and is a corridor
        self.assertTrue((tok["key"] >= 0).all() and (tok["key"] < 9 * 9 * 8).all())
        vis = tok["vis"]
        for g in range(3):
            b = M.N_GRID + 3 * g
            np.testing.assert_array_equal(tok["o"][:, b + 1], vis[:, g])

    def test_visibility_blocked_by_wall(self):
        lay = np.ones((3, 3), dtype=np.uint8)
        lay[1, 1] = 0
        pos = np.array([[0.5, 1.5]])
        d = np.array([[1.0, 0.0]])
        tpos = np.array([[2.5, 1.5]])
        tvec = M.egocentric(tpos[None] - pos[:, None], d[:, None], "right_fwd")
        self.assertFalse(M.object_visibility(lay, pos, d, tpos, tvec, "right_fwd")[0, 0])
        lay[1, 1] = 1
        self.assertTrue(M.object_visibility(lay, pos, d, tpos, tvec, "right_fwd")[0, 0])

    def test_revisit_events(self):
        lay = np.zeros((9, 9), dtype=np.uint8)
        lay[0] = 1
        xs = np.concatenate([np.linspace(0.5, 8.5, 60), np.linspace(8.5, 0.5, 60)])
        pos = np.stack([xs, np.full_like(xs, 0.5)], -1)
        d = np.tile([1.0, 0.0], (len(xs), 1))
        ev = M.revisit_events(pos, d, lay, min_gap=50, min_bfs=2)
        self.assertTrue(len(ev) > 0)
        for (t, lf, _v) in ev:
            self.assertGreaterEqual(t - lf, 50)


class TestCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        data = cls.tmp / "data"
        fake_maze.write_fake_dataset(data, n_train=10, n_eval=2, images=False)
        cls.res = prepare.prepare(data, cls.tmp / "cache", splits=("train",), procs=2)
        files = prepare.discover(data)
        cal = json.loads((cls.tmp / "cache" / "calibration.json").read_text())
        prepare.build_split([f for f in files if f["split"] == "train"][:6], cls.tmp / "cache" / "dev", cal, procs=2)
        cls.store = Dm.Store(cls.tmp / "cache", "dev")
        cls.files = files

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_splits_and_cache(self):
        self.assertTrue(all(f["split"] == "test" for f in self.files if "/eval/" in f["path"]))
        self.assertEqual(self.store.meta["n_ok"], 6)
        self.assertEqual(self.store.D, M.value_dim(3))

    def test_conditions(self):
        st = self.store
        ts = prereg.DATA["splice_ts"]
        for cond in ("STATIC", "CONT", "EXACT", "NEWPATH", "CHANGE"):
            spec = Dm.plan(st, cond, 2)[0]
            it = Dm.build_item(st, spec, {"kind": "l0"})
            T = len(it["y"])
            self.assertEqual(T, 1000 if cond == "STATIC" else 799)
            if cond in ("EXACT", "NEWPATH", "CHANGE"):
                self.assertEqual(it["meta"]["ts"], ts - 1)
                self.assertEqual(it["w"][ts - 1], 0.0)
                self.assertEqual(it["w"].sum(), T - 1)
            if cond == "EXACT":
                np.testing.assert_array_equal(it["y"][ts:], it["y"][: ts - 1])
            self.assertEqual(it["zin"].shape[1], 14 + 2 * st.D)

    def test_l1_perturbation_deterministic(self):
        st = self.store
        sv = np.full((st.G, 2), 0.3, dtype=np.float32)
        spec = Dm.plan(st, "EXACT", 1)[0]
        a = Dm.build_item(st, spec, {"kind": "l1"}, sigma_vec=sv)
        b = Dm.build_item(st, spec, {"kind": "l1"}, sigma_vec=sv)
        np.testing.assert_array_equal(a["y"], b["y"])
        np.testing.assert_array_equal(a["zin"], b["zin"])
        ts = prereg.DATA["splice_ts"]
        self.assertFalse(np.array_equal(a["y"][ts:], a["y"][: ts - 1]))    # fresh noise on the repeat

    def test_move_recomputes_object(self):
        st = self.store
        specs = Dm.plan(st, "MOVE", 3)
        if not specs:
            self.skipTest("no movable object in the tiny fake set")
        spec = specs[0]
        ep = st.episode(spec["A"])
        tr_frame = Dm.apply_move(ep, spec, st.conv, st.fov, st.max_dist)
        g, tm, new = spec["g"], spec["tm"], np.asarray(spec["new"])
        want = M.egocentric(new[None] - ep["pos"][tm:].astype(np.float64), ep["dir"][tm:].astype(np.float64),
                            st.conv)
        np.testing.assert_allclose(ep["tvec"][tm:, g], want, atol=1e-5)
        if tr_frame is not None:
            self.assertTrue(ep["vis"][tr_frame, g])
            self.assertFalse(ep["vis"][tm:tr_frame, g].any())

    def test_residual_noise_seeded_by_item(self):
        st = self.store
        spec = Dm.plan(st, "STATIC", 1)[0]
        a = Dm.build_item(st, spec, {"kind": "l0"}, rng_salt="dev/ns0", sbar_inj=0.5)
        b = Dm.build_item(st, spec, {"kind": "l0"}, rng_salt="dev/ns1", sbar_inj=0.5)
        self.assertFalse(np.array_equal(a["eps"], b["eps"]))
        np.testing.assert_array_equal(a["y"], b["y"])


if __name__ == "__main__":
    unittest.main()
