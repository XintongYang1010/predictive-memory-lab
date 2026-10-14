"""V6.1 pre-registration (mirror of docs/PREREGISTRATION_V6.1.md).

Status ``approved-dev``: the user approved running the development stages
S0-S6 on 2026-10-14 ("研究完之后更新v6 ... 直接开始跑").  Thresholds below are
binding for those stages from the commit that introduced this file.  FINAL
(the only stage that opens Memory Maze test or scores VSC-Repeat) additionally
needs the human checks listed in the pre-registration and an explicit
``--final-ack`` from the user.

Never lower a threshold or re-seed to obtain a pass (AGENTS.md rule 6).  A
failed gate is first debugged as a possible bug, then reported as a finding.
Changing anything here is a protocol change: new version, user approval.
"""

STATUS = "approved-dev"
VERSION = "6.1"

RULES = {
    "bootstrap_draws": 1000, "ci": 0.95, "unit": "episode (VSC: video)", "paired": True,
    "pass_requires": "point estimate meets threshold AND paired CI on the correct side of 0",
    # T1-T3: learned keys, 5 phi seeds, pass in >= 4.  T0: oracle keys, 3 noise
    # levels x 3 noise seeds, pass at every level in >= 2 of 3 seeds.
    "phi_seeds": 5, "phi_seed_pass_min": 4,
    "learned_gate_seeds": 5,
    "t0_noise_seeds": 3, "t0_noise_seed_pass_min": 2,
}

DATA = {
    "maze": "9x9",                     # primary; 15x15 is an optional FINAL replication
    "dev_mod": 29,                     # dev = train files with int(sha1(basename)) % 29 == 0
    "train_max_episodes": 12000,       # first N train files in sha1 order (bounds prep time/disk)
    "splice_ts": 400, "post_len": 200, "stat_window": (500, 1000), "rev_window": 10,
    "dev_items_per_condition": 500, "test_items_per_condition": 1000,
    "fov_deg": 80.0,                   # dm_control jumping_ball_with_head.xml egocentric fovy
    "revisit_min_gap": 50, "revisit_min_bfs": 2, "view_revisit_cos": 0.7,
}

CALIB = {
    "pi": 1e-3, "pi_robustness": (1e-4, 1e-2), "pi_vsc": 5e-3,
    "sbar_grid": tuple(round(0.05 * i, 2) for i in range(1, 20)),
    "t0_sbar_inj": (0.25, 0.5, 0.75), "t1_sbar_inj": 0.5,
    "dk_grid": (64, 128, 256, 512), "dk_frac_of_oracle_gain": 0.80,
    "ell_clip": 50.0, "n_max": None, "lambda_temper_fallback": "only if C1 fails (needs approval)",
    "a1_beta0": (0.05, 0.1, 0.2, 0.4, 0.8, 1.0), "a6_lambda": (0.99, 0.999),
    "mlii_train_episodes": 1000,
}

L1 = {
    # Observation-space perturbations.  The object-vector noise s.d. is set per
    # dimension so that it makes up `obs_noise_frac` of the standardized
    # residual variance of the clean (T1) slow model.
    "obs_noise_frac": 0.5, "pose_noise_cells": 0.10, "dir_noise_rad": 0.05,
    "vis_dropout": 0.05, "vis_false_pos": 0.01, "flicker_var": 4.0,
    "move_window": (400, 600), "move_hidden_before": 50, "move_hidden_after": 20, "move_min_bfs": 3,
}

TRAIN = {
    "slow": {"hidden": 256, "steps": 20000, "batch": 4096, "lr": 1e-3, "wd": 1e-4},
    "phi": {"hidden": 256, "steps": 10000, "batch": 32, "lr": 3e-4, "warmup": 500, "wd": 0.1,
            "clip": 1.0, "chunk": 64, "window": 1000},
    "gate": {"A2_steps": 2000, "A2P_steps": 2000, "A3A4_steps": 400, "batch": 32, "lr": 1e-3,
             "tbptt": 250, "hidden": 64},
    "probe": {"layers": 4, "hidden": 1024, "steps": 5000, "batch": 256, "lr": 3e-4, "eps": 1e-5,
              "train_episodes": 1024, "states_per_episode": 20, "eval_episodes": 256, "eval_stride": 10},
}

# H1 is judged on the global CHANGE split, where the natural member of the
# BF-Delta family is the global form A8g: the local form A8 must pay one stale
# error per old key on its first revisit, so nu_C(A8) < nu_C(A8g) by
# construction (predicted analytically in docs/research/V6_1_FINAL_REVIEW.md
# and confirmed on a synthetic one-hot world before any Memory Maze run:
# A8g (0.88, 0.96), A8 (0.30, 1.00)).  A8 is reported; locality is H4.
H1 = {"not_dominated": True, "nuR_margin_at_matched_nuC": 0.20, "a1bf_ge_a1_min_count": 5,
      "a1bf_tie_tol": 0.01, "t0_nuC_min": 0.80, "t0_nuR_min": 0.80, "applies_to": "A8g"}
H2 = {"nuC_gap_vs_A2_min": 0.30, "nuC_vs_A2plus_min_diff": -0.10, "nuR_vs_A2plus_min_diff": -0.10,
      "applies_to": "better of A8, A8g (chosen on dev nu_C, reported)"}
H3 = {"flicker_false_change_ratio_max": 0.50, "auroc_gap_vs_A4_min": 0.05}
H4 = {"nuR_unch_gap_vs_A8g_min": 0.30, "moved_reveal_err_ratio_max": 1.10, "config": "single head"}
H5 = {"walls_pts_tol": 1.0, "objmse_rel_tol": 0.10, "ref_arms": ("A1_best", "A6", "A2"),
      "all_fast_beat_A0": True}
H6 = {"repro_tol_mra": 2.0, "mra_rep_frac_of_own_base_min": 0.80, "mra_base_drop_max_pts": 5.0,
      "median_pred_gt_N5_max": 1.30, "veto_rate_N0_max": 0.05, "key_overlap_max": 0.10,
      "fam_auroc_exact_min": 0.90, "fam_auroc_phase_min": 0.80, "false_familiar_calib": 0.05,
      "kill_if": "A9 equal-byte >= A8 on exact, phase-shift and reversed"}

DIAG = {"C1_fa_per_episode_max": 1.0, "C2_sbar_tol": 0.05, "C4_soft_band": (0.05, 0.95),
        "F1_key_overlap_max": 0.10, "aliasing_cos": 0.9}

R_GATE = {"encoder": "dinov2_vits14", "image_px": 224, "crop_border_px": 2, "pca_dim": 64,
          "pca_fit_frames": 1_000_000, "probe_mse_frac_of_const_max": 0.50,
          "A0_next_latent_r2_min": 0.30, "runs": 1}

GATES = {"s0_vis_agreement_min": 0.95, "overfit_ratio_max": 0.50, "unit_tol_fp64": 1e-6,
         "gdn_vs_ref_max_abs": 1e-5, "chunk_vs_scan_max_abs": 1e-4, "profile_dev_max_factor": 2.0,
         "mra_cases": {(10, 10): 1.0, (15, 10): 0.1, (10.5, 10): 1.0, (0, 10): 0.0},
         "axis_fit_max_err_cells": 0.05}

BUDGET_GPUH = {"S1": 2, "S2": 4, "S3": 16, "S4": 8, "S5": 40, "S6": 6, "FINAL": 30, "reserve": 25,
               "first_decision_point": 30, "hard_cap": 160}

STAGE_ORDER = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "FINAL")


def as_dict():
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
