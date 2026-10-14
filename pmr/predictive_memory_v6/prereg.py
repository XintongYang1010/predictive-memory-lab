"""Pre-registered thresholds (mirror of docs/PREREGISTRATION_V6.md).

Status: PROPOSED.  They become binding when the user confirms them and the
commit hash of this file is recorded in docs/PREREGISTRATION_V6.md (the
runner refuses S2+ without ``--prereg-commit``).  After freezing: never lower
a threshold, never re-seed to obtain a pass; a failed gate is first debugged
as a possible implementation error, then reported as a finding.
"""

STATUS = "proposed"

# H1 protocol validity (L0, train length T=64, identifiable predictable hidden steps)
H1 = {"ours_er_range": (0.85, 1.15), "ours_nhe_max": 0.30,
      "pausing_variants": ("static_delta", "gdn", "gdn_neg", "fifo", "lact"), "pausing_er_max": 0.20,
      "abandon_if": "gdn_neg or lact ER >= 0.8 at 9T"}

# H2 structure necessity / equivalence
H2 = {"vo_er_max": 0.30, "frame_ci_contains_zero_at": (64, 192),
      "tpa_nhe_ratio_at_T_max": 1.2, "tpa_nhe_ratio_at_9T_min": 2.0}

# H3 correction and non-interference (l0_ctrl + change pairs)
H3 = {"post3_vs_kf_max_ratio": 2.0, "never_recovered_max": 0.05, "lemma4_slope_tol": 0.20,
      "interference_rel_max_at_low_rho": 0.10, "low_rho": 0.5}

# H4 extrapolation and numerics
H4 = {"ours_nhe_9T_over_T_max": 1.5, "recurrent_nhe_9T_over_T_min": 3.0, "fp64_replay_max_dev": 1e-3}

# H5 memory-aware surprise (Saining / Cambrian-S analogue)
H5 = {"lfp_mra_drop_k3_min_frac": 0.50, "memory_mra_drop_k3_max_points": 0.10,
      "auroc_ours_min": 0.90, "auroc_static_max": 0.75, "auroc_lfp_max": 0.65,
      "f1_transfer_drop_max": 0.05, "narrow_if_auroc_gap_below": 0.05}

# H6 (secondary) reveal-error TTT
H6 = {"hidden_error_reduction_min": 0.30, "ttt_lrs": (0.1, 0.3, 1.0)}

# L1 replication
L1 = {"seeds_same_direction": 3}

# H7 real-token validation (nuScenes primary, MOT direction only)
H7 = {"ours_er_range": (0.8, 1.2), "ade_vs_cv_kf_max_ratio": 1.1, "gap_closure_min": 0.6,
      "static_er_max": 0.2, "min_natural_events": 200}

# S0/S1 software gates
GATES = {"kf_hidden_rmse_max": 1e-3, "mra_cases": {(10, 10): 1.0, (15, 10): 0.1, (10.5, 10): 1.0, (0, 10): 0.0},
         "gdn_vs_fla_max_abs": 1e-5, "ttt_equivalence_max_abs": 1e-6, "overfit_ratio_max": 0.5,
         "l1_depth_agreement_min": 0.95, "spectral_abscissa_max": 1e-6}

BUDGET_GPUH = {"S1": 2, "S2": 30, "S3": 8, "S4": 18, "S5": 35, "S6": 20, "S7": 6, "reserve": 25,
               "first_decision_point": 40, "hard_cap": 160}
