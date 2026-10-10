# V4 State Memory Mechanism Experiment

Status: **complete**. Smoke only: **False**.

Supervised synthetic 2D motion with stable object IDs, measured velocities and occlusion. Not an AV2 result, visual world model, JEPA reproduction, or novelty claim.

Learned arms see identical observations and train on multi-horizon position loss. Only clean Dev selects checkpoints. History corruption is an evaluation stress test. Seeds share data; uncertainty intervals are per-seed paired episode bootstraps, not independent seed evidence. Future MSE includes reconstruction at h=0 alongside h=1,4. Hidden RMSE uses the same natural occlusion mask across history-corruption conditions, excluding the deleted initial frames.

| Model | Seed | Length | MSE (h=0,1,4) | Hidden RMSE | Changed RMSE | Preserved RMSE | Never recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| window | 42 | 24 | 0.478939 | 1.154884 | 1.213372 | 1.295135 | 0.142578 |
| window | 42 | 48 | 1.501525 | 2.000190 | 2.494936 | 2.252788 | 0.378906 |
| window | 42 | 96 | 6.033364 | 3.919499 | 5.091747 | 4.600510 | 0.792969 |
| retrieval | 42 | 24 | 0.351477 | 0.988199 | 0.855882 | 1.297204 | 0.111328 |
| retrieval | 42 | 48 | 1.400606 | 1.943650 | 2.374192 | 2.245961 | 0.291016 |
| retrieval | 42 | 96 | 5.988697 | 3.943719 | 5.068553 | 4.666433 | 0.728516 |
| gru | 42 | 24 | 0.018250 | 0.184276 | 0.255142 | 0.244992 | 0.373047 |
| gru | 42 | 48 | 0.233701 | 0.767277 | 0.832971 | 1.032960 | 0.693359 |
| gru | 42 | 96 | 2.646232 | 2.653260 | 3.147156 | 3.594588 | 0.943359 |
| slot | 42 | 24 | 0.005949 | 0.108917 | 0.071033 | 0.168265 | 0.000000 |
| slot | 42 | 48 | 0.027414 | 0.246115 | 0.170726 | 0.382889 | 0.003906 |
| slot | 42 | 96 | 1.015558 | 1.481270 | 0.904262 | 2.652507 | 0.193359 |
| analytic | 42 | 24 | 0.008533 | 0.138561 | 0.063964 | 0.214239 | 0.000000 |
| analytic | 42 | 48 | 0.031476 | 0.275526 | 0.141030 | 0.433835 | 0.000000 |
| analytic | 42 | 96 | 0.125740 | 0.564435 | 0.328079 | 0.888650 | 0.000000 |
| window | 43 | 24 | 0.477907 | 1.157183 | 1.216227 | 1.294736 | 0.197266 |
| window | 43 | 48 | 1.487509 | 1.995099 | 2.492026 | 2.240459 | 0.417969 |
| window | 43 | 96 | 6.000489 | 3.894367 | 5.054192 | 4.578805 | 0.820312 |
| retrieval | 43 | 24 | 0.344894 | 0.982898 | 0.851516 | 1.294262 | 0.064453 |
| retrieval | 43 | 48 | 1.388436 | 1.932040 | 2.352080 | 2.242608 | 0.300781 |
| retrieval | 43 | 96 | 5.866376 | 3.863874 | 4.980394 | 4.574111 | 0.751953 |
| gru | 43 | 24 | 0.017463 | 0.179554 | 0.258695 | 0.234835 | 0.417969 |
| gru | 43 | 48 | 0.211811 | 0.726502 | 0.805633 | 0.972657 | 0.658203 |
| gru | 43 | 96 | 2.583445 | 2.631974 | 3.112895 | 3.565532 | 0.951172 |
| slot | 43 | 24 | 0.005593 | 0.104166 | 0.073585 | 0.157916 | 0.000000 |
| slot | 43 | 48 | 0.026516 | 0.251029 | 0.197797 | 0.383318 | 0.001953 |
| slot | 43 | 96 | 0.979967 | 1.506818 | 1.134146 | 2.586453 | 0.246094 |
| analytic | 43 | 24 | 0.008533 | 0.138561 | 0.063964 | 0.214239 | 0.000000 |
| analytic | 43 | 48 | 0.031476 | 0.275526 | 0.141030 | 0.433835 | 0.000000 |
| analytic | 43 | 96 | 0.125740 | 0.564435 | 0.328079 | 0.888650 | 0.000000 |
| window | 44 | 24 | 0.478192 | 1.154880 | 1.211665 | 1.293679 | 0.156250 |
| window | 44 | 48 | 1.488153 | 1.990226 | 2.487889 | 2.230222 | 0.425781 |
| window | 44 | 96 | 6.058824 | 3.901779 | 5.046368 | 4.592251 | 0.820312 |
| retrieval | 44 | 24 | 0.346646 | 0.981056 | 0.852388 | 1.290153 | 0.089844 |
| retrieval | 44 | 48 | 1.393416 | 1.937010 | 2.361832 | 2.240003 | 0.300781 |
| retrieval | 44 | 96 | 5.912123 | 3.893942 | 5.024215 | 4.590623 | 0.742188 |
| gru | 44 | 24 | 0.018316 | 0.185019 | 0.274245 | 0.230859 | 0.476562 |
| gru | 44 | 48 | 0.207542 | 0.719027 | 0.777098 | 0.976951 | 0.632812 |
| gru | 44 | 96 | 2.398462 | 2.535278 | 3.042669 | 3.411755 | 0.931641 |
| slot | 44 | 24 | 0.006850 | 0.116990 | 0.073694 | 0.184242 | 0.000000 |
| slot | 44 | 48 | 0.039483 | 0.297719 | 0.151702 | 0.502672 | 0.003906 |
| slot | 44 | 96 | 2.004979 | 2.080200 | 0.748191 | 3.922922 | 0.130859 |
| analytic | 44 | 24 | 0.008533 | 0.138561 | 0.063964 | 0.214239 | 0.000000 |
| analytic | 44 | 48 | 0.031476 | 0.275526 | 0.141030 | 0.433835 | 0.000000 |
| analytic | 44 | 96 | 0.125740 | 0.564435 | 0.328079 | 0.888650 | 0.000000 |

Primary mechanism comparison: clean T=48, GRU minus slot hidden-position RMSE. Positive gain favors slot. Each interval resamples paired episodes within one model seed.

| Seed | Gain | 95% CI | Paired episodes |
|---:|---:|---|---:|
| 42 | 0.521162 | [0.49809314139166466, 0.5423483566567494] | 512 |
| 43 | 0.475473 | [0.45047069195484424, 0.49898547207293686] | 512 |
| 44 | 0.421307 | [0.39785963949312037, 0.444411051924089] | 512 |

See `summary.json` for all conditions, recovery and preservation metrics, `episode_metrics.jsonl` for exact paired records, `training_curves.png` and `training.jsonl` for training/Dev curves, `trajectories/*.npz` for observations/targets/predictions, and `bad_cases/` for failures.

Interpretation: the analytic tracker uses the known constant-velocity law and is a causal solvability control. A low collateral-change score must be read alongside state accuracy; a model ignoring all memory could otherwise appear stable. Object slots have an identity and conditional-independence prior. Report parameter counts and memory scalars; these arms are not parameter- or FLOP-matched. Larger T also changes the state range, so length generalization alone does not isolate retention from extrapolation.

Do not select architectures or thresholds on these Eval results and reuse them as confirmation. Use fresh data namespaces for a subsequent confirmatory run.
