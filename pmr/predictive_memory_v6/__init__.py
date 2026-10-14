"""V6.1: Bayes-factor gated fast-weight memory for world models (BF-Delta).

A frozen memoryless predictor f_theta (the role of Cambrian-S's LFP head) is
the prior; a test-time delta-rule fast weight S on its standardized residual is
the belief; the signed log Bayes factor between the two computes, in closed
form, the decay gate 1 - P(change) and the write gate (Kalman gain) that Gated
DeltaNet otherwise learns.  Studied on Memory Maze ground-truth tokens (L0),
perturbed tokens (L1), frozen DINOv2 features (R) and, once, on Cambrian-S
VSC-Repeat counting.

numpy-only modules: prereg, maze, fake_maze, prepare, data, bf, metrics, vsc.
torch modules (imported lazily): models, train, evaluate, probes, features_r.
Protocol: docs/V6_BF_DELTA.md, docs/PREREGISTRATION_V6.1.md.
"""

SCHEMA = "pmr-v6.1"
