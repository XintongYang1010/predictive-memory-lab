# BR-0.1 frozen-residual bank development results

This is an L0-derived frozen-residual diagnostic, not complete L0/visual evaluation or paper reproduction.
Source-disjoint train/dev: 69 train streams, 24 dev streams, 3 boundary-purged streams. Reused exploratory dev; seed 17 only.
No original V6 gate changed, no FINAL/test opened, no GPU or data restoration.

## Held-out development MPE (lower is better)

| Method | EXACT POST | NEWPATH POST | CHANGE POST | CHANGE first20 |
|---|---:|---:|---:|---:|
| K16/FIFO | 1.387291 | 1.358019 | 1.327830 | 1.395833 |
| K16/PaFu | 1.366133 | 1.357857 | 1.339790 | 1.392762 |
| K16/WR | 1.346264 | 1.368552 | 1.347817 | 1.440330 |
| K4/FIFO | 1.385939 | 1.357015 | 1.327830 | 1.395833 |
| K4/PaFu | 1.381960 | 1.359695 | 1.327687 | 1.395833 |
| K4/WR | 1.376746 | 1.358555 | 1.331767 | 1.427490 |
| K64/FIFO | 1.379673 | 1.356089 | 1.332790 | 1.429035 |
| K64/PaFu | 1.297827 | 1.341295 | 1.340927 | 1.404844 |
| K64/WR | 1.294593 | 1.381399 | 1.367625 | 1.469842 |
| REF/A1 | 0.909317 | 0.953850 | 0.940023 | 1.000573 |
| REF/A2 | 0.903209 | 0.953026 | 0.936854 | 0.991557 |
| REF/A6 | 0.886180 | 0.952821 | 0.943129 | 1.003706 |
| REF/all_observations_mean | 0.907190 | 1.220183 | 1.294315 | 1.382852 |
| REF/no_memory | 1.005086 | 0.973741 | 0.946773 | 1.008666 |

## Same-parent frozen-bank oracle

The oracle enumerates every deletion in each policy's own K+1 parent, with the same R=4 recent entries and six future queries. No intervening writes. It is a local privileged diagnostic, not a globally optimal online policy.

| Policy/condition | Parents | Positive-regret parents | Chosen MPE | Oracle MPE | Regret |
|---|---:|---:|---:|---:|---:|
| K16/FIFO/EXACT | 32 | 0 | 1.149076 | 1.149076 | 0.000000 |
| K16/FIFO/NEWPATH | 32 | 1 | 1.151307 | 1.134995 | 0.016312 |
| K16/FIFO/CHANGE | 32 | 1 | 1.124743 | 1.122720 | 0.002022 |
| K16/PaFu/EXACT | 32 | 5 | 1.114030 | 1.089725 | 0.024305 |
| K16/PaFu/NEWPATH | 32 | 4 | 1.171168 | 1.130629 | 0.040540 |
| K16/PaFu/CHANGE | 32 | 5 | 1.204469 | 1.115278 | 0.089191 |
| K16/WR/EXACT | 32 | 6 | 1.141029 | 1.103580 | 0.037449 |
| K16/WR/NEWPATH | 32 | 5 | 1.181478 | 1.132580 | 0.048898 |
| K16/WR/CHANGE | 32 | 5 | 1.220042 | 1.122720 | 0.097322 |
| K4/FIFO/EXACT | 32 | 0 | 1.149076 | 1.149076 | 0.000000 |
| K4/FIFO/NEWPATH | 32 | 0 | 1.132580 | 1.132580 | 0.000000 |
| K4/FIFO/CHANGE | 32 | 1 | 1.124743 | 1.122720 | 0.002022 |
| K4/PaFu/EXACT | 32 | 1 | 1.179483 | 1.149076 | 0.030408 |
| K4/PaFu/NEWPATH | 32 | 1 | 1.133380 | 1.132580 | 0.000800 |
| K4/PaFu/CHANGE | 32 | 1 | 1.126477 | 1.122720 | 0.003757 |
| K4/WR/EXACT | 32 | 3 | 1.161591 | 1.131270 | 0.030322 |
| K4/WR/NEWPATH | 32 | 1 | 1.133380 | 1.132580 | 0.000800 |
| K4/WR/CHANGE | 32 | 2 | 1.157414 | 1.122720 | 0.034693 |
| K64/FIFO/EXACT | 32 | 3 | 1.179833 | 1.151297 | 0.028536 |
| K64/FIFO/NEWPATH | 32 | 2 | 1.152770 | 1.134058 | 0.018712 |
| K64/FIFO/CHANGE | 32 | 3 | 1.139066 | 1.122631 | 0.016435 |
| K64/PaFu/EXACT | 32 | 21 | 1.193820 | 1.080719 | 0.113101 |
| K64/PaFu/NEWPATH | 32 | 13 | 1.167879 | 1.097786 | 0.070093 |
| K64/PaFu/CHANGE | 32 | 6 | 1.193547 | 1.137185 | 0.056362 |
| K64/WR/EXACT | 32 | 16 | 1.256209 | 1.090004 | 0.166205 |
| K64/WR/NEWPATH | 32 | 11 | 1.320809 | 1.151017 | 0.169793 |
| K64/WR/CHANGE | 32 | 12 | 1.297229 | 1.143468 | 0.153761 |

## Limits

- WR is a causal Gaussian/MAD/peak/NMS adapter on residual/mask/key features, not the visual paper system.
- PaFu is trained on the fixed reader's real conditional long-memory attention; only future groups exist here. No deletion utility is used as teacher. Teacher coverage and nonuniformity are in summary.json.
- Within K, item count and reserved controller envelope match. Actual dynamic bytes and extra student weights differ and are itemized; unused reservation is not allocated memory.
- A1/A2/A6 are 90,720-byte matrix-state references, not equal-byte bank competitors. The all-observations mean uses 352,512 bytes of float64 sufficient statistics.
- A2 replays frozen gates originally produced from full Z, which is absent here; it is not a deployable input-parity comparison with these bank policies.
- The residual/mask/key embedding includes a scalar key/647. Its ordinal geometry is an adaptation limitation; failures do not isolate the original papers' memory mechanisms.
- Parent oracles across policies can have different past candidate sets. Compare paired regret within a parent; do not infer global optimality or a common-history advantage from cross-parent oracle scores.
- Only eight source A clusters are evaluated; adjacent CHANGE pairs still share sources within the dev partition. Paired intervals in summary.json are exploratory, not independent confirmation.
- No RGB, original actions, continuous pose, or Z is available; top bad cases are timestamp/key/loss traces, not visual explanations.
