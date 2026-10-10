# V5.4 posthoc frozen FIT replay

Each unchanged reader was replayed on its own 256 saved FIT worlds (4096 views). Scoring uses the same frozen joint N/W target mean as the official evaluation. The script loaded no encoder, generated no RGB, and performed no refitting or alpha selection.

| Reader | Own-FIT motion skill | dx MSE | Direction error (A_dx, A_dy, B_dx, B_dy) | Absolute amplitude ratio |
| --- | --- | --- | --- | --- |
| N | 0.999493 | 1.311355e-07 | 0.0000, 0.0000, 0.0000, 0.0000 | 0.9859, 0.9939, 0.9857, 0.9937 |
| W | 0.999508 | 1.1714485e-07 | 0.0000, 0.0000, 0.0000, 0.0000 | 0.9855, 0.9919, 0.9856, 0.9918 |

High own-FIT accuracy with weaker held-out accuracy is evidence of an observed generalization gap. Poor own-FIT accuracy instead shows that this fixed reader does not fit even its training inputs well. These are descriptive distinctions, not causal conclusions or a replacement pass gate. Per-axis denominators, errors, direction ties, amplitude/RMS ratios, static/moving groups, condition metrics and all input hashes are preserved in report.json.

Official anchor/bridge report comparisons were supplied and bound to the same model/publication hashes; their differences are in official_evaluation_comparisons.
