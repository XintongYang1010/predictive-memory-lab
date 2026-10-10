# V5 probe follow-up: CPU diagnostics

These are exploratory diagnostics of the previously inspected Train/Dev probe
cache. They do not change the original failed gate or authorize downstream
training. No new GPU inference or Test access was used.

The first round failed at pooled displacement R² = −0.020927. It did not evaluate
the memory architectures. This follow-up investigates what that prerequisite
actually measures and whether its failure reflects a correctable implementation
or statistical issue.

| Diagnostic | Pooled displacement R² |
|---|---:|
| Original two-frame concatenation, alpha 0.01 | −0.020927 |
| Same features, alpha 1 selected only by grouped inner-Train CV | 0.243895 |
| Direct temporal difference, alpha 0.01 | −0.282749 |
| Direct temporal difference, Train-CV alpha 10 | 0.010472 |
| Shared position decoder, Train-CV alpha 0.01 | −1.861991 |

The fixed ridge setting substantially affects this readout. A positive score
after changing the probe is not yet evidence that temporal information, rather
than position priors, accounts for the gain. Matched-CV single-frame and
cue-order controls are being checked separately before drawing that conclusion.

- [All 22 planned fits, Train-only selections and exploratory intervals](core_probe/README.md)
- [Input, repooling, single-frame and renderer controls](input_audit/README.md)
- [Original-Dev generator, pixel direction and bad-case audit](generator_audit/generator_analysis.md)
- [Plan fixed before the core suite](../../../..//docs/VISUAL_V5_PROBE_DIAGNOSTIC_PLAN.md)
- [Original stopped run](../visual-v5-dgx-20261009/README.md)

The original 256 cache tensor hashes were verified before analysis. Reconstructed
raw/pooled baseline R² values differ from the original by less than 1e−12. The
input audit shows that repooling the cached raw tokens leaves the pooled gate
negative; batch/quantization differences tested here do not resolve the failure.
RGB motion is visible and agrees with the labels. A renderer-specific two-frame
color-centroid oracle obtains displacement R² = 0.977881, while a single-frame
centroid followed by Train-fitted linear ridge obtains 0.314306. These privileged
diagnostic controls expose a generator prior; they are never model inputs.

Source code and results are synchronized through the
`diagnostics/visual-v5-probe-20261009` branch. Consultation with the original
ChatGPT design conversation is ongoing; its review will be linked after receipt
and independent verification. Original remote contents and IndexMem are retained.
