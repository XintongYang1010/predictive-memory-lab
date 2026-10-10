# V5.5 module localization: initial completed results

Fixed lane-restricted reading on the same frozen encoder/cache reduces original horizontal MSE by29.78% (0.0001184554 to0.0000831771); original512-episode any-horizontal-error count falls196 to139. Whole-world descriptive95% interval of MSE reduction is[2.8391e-5,4.2310e-5]. This is posthoc diagnosis on already inspected Dev, not a new confirmation or P0 pass.

GPU step415453.40 and CPU step415453.41 completed0:0. Original feature parity is bitwise exact and old N/W predictions match exactly. First extraction charged0.681470273191161 GPUh.

The first factorial cache used batch4 for original endpoints and batch12 for edited endpoints. Full-population matched-batch4 validation is in progress under the original1.2 GPUh round cap; factorial mechanism claims are provisional until that check. The original same-cache lane comparison is unaffected.

The full learned lane model was uploaded and read back before factorial generation: https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_5-fit-20261010 . Complete final-scene archival and cleanup are pending. Prior remote scenes remain preserved.
