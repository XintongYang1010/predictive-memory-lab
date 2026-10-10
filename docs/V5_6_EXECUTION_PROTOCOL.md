# V5.6: controlled ablations and bounded optimization prototypes

Registered before scientific fitting, new image extraction, and new known-Dev scoring. This round implements the user-requested trials; it does not alter frozen V5.1–V5.5 code, reopen original Test, or authorize P0. Original memory writer is not on the reader diagnostic path. Existing known Dev640–895 is exploratory: 256 paired worlds /512 episodes, not thousands of independent interventions. Fresh64 confirms only object-sign composition. All negative outcomes and timing failures remain archived.

## Four mechanistic interventions

1. Reader access and geometry: the exact fixed arms and FIT-only df matching are specified below. No Dev hyperparameter search.
2. Pooling: the same saved prequantization raw I-JEPA tokens generate the pooled comparison. Physical query grid and matcher are fixed; a failed matcher is not proof of absent information. Equal-byte banks are the adaptive-resolution prototype, not long-term memory.
3. Object-combination coverage: reuse all256 wide FIT worlds and original six images; append independent A/B sign combinations, two additional poses times two yellow positions. J repeats the old16 views twice; X uses old16 plus16 mixed-object views. Both8192 weighted views, same per-object position/motion/yellow marginals, mean-loss alpha7.8125e-7, and full X8192 normalization reference. J duplicates may collapse to4096 unique rows with exactly equivalent regularization. Fit global quadratic readers with free intercept; no tuning. Fresh64 worlds use the new `v5.6-axis-object-1/confirm` SHA256 seed namespace,10 images and32 views each. Freeze all fitted model/basis/config files, publish to GitHub, download every byte and verify before generating fresh images. Report first16 versus mixed16 separately and whole-world paired intervals. Fresh64 does not confirm direct axis interventions, layout swaps or a memory benefit.
4. Axis mixing: for all512 existing episodes encode Ax,Ay,Bx,By; reuse none/Axy/Bxy from the original cache. Other object/yellow fixed. Compare frozen global/lane readers' x-only, y-only false horizontal response, xy and exact interaction remainder. Large-axis sign counts require |ground truth|>=1px, threshold fixed from the preceding RGB audit. Report raw effects rather than unstable tiny-motion divisions. These mixed images are interventions, not claimed physically integrated reflected trajectories.

## Complete transitions and contextual feature paths

Use all8 endpoint images to evaluate every ordered pair (8x8) with the frozen global and lane readers. Targets use bit differences times the authoritative old displacement, avoiding rounded-position subtraction. Original000→all8 predictions must reproduce archived values within1e-7. Save all predictions and absolute/horizontal/4D motion errors, identity and reversal checks. All queries remain256 paired worlds.

For each fixed A/B endpoint configuration, use yellow-a/b donors from the same episode. Replace four top-row tokens for A or four bottom-row tokens for B, or their complements, to form the2x2 feature replacement table. These outer rows exclude pooled cells containing direct yellow pixels, but their encoder outputs may already carry contextual information. Save all four predictions and direct/complement/interaction effects, including both replacement directions obtainable from the table. Feature hybrids are off-manifold; a pathway effect is not a claim that a specific attention head is defective. The other target's four-row region provides an equal-size spatial comparison across the two recorded tables.

## Trusted-update prototype

Use the frozen lane position output for every policy, and the complete64 transition table. Replay endpoint masks `[0,1,0,1,4,5,4,5,6,7,6,7,2,3,2,3,0,1,0]`. It contains target changes and intervening yellow-only changes. Three observation-free hold ticks follow each fourth observation; hold gaps are recorded separately, not counted as additional samples. No future state, true motion or label enters a policy. Initialize from the frozen identity-pair position output.

Each object has two2D FP32 stored states; total32 payload bytes plus8 common metadata bytes per episode. All policies read the mean of the two slots. Latest duplicates the newest candidate; FIFO shifts on every observation; surprise-only shifts if full-feature change exceeds the95th percentile of archived W FIT target-static/yellow-change differences; trusted-local applies the corresponding95th percentile to A's top row or B's bottom row independently. Thresholds use all three poses of all256 W FIT worlds and are frozen before Dev. Report position error on actual-change and unchanged-object steps, gap error, write counts, missed-change writes and stale-state behavior. This deliberately small stress test can reject a naive gate. It does not establish learned identity, uncertainty calibration, full-history sufficiency or long-term memory gains.

## Execution and budgets

Allocation415453, account hansirui, partition/QOS llm, node dgx-097 currently provides eight H80080GB. It is a shared existing allocation; no IndexMem process or parent allocation may be terminated. Eight encoder workers run concurrently, BF16 with the original official I-JEPA checkpoint and existing4x4 pooling, BF16 storage rounding then FP32 cache. FIT uses batch6 [four new images,old0,old1], matching the archived FIT batch shape; both old features must match exactly for all256 worlds (512 checks). Axis/confirmation use batch4, with32 original Dev parity slots. New images:1024 FIT,2048 axis,640 confirmation, total3712 plus544 parity slots =4256 encoded slots. Exact parity failures preserve the scene; do not relax tolerances.

Cumulative inherited charge5.32908766817043 GPUh; new envelope0.65; diagnostic cap6; original round cap60. Two GPU steps use the existing exact-numeric-step ownership monitor. Charge eight GPUs times controller wall time through confirmed terminal state; no capping of actual overruns. Stage deadlines reserve exit-accounting margin. No automatic scientific retry. A failed run keeps its scene and evidence. CPU work uses no additional GPU allocation. The6GPUh limit is the existing diagnostic reserve protocol, distinct from the user's60GPUh total.

New owned work root and reference staging root are recorded with inode/device/token. The restored V5.5 reference contains only authenticated needed files; its original directory stays deleted. All old remote content and shared assets remain unchanged. Upload full new results, models, configs, software/source and figures; independently re-download and check exact inventory/hashes before deleting only new owned roots. Failure of execution or archiving means preserve the scene.

## Reader specification appendix

This is a prespecified, exploratory intervention on already opened Dev640–895. No Test, GPU, new images, writer training, model selection on Dev, or old-gate update. Scientific FIT/evaluation has not started. The parent publishes and byte-verifies source plus `docs/V5_6_EXECUTION_PROTOCOL.md` before invoking `fit`.

All arms use the same authenticated V5.4 frozen N+W basis: 512 matched worlds / 3,072 unique images / 8,192 reference views. Only W's 4,096 views from 256 worlds supervise eight original outputs. Each world has equal total loss. The A output group is columns [0,1,4,5]; B is [2,3,6,7]. All fits have an unpenalized intercept. Original mean-loss alpha is 7.8125e-7, so ridge diagonal is 0.0032. Four BLAS threads. Shared reference/score baseline is never recomputed from Dev.

## Frozen arms

Six paired-output arms, each reported with original alpha and per-output-group FIT-only df-matched alpha:

| arm | A kernel | B kernel |
|---|---|---|
| global | complete global | complete global |
| additive | upper+lower independent terms | same |
| target_common | upper term, original global scales | lower term, original global scales |
| correct_lane | native normalized upper 8 tokens | native normalized lower 8 tokens |
| opposite_lane | native normalized lower | native normalized upper |
| checkerboard | even (row+column) parity 8 tokens | odd parity 8 tokens |

The four native masks use the exact same two-stage center/trace recipe on all 8,192 FIT reference views. Checkerboard assignments are fixed, not selected. No random projection or regularization grid is included.

For common-scale kernels, let H center all reference views, a=H G_upper H/s_linear and b=H G_lower H/s_linear. The components are upper=2a+a², lower=2b+b², cross=2ab. Each component is separately centered with H and divided by the original global s_quadratic. Full=(1+a+b)² is centered in the same way; additive=upper+lower. Each component has its own FIT column/overall means, both global scales are frozen. Query means use the complete FIT reference. Full reconstruction and predictions must reproduce archived global W; native correct lane must reproduce archived lane W. Registered absolute prediction tolerance 1e-7, rtol zero, for every output of every one of the 4,096 known queries. No interpretation if parity fails; retain all outputs.

DF for an n=4,096 W kernel is 1+sum(eig(H_W K_WW H_W)/(eig+n*alpha)). A's target is the native upper kernel at original alpha, B's target the native lower kernel at original alpha. Solve each kernel/group alpha by monotone bisection on log(alpha) in [-40,40], fixed absolute df tolerance 1e-6. A target outside attainable endpoint dfs is recorded as unavailable; do not change the target, threshold, or omit the failed condition from the report. Material negative eigenvalue below -1e-10*max(1,maxabs spectrum) aborts; only negative numerical roundoff is clipped to zero. Save the complete eigenvalue vector, negative counts, spectrum summaries, fitted alphas, root residuals and normal-equation residuals. Equal df does not mean identical directional shrinkage, and failure at one matched alpha cannot rule out all scalar regularization explanations.

## Evaluation

All 256 worlds x2 variants x8 endpoint masks are retained. First frame is original mask000; mask111 is original t7. Existing target arrays and mask identities remain authoritative. Report each mask's dx MSE, all axes, direction, absolute amplitude, zero-motion false response RMS, plus original111 motion skills using the unchanged joint FIT mean. Save all eight-output predictions, all rows of a per-episode CSV, frozen kernels/coefficients and original baseline parity. Per-mask paired differences versus original global are bootstrapped as 256 whole worlds (both variants, all masks kept together), 10,000 replicates, fixed seed7816. These are conditional exploratory intervals, not multiplicity-adjusted confirmation claims or P0 gates.

## Source / publication contract

`--publication` JSON must have `verified_readback: true`, `source_commit` (or `commit`), and `files` keyed by repo-relative source paths. Each value is either the SHA256 string or `{sha256: ..., bytes: ...}`. Required paths are the script's explicit `SOURCE_NAMES` plus `docs/V5_6_EXECUTION_PROTOCOL.md`; the function `sources()` returns the exact current map. `publication()` validates them before FIT and score. The complete prior V5.4 manifest is pinned to ec7bc9c17da8417e9ecd11ba0f21415a2df0ee7af620af9d12f911c3ab26e75c; V5.5 to b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663. Used files are checked against their sealed manifests and input receipts. No old files are written.

FIT creates a separate model directory, serializes all 12 planned controls, configuration, software checks, basis, spectra and numerical checks before any V5.5 evaluation cache is opened. A final freeze receipt hashes all model files. Score verifies this exact inventory before use. A failed/incomplete fit cannot be loaded. Source changes invalidate the freeze.

Run after the parent source-publication signal, from repo root:

```sh
python3 scripts/v5_6_reader_ablation.py fit \
  --scene execution/visual-v5_4-20261010/diagnostic-20261010-v54 \
  --out execution/visual-v5_6-20261010/reader_ablation/model \
  --publication execution/visual-v5_6-20261010/registration_publication.json
python3 scripts/v5_6_reader_ablation.py score \
  --scene execution/visual-v5_4-20261010/diagnostic-20261010-v54 \
  --v55-scene execution/visual-v5_5-20261010/diagnostic-20261010-v55 \
  --model execution/visual-v5_6-20261010/reader_ablation/model \
  --out execution/visual-v5_6-20261010/reader_ablation/report \
  --publication execution/visual-v5_6-20261010/registration_publication.json
```

## Pure-math reuse for a separate J/X driver

The parent may reuse `fit_math(base, output)` with explicitly supplied FIT-only arrays; it does not read any dataset paths or queries. `base` keys: `fit_features [worlds,unique_images,20480]`, `fit_targets [...,8]` flattened in reference-row order, `fit_view_indices [reference_views,2]` indexing flattened unique images, `W_rows [supervised_rows]`, `fit_block_ids`, `shared_target_mean`, `fit_motion_rms`, and `concat__linear_scale/concat__quadratic_scale` already computed on the intended shared reference. It fits exactly the rows supplied as W_rows; the pure math supports other n, but the present production contract remains W4096. The parent J/X experiment must document a separate config, provenance and row weights; do not load its outputs through this V5.6 reader production wrapper. Lower-level `construct_kernel`, `eigensystem`, `solve`, `effective_df` and `match_df` are separately callable.

Software checks compare all eight kernels and held-out query kernels to explicit feature maps, free-intercept solutions to an independent augmented linear system, the additive/cross decomposition to the full kernel, and rejection of an impossible df target. NumPy Accelerate on this macOS host emits floating-status RuntimeWarnings for some finite matrix multiplies; all checked outputs are finite and the complete unsuppressed stderr is retained. These warnings do not authorize relaxing any numerical or parity check.

## Correspondence and storage specification appendix

This is a posthoc, CPU-only diagnostic on the already inspected Dev worlds 640–895 (256 worlds, both variants; 512 episodes). No training, Test, new encoder execution, method selection or protocol gate is performed. Scientific scoring starts only after GitHub source/design readback. All six methods and all examples are retained, including negative outcomes and ambiguous matches.

## Exact inputs and authorization

Input scene: `execution/visual-v5_5-20261010/diagnostic-20261010-v55`.

- `archive_manifest.json`: `b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663`.
- `counterfactual_cache/receipt.json`: `000ec48d9a2a7ba7ea07025f87e246ae87045d3a122ca478f6d7b10c4e5d4f46`.
- `report/predictions.npz`: `f3f66de4fb643ccbf4bb011761111d24aa49170e539c186a41d93068026eb294`.

Every opened artifact must match the sealed scene inventory. Every raw array must match its receipt's array shape, dtype and SHA256. Only original endpoints 000/111 are used: each pair raw shard has shape `[2 variants,2 frames,256 spatial tokens,1280 channels]`, FP32 storage from the archived BF16 encoder forward. FP32 storage does not imply FP32 encoder inference. Source RGB is reconstructed solely to authenticate the original hash and extract the fixed privileged ROI. Archived continuous displacement labels are independently checked against the original generator. They never choose a displacement, ROI, token or parameter.

The publication JSON must contain `verified_readback: true`, `verified_unix >= plan.created_unix`, and `files[repo_relative_path]` binding both this script and the design JSON to SHA256 (a string or a `{sha256: ...}` record). The design JSON records exact source hashes. Results must be outside the immutable scene and in a new directory.

## Fixed matcher

Source-frame object colors define the diagnostic ROIs: A uses R=235/B=60; B uses R=60/B=235. Green is not used. For each color bounding box, query the Cartesian 3×3 points at 25%, 50%, 75% of its x/y extent. This uses source RGB only, not target RGB or latent positions, and is explicitly a privileged diagnostic.

Candidate motion is the physical-pixel Cartesian grid `dx,dy ∈ {-6,-5.5,...,5.5,6}`: 625 candidates in dy-major/dx-minor order. Query frame 0 at each ROI point and frame 1 at that point plus candidate motion. Native spatial token centers are `(i+0.5)*224/side-0.5`: 16×16 raw and 4×4 pooled. Bilinear interpolation replicates border values; clamped query counts are saved.

Each interpolated descriptor is centered over its own 1280 channels. Cost is mean cosine distance across the nine points. A squared descriptor norm ≤1e−24 is undefined and contributes similarity zero; counts are saved. Candidates within absolute cost 1e−12 of the minimum tie; choose smallest displacement squared norm, then dx, then dy. Save the top-two cost gap, exact-tie count, near-best count within 1e−6, and boundary-optimum flag. No softness or confidence cutoff removes predictions.

## Six prespecified methods

1. `raw_native`: all 256 raw spatial descriptors, native bilinear reconstruction.
2. `pool_native`: same raw descriptors spatially averaged into 4×4. Means accumulate in FP64 and store FP32; no final BF16 quantization. This is same-source pooling, not an assertion of bitwise GPU mean-pooling parity. Native bilinear reconstruction.
3. `uniform_budget`: sixteen 4×4 mean descriptors.
4. `roi_adaptive_budget`: four 2×2 mean descriptors plus twelve raw descriptors. Select six raw cells per object by descending source-color pixel coverage, then distance to source-color centroid, then raster index, without replacement.
5. `random_budget`: four 2×2 means plus twelve raw cells sampled without replacement with `default_rng(5617 + 2*pair_id + variant)`.
6. `change_priority_budget`: four 2×2 means plus twelve largest raw frame-difference squared-L2 cells; raster order resolves ties. This selection reads both frames and is offline lookahead, not an online memory policy.

All four budget methods share one bank per frame for both objects: exactly sixteen 1280-dimensional FP32 vectors plus sixteen uint16 location codes = 81,952 bytes/frame, 163,904 bytes/two frames. Shared method/grid headers are excluded equally. The same selected indices are used in both frames; there is no extra bank per object. Location codes are raw 0–255, 2×2 means 256–259, 4×4 means 260–275.

All four budget methods use the same fixed inverse-distance-squared interpolation from stored physical centers to the 16×16 centers (exact at a stored center), followed by the same bilinear/cosine matcher. `uniform_budget` therefore differs from the native interpolation baseline `pool_native`. Selected codes and descriptor hashes are persisted, allowing exact reconstruction from the frozen raw inputs without duplicating the banks.

## Scores and outputs

Predicted coordinates are `[A_dx,A_dy,B_dx,B_dy]`; pixel estimates are divided by 224 for comparison with archived continuous targets. Skills use the original V5.5 lane-model `shared_target_mean`, inherited from joint N+W FIT, never the opened Dev mean. Save per-axis MSE/skill, mean 4D skill, primary horizontal MSE, direction-error count, predicted-zero count and absolute-amplitude ratio. No winner is selected from these diagnostic scores.

- `costs_and_predictions.npz`: costs `[512,6,2,625]` FP64; predictions `[512,6,4]`; targets `[512,4]`; offsets `[625,2]`; method names, ordered pair IDs/variants and frozen FIT mean.
- `per_episode_object.csv`: 512×6×2 rows with targets, predictions, all ambiguity/clamping diagnostics.
- `selected_banks.json`: selected codes, per-frame descriptor hashes, metadata hashes, bytes and source-RGB hash for all four budget methods.
- `report.json`: scores, fixed design, complete checked-input inventory, source/publication/plan hashes, command and runtime versions.
- `failure.json`, on a failure in a fresh authorized output directory; preserve partial outputs and original scene.

These results are neither bounds on decoder capacity nor proof of encoder information loss. Sparse reconstruction, ROI privilege, descriptor choice, spatial smoothing and the fixed motion range constrain interpretation. A two-frame storage experiment is not evidence of long-term memory.

## Reproduction

Run from the repository root with at most four BLAS threads (the script enforces them before NumPy import):

```bash
/Library/Developer/CommandLineTools/usr/bin/python3 scripts/v5_6_correspondence.py self-test
/Library/Developer/CommandLineTools/usr/bin/python3 scripts/v5_6_correspondence.py freeze-design --out scripts/v5_6_correspondence_plan.json
/Library/Developer/CommandLineTools/usr/bin/python3 scripts/v5_6_correspondence.py run --scene execution/visual-v5_5-20261010/diagnostic-20261010-v55 --out execution/visual-v5_6-20261010/correspondence/results --publication execution/visual-v5_6-20261010/registration_publication.json --plan scripts/v5_6_correspondence_plan.json
```

The last command waits for the parent controller's explicit publication/readback signal. Software verification uses only synthetic arrays: known affine descriptor translation, an independently explicit descriptor/cosine calculation, exact constant-field ambiguity, stationary banks, equal payload sizes, and spatial raster indexing. macOS NumPy 2.0.2 emits matmul RuntimeWarnings in the software fixture; warnings remain visible, all checked costs are finite, and explicit computations agree. No scientific feature scoring has occurred at design freeze.
