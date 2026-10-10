# Upstream sources

- RealMotion: https://github.com/fudan-zvg/RealMotion
  - Pinned commit: `e292609f03ccd415c4f2ac6dd7fd907c098508e8`.
  - Paper: Song et al., *Motion Forecasting in Continuous Driving*, NeurIPS 2024.
  - Code and checkpoint are downloaded from the authors at setup time and excluded from this repository and release ZIP.
  - No blanket license is asserted over upstream code or weights. Check the authors' terms before redistribution.
- NATTEN 0.17.3: https://github.com/SHI-Labs/NATTEN/tree/v0.17.3
  - `pmr.compat.NeighborhoodAttention1D` is an independent pure-PyTorch implementation of the noncausal, dilation-one operation used here, with compatible parameter names.
  - Window and relative-bias indexing were checked against NATTEN's published implementation. This is not the NATTEN CUDA kernel or a performance-equivalent implementation.
- AV2 API: https://github.com/argoverse/av2-api
  - `av2==0.3.6` is used for optional raw-data preprocessing.
  - Data and third-party test fixtures are not redistributed in this repository.

The repository defaults to private. No new open-source license is assigned to the research code without an author decision.
