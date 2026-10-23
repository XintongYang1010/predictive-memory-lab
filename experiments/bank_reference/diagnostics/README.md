# Existing-case connection diagnostic

This directory implements BR-0.2, a frozen-parent signal/noise accounting of
previously published BR-0.1 cases. It does not train a policy, rerun V6 gates or
start a perception study. Read [PROTOCOL.md](PROTOCOL.md) before interpreting the
[Chinese assessment](../results/br-0.2-connect-20261022/ASSESSMENT_ZH.md).

Run with NumPy from a new output directory and the local original evidence root:

```sh
python experiments/bank_reference/diagnostics/connect_badcases.py \
  --source-root /absolute/path/to/predictive-memory-v6-h800 \
  --out /new/empty/output
python experiments/bank_reference/diagnostics/summarize_connections.py \
  --input /new/empty/output
python -m pytest -q experiments/bank_reference/diagnostics/test_connect_badcases.py
```

The first command requires the previously retained R25 and R50 NPZ archives and
manifests at their recorded deployment-relative locations, plus the existing A2
conclusion and component report. A Git clone alone does not contain these raw
inputs. Their hashes, identity checks and exact loss reconstruction are recorded
in the frozen protocol and result summary. No remote data is fetched. The second
command only summarizes published parent JSON and requires Matplotlib.
