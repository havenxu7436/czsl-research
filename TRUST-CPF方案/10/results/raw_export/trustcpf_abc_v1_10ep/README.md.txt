# TRUST-CPF ABC v1, 10 epochs

A: amplitude calibration. B: finite-mixture deployment utility. C: two object contexts.

Three new Uniform trajectories (3351/3352/3353), each trained for ten complete epochs from the existing strong CPF reference. All policies share each seed's frozen best checkpoint. Fifteen Stage teachers and their original paired interventions are reused. A's Q4 amplitude selection does not set B/C strength.

- Read RUNBOOK.md for the complete Windows upload, Linux training and Windows download commands.
- Read FORMULAS.md for the derivations, experimental comparisons and source-extracted implementation.
- Source: /data/xhw/experiments/trustcpf_stage_v1_10ep
- Cache: /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
- Default physical GPUs: 1 2 3; change the run_all.sh arguments to 0 1 2 if desired.

## Production completion

36 stages; 936 recognition rows; 3360 Q4 rows; 3504 Q5 rows; 1056 same-measure-image directional rows; 33 epoch rows. The three seeds share a predictor fitted from nine Q1-Q3 teachers. Graph banks are averaged within seed.

## Local validation

`python -m unittest discover -s tests -p test_abc.py -v`

Five tests cover gradient identities, context nesting and budgets, the full three-seed synthetic pipeline including ten-epoch training and archive roundtrip, and CUDA resume plus sixteen-graph routing. The pipeline test uses a small synthetic cache and two graphs; the CUDA route test uses sixteen graphs. These are implementation checks, not UT-Zappos experimental results. Validation JSON files and the test log record the performed checks. Full server training is launched with the runbook commands.
