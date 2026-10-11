# UT semantic attribution read-only audit v2

See the Chinese operation plan in `PLAN.md` for upload/run commands.

Run `audit_semantic_gate.py` using the existing CPF Python interpreter. Requires
Python 3.8+, torch, numpy, PyYAML and existing YACS. No GPU or administrator rights.
The audit writes only a fresh `--out` directory and never trains/evaluates models.

Exit codes: 0 = artifact checks passed; 2 = recorded check failures;
1 = fatal setup error. A negative research gate is not an artifact failure.

Outputs: `audit.json`, `audit_summary.md`, and, when the bank check succeeds,
`description_review.md`. Keep the old report, old gate and failed runs.

Local validation: `python -B test_audit.py` covers 10 failure-detection and
end-to-end tests. Fixtures contain the original UT C / seed3351
validation log excerpt and eight numeric label/prediction rows, rather than weights.
Synthetic tests do not establish the latest server experiment's correctness.

Limits: no full GloVe rescan or sentence re-encoding, no GPU checkpoint replay,
no AUC/HM computation from top1 predictions, no claim of semantic quality or
statistical significance from checksum/engineering-gate success.

v2 fixes native print(cfg) vs cfg.dump() scalar serialization compatibility and
legacy top5_indices_at_bias0 prediction loading. Real parameter differences and
artifact corruption still fail. v1 outputs and the research gate remain unchanged.
