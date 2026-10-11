# SG-EKR seed-0 pilot

This release runs a bounded support-gated fixed-value EKR pilot on the reviewed UT-Zappos
artifacts. It preserves the old name pilot, failed v2 result, source tree, banks, checkpoints,
and gates. It does not use the random semantic graph route.

The prepare phase validates every pinned input, reconstructs a deterministic 3/3 image split
inside each six-image TRAIN pair, computes object-conditioned support for every AE slot, and
freezes slots whose support is at least 0.005 on both halves. VAL and TEST are not used to
construct the mask.

The three matched variants are `G_name`, `G_support`, and `G_shuffle`. All use fixed raw values
and train only 28,834 query/key/null/gain parameters for three epochs. Epoch zero is always a
selection candidate. The single-seed operational threshold is +0.10 CW AUC percentage points
over the common name anchor, the matched name router, and the matched shuffle router.

Run phases separately after extraction and release verification:

```bash
bash /tmp/sgekr_support_mask_20261008/run.sh prepare
bash /tmp/sgekr_support_mask_20261008/run.sh train
bash /tmp/sgekr_support_mask_20261008/run.sh report
```

If training is interrupted after at least one committed epoch:

```bash
bash /tmp/sgekr_support_mask_20261008/run.sh resume
bash /tmp/sgekr_support_mask_20261008/run.sh report
```

Read the outcome:

```bash
cat /var/tmp/czsl_xhw_sgekr_20261008/reports/decision.json
cat /var/tmp/czsl_xhw_sgekr_20261008/reports/results.csv
cat /var/tmp/czsl_xhw_sgekr_20261008/support_mask.json
```

The report phase creates `/var/tmp/czsl_xhw_sgekr_20261008_review.zip`. The archive excludes
weights; retain all server checkpoints. Do not run the old stage C even if this pilot passes.
A passing result only permits a separately frozen router-seed confirmation protocol.
