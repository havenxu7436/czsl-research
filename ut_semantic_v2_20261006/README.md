# UT semantic v2 — reviewed dictionary and matched experiment package

Release date: 2026-10-06. User-authorized assistant review; NOT human expert approval.

All 28 TRAIN contact sheets / 84 sentences were reviewed: 30 kept, 54 rewritten.
The knowledge folder contains the v2 AE/OE dictionaries, unchanged OI/labels, and
per-sentence evidence, exceptions, terminology sources, and review provenance.

Read RUN_UT_V2.md for complete Chinese Windows/server commands and formulas.
Default output workspace: /var/tmp/czsl_xhw_ut_v2_20261006 (avoids full /data).

1. Verify upload and extract this bundle; `python verify_release.py`.
2. `bash prepare.sh`: immutable source/data/GloVe lineage checks, all TRAIN image
   decode checks, real v2 cache/bank construction, 12 frozen configs.
3. Inside tmux with a free GPU selected: `bash run_paired.sh`.
   Real GPU smoke precedes 12 fresh native CPF runs (4 controls x 3 seeds x 10 epochs).
   The script audits, summarizes and writes a new research gate automatically.
4. ONLY if the v2 gate passes: `bash enter_ekr_pilot.sh` for R0/R3/R5/R7 seed0,
   3 epochs, frozen v2 C anchor, in a separate experiment directory.

Static K calibration stays off. No test image evaluation. No USG semantic graph.
V1 knowledge/results/gates are preserved and never included as new v2 results.
No manual JSON/YAML edits are needed. Stop rather than bypass an integrity failure.

Local tests use real week5 source and UT metadata, SYNTHETIC GloVe and audit
fixtures. Real server embeddings, images, CUDA, and training are not claimed tested.
See release_check.json and release_tests/v2_cpu.log for the exact scope.

review_evidence/selected_train_images.json and review_template.md preserve the
ORIGINAL v1 export context. Completed review is knowledge/ut-zap50k/item_review.json;
the authoritative revised text is knowledge/ut-zap50k/ae_dict.json and oe_dict.json.
reference/, older test logs and reference/legacy_test_release_integration.py are historical
v1 supporting material, not v2 performance results.
