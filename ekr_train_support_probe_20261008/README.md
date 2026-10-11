# EKR TRAIN support probe

This is a read-only CPU diagnostic for the completed failed v2 seed-0 pilot. It uses the
already exported balanced TRAIN features and the frozen reviewed GloVe bank. It performs
no optimizer step, does not load model weights, does not read VAL or TEST features, does
not change either old gate, and does not authorize a new EKR training phase.

Run with the absolute CPF interpreter:

```bash
/data/xhw/miniconda3/envs/cpf/bin/python /tmp/ekr_train_support_probe_20261008/verify_release.py
bash /tmp/ekr_train_support_probe_20261008/run.sh
```

The primary comparison is correct description evidence versus the fixed shuffled-description
control among the attributes available for the same object. Results are also compared with
the attribute-name vector and with a cross-fitted visual-prototype reference. The two folds
are deterministic three-image halves of each reviewed six-image TRAIN pair.

Read:

```bash
cat /var/tmp/czsl_xhw_ekr_train_support_20261008/decision.json
cat /var/tmp/czsl_xhw_ekr_train_support_20261008/summary.csv
```

Download:

```powershell
scp "xhw@211.87.232.122:/var/tmp/czsl_xhw_ekr_train_support_20261008_review.zip" "C:/Users/llenovo/Desktop/UT_description_review/"
```

`support_screen_passed` is an operational diagnostic, not a paper result or a statistical
claim. A positive result only permits designing a separate bounded fixed-value pilot. A
negative result means blind tuning of this GloVe score-residual route should stop. The
random semantic graph route is outside this bundle.
