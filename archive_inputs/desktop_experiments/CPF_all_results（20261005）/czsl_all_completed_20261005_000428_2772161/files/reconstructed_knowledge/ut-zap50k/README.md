# UT-Zappos LOGICZSL reconstructed knowledge

This directory contains a paper-guided reconstruction of the three primitive
concept trees described by LOGICZSL:

- `oi_dict.json`: object-induction groups;
- `oe_dict.json`: exactly three distinguishing visual texts per object;
- `ae_dict.json`: exactly three visual/semantic extension texts per attribute.

These are **not** the authors' unpublished GPT-4 outputs. Experiments using them
must be identified as `LOGICZSL-reconstructed`, not official LOGICZSL results.

`compatibility.csv` is intentionally not pre-generated. Build it from the exact
pair lists installed on the experiment server:

```bash
python build_compatibility.py /data/xhw/czsl_repro/data/ut-zap50k \
  --output compatibility.csv
```

Then validate all files against the same dataset installation:

```bash
python validate_knowledge.py /data/xhw/czsl_repro/data/ut-zap50k
```

The validator checks exact label coverage and ordering, exactly three extension
texts per primitive, object-induction uniqueness, and the compatibility matrix.

Do not enable the Logic-Troika code path in the already repaired Troika baseline
directory. Preserve that baseline, then create a separate experiment checkout or
copy for the reconstructed logic branch.
