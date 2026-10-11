# CPF + LOGICZSL B/F actual training and VAL summary

These are reconstructed-knowledge, native CPF results, not original Logic-Troika paper scores.
First-round protocol: 10 epochs, beta=0.01, q=3; F keeps legacy lambda=0.1. No TEST metrics.

| seed | variant | setting | AUC% | HM% | Seen% | Unseen% | best epoch |
|---|---|---|---:|---:|---:|---:|---:|
| 3351 | A_ref | cw | 40.614 | 54.747 | 67.845 | 67.009 | 5 |
| 3351 | A_ref | ow | 24.040 | 37.369 | 67.845 | 49.765 | 5 |
| 3351 | B | cw | 40.259 | 54.425 | 67.161 | 66.709 | 5 |
| 3351 | B | ow | 23.682 | 36.987 | 67.161 | 48.481 | 5 |
| 3351 | E_ref | cw | 37.955 | 54.542 | 66.819 | 65.297 | 6 |
| 3351 | E_ref | ow | 25.140 | 38.107 | 66.819 | 54.857 | 6 |
| 3351 | F | cw | 37.365 | 54.564 | 65.678 | 64.784 | 6 |
| 3351 | F | ow | 24.087 | 36.855 | 65.678 | 52.888 | 6 |

## Paired differences (percentage points)

| seed | setting | B - A_ref AUC | F - E_ref AUC | F - A_ref AUC |
|---|---|---:|---:|---:|
| 3351 | cw | -0.355 | -0.591 | -3.250 |
| 3351 | ow | -0.358 | -1.053 | +0.047 |

## Training evidence / incomplete jobs

- seed=3351 B: 20 logged logic observations (not every batch)
-   last: epoch=10, cpf=2.3213458, oi=0.31572402, oe=0.25647905, ae=0.14271738, weighted=0.0071492037
- seed=3351 F: 20 logged logic observations (not every batch)
-   last: epoch=10, cpf=2.3149185, oi=0.31539065, oe=0.20903024, ae=0.12833206, weighted=0.0065275291

## Interpretation limits

- Positive F-E_ref with F-A_ref <= 0 means logic may help E but does not establish an overall improvement over CPF.
- Single-seed changes do not prove stable improvement. Keep beta/knowledge unchanged before paired seed 0 and 2026 runs.
- Do not compare these native CPF/GloVe VAL values directly against original CLIP-based TEST values.
- The old static calibration/text findings are not evidence that this logic-loss combination succeeds or fails.
