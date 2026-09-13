# WP T10 mismatch diagnostics (handoff to WP T7)

Gap = observed rejection rate - Gaussian-shift prediction. SE(gap) = sqrt(SE_obs^2 + SE_pred^2), where SE_obs is the binomial MC SE of the empirical rate and SE_pred is the MC SE of the 20,000-draw Gaussian-shift power. A row is flagged when |gap| > 2.0 SE(gap); the pre-registered tolerance is 0.06 absolute.

Scored rows: 51; over tolerance: 9; flagged at 2 SE: 17; max |gap| = 0.275.

Full-fleet verdict: confirmed=False.
Recorded (partial) verdict confirmed=False.

| Setting | Direction | n | reps | Test | Observed | Predicted | Gap | SE(gap) | z | 2SE? |
|---|---|---|---|---|---|---|---|---|---|---|
| NONPROG0 | bump | 200 | 300 | dr | 0.417 | 0.395 | +0.022 | 0.0287 | 0.77 | no |
| NONPROG0 | bump | 200 | 300 | unadjusted | 0.567 | 0.556 | +0.010 | 0.0288 | 0.35 | no |
| NONPROG0 | bump | 500 | 300 | dr | 0.410 | 0.420 | -0.010 | 0.0286 | -0.35 | no |
| NONPROG0 | bump | 500 | 300 | unadjusted | 0.550 | 0.556 | -0.006 | 0.0289 | -0.22 | no |
| NONPROG0 | freq | 200 | 300 | dr | 0.877 | 0.816 | +0.060 | 0.0192 | 3.14 | YES |
| NONPROG0 | freq | 200 | 300 | unadjusted | 0.953 | 0.985 | -0.032 | 0.0122 | -2.63 | YES |
| NONPROG0 | freq | 500 | 300 | dr | 0.897 | 0.866 | +0.031 | 0.0177 | 1.75 | no |
| NONPROG0 | freq | 500 | 300 | unadjusted | 0.957 | 0.984 | -0.028 | 0.0118 | -2.34 | YES |
| NONPROG0 | mean | 200 | 300 | dr | 0.563 | 0.580 | -0.017 | 0.0288 | -0.59 | no |
| NONPROG0 | mean | 200 | 300 | unadjusted | 0.753 | 0.751 | +0.002 | 0.0251 | 0.08 | no |
| NONPROG0 | mean | 500 | 300 | dr | 0.567 | 0.609 | -0.043 | 0.0288 | -1.48 | no |
| NONPROG0 | mean | 500 | 300 | unadjusted | 0.707 | 0.749 | -0.042 | 0.0265 | -1.61 | no |
| NONPROG1 | bump | 200 | 300 | dr | 0.280 | 0.269 | +0.011 | 0.0261 | 0.42 | no |
| NONPROG1 | bump | 200 | 300 | unadjusted | 0.467 | 0.512 | -0.046 | 0.0290 | -1.57 | no |
| NONPROG1 | bump | 500 | 300 | dr | 0.293 | 0.291 | +0.002 | 0.0265 | 0.07 | no |
| NONPROG1 | bump | 500 | 300 | unadjusted | 0.553 | 0.511 | +0.042 | 0.0289 | 1.47 | no |
| NONPROG1 | freq | 200 | 300 | dr | 0.673 | 0.496 | +0.177 | 0.0273 | 6.50 | YES |
| NONPROG1 | freq | 200 | 300 | unadjusted | 0.913 | 0.959 | -0.046 | 0.0163 | -2.80 | YES |
| NONPROG1 | freq | 500 | 300 | dr | 0.723 | 0.633 | +0.090 | 0.0261 | 3.45 | YES |
| NONPROG1 | freq | 500 | 300 | unadjusted | 0.930 | 0.960 | -0.030 | 0.0148 | -2.00 | no |
| NONPROG1 | mean | 200 | 300 | dr | 0.457 | 0.416 | +0.040 | 0.0290 | 1.40 | no |
| NONPROG1 | mean | 200 | 300 | unadjusted | 0.673 | 0.706 | -0.032 | 0.0273 | -1.18 | no |
| NONPROG1 | mean | 500 | 300 | dr | 0.463 | 0.451 | +0.012 | 0.0290 | 0.42 | no |
| NONPROG1 | mean | 500 | 300 | unadjusted | 0.763 | 0.703 | +0.060 | 0.0248 | 2.43 | YES |
| PROG0 | bump | 100 | 300 | dr | 0.390 | 0.371 | +0.019 | 0.0284 | 0.67 | no |
| PROG0 | bump | 100 | 300 | unadjusted | 0.040 | 0.060 | -0.020 | 0.0114 | -1.79 | no |
| PROG0 | bump | 200 | 300 | dr | 0.387 | 0.387 | -0.001 | 0.0283 | -0.02 | no |
| PROG0 | bump | 200 | 300 | unadjusted | 0.063 | 0.061 | +0.003 | 0.0142 | 0.18 | no |
| PROG0 | bump | 500 | 300 | dr | 0.397 | 0.410 | -0.014 | 0.0285 | -0.48 | no |
| PROG0 | bump | 500 | 300 | unadjusted | 0.050 | 0.061 | -0.011 | 0.0127 | -0.84 | no |
| PROG0 | freq | 100 | 300 | dr | 0.820 | 0.748 | +0.072 | 0.0224 | 3.20 | YES |
| PROG0 | freq | 100 | 300 | unadjusted | 0.063 | 0.099 | -0.036 | 0.0142 | -2.54 | YES |
| PROG0 | freq | 200 | 300 | dr | 0.853 | 0.806 | +0.048 | 0.0206 | 2.31 | YES |
| PROG0 | freq | 200 | 300 | unadjusted | 0.083 | 0.100 | -0.016 | 0.0161 | -1.02 | no |
| PROG0 | freq | 500 | 300 | dr | 0.917 | 0.867 | +0.050 | 0.0161 | 3.10 | YES |
| PROG0 | freq | 500 | 300 | unadjusted | 0.083 | 0.100 | -0.016 | 0.0161 | -1.01 | no |
| PROG0 | mean | 100 | 300 | dr | 0.550 | 0.581 | -0.031 | 0.0289 | -1.07 | no |
| PROG0 | mean | 100 | 300 | unadjusted | 0.050 | 0.076 | -0.026 | 0.0127 | -2.04 | YES |
| PROG0 | mean | 200 | 300 | dr | 0.570 | 0.594 | -0.024 | 0.0288 | -0.84 | no |
| PROG0 | mean | 200 | 300 | unadjusted | 0.057 | 0.076 | -0.020 | 0.0135 | -1.46 | no |
| PROG0 | mean | 500 | 300 | dr | 0.623 | 0.628 | -0.005 | 0.0282 | -0.17 | no |
| PROG0 | mean | 500 | 300 | unadjusted | 0.047 | 0.076 | -0.029 | 0.0123 | -2.36 | YES |
| PROG1 | bump | 100 | 300 | dr | 0.277 | 0.257 | +0.020 | 0.0260 | 0.76 | no |
| PROG1 | bump | 200 | 300 | dr | 0.270 | 0.270 | -0.000 | 0.0258 | -0.00 | no |
| PROG1 | bump | 500 | 300 | dr | 0.277 | 0.285 | -0.008 | 0.0260 | -0.32 | no |
| PROG1 | freq | 100 | 300 | dr | 0.637 | 0.407 | +0.230 | 0.0280 | 8.20 | YES |
| PROG1 | freq | 200 | 300 | dr | 0.697 | 0.560 | +0.136 | 0.0268 | 5.09 | YES |
| PROG1 | freq | 500 | 300 | dr | 0.730 | 0.628 | +0.102 | 0.0259 | 3.94 | YES |
| PROG1 | mean | 100 | 300 | dr | 0.460 | 0.185 | +0.275 | 0.0289 | 9.52 | YES |
| PROG1 | mean | 200 | 300 | dr | 0.427 | 0.448 | -0.021 | 0.0288 | -0.73 | no |
| PROG1 | mean | 500 | 300 | dr | 0.410 | 0.450 | -0.040 | 0.0286 | -1.40 | no |

Rows over the 0.06 absolute tolerance:
- NONPROG0/freq/n=200 (dr): observed 0.877, predicted 0.816, gap +0.060, SE 0.0192.
- NONPROG1/freq/n=200 (dr): observed 0.673, predicted 0.496, gap +0.177, SE 0.0273.
- NONPROG1/freq/n=500 (dr): observed 0.723, predicted 0.633, gap +0.090, SE 0.0261.
- NONPROG1/mean/n=500 (unadjusted): observed 0.763, predicted 0.703, gap +0.060, SE 0.0248.
- PROG0/freq/n=100 (dr): observed 0.820, predicted 0.748, gap +0.072, SE 0.0224.
- PROG1/freq/n=100 (dr): observed 0.637, predicted 0.407, gap +0.230, SE 0.0280.
- PROG1/freq/n=200 (dr): observed 0.697, predicted 0.560, gap +0.136, SE 0.0268.
- PROG1/freq/n=500 (dr): observed 0.730, predicted 0.628, gap +0.102, SE 0.0259.
- PROG1/mean/n=100 (dr): observed 0.460, predicted 0.185, gap +0.275, SE 0.0289.

Rows beyond 2 SE(gap):
- NONPROG0/freq/n=200 (dr): z=3.14, gap +0.060.
- NONPROG0/freq/n=200 (unadjusted): z=-2.63, gap -0.032.
- NONPROG0/freq/n=500 (unadjusted): z=-2.34, gap -0.028.
- NONPROG1/freq/n=200 (dr): z=6.50, gap +0.177.
- NONPROG1/freq/n=200 (unadjusted): z=-2.80, gap -0.046.
- NONPROG1/freq/n=500 (dr): z=3.45, gap +0.090.
- NONPROG1/mean/n=500 (unadjusted): z=2.43, gap +0.060.
- PROG0/freq/n=100 (dr): z=3.20, gap +0.072.
- PROG0/freq/n=100 (unadjusted): z=-2.54, gap -0.036.
- PROG0/freq/n=200 (dr): z=2.31, gap +0.048.
- PROG0/freq/n=500 (dr): z=3.10, gap +0.050.
- PROG0/mean/n=100 (unadjusted): z=-2.04, gap -0.026.
- PROG0/mean/n=500 (unadjusted): z=-2.36, gap -0.029.
- PROG1/freq/n=100 (dr): z=8.20, gap +0.230.
- PROG1/freq/n=200 (dr): z=5.09, gap +0.136.
- PROG1/freq/n=500 (dr): z=3.94, gap +0.102.
- PROG1/mean/n=100 (dr): z=9.52, gap +0.275.
