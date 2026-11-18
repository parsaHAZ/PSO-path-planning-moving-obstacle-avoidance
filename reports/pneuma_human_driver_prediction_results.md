# Human-Driver Motion Prediction: Experimental Results

> **Status: provisional results report ? not yet submission-ready.** The metrics below were produced using the previous heading preprocessing. The preprocessing has since been corrected to avoid future-derived first-observation headings, but the strict 80 ms benchmark has not yet been rerun with the corrected features. Do not present these metrics as results from the corrected pipeline.

## 1. Scope

This report summarizes existing pNEUMA trajectory-prediction experiments
for the project:

*Human-Preference-Guided Trajectory Planning for Autonomous Vehicles in
Dynamic Traffic Using Graph Neural Networks and Multi-Objective Particle
Swarm Optimization.*

The immediate research question is whether incorporating neighboring
vehicles through a graph neural network improves prediction of recorded
vehicle motion over simpler baselines. The predicted variables are next
speed, longitudinal acceleration, and lateral acceleration.

These experiments concern recorded-motion prediction. They do not, by
themselves, establish causal human preferences, driver intent, or improved
autonomous-vehicle safety.

## 2. Data split and evaluation

For the strict 80 ms benchmark, the audited day-level split is:

- Training: D1–D7, using valid approximately 80 ms targets only
- Validation: D8
- Held-out testing: D9
- D10 is excluded because its saved target interval is approximately 83.417 ms.

Other experiments below use different target construction, horizons, or evaluation procedures and must be interpreted separately.

The original graph pipeline uses a 40 m interaction radius and retains at
most 80 vehicles per timestamp, selected by track ID after sorting. The
existing benchmark therefore evaluates a truncated subset of each traffic
scene, not every recorded vehicle.

The reported benchmark uses next recorded speed and acceleration targets.
Metrics below are mean absolute error (MAE) and root mean squared error
(RMSE). Lower values are better.

## 3. Strict 80 ms capped-graph benchmark

### 3.1 Row-weighted validation and held-out test metrics

| Split | Model | Rows | Speed MAE | Longitudinal acceleration MAE | Lateral acceleration MAE | Speed RMSE | Longitudinal acceleration RMSE | Lateral acceleration RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D8 | Persistence | 1,791,600 | 0.034722 | 0.032780 | 0.011659 | 0.061130 | 0.096926 | 0.032635 |
| D8 | Ego-only MLP | 1,791,600 | 0.024780 | 0.039959 | 0.014234 | 0.038047 | 0.097080 | 0.033823 |
| D8 | Interaction GNN | 1,791,600 | 0.026952 | 0.036887 | 0.012556 | 0.040744 | 0.094102 | 0.031963 |
| D9 | Persistence | 1,732,933 | 0.025775 | 0.031334 | 0.009148 | 0.060103 | 0.270908 | 0.078307 |
| D9 | Ego-only MLP | 1,732,933 | 0.014887 | 0.033204 | 0.010555 | 0.039227 | 0.265551 | 0.078304 |
| D9 | Interaction GNN | 1,732,933 | 0.019324 | 0.032617 | 0.010703 | 0.046970 | 0.267457 | 0.078020 |

**Interpretation.** On D9, the ego-only MLP has lower row-weighted speed MAE than the
interaction GNN. The GNN has slightly lower longitudinal-acceleration MAE
than the ego-only model, but higher lateral-acceleration MAE. It is therefore
not accurate to claim that the GNN consistently outperforms the simpler model.

D8 results show a similar trade-off: the GNN improves the two acceleration
RMSE values relative to the ego-only MLP, while the ego-only MLP has better
speed MAE and speed RMSE.

### 3.2 Equal-track results on the strict 80 ms D9 test

The following values are calculated from
`artifacts/human_driving_validated_80ms/paired_track_metrics.csv`.
Each of the 1,062 D9 tracks contributes one track-level MAE per target
and model, and the reported value is the arithmetic mean across tracks.

| Split | Model | Speed MAE | Longitudinal acceleration MAE | Lateral acceleration MAE |
|---|---|---:|---:|---:|
| D9 | Persistence | 0.035362 | 0.046184 | 0.013388 |
| D9 | Ego-only MLP | 0.028709 | 0.047880 | 0.014931 |
| D9 | Interaction GNN | 0.026265 | 0.047857 | 0.014072 |

Under equal-track weighting, the interaction GNN has lower mean speed
and lateral-acceleration MAE than the ego-only MLP. The longitudinal-
acceleration means are nearly identical (0.047857 versus 0.047880).
These equal-track metrics differ from the row-weighted metrics in
Section 3.1 because each track receives equal weight here, regardless
of how many valid observations it contains.

D10 equal-track values are not included in this strict benchmark:
D10's saved target interval is approximately 83.417 ms, outside the
benchmark's 80 ms tolerance. Do not interpret D10 results from other
experiments as strict 80 ms test results.

### 3.3 Track-level bootstrap comparison on D9

Previously computed differences in mean-track MAE (GNN minus ego-only MLP):

| Target | Difference | 95% bootstrap interval |
|---|---:|---:|
| Speed | -0.002444 | [-0.003304, -0.001595] |
| Longitudinal acceleration | -0.000023 | [-0.000511, +0.000429] |
| Lateral acceleration | -0.000859 | [-0.001054, -0.000673] |

Negative differences favor the GNN. The speed and lateral-acceleration
intervals exclude zero in this reported comparison; the longitudinal-
acceleration interval includes zero. These results should be accompanied
by the exact bootstrap procedure, number of resamples, and track inclusion
rules from the analysis script before publication.

## 4. Separate interaction-regression experiment

A separate experiment compared ego-only ridge regression against
interaction-feature ridge regression on held-out recorded motion.

Previously reported pooled MAEs:

| Model | Speed MAE | Longitudinal acceleration MAE | Lateral acceleration MAE |
|---|---:|---:|---:|
| Ego-only ridge | 0.005141 | 0.021129 | 0.004209 |
| Interaction ridge | 0.005056 | 0.017387 | 0.004178 |

These results suggest that the selected interaction features may provide
useful predictive information, especially for longitudinal acceleration.
They are from a separate experiment and evaluation setup; they must not be
combined numerically with the capped-graph GNN benchmark as though they
were the same test.

A previously reported track-bootstrap difference for longitudinal-
acceleration MAE (interaction ridge minus ego-only ridge) was -0.003742
pooled across D9 and D10. The day-specific differences were -0.000374
for D9 and -0.005578 for D10. The pooled value alone does not demonstrate
that the effect is equally strong on both days.

## 5. Fixed-horizon experiment

A separate 200 ms horizon experiment reported:

| Metric | Ego-only ridge | Interaction ridge |
|---|---:|---:|
| Speed MAE | 0.011431 | 0.011412 |
| Longitudinal acceleration MAE | 0.098144 | 0.097417 |
| Lateral acceleration MAE | 0.031312 | 0.031847 |

The reported track-bootstrap difference for longitudinal-acceleration
MAE (interaction minus ego-only) was -0.000727, with interval
[-0.001061, -0.000391].

This is evidence of a modest improvement for longitudinal acceleration
in that experiment, not evidence of improvement in every target. The
lateral-acceleration MAE was slightly worse for the interaction model.

The 200 ms results must remain separate from the 80 ms benchmark because
the prediction horizons and experiment configurations differ.

## 6. Dataset and validity limitations

### 6.1 Vehicle cap

The original graph files cap each timestamp at 80 vehicles. A source-to-
graph audit found that 209,780 of 236,065 timestamps exceeded the cap.
Across the ten days, 21,075,422 source vehicle observations were omitted;
17,132,870 omitted observations were within 40 m of a retained vehicle.

Consequently, the original GNN benchmark cannot establish performance
when all nearby vehicles are represented. The cap may remove relevant
neighbors and changes the interaction graph supplied to the model.

### 6.2 Target and split checks

A saved-target audit compared sampled saved labels against future
same-track observations and found zero difference for the checked pairs.
This was a sampled audit, not a proof that every saved target is correct.
The strict 80 ms audit reports selected valid targets at approximately 0.08 s for training, validation, and D9 test, within a 0.001 s tolerance. D10 is excluded because its saved target interval is approximately 0.083417 s. This interval check alone does not independently verify every label against raw trajectories or rule out all future-information leakage.

The day-level split separates recording days, which is preferable to
randomly splitting individual observations from the same tracks. The
training, validation, and test boundaries, preprocessing, feature
normalization, checkpoint selection, and all filtering rules should be
confirmed from the executable evaluation pipeline before publication.

### 6.3 Metrics and units

Speed errors are in m/s. Acceleration errors are in m/s². Row-weighted
and equal-track metrics answer different questions and must be labeled
separately. RMSE can be disproportionately affected by large errors;
the relatively high held-out longitudinal-acceleration RMSE warrants
additional inspection of outliers and data quality.

### 6.4 Interpretation

Predicting recorded motion is not equivalent to identifying a driver's
intent, preferences, or causal response to a neighbor. The current
results support only predictive comparisons under the tested dataset,
features, horizons, and evaluation procedure.

## 7. Conclusions supported by the existing results

1. On the strict 80 ms D9 test, the interaction GNN has worse speed MAE than the ego-only MLP, slightly lower longitudinal-acceleration MAE, and slightly worse lateral-acceleration MAE.
2. On strict 80 ms D9, equal-track mean MAE favors the GNN for speed and lateral acceleration; longitudinal-acceleration mean MAE is effectively tied.
3. A separate interaction-ridge experiment reports lower pooled MAE for
   all three targets, with the largest reported difference for
   longitudinal acceleration. Its setup must be described separately.
4. The original graph cap is a material limitation because it removes
   many vehicles, including many within the interaction radius.
5. These prediction results alone do not establish better obstacle
   avoidance, safer planning, inferred human preferences, or causal
   driver behavior.

## 8. Required checks before submission

- Reproduce all reported metrics from the saved predictions and evaluation
  scripts.
- Verify that baseline and GNN metrics use identical held-out observations.
- Confirm identical target rows across models in the strict 80 ms benchmark; the audited D9 count is 1,732,933 valid targets.
- Document the equal-track aggregation and track-bootstrap implementation.
- Audit all target timestamps and confirm no future-derived feature leaks
  into model inputs.
- Clearly distinguish the strict 80 ms D9 benchmark from separate equal-track, interaction-regression, and 200 ms horizon experiments.
- Treat the uncapped graph build as unfinished until it has been built,
  independently audited, and evaluated.
- Report downstream planning results separately from human-motion
  prediction results.

---

**Reproducibility note:** This report records previously available result
tables and audits. It does not rerun experiments. All values should be
reproduced from the corresponding saved artifacts before submission.
