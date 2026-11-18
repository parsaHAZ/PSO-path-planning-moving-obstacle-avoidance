# Human-Driver Motion Prediction from pNEUMA Trajectories: Evaluating Graph-Based Interaction Learning Against Ego-Only Baselines

## Abstract

Reliable prediction of nearby human-driven vehicle motion is a relevant component of autonomous-vehicle planning in dynamic traffic. This study evaluates whether explicitly representing neighboring vehicles with a graph neural network (GNN) improves prediction of recorded vehicle motion over persistence and an ego-only multilayer perceptron (MLP). The evaluation uses the pNEUMA trajectory dataset and separates training days (D1–D7), validation day (D8), and held-out day (D9), with D10 excluded from the strict 80 ms benchmark. The strict 80 ms benchmark reports next-motion prediction errors for speed, longitudinal acceleration, and lateral acceleration. On held-out day D9, the interaction GNN has lower equal-track mean absolute error (MAE) than the ego-only MLP for speed and lateral acceleration, while the longitudinal-acceleration difference is negligible. Track-level bootstrap intervals support a nonzero improvement for speed and lateral acceleration, but not longitudinal acceleration. These results are provisional with respect to a subsequently identified causal-heading preprocessing issue: the existing processed data, graph files, checkpoints, and predictions were not regenerated after the source correction. The study therefore documents the current evidence without claiming that the final corrected pipeline has been validated.

**Keywords:** autonomous driving; motion prediction; pNEUMA; graph neural network; vehicle interaction; trajectory data; held-out evaluation

## 1. Introduction

Autonomous vehicles operate among human-driven vehicles whose observed motion is influenced by traffic conditions and nearby road users. Predicting the next recorded motion of surrounding vehicles can provide an input to downstream planning systems that must reason about moving obstacles. A useful prediction model should be evaluated against simple baselines and on held-out data, rather than judged only by its training loss or by a visually plausible trajectory.

An ego-only model represents a vehicle using its own observed state. An interaction model additionally represents nearby vehicles and their relative states. Graph neural networks provide one way to encode this relational information. However, the added complexity is useful only if it improves held-out prediction under a consistent data split and target definition.

This work investigates the following empirical question:

**Does graph-based neighboring-vehicle information improve prediction of recorded vehicle motion compared with persistence and an ego-only neural baseline?**

The study deliberately limits its claims to prediction of recorded motion. It does not infer driver intent, establish causal effects of interactions, or demonstrate improvements in an autonomous-vehicle planner. Those claims require separate evidence.

## 2. Research questions

1. How do persistence, an ego-only MLP, and an interaction GNN compare on held-out prediction errors?
2. Do the differences remain when evaluation gives each vehicle track equal weight rather than weighting every recorded observation equally?
3. Which target variables show the clearest evidence of improvement, and what limitations constrain interpretation?

## 3. Data and experimental design

### 3.1 Dataset and split

The experiment uses pNEUMA recorded vehicle trajectories, organized into day-specific files. The intended split is by recording day: D1–D7 for training, D8 for validation, and D9–D10 for held-out evaluation. The held-out days are not used for model selection.

The strict benchmark uses a nominal prediction horizon of 80 ms with a strict target-time tolerance. The recorded audit excludes D10 from the strict metric evaluation because its target time interval does not satisfy the configured target-time mask. Consequently, D9 is the held-out day used for the reported strict benchmark comparison and track-level bootstrap analysis. The existence of exported predictions for D10 does not mean D10 is included in these strict benchmark metrics.

### 3.2 Input representation

The graph representation includes vehicle position, speed, longitudinal acceleration, lateral acceleration, and heading represented by sine and cosine components. Edges connect vehicles within the configured spatial radius. Relative edge features encode longitudinal and lateral displacement, distance, and relative speed. The edge direction follows the builder's destination-minus-source convention.

### 3.3 Prediction targets

The three reported targets are next recorded speed, longitudinal acceleration, and lateral acceleration. The models are evaluated as predictors of observed motion at the specified horizon. The task is not treated as a reconstruction of latent driver intention.

### 3.4 Compared models

- **Persistence:** uses the current recorded state as the prediction baseline where applicable.
- **Ego-only MLP:** predicts from the target vehicle's own node features without graph-based neighbor aggregation.
- **Interaction GNN:** uses the graph representation to incorporate neighboring-vehicle information.

The strict training script's recorded configuration uses seed 2026, graph batches of 32, a maximum of 10 epochs, patience of 3, and learning rate of 0.001. These are descriptions of the existing run configuration, not evidence that the run has been repeated after the preprocessing correction.

### 3.5 Metrics and statistical comparison

Prediction quality is summarized with mean absolute error (MAE) and root mean squared error (RMSE). Row-weighted metrics give more influence to tracks with more eligible observations. Equal-track metrics first calculate each track's error and then average across tracks, preventing long tracks from dominating the summary.

The paired track bootstrap resamples held-out tracks with replacement and evaluates the difference between the interaction GNN and ego-only MLP. A negative GNN-minus-MLP difference favors the GNN. The existing bootstrap uses 2,000 resamples and seed 20261023. Its intervals quantify variation across the sampled tracks under this procedure; they do not account for all dataset, preprocessing, or model-selection uncertainty.

## 4. Results

### 4.1 Row-weighted held-out metrics

The table below reproduces the existing metrics file. Values are retained as recorded in the artifact rather than recomputed.

| model           | split | n       | speed_mae   | lon_acc_mae | lat_acc_mae  | speed_rmse  | lon_acc_rmse | lat_acc_rmse |
| --------------- | ----- | ------- | ----------- | ----------- | ------------ | ----------- | ------------ | ------------ |
| persistence     | D8    | 1791600 | 0.034722441 | 0.032780262 | 0.011658958  | 0.061130463 | 0.096925897  | 0.032635366  |
| persistence     | D9    | 1732933 | 0.025775011 | 0.031333903 | 0.0091479241 | 0.06010308  | 0.27090805   | 0.078307098  |
| ego_mlp         | D8    | 1791600 | 0.024780156 | 0.0399593   | 0.014234242  | 0.038047482 | 0.097080481  | 0.033822883  |
| ego_mlp         | D9    | 1732933 | 0.014886669 | 0.03320384  | 0.01055494   | 0.039227095 | 0.26555147   | 0.078303644  |
| interaction_gnn | D8    | 1791600 | 0.026952432 | 0.036887318 | 0.012555636  | 0.040744087 | 0.094101561  | 0.031962572  |
| interaction_gnn | D9    | 1732933 | 0.01932418  | 0.032616698 | 0.010703312  | 0.046969618 | 0.26745702   | 0.078019904  |

### 4.2 Equal-track MAE on held-out day D9

The existing paired-track metrics contain 1,062 held-out tracks. Their equal-track mean MAEs are:

| Model | Speed MAE (m/s) | Longitudinal acceleration MAE (m/s²) | Lateral acceleration MAE (m/s²) |
|---|---:|---:|---:|
| Persistence | 0.035362150 | 0.046184413 | 0.013388335 |
| Ego-only MLP | 0.028708936 | 0.047879957 | 0.014931116 |
| Interaction GNN | 0.026265236 | 0.047856990 | 0.014071890 |

Relative to the ego-only MLP, the interaction GNN's equal-track MAE is lower for speed and lateral acceleration. The longitudinal-acceleration difference is very small. These comparisons describe the existing held-out predictions and should not be interpreted as evidence that graph interactions improve every target.

### 4.3 Paired track-bootstrap uncertainty

For the GNN-minus-MLP equal-track MAE difference, the existing 2,000-resample bootstrap gives:

| Target | Mean paired difference | 95% bootstrap interval | Interpretation of interval |
|---|---:|---:|---|
| Speed | -0.002443700 m/s | [-0.003304298, -0.001595024] | Entire interval below zero |
| Longitudinal acceleration | -0.000022967 m/s² | [-0.000511291, 0.000429397] | Interval includes zero |
| Lateral acceleration | -0.000859227 m/s² | [-0.001054198, -0.000672800] | Entire interval below zero |

Under this track-bootstrap procedure, the equal-track differences favor the interaction GNN for speed and lateral acceleration. The longitudinal-acceleration interval includes zero and therefore does not provide clear evidence of an improvement for that target.

## 5. Discussion

The existing results do not support a blanket claim that graph-based interaction learning improves every motion-prediction target. They suggest a more specific pattern: equal-track speed and lateral-acceleration MAE favor the interaction GNN, whereas the longitudinal-acceleration difference is practically small and its bootstrap interval includes zero.

The row-weighted and equal-track views answer different questions. Row-weighted scores describe average error over eligible observations; equal-track scores describe average performance over tracks. Reporting both helps expose whether a result is driven by a small number of long tracks.

These findings are predictive rather than causal. A performance difference between the ego-only MLP and interaction GNN does not establish that a neighboring vehicle caused a change in a driver's behavior. Nor does lower prediction error alone show that using the model makes an autonomous-vehicle planner safer or more effective. Such claims would require a separate, controlled downstream planning evaluation.

## 6. Data integrity and limitations

### 6.1 Causal-heading preprocessing issue

A later source review identified that the earlier preprocessing implementation interpolated missing heading values within each track. Because the missing values occurred at the first observation of tracks, interpolation could use a later observation to fill an earlier feature. The preprocessing source was subsequently changed to use a neutral zero heading when no previous position is available.

That source edit did **not** automatically regenerate the already-processed CSVs, graph files, model checkpoints, or held-out predictions used for the metrics in this manuscript. Therefore, the existing numerical results and their internal statistical checks should be distinguished from end-to-end validation of the corrected preprocessing pipeline. The reported scores are retained as existing experimental evidence but are provisional with respect to this issue.

The appropriate interpretation is not that every score is necessarily wrong. Rather, the size and direction of any effect on model performance have not been established from the existing artifacts alone.

### 6.2 Graph-path provenance

The graph-building script and training script have been observed to reference different graph directories (`graphs_all_vehicles_80ms` and `graphs_fixed_80ms`, respectively). The exact relationship between those directories and the graph files used for the existing checkpoint must be established before asserting complete end-to-end provenance. This is a provenance limitation, not by itself proof that the recorded metrics were calculated incorrectly.

### 6.3 Temporal coverage and held-out evaluation

The strict benchmark's D10 exclusion means the strict held-out comparison presented here is based on D9. It should not be generalized to D10 or all pNEUMA recording conditions. The existence of a D10 prediction export from another stage does not replace evaluation under the strict benchmark's target-time eligibility criteria.

### 6.4 Scope of claims

This experiment evaluates prediction of recorded motion, not inferred intention or human preference. It does not establish that the model is suitable for safety-critical deployment. The paper also does not claim a downstream planning improvement, because no corresponding planner comparison is reported here.

## 7. Reproducibility

The current run's artifacts are retained in:

- `artifacts/human_driving_validated_80ms/metrics.csv`
- `artifacts/human_driving_validated_80ms/paired_track_metrics.csv`
- `artifacts/human_driving_validated_80ms/paired_track_bootstrap.csv`
- `artifacts/human_driving_validated_80ms/data_audit.json`
- `artifacts/human_driving_validated_80ms/heldout_paired_predictions.csv`

The paired predictions file is large and need not be embedded in the manuscript. The compact metrics, paired-track summary, audit, and bootstrap files should be archived with the code and configuration needed to reproduce the experiment.

The existing validation checks reported for the artifacts include duplicate-key checks, finite-prediction checks, reconciliation of track metrics, and bootstrap reproducibility. These checks support the integrity of the saved evaluation artifacts, but do not independently resolve the later-discovered preprocessing and graph-path provenance questions.

## 8. Conclusion

This report documents an existing pNEUMA held-out motion-prediction experiment comparing persistence, an ego-only MLP, and an interaction GNN. On D9, equal-track MAE favors the GNN for speed and lateral acceleration, while longitudinal acceleration shows no clear difference under the saved track-bootstrap interval. The evidence therefore supports a target-specific conclusion rather than a universal advantage for graph-based prediction.

Because the causal-heading correction has not been propagated through the saved processed data, graph inputs, checkpoints, and predictions, the results must remain provisional with respect to the corrected pipeline. The current findings are useful as a documented baseline and should not be represented as final validation of the corrected preprocessing implementation.

## References

The final submission must include the authoritative citation for the pNEUMA dataset and citations for the specific GNN and baseline architectures used. Those bibliographic details are intentionally not fabricated here. Add the exact dataset citation and model references from the project's existing bibliography before submission.
