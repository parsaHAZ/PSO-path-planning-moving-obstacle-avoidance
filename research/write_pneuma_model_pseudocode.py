from pathlib import Path

root = Path.cwd()
out = root / "reports" / "pneuma_model_pseudocode.md"
out.parent.mkdir(parents=True, exist_ok=True)

text = r"""# Model Pseudocode: pNEUMA Human-Driver Motion Prediction

## 1. Task definition and notation

For a target vehicle \(i\), use its observed state and, for the interaction
model, the observed states of nearby vehicles to predict the next recorded
motion target:

\[
\hat{\mathbf y}_i =
[\hat v_i,\widehat{a}_{x,i},\widehat{a}_{y,i}]
\]

where the three outputs correspond to speed, longitudinal acceleration, and
lateral acceleration.

The graph node feature vector contains seven features:

\[
\mathbf x_i =
[x_i,y_i,v_i,a_{x,i},a_{y,i},\sin(\psi_i),\cos(\psi_i)]
\]

Each directed edge contains four relative features:

\[
\mathbf e_{ji} =
[\Delta x_{ji},\Delta y_{ji},d_{ji},\Delta v_{ji}]
\]

The graph builder uses destination-minus-source relative quantities. Thus,
for an edge from neighbor \(j\) to target \(i\), the relative quantities are
defined as target \(i\) minus neighbor \(j\).

The following algorithms describe the experimental model families. The exact
layer widths, activations, aggregation operators, normalization, and loss
function must match the implementation in the training script before this
document is used as a layer-by-layer specification.

## 2. Model A — Persistence baseline

**Purpose:** provide a simple reference that does not learn a mapping from
training data.

```text
ALGORITHM PersistencePrediction

INPUT:
    Current observed state of target vehicle i
    Current speed v_i
    Current longitudinal acceleration a_x_i
    Current lateral acceleration a_y_i

OUTPUT:
    Predicted next-state target y_hat_i

BEGIN
    predicted_speed <- current_speed
    predicted_longitudinal_acceleration <- current_longitudinal_acceleration
    predicted_lateral_acceleration <- current_lateral_acceleration

    y_hat_i <- [
        predicted_speed,
        predicted_longitudinal_acceleration,
        predicted_lateral_acceleration
    ]

    RETURN y_hat_i
END
```

Persistence assumes the target variables remain unchanged over the prediction
horizon. It has no trainable parameters.

## 3. Model B — Ego-only multilayer perceptron

**Purpose:** learn a nonlinear mapping from the target vehicle's own observed
features to its next recorded motion, without aggregating neighbor information.

```text
ALGORITHM EgoOnlyMLP

INPUT:
    Target vehicle feature vector x_i
    Trained MLP parameters theta_ego

OUTPUT:
    Predicted next recorded motion y_hat_i

BEGIN
    h <- x_i

    FOR each hidden layer l in the implemented MLP:
        h <- implemented_layer_l(h)
        h <- implemented_activation_l(h)
    END FOR

    y_hat_i <- implemented_output_layer(h)

    RETURN y_hat_i
END
```

Training procedure:

```text
ALGORITHM TrainEgoOnlyMLP

INPUT:
    Training examples from days D1-D7
    Validation examples from day D8
    Implemented optimizer, loss, and training configuration

BEGIN
    Initialize MLP parameters theta_ego

    FOR each training epoch:
        FOR each training batch:
            x_batch, y_batch <- obtain ego features and recorded targets
            y_hat_batch <- EgoOnlyMLP(x_batch, theta_ego)
            loss <- implemented_training_loss(y_hat_batch, y_batch)

            Compute gradients of loss with respect to theta_ego
            Update theta_ego using the implemented optimizer
        END FOR

        Evaluate the model on validation day D8
        Apply the implemented checkpoint-selection and early-stopping rules
    END FOR

    RETURN selected ego-only checkpoint
END
```

The actual implementation determines the number of layers, hidden dimensions,
activations, loss, and any target normalization. These must be reported from
the code rather than inferred from this high-level pseudocode.

## 4. Model C — Interaction graph neural network

**Purpose:** use the target vehicle's own features and graph-connected nearby
vehicles to predict the target's next recorded motion.

Let \(G=(V,E)\) denote a traffic graph. Each node represents a vehicle; an
edge represents a neighbor relationship established by the graph builder.

```text
ALGORITHM InteractionGNN

INPUT:
    Node features X
    Directed edge index E
    Edge attributes A
    Index i of target vehicle
    Trained GNN parameters theta_gnn

OUTPUT:
    Predicted next recorded motion y_hat_i

BEGIN
    H <- X

    FOR each graph/message-passing layer l in the implemented GNN:
        FOR each directed edge (j -> k):
            Compute the implemented edge/message representation
                from source representation H_j,
                destination representation H_k,
                and edge attributes A_jk
        END FOR

        Aggregate incoming messages for each destination node
        Update node representations using the implemented layer
        Apply any activation, normalization, or residual operation
            specified by the implementation
    END FOR

    z_i <- final representation of target node i
    y_hat_i <- implemented prediction head(z_i)

    RETURN y_hat_i
END
```

Training procedure:

```text
ALGORITHM TrainInteractionGNN

INPUT:
    Training traffic graphs from days D1-D7
    Validation graphs from day D8
    Recorded next-motion targets
    Implemented optimizer, loss, and training configuration

BEGIN
    Initialize GNN parameters theta_gnn

    FOR each training epoch:
        FOR each batch of traffic graphs:
            Load node features, edge indices, edge attributes, and targets
            Compute predictions with InteractionGNN
            Calculate the implemented training loss
            Compute gradients
            Update theta_gnn using the implemented optimizer
        END FOR

        Evaluate the checkpoint on validation day D8
        Apply the implemented checkpoint-selection and early-stopping rules
    END FOR

    RETURN selected interaction-GNN checkpoint
END
```

Unlike the ego-only model, the interaction GNN can represent neighboring-vehicle
information through message passing. Whether this representation improves
held-out accuracy is an empirical question evaluated using the same target
definition and held-out examples.

## 5. Evaluation protocol

```text
ALGORITHM ComparePredictionModels

INPUT:
    Selected persistence, ego-only MLP, and interaction-GNN predictors
    Eligible held-out examples from D9
    Recorded next-motion targets

BEGIN
    FOR each eligible held-out example:
        Obtain predictions from all three models
        Store predictions alongside the recorded target and track identifier
    END FOR

    FOR each model and each target variable:
        Compute row-weighted MAE
        Compute row-weighted RMSE
    END FOR

    FOR each held-out vehicle track:
        Compute each model's per-track MAE for every target variable
    END FOR

    Compute equal-track mean MAE for each model

    Resample held-out tracks with replacement for the paired bootstrap
    For every resample:
        Compute equal-track MAE difference:
            interaction_GNN minus ego_only_MLP
    Summarize the bootstrap distribution and confidence interval

    RETURN saved metrics, paired predictions, and bootstrap summaries
END
```

The existing benchmark's paired bootstrap uses 2,000 resamples and seed
20261023. The strict benchmark's reported held-out metrics use D9; D10 is
excluded from those strict metrics by the target-time eligibility mask.

## 6. Comparison of model families

| Property | Persistence | Ego-only MLP | Interaction GNN |
|---|---|---|---|
| Trainable parameters | No | Yes | Yes |
| Uses target vehicle's own state | Yes | Yes | Yes |
| Uses neighboring vehicles explicitly | No | No | Yes |
| Learns nonlinear mapping | No | Yes | Yes |
| Learns graph-based aggregation | No | No | Yes |
| Evaluated against recorded motion | Yes | Yes | Yes |

## 7. Implementation fidelity checklist

Before treating this pseudocode as an exact implementation description,
confirm the following directly against the actual training code:

- [ ] Exact node-feature order and units.
- [ ] Exact edge-feature order, sign convention, and neighbor radius.
- [ ] Exact GNN message-passing layer type and number of layers.
- [ ] Exact hidden dimensions and activation functions.
- [ ] Exact ego-only MLP architecture.
- [ ] Exact prediction-head dimensions.
- [ ] Exact loss function and target normalization.
- [ ] Exact optimizer, learning rate, batch size, epoch limit, and early stopping.
- [ ] Exact checkpoint used to generate the saved held-out predictions.
- [ ] Exact graph directory from which the checkpoint's training data originated.

## 8. Interpretation and scope

These models predict recorded vehicle motion. Their outputs should not be
described as inferred driver intentions, causal responses to neighbors, or
human preferences. Prediction performance alone does not establish downstream
autonomous-vehicle planning or safety improvements.

The existing results were generated before the corrected causal-heading
preprocessing was propagated through processed data, graphs, checkpoints, and
predictions. Consequently, the numerical results remain provisional with
respect to that preprocessing correction. This does not erase the existing
evaluation checks, but it limits claims of end-to-end validation of the
corrected pipeline.
"""

out.write_text(text, encoding="utf-8")
print("Created:", out)
print("Bytes:", out.stat().st_size)
