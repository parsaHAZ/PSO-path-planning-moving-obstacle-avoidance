# Human-Driven Trajectory Prediction: History and Traffic-Context Evaluation

Generated: 2026-10-20T02:04:15+03:30

## Abstract

This report consolidates existing experiments on predicting human-driven vehicle motion from pNEUMA trajectories. It examines temporal history, a shuffled-history control, and whether the relative predictive performance of ego-history and interaction-history models varies with graph-based local traffic context. D1-D7 are used for training, D8 for validation, and D9-D10 as held-out drones. Results are predictive associations, not evidence of causal effects, driver intentions, or human preference labels.

## Data and protocol

- Dataset: pNEUMA, 2018-11-01, 08:00-08:30.
- Split: D1-D7 train; D8 validation; D9-D10 held-out evaluation.
- Prediction horizons: 200, 500, and 1000 ms.
- Targets: speed, longitudinal acceleration, and lateral acceleration.
- Graph connectivity radius: 40 m.
- Broad context robustness grouping: sparse degree 0-5; dense degree 6+.
- Uncertainty: paired track-cluster bootstrap where reported.

## History prediction metrics

These metrics compare models on held-out predictions. MAE and RMSE are reported in the units of each target. Interpret comparisons by target, horizon, and drone.

Source: `artifacts/human_driving_history_v1/all_horizons_metrics.csv`

| Horizon (ms) | Drone | Target | Model | N | MAE | RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| 200 | D9 | next_speed_mps | ego_snapshot | 1730556 | 0.0098 | 0.0969 |
| 200 | D9 | next_speed_mps | interaction_snapshot | 1730556 | 0.0097 | 0.0969 |
| 200 | D9 | next_speed_mps | ego_history | 1730556 | 0.0055 | 0.0727 |
| 200 | D9 | next_speed_mps | interaction_history | 1730556 | 0.0055 | 0.0727 |
| 200 | D9 | next_lon_acc | ego_snapshot | 1730556 | 0.0885 | 0.4721 |
| 200 | D9 | next_lon_acc | interaction_snapshot | 1730556 | 0.0871 | 0.4719 |
| 200 | D9 | next_lon_acc | ego_history | 1730556 | 0.0551 | 0.5378 |
| 200 | D9 | next_lon_acc | interaction_history | 1730556 | 0.0550 | 0.5373 |
| 200 | D9 | next_lat_acc | ego_snapshot | 1730556 | 0.0226 | 0.1317 |
| 200 | D9 | next_lat_acc | interaction_snapshot | 1730556 | 0.0234 | 0.1317 |
| 200 | D9 | next_lat_acc | ego_history | 1730556 | 0.0117 | 0.1663 |
| 200 | D9 | next_lat_acc | interaction_history | 1730556 | 0.0119 | 0.1663 |
| 200 | D10 | next_speed_mps | ego_snapshot | 3169342 | 0.0124 | 0.0374 |
| 200 | D10 | next_speed_mps | interaction_snapshot | 3169342 | 0.0124 | 0.0373 |
| 200 | D10 | next_speed_mps | ego_history | 3169342 | 0.0139 | 0.0327 |
| 200 | D10 | next_speed_mps | interaction_history | 3169342 | 0.0139 | 0.0327 |
| 200 | D10 | next_lon_acc | ego_snapshot | 3169342 | 0.1034 | 0.3197 |
| 200 | D10 | next_lon_acc | interaction_snapshot | 3169342 | 0.1030 | 0.3188 |
| 200 | D10 | next_lon_acc | ego_history | 3169342 | 0.0656 | 0.2325 |
| 200 | D10 | next_lon_acc | interaction_history | 3169342 | 0.0660 | 0.2323 |
| 200 | D10 | next_lat_acc | ego_snapshot | 3169342 | 0.0360 | 0.0927 |
| 200 | D10 | next_lat_acc | interaction_snapshot | 3169342 | 0.0365 | 0.0927 |
| 200 | D10 | next_lat_acc | ego_history | 3169342 | 0.0155 | 0.0701 |
| 200 | D10 | next_lat_acc | interaction_history | 3169342 | 0.0156 | 0.0701 |
| 500 | D9 | next_speed_mps | ego_snapshot | 1722084 | 0.0547 | 0.2474 |
| 500 | D9 | next_speed_mps | interaction_snapshot | 1722084 | 0.0541 | 0.2470 |
| 500 | D9 | next_speed_mps | ego_history | 1722084 | 0.0385 | 0.2418 |
| 500 | D9 | next_speed_mps | interaction_history | 1722084 | 0.0384 | 0.2413 |
| 500 | D9 | next_lon_acc | ego_snapshot | 1722084 | 0.2071 | 0.8091 |
| 500 | D9 | next_lon_acc | interaction_snapshot | 1722084 | 0.2043 | 0.8078 |
| 500 | D9 | next_lon_acc | ego_history | 1722084 | 0.1644 | 0.8063 |
| 500 | D9 | next_lon_acc | interaction_history | 1722084 | 0.1642 | 0.8048 |
| 500 | D9 | next_lat_acc | ego_snapshot | 1722084 | 0.0552 | 0.2291 |
| 500 | D9 | next_lat_acc | interaction_snapshot | 1722084 | 0.0570 | 0.2292 |
| 500 | D9 | next_lat_acc | ego_history | 1722084 | 0.0379 | 0.3287 |
| 500 | D9 | next_lat_acc | interaction_history | 1722084 | 0.0386 | 0.3286 |
| 500 | D10 | next_speed_mps | ego_snapshot | 3150361 | 0.0589 | 0.1706 |
| 500 | D10 | next_speed_mps | interaction_snapshot | 3150361 | 0.0589 | 0.1699 |
| 500 | D10 | next_speed_mps | ego_history | 3150361 | 0.0577 | 0.1340 |
| 500 | D10 | next_speed_mps | interaction_history | 3150361 | 0.0580 | 0.1338 |
| 500 | D10 | next_lon_acc | ego_snapshot | 3150361 | 0.2314 | 0.6146 |
| 500 | D10 | next_lon_acc | interaction_snapshot | 3150361 | 0.2317 | 0.6121 |
| 500 | D10 | next_lon_acc | ego_history | 3150361 | 0.1920 | 0.4954 |
| 500 | D10 | next_lon_acc | interaction_history | 3150361 | 0.1936 | 0.4943 |
| 500 | D10 | next_lat_acc | ego_snapshot | 3150361 | 0.0831 | 0.1965 |
| 500 | D10 | next_lat_acc | interaction_snapshot | 3150361 | 0.0841 | 0.1964 |
| 500 | D10 | next_lat_acc | ego_history | 3150361 | 0.0489 | 0.1560 |
| 500 | D10 | next_lat_acc | interaction_history | 3150361 | 0.0493 | 0.1561 |
| 1000 | D9 | next_speed_mps | ego_snapshot | 1709424 | 0.1745 | 0.4755 |
| 1000 | D9 | next_speed_mps | interaction_snapshot | 1709424 | 0.1724 | 0.4732 |
| 1000 | D9 | next_speed_mps | ego_history | 1709424 | 0.1423 | 0.5112 |
| 1000 | D9 | next_speed_mps | interaction_history | 1709424 | 0.1421 | 0.5090 |
| 1000 | D9 | next_lon_acc | ego_snapshot | 1709424 | 0.3227 | 0.9401 |
| 1000 | D9 | next_lon_acc | interaction_snapshot | 1709424 | 0.3199 | 0.9361 |
| 1000 | D9 | next_lon_acc | ego_history | 1709424 | 0.2989 | 0.9058 |
| 1000 | D9 | next_lon_acc | interaction_history | 1709424 | 0.2980 | 0.9021 |
| 1000 | D9 | next_lat_acc | ego_snapshot | 1709424 | 0.0905 | 0.3170 |
| 1000 | D9 | next_lat_acc | interaction_snapshot | 1709424 | 0.0936 | 0.3170 |
| 1000 | D9 | next_lat_acc | ego_history | 1709424 | 0.0795 | 0.4272 |
| 1000 | D9 | next_lat_acc | interaction_history | 1709424 | 0.0809 | 0.4271 |
| 1000 | D10 | next_speed_mps | ego_snapshot | 3118043 | 0.2048 | 0.5080 |
| 1000 | D10 | next_speed_mps | interaction_snapshot | 3118043 | 0.2053 | 0.5055 |
| 1000 | D10 | next_speed_mps | ego_history | 3118043 | 0.2273 | 0.4513 |
| 1000 | D10 | next_speed_mps | interaction_history | 3118043 | 0.2285 | 0.4500 |
| 1000 | D10 | next_lon_acc | ego_snapshot | 3118043 | 0.3639 | 0.8227 |
| 1000 | D10 | next_lon_acc | interaction_snapshot | 3118043 | 0.3665 | 0.8180 |
| 1000 | D10 | next_lon_acc | ego_history | 3118043 | 0.4393 | 0.8313 |
| 1000 | D10 | next_lon_acc | interaction_history | 3118043 | 0.4421 | 0.8280 |
| 1000 | D10 | next_lat_acc | ego_snapshot | 3118043 | 0.1464 | 0.3316 |
| 1000 | D10 | next_lat_acc | interaction_snapshot | 3118043 | 0.1481 | 0.3316 |
| 1000 | D10 | next_lat_acc | ego_history | 3118043 | 0.1172 | 0.3012 |
| 1000 | D10 | next_lat_acc | interaction_history | 3118043 | 0.1179 | 0.3012 |


## Shuffled-history ablation

The shuffled-history control tests whether temporal alignment of the history features matters. A rise in MAE after shuffling is consistent with useful temporal information in aligned history. Shuffling is a diagnostic control, not a realistic deployment baseline.

### 200 ms

Source: `artifacts/human_driving_history_ablation_v2/ablation_comparison_200ms.csv`

| horizon_ms | drone | target | model | aligned_mae | shuffled_mae | mae_increase | mae_increase_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | D10 | next_lat_acc | ego_history | 0.015453396532803705 | 0.29364132202298954 | 0.27818792549018584 | 1800.173346355685 |
| 200 | D10 | next_lat_acc | interaction_history | 0.015595333797240834 | 0.29375979430979526 | 0.27816446051255445 | 1783.6390302961518 |
| 200 | D10 | next_lon_acc | ego_history | 0.06564491022126617 | 0.49529288356123014 | 0.42964797333996396 | 654.5031014465097 |
| 200 | D10 | next_lon_acc | interaction_history | 0.06598298236900113 | 0.4954498297202372 | 0.4294668473512361 | 650.8751680084713 |
| 200 | D10 | next_speed_mps | ego_history | 0.013870586818455143 | 2.9487495953737106 | 2.9348790085552556 | 21159.011128861035 |
| 200 | D10 | next_speed_mps | interaction_history | 0.013896112831250787 | 2.9490278347303414 | 2.9351317218990904 | 21121.962361289345 |
| 200 | D9 | next_lat_acc | ego_history | 0.011706544951161135 | 0.15917627153812658 | 0.14746972658696544 | 1259.7203291167339 |
| 200 | D9 | next_lat_acc | interaction_history | 0.011944266101093674 | 0.15932773081453638 | 0.1473834647134427 | 1233.9265005151517 |
| 200 | D9 | next_lon_acc | ego_history | 0.05514909391334937 | 0.4527487773911262 | 0.3975996834777768 | 720.954154029236 |
| 200 | D9 | next_lon_acc | interaction_history | 0.05501929129515914 | 0.4524810487504646 | 0.39746175745530543 | 722.4043569064947 |
| 200 | D9 | next_speed_mps | ego_history | 0.005502888079585147 | 2.7277629512125543 | 2.722260063132969 | 49469.66072655789 |
| 200 | D9 | next_speed_mps | interaction_history | 0.005481222939636154 | 2.728002400756873 | 2.722521177817237 | 49669.958835827965 |
| 200 | POOLED | next_lat_acc | ego_history | 0.014130075825776528 | 0.24615068040006566 | 0.23202060457428914 | 1642.0336835774785 |
| 200 | POOLED | next_lat_acc | interaction_history | 0.014305842238850494 | 0.2462808030993451 | 0.23197496086049463 | 1621.5400462792627 |
| 200 | POOLED | next_lon_acc | ego_history | 0.061937976344976615 | 0.4802670691060492 | 0.4183290927610726 | 675.4000008510135 |
| 200 | POOLED | next_lon_acc | interaction_history | 0.062110803525689734 | 0.4802740277504967 | 0.418163224224807 | 673.2536056337606 |
| 200 | POOLED | next_speed_mps | ego_history | 0.010915265858969079 | 2.870701080287694 | 2.859785814428725 | 26199.873199413076 |
| 200 | POOLED | next_speed_mps | interaction_history | 0.010924124803893272 | 2.8709656197790507 | 2.8600414949751576 | 26180.966862955112 |


### 500 ms

Source: `artifacts/human_driving_history_ablation_v2/ablation_comparison_500ms.csv`

| horizon_ms | target | ego_history_aligned | ego_history_shuffled | interaction_history_aligned | interaction_history_shuffled |
| --- | --- | --- | --- | --- | --- |
| 500 | next_lat_acc | 0.0451 | 0.2500 | 0.0455 | 0.2502 |
| 500 | next_lon_acc | 0.1823 | 0.4738 | 0.1832 | 0.4730 |
| 500 | next_speed_mps | 0.0509 | 2.8723 | 0.0511 | 2.8746 |


### 1000 ms

Source: `artifacts/human_driving_history_ablation_v2/ablation_comparison_1000ms.csv`

| horizon_ms | target | ego_history_aligned | ego_history_shuffled | interaction_history_aligned | interaction_history_shuffled |
| --- | --- | --- | --- | --- | --- |
| 1000 | next_lat_acc | 0.1038 | 0.2536 | 0.1048 | 0.2541 |
| 1000 | next_lon_acc | 0.3895 | 0.4835 | 0.3911 | 0.4827 |
| 1000 | next_speed_mps | 0.1972 | 2.8769 | 0.1979 | 2.8882 |


### Ablation robustness summary

Source: `artifacts/human_driving_history_ablation_v2/robustness_summary.csv`

| Horizon (ms) | Drone | Target | Ego aligned MAE | Interaction aligned MAE | Ego shuffled MAE | Shuffle increase (%) | Interaction change (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | D9 | next_speed_mps | 0.0055 | 0.0054812229396361 | 2.7278 | 49469.6607 | -0.3937 |
| 200 | D9 | next_lon_acc | 0.0551 | 0.0550192912951591 | 0.4527 | 720.9542 | -0.2354 |
| 200 | D9 | next_lat_acc | 0.0117 | 0.0119442661010936 | 0.1592 | 1259.7203 | 2.0307 |
| 200 | D10 | next_speed_mps | 0.0139 | 0.0138961128312507 | 2.9487 | 21159.0111 | 0.1840 |
| 200 | D10 | next_lon_acc | 0.0656 | 0.0659829823690011 | 0.4953 | 654.5031 | 0.5150 |
| 200 | D10 | next_lat_acc | 0.0155 | 0.0155953337972408 | 0.2936 | 1800.1733 | 0.9185 |
| 200 | POOLED | next_speed_mps | 0.0109 | 0.0109241248038932 | 2.8707 | 26199.8732 | 0.0812 |
| 200 | POOLED | next_lon_acc | 0.0619 | 0.0621108035256897 | 0.4803 | 675.4000 | 0.2790 |
| 200 | POOLED | next_lat_acc | 0.0141 | 0.0143058422388504 | 0.2462 | 1642.0337 | 1.2439 |
| 500 | POOLED | next_speed_mps | 0.0509 | 0.0510937265530283 | 2.8723 | 5541.0875 | 0.3449 |
| 500 | POOLED | next_lon_acc | 0.1823 | 0.1831915672598664 | 0.4738 | 159.9272 | 0.5042 |
| 500 | POOLED | next_lat_acc | 0.0451 | 0.0455060034730206 | 0.2500 | 454.9422 | 0.9953 |
| 1000 | POOLED | next_speed_mps | 0.1972 | 0.1978823005053645 | 2.8769 | 1358.7351 | 0.3378 |
| 1000 | POOLED | next_lon_acc | 0.3895 | 0.3910513861085252 | 0.4835 | 24.1178 | 0.3882 |
| 1000 | POOLED | next_lat_acc | 0.1038 | 0.104802707468825 | 0.2536 | 144.3002 | 0.9579 |


## Traffic-context evaluation

Graph degree represents connectivity under the chosen 40 m graph construction; it should not be treated as a universal measure of traffic density. Negative interaction-change percentages mean lower MAE for the interaction-history model than for the ego-history model; positive percentages mean higher MAE.

### 200 ms descriptive metrics

Source: `artifacts/human_driving_condition_analysis_v1/condition_metrics_200ms.csv`

| Horizon | Drone | Degree bin | Target | N | Ego MAE | Interaction MAE | MAE difference | Change (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | D10 | 0 | lat_acc | 64415 | 0.0096 | 0.0099 | 0.0003 | 3.4745 |
| 200 | D10 | 0 | lon_acc | 64415 | 0.0475 | 0.0436 | -0.0039 | -8.1433 |
| 200 | D10 | 0 | speed | 64415 | 0.0092 | 0.0089 | -0.0003 | -3.3201 |
| 200 | D10 | 1-2 | lat_acc | 193519 | 0.0121 | 0.0123 | 0.0002 | 1.7955 |
| 200 | D10 | 1-2 | lon_acc | 193519 | 0.0574 | 0.0562 | -0.0011 | -1.9956 |
| 200 | D10 | 1-2 | speed | 193519 | 0.0115 | 0.0115 | -0.0001 | -0.6784 |
| 200 | D10 | 3-5 | lat_acc | 464674 | 0.0130 | 0.0131 | 0.0002 | 1.1760 |
| 200 | D10 | 3-5 | lon_acc | 464674 | 0.0607 | 0.0602 | -0.0005 | -0.8203 |
| 200 | D10 | 3-5 | speed | 464674 | 0.0127 | 0.0127 | -0.0000 | -0.2775 |
| 200 | D10 | 6+ | lat_acc | 2446734 | 0.0163 | 0.0165 | 0.0001 | 0.7890 |
| 200 | D10 | 6+ | lon_acc | 2446734 | 0.0677 | 0.0684 | 0.0007 | 1.0704 |
| 200 | D10 | 6+ | speed | 2446734 | 0.0144 | 0.0145 | 0.0001 | 0.3751 |
| 200 | D9 | 0 | lat_acc | 90231 | 0.0095 | 0.0099 | 0.0004 | 4.1600 |
| 200 | D9 | 0 | lon_acc | 90231 | 0.0413 | 0.0367 | -0.0046 | -11.2209 |
| 200 | D9 | 0 | speed | 90231 | 0.0042 | 0.0038 | -0.0004 | -9.2164 |
| 200 | D9 | 1-2 | lat_acc | 246152 | 0.0108 | 0.0113 | 0.0004 | 3.8652 |
| 200 | D9 | 1-2 | lon_acc | 246152 | 0.0487 | 0.0469 | -0.0019 | -3.8038 |
| 200 | D9 | 1-2 | speed | 246152 | 0.0051 | 0.0049 | -0.0002 | -3.6462 |
| 200 | D9 | 3-5 | lat_acc | 334773 | 0.0122 | 0.0124 | 0.0002 | 1.7206 |
| 200 | D9 | 3-5 | lon_acc | 334773 | 0.0665 | 0.0659 | -0.0007 | -0.9794 |
| 200 | D9 | 3-5 | speed | 334773 | 0.0067 | 0.0067 | -0.0001 | -0.7495 |
| 200 | D9 | 6+ | lat_acc | 1059400 | 0.0119 | 0.0121 | 0.0002 | 1.5991 |
| 200 | D9 | 6+ | lon_acc | 1059400 | 0.0542 | 0.0550 | 0.0008 | 1.5116 |
| 200 | D9 | 6+ | speed | 1059400 | 0.0053 | 0.0054 | 0.0001 | 1.0654 |


## Track-bootstrap results: original degree bins

The following files contain track-bootstrap summaries within the original graph-degree strata. The confidence intervals shown are for the interaction-versus-ego MAE change.

### `paired_track_bootstrap_200ms.csv`

| Horizon | Drone | Degree bin | Target | Tracks | Rows | Ego MAE | Interaction MAE | Change (%) | CI low (%) | CI high (%) | Bootstrap fraction improved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | D10 | 0 | lat_acc | 156 | 64415 | 0.009583238979164617 | 0.009916206410764612 | 3.4745 | 1.1133 | 7.8696 | 0.0 |
| 200 | D10 | 0 | lon_acc | 156 | 64415 | 0.0474635350424059 | 0.043598417139285174 | -8.1433 | -15.1693 | -2.9120 | 1.0 |
| 200 | D10 | 0 | speed | 156 | 64415 | 0.009216611174871613 | 0.008910606518889653 | -3.3201 | -6.6477 | -1.0658 | 1.0 |
| 200 | D10 | 1-2 | lat_acc | 529 | 193519 | 0.012085820508752221 | 0.012302824803787074 | 1.7955 | 1.1648 | 2.7321 | 0.0 |
| 200 | D10 | 1-2 | lon_acc | 529 | 193519 | 0.057384828358097315 | 0.05623966681566679 | -1.9956 | -3.3648 | -0.8441 | 1.0 |
| 200 | D10 | 1-2 | speed | 529 | 193519 | 0.011545444163708063 | 0.011467123209658401 | -0.6784 | -1.2094 | -0.2163 | 0.998 |
| 200 | D10 | 3-5 | lat_acc | 934 | 464674 | 0.012966065328781421 | 0.01311855033124288 | 1.1760 | 0.8934 | 1.5343 | 0.0 |
| 200 | D10 | 3-5 | lon_acc | 934 | 464674 | 0.060710894668081744 | 0.0602128755486911 | -0.8203 | -1.4851 | -0.1299 | 0.987 |
| 200 | D10 | 3-5 | speed | 934 | 464674 | 0.012700731946661237 | 0.01266548291122195 | -0.2775 | -0.6148 | 0.0031 | 0.974 |
| 200 | D10 | 6+ | lat_acc | 1252 | 2446734 | 0.01634667479546253 | 0.016475642387642712 | 0.7890 | 0.6586 | 0.9396 | 0.0 |
| 200 | D10 | 6+ | lon_acc | 1252 | 2446734 | 0.06771393162624427 | 0.06843876103839551 | 1.0704 | 0.7682 | 1.3970 | 0.0 |
| 200 | D10 | 6+ | speed | 1252 | 2446734 | 0.014399188034622281 | 0.014453197939098765 | 0.3751 | 0.2501 | 0.5225 | 0.0 |
| 200 | D9 | 0 | lat_acc | 211 | 90231 | 0.009520769418705836 | 0.00991683456956819 | 4.1600 | 2.4067 | 6.5491 | 0.0 |
| 200 | D9 | 0 | lon_acc | 211 | 90231 | 0.04133016322372751 | 0.036692541000868835 | -11.2209 | -16.0887 | -7.0679 | 1.0 |
| 200 | D9 | 0 | speed | 211 | 90231 | 0.004219929479749372 | 0.0038310054075064775 | -9.2164 | -12.7059 | -6.0877 | 1.0 |
| 200 | D9 | 1-2 | lat_acc | 625 | 246152 | 0.010837303626772203 | 0.01125618998775346 | 3.8652 | 2.3332 | 6.2067 | 0.0 |
| 200 | D9 | 1-2 | lon_acc | 625 | 246152 | 0.04873112302901015 | 0.046877476644805065 | -3.8038 | -6.0389 | -1.8193 | 1.0 |
| 200 | D9 | 1-2 | speed | 625 | 246152 | 0.00507053614856117 | 0.004885654080808127 | -3.6462 | -6.0928 | -1.6233 | 1.0 |
| 200 | D9 | 3-5 | lat_acc | 915 | 334773 | 0.012236325611194535 | 0.01244686153775455 | 1.7206 | 1.2754 | 2.2984 | 0.0 |
| 200 | D9 | 3-5 | lon_acc | 915 | 334773 | 0.06653697569822806 | 0.065885310030297 | -0.9794 | -1.7150 | -0.3166 | 0.999 |
| 200 | D9 | 3-5 | speed | 915 | 334773 | 0.006747353677399316 | 0.006696779402751237 | -0.7495 | -1.3492 | -0.2462 | 0.998 |
| 200 | D9 | 6+ | lat_acc | 873 | 1059400 | 0.011927267939926359 | 0.01211799942678619 | 1.5991 | 1.2462 | 2.0227 | 0.0 |
| 200 | D9 | 6+ | lon_acc | 873 | 1059400 | 0.054218696568476984 | 0.05503827866823387 | 1.5116 | 0.9592 | 2.0846 | 0.0 |
| 200 | D9 | 6+ | speed | 873 | 1059400 | 0.0053193628024745775 | 0.005376036741120539 | 1.0654 | 0.7269 | 1.4146 | 0.0 |


### `paired_track_bootstrap_500ms.csv`

| Horizon | Drone | Degree bin | Target | Tracks | Rows | Ego MAE | Interaction MAE | Change (%) | CI low (%) | CI high (%) | Bootstrap fraction improved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 500 | D10 | 0 | lat_acc | 150 | 63067 | 0.029145408305470386 | 0.029840300609909966 | 2.3842 | 0.6755 | 5.0547 | 0.0 |
| 500 | D10 | 0 | lon_acc | 150 | 63067 | 0.13603435084516294 | 0.11690345617116045 | -14.0633 | -25.3805 | -4.8822 | 1.0 |
| 500 | D10 | 0 | speed | 150 | 63067 | 0.03882224924660086 | 0.035065539512700736 | -9.6767 | -17.8283 | -3.4927 | 1.0 |
| 500 | D10 | 1-2 | lat_acc | 513 | 189490 | 0.03699792727517651 | 0.03761453716205128 | 1.6666 | 1.0648 | 2.5509 | 0.0 |
| 500 | D10 | 1-2 | lon_acc | 513 | 189490 | 0.16469461303125377 | 0.16042545192131885 | -2.5922 | -4.2441 | -0.9909 | 1.0 |
| 500 | D10 | 1-2 | speed | 513 | 189490 | 0.04866073477734793 | 0.047928167933879794 | -1.5055 | -2.5985 | -0.3994 | 0.995 |
| 500 | D10 | 3-5 | lat_acc | 928 | 460073 | 0.040410965866291164 | 0.04077022683938842 | 0.8890 | 0.6202 | 1.2104 | 0.0 |
| 500 | D10 | 3-5 | lon_acc | 928 | 460073 | 0.17781209189469227 | 0.1763894422197517 | -0.8001 | -1.5590 | -0.0415 | 0.981 |
| 500 | D10 | 3-5 | speed | 928 | 460073 | 0.05308007124114569 | 0.05278925100340042 | -0.5479 | -1.0927 | 0.0236 | 0.971 |
| 500 | D10 | 6+ | lat_acc | 1246 | 2437731 | 0.051994811311499314 | 0.05231614437133437 | 0.6180 | 0.5160 | 0.7381 | 0.0 |
| 500 | D10 | 6+ | lon_acc | 1246 | 2437731 | 0.1982997641357151 | 0.20136000569975387 | 1.5432 | 1.1335 | 1.9671 | 0.0 |
| 500 | D10 | 6+ | speed | 1246 | 2437731 | 0.05980655723135905 | 0.06036888873688521 | 0.9403 | 0.6413 | 1.2711 | 0.0 |
| 500 | D9 | 0 | lat_acc | 198 | 89338 | 0.029440244167721407 | 0.03026283313201941 | 2.7941 | 1.6434 | 4.3211 | 0.0 |
| 500 | D9 | 0 | lon_acc | 198 | 89338 | 0.1192111333515468 | 0.0963539259938506 | -19.1737 | -26.4181 | -12.7447 | 1.0 |
| 500 | D9 | 0 | speed | 198 | 89338 | 0.028033602671009652 | 0.023634044872154036 | -15.6939 | -21.6134 | -10.1019 | 1.0 |
| 500 | D9 | 1-2 | lat_acc | 601 | 243130 | 0.0319509424517908 | 0.03328380557882929 | 4.1716 | 2.4886 | 6.4600 | 0.0 |
| 500 | D9 | 1-2 | lon_acc | 601 | 243130 | 0.14224812463933179 | 0.13593890872870185 | -4.4354 | -6.6106 | -2.2524 | 1.0 |
| 500 | D9 | 1-2 | speed | 601 | 243130 | 0.03373257452286192 | 0.03279917468724228 | -2.7671 | -4.6765 | -0.5750 | 0.996 |
| 500 | D9 | 3-5 | lat_acc | 914 | 331664 | 0.03838762317636555 | 0.038946802736645234 | 1.4567 | 1.0656 | 1.9221 | 0.0 |
| 500 | D9 | 3-5 | lon_acc | 914 | 331664 | 0.18672031179743245 | 0.1844993023051056 | -1.1895 | -1.9430 | -0.4981 | 1.0 |
| 500 | D9 | 3-5 | speed | 914 | 331664 | 0.04390582242655765 | 0.043565005062979674 | -0.7762 | -1.5003 | 0.0039 | 0.974 |
| 500 | D9 | 6+ | lat_acc | 872 | 1057952 | 0.039896378107860765 | 0.04037372141922408 | 1.1965 | 0.9183 | 1.5378 | 0.0 |
| 500 | D9 | 6+ | lon_acc | 872 | 1057952 | 0.16624791839137867 | 0.17010049485997744 | 2.3174 | 1.6236 | 2.9866 | 0.0 |
| 500 | D9 | 6+ | speed | 872 | 1057952 | 0.038719607540255975 | 0.03939114475800582 | 1.7344 | 1.1465 | 2.2904 | 0.0 |


### `paired_track_bootstrap_1000ms.csv`

| Horizon | Drone | Degree bin | Target | Tracks | Rows | Ego MAE | Interaction MAE | Change (%) | CI low (%) | CI high (%) | Bootstrap fraction improved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | D10 | 0 | lat_acc | 133 | 60976 | 0.06581493039720288 | 0.06687636037886446 | 1.6127 | 0.5169 | 3.6115 | 0.0 |
| 1000 | D10 | 0 | lon_acc | 133 | 60976 | 0.2882057704735399 | 0.23810172896362833 | -17.3848 | -30.5539 | -6.2188 | 1.0 |
| 1000 | D10 | 0 | speed | 133 | 60976 | 0.15121482860706995 | 0.1311228528340342 | -13.2870 | -24.5368 | -5.1075 | 1.0 |
| 1000 | D10 | 1-2 | lat_acc | 476 | 182812 | 0.08444868198691237 | 0.08571184681822583 | 1.4958 | 0.8733 | 2.2810 | 0.0 |
| 1000 | D10 | 1-2 | lon_acc | 476 | 182812 | 0.3796943031898488 | 0.36858517571767396 | -2.9258 | -4.6150 | -1.2109 | 1.0 |
| 1000 | D10 | 1-2 | speed | 476 | 182812 | 0.19416722689171892 | 0.18985507805320834 | -2.2208 | -3.6650 | -0.8548 | 0.999 |
| 1000 | D10 | 3-5 | lat_acc | 899 | 452384 | 0.09338536719284585 | 0.09411232179029941 | 0.7784 | 0.5089 | 1.0949 | 0.0 |
| 1000 | D10 | 3-5 | lon_acc | 899 | 452384 | 0.4053259608512149 | 0.4019082588599493 | -0.8432 | -1.5263 | -0.1578 | 0.99 |
| 1000 | D10 | 3-5 | speed | 899 | 452384 | 0.2089899333467331 | 0.20750362718886198 | -0.7112 | -1.2907 | -0.0920 | 0.991 |
| 1000 | D10 | 6+ | lat_acc | 1241 | 2421871 | 0.12535811332764896 | 0.12609862187478393 | 0.5907 | 0.4777 | 0.7090 | 0.0 |
| 1000 | D10 | 6+ | lon_acc | 1241 | 2421871 | 0.4539542065571886 | 0.4602670778874727 | 1.3906 | 0.9527 | 1.8611 | 0.0 |
| 1000 | D10 | 6+ | speed | 1241 | 2421871 | 0.23510678808590305 | 0.23775959080971137 | 1.1283 | 0.7709 | 1.4854 | 0.0 |
| 1000 | D9 | 0 | lat_acc | 190 | 88157 | 0.05864476203495695 | 0.05985330128149638 | 2.0608 | 1.1870 | 3.2097 | 0.0 |
| 1000 | D9 | 0 | lon_acc | 190 | 88157 | 0.21486614127669765 | 0.15588617652823988 | -27.4496 | -36.0092 | -18.6155 | 1.0 |
| 1000 | D9 | 0 | speed | 190 | 88157 | 0.10269539472033543 | 0.07908005113850443 | -22.9955 | -30.8794 | -15.1433 | 1.0 |
| 1000 | D9 | 1-2 | lat_acc | 569 | 238853 | 0.0612994179375502 | 0.06421780264370265 | 4.7609 | 2.6755 | 7.0378 | 0.0 |
| 1000 | D9 | 1-2 | lon_acc | 569 | 238853 | 0.25693817995489054 | 0.23997613298160214 | -6.6016 | -9.3650 | -4.1388 | 1.0 |
| 1000 | D9 | 1-2 | speed | 569 | 238853 | 0.12187265080203852 | 0.11564473794301153 | -5.1102 | -7.6095 | -2.7197 | 1.0 |
| 1000 | D9 | 3-5 | lat_acc | 899 | 326730 | 0.07801416518047515 | 0.07921512733652103 | 1.5394 | 1.0701 | 2.0535 | 0.0 |
| 1000 | D9 | 3-5 | lon_acc | 899 | 326730 | 0.32544336269473567 | 0.3198860287613072 | -1.7076 | -2.6929 | -0.7829 | 1.0 |
| 1000 | D9 | 3-5 | speed | 899 | 326730 | 0.15342341649066712 | 0.1512956864876099 | -1.3868 | -2.3124 | -0.5079 | 1.0 |
| 1000 | D9 | 6+ | lat_acc | 872 | 1055684 | 0.08575966485527926 | 0.08687924584153521 | 1.3055 | 1.0308 | 1.6547 | 0.0 |
| 1000 | D9 | 6+ | lon_acc | 872 | 1055684 | 0.30715730337060604 | 0.3161831333996495 | 2.9385 | 2.0726 | 3.7429 | 0.0 |
| 1000 | D9 | 6+ | speed | 872 | 1055684 | 0.14685944655907884 | 0.15047014155748342 | 2.4586 | 1.7113 | 3.2311 | 0.0 |


## Robustness: sparse degree 0-5 versus dense degree 6+

Source: `artifacts/human_driving_condition_analysis_v1/paired_track_bootstrap_sparse0_5_vs_dense6plus.csv`

| Drone | Horizon (ms) | Target | Sparse tracks | Dense tracks | Sparse change (%) | Dense change (%) | Contrast (pp) | 95% CI low | 95% CI high | Valid bootstrap replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D9 | 200 | lat_acc | 993 | 873 | 2.7460 | 1.5991 | 1.1469 | 0.3797 | 2.0861 | 2000 |
| D9 | 200 | lon_acc | 993 | 873 | -2.8761 | 1.5116 | -4.3877 | -5.6544 | -3.2959 | 2000 |
| D9 | 200 | speed | 993 | 873 | -2.5087 | 1.0654 | -3.5742 | -4.7262 | -2.5703 | 2000 |
| D9 | 500 | lat_acc | 990 | 872 | 2.5206 | 1.1965 | 1.3241 | 0.5986 | 2.2389 | 2000 |
| D9 | 500 | lon_acc | 990 | 872 | -4.0243 | 2.3174 | -6.3417 | -7.8360 | -5.0150 | 2000 |
| D9 | 500 | speed | 990 | 872 | -2.9010 | 1.7344 | -4.6354 | -5.8895 | -3.5191 | 2000 |
| D9 | 1000 | lat_acc | 973 | 872 | 2.6401 | 1.3055 | 1.3346 | 0.4987 | 2.2564 | 2000 |
| D9 | 1000 | lon_acc | 973 | 872 | -5.9293 | 2.9385 | -8.8678 | -10.8166 | -7.1135 | 2000 |
| D9 | 1000 | speed | 973 | 872 | -4.8302 | 2.4586 | -7.2888 | -8.9160 | -5.8371 | 2000 |
| D10 | 200 | lat_acc | 974 | 1252 | 1.4953 | 0.7890 | 0.7064 | 0.2732 | 1.2051 | 2000 |
| D10 | 200 | lon_acc | 974 | 1252 | -1.6567 | 1.0704 | -2.7271 | -3.6042 | -1.9164 | 2000 |
| D10 | 200 | speed | 974 | 1252 | -0.5870 | 0.3751 | -0.9621 | -1.3494 | -0.6501 | 2000 |
| D10 | 500 | lat_acc | 965 | 1246 | 1.1878 | 0.6180 | 0.5698 | 0.1943 | 1.0348 | 2000 |
| D10 | 500 | lon_acc | 965 | 1246 | -2.1958 | 1.5432 | -3.7391 | -4.9926 | -2.6698 | 2000 |
| D10 | 500 | speed | 965 | 1246 | -1.4119 | 0.9403 | -2.3521 | -3.1381 | -1.6382 | 2000 |
| D10 | 1000 | lat_acc | 932 | 1241 | 1.0122 | 0.5907 | 0.4215 | 0.0910 | 0.8219 | 2000 |
| D10 | 1000 | lon_acc | 932 | 1241 | -2.4532 | 1.3906 | -3.8438 | -5.1921 | -2.6260 | 2000 |
| D10 | 1000 | speed | 932 | 1241 | -1.9286 | 1.1283 | -3.0570 | -4.1554 | -2.0837 | 2000 |


### Interpretation

- Speed: all six sparse-minus-dense contrasts were negative, with reported 95% confidence intervals below zero.
- Longitudinal acceleration: all six contrasts were negative, with reported 95% confidence intervals below zero.
- Lateral acceleration: all six contrasts were positive, with reported 95% confidence intervals above zero.

This supports a target-dependent association between graph context and the relative predictive value of neighbor information. It does not establish that surrounding vehicles cause the observed prediction differences.

## Limitations

- Only two held-out drones are evaluated; generalization to other days, sites, and collection conditions is unknown.
- Graph degree depends on the 40 m graph radius and graph-construction choices.
- Track bootstrap intervals quantify resampling uncertainty within these data; they do not eliminate dataset shift or every form of temporal dependence.
- The interaction-history model does not improve every target in every context; lateral acceleration is a consistent weakness in the reported comparisons.
- Shuffling history creates potentially unrealistic feature combinations and should be treated as an ablation control.
- No pNEUMA human-preference labels or explicit driver-intention labels are established by these experiments.

## Conclusions

The existing experiments indicate that temporally aligned history carries useful predictive information. The shuffled-history control supports the importance of temporal alignment. Across the broader sparse-versus-dense grouping, the interaction-history model's relative performance is more favorable for speed and longitudinal acceleration in sparse contexts, but less favorable for lateral acceleration. These findings should be reported as held-out predictive associations, not causal explanations of driver behavior.

## Source scripts

- `research/human_driving_history_v1.py`: present
- `research/human_driving_condition_analysis_v1.py`: present
- `research/human_driving_condition_bootstrap_v1.py`: present
- `research/human_driving_condition_bootstrap_v2.py`: present
- `research/human_driving_sparse_dense_contrast_v1.py`: present
- `research/human_driving_sparse_dense_bootstrap_v2.py`: present
- `research/human_driving_sparse_dense_bootstrap_v3.py`: present
