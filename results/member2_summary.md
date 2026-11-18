# PathSnap model experiments

Five-fold validation keeps complete trips together. The comparison uses every tenth training observation and evaluates all ten student-test trips. The nearest candidate appearing in the reference route supplies the pointwise label. Scores measure agreement with that proxy.

| Model | CV Macro F1 | Training Macro F1 | Student-test Macro F1 | Top-1 | Log loss | Continuity | Inference seconds |
|---|---|---|---|---|---|---|---|
| gnb | 0.7401 | 0.7410 | 0.7318 | 0.6018 | 0.4656 | 0.8202 | 0.0523 |
| logreg | 0.7400 | 0.7409 | 0.7494 | 0.6018 | 0.3415 | 0.8202 | 0.0171 |
| rf | 0.7346 | 0.7598 | 0.7435 | 0.6051 | 0.3190 | 0.8209 | 0.1140 |
| et | 0.7344 | 0.7492 | 0.7422 | 0.6017 | 0.3243 | 0.8202 | 0.1600 |
| dtree | 0.7184 | 0.7719 | 0.7345 | 0.5856 | 0.7458 | 0.8078 | 0.0275 |

## Bayesian tuning

| Variant | Model | CV Macro F1 | Student-test Top-1 | Student-test Macro F1 | Log loss |
|---|---|---:|---:|---:|---:|
| untuned | gnb | 0.7401 | 0.6018 | 0.7318 | 0.4656 |
| tuned | gnb | 0.7401 | 0.6018 | 0.7318 | 0.4656 |

Student-test GPS observations: 19160. Observations without candidates: 0.

Full fold scores, continuity, timing and model configurations are in model_comparison.csv. Trial parameters and objectives are in optuna_trials.csv.
