# Member 2 training and tuning

Run the commands below from the repository root using the same Python environment throughout. Installing requirements.txt covers raw-trip preprocessing, Parquet feature creation, training, tuning, maps and API dependencies.

    python -m src.data.preprocess
    python -m src.features.build_features
    python -m src.models.train_zoo
    python -m src.models.optimize
    python -c "from maps.make_maps import plot_model_comparison; plot_model_comparison()"

Raw data lives in train/ and student_test/. Place the trip folders directly under each partition. The model comparison uses five classifiers on the shared Parquet features. Every tenth training observation is retained with all its candidates to limit CPU cost. Five-fold cross-validation groups whole trips. All student-test observations contribute to Top-1 accuracy, including empty candidate sets. The target is the nearest candidate belonging to the reference route, a proxy for the true segment at that timestamp. Ties follow candidate order.

Selection maximises mean CV candidate-level Macro F1. Optuna runs twelve trials on the selected family using TPE, four initial random trials, seed 42 and identical folds. Compare the selected model before and after tuning in results/tuned_vs_untuned.csv. Results before HMM decoding measure the classifier alone. Continuity counts consecutive represented observations whose selected directed arcs match or share an end-to-start node.

For inference load artifacts/model_tuned.joblib and artifacts/feature_columns.json and call model.predict_proba(candidates[feature_columns])[:,1]. Each input row is a candidate. Keep its trip_id, obs_id and candidate_id alongside the resulting score. Match the training software versions when loading the artifact.

## Comparing model behaviour

| Model | Why it was tested | What to inspect |
|---|---|---|
| Logistic regression | Tests whether an additive linear boundary in scaled features explains the candidate labels. Optimises regularised log loss. | Compare the training and CV scores to the nonlinear models. |
| Gaussian naive Bayes | Fits a class prior and one Gaussian distribution per feature and class. Provides a low-cost probabilistic baseline. | Rank and distance are related, so the conditional independence assumption can affect log loss. |
| Decision tree | Uses Gini splits to learn thresholds in rank, distance and motion. | A training-to-CV gap can indicate overly specific leaves. |
| Random forest | Averages Gini trees fitted to bootstrap samples with feature subsampling. | Compare its validation gain, model size and inference latency with those of one tree. |
| Extra trees | Averages trees using randomised thresholds. | Compare fold variation and test log loss with the random forest. |

Generate the review tables with python -m src.models.report_results after tuning. Consult results/model_comparison.csv for the configuration, runtime, training Macro F1 and individual cross-validation scores of each model. For Windows with 64-bit Python 3.12, requirements-windows-py312.lock records the installed versions.
