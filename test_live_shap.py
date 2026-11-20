
import joblib
import shap
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# 1. Load the saved models and SHAP background
models = joblib.load(
    "results/stability_experiment/trained_run_models.pkl"
)

background = joblib.load(
    "data/processed/shap_background.pkl"
)

# 2. Load one test record
X_test = pd.read_csv(
    "data/processed/X_test.csv",
    index_col=0
)

sample = X_test.iloc[[0]]

print("Input shape:", sample.shape)

# 3. Generate a fresh SHAP explanation for each model
shap_results = []
predictions = []

for run_number, model in enumerate(models, start=1):

    def predict_malignant(data):
        data = pd.DataFrame(
            data,
            columns=background.columns
        )
        return model.predict_proba(data)[:, 1]

    explainer = shap.Explainer(
        predict_malignant,
        shap.maskers.Independent(background),
        algorithm="permutation",
        seed=42
    )

    explanation = explainer(
        sample,
        max_evals=305
    )

    shap_results.append(explanation.values[0])

    malignant_probability = model.predict_proba(sample)[0, 1]
    predictions.append(malignant_probability)

    print(
        f"Run {run_number}: "
        f"Malignant probability = {malignant_probability:.6f}"
    )

# 4. Combine all ten explanations
values = np.array(shap_results)

print("\nSHAP matrix shape:", values.shape)
print("All SHAP values finite:", np.isfinite(values).all())

# 5. Compare global feature-importance rankings
rank_correlations = []

for i in range(10):
    for j in range(i + 1, 10):

        correlation, _ = spearmanr(
            np.abs(values[i]),
            np.abs(values[j])
        )

        rank_correlations.append(correlation)

# 6. Compare the top five features
rankings = np.argsort(-np.abs(values), axis=1)

top5_sets = [
    set(rankings[i, :5])
    for i in range(10)
]

jaccard_scores = []

for i in range(10):
    for j in range(i + 1, 10):

        intersection = len(top5_sets[i] & top5_sets[j])
        union = len(top5_sets[i] | top5_sets[j])

        jaccard_scores.append(intersection / union)

# 7. Calculate SHAP variation
mean_shap_variation = np.mean(
    np.std(values, axis=0)
)

print("\n--- LIVE EXPLANATION STABILITY ---")
print("Mean pairwise Spearman:", np.mean(rank_correlations))
print("Mean pairwise Top-5 Jaccard:", np.mean(jaccard_scores))
print("Mean absolute SHAP variation:", mean_shap_variation)
