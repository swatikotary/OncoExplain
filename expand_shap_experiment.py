
import joblib
import shap
import numpy as np
import pandas as pd
import warnings

from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
STABILITY = BASE / "results" / "stability_experiment"

OUTPUT = STABILITY / "expanded_114"
OUTPUT.mkdir(parents=True, exist_ok=True)

MAX_EVALS = 305
SEED = 42

# ============================================================
# 2. LOAD EXPERIMENT DATA
# ============================================================

print("Loading test data...")

X_test = pd.read_csv(DATA / "X_test.csv", index_col=0)
background = joblib.load(DATA / "shap_background.pkl")

models = joblib.load(
    STABILITY / "trained_run_models.pkl"
)

print("Test data:", X_test.shape)
print("Background:", background.shape)
print("Models:", len(models))

assert X_test.shape == (114, 30)
assert background.shape[1] == 30
assert len(models) == 10
assert list(X_test.columns) == list(background.columns)

# ============================================================
# 3. COMPUTE SHAP FOR ALL 114 SAMPLES
# ============================================================

all_shap = []

for run_number, model in enumerate(models, start=1):

    print(
        f"\nComputing SHAP: Model {run_number}/10"
    )

    explainer = shap.PermutationExplainer(
        model.predict_proba,
        shap.maskers.Independent(background),
        seed=SEED
    )

    explanation = explainer(
        X_test,
        max_evals=MAX_EVALS
    )

    # Extract malignant class (class index 1).
    values = explanation.values[:, :, 1]

    assert values.shape == (114, 30)

    all_shap.append(values)

    # Save each run separately.
    joblib.dump(
        values,
        OUTPUT / f"run_{run_number}_shap_114.pkl"
    )

    print("Saved shape:", values.shape)

# Shape: models x samples x features
shap_array = np.stack(all_shap, axis=0)

print("\nFinal SHAP array:", shap_array.shape)

joblib.dump(
    shap_array,
    OUTPUT / "shap_array_114.pkl"
)

# ============================================================
# 4. CALCULATE GLOBAL AND INDIVIDUAL STABILITY
# ============================================================

spearman_rows = []
jaccard_rows = []
variation_rows = []

for sample_position, sample_id in enumerate(X_test.index):

    sample_values = shap_array[:, sample_position, :]

    absolute_values = np.abs(sample_values)

    spearman_scores = []
    jaccard_scores = []

    for run_a, run_b in combinations(range(10), 2):

        # Spearman correlation of absolute SHAP contributions.
        correlation = spearmanr(
            absolute_values[run_a],
            absolute_values[run_b]
        ).statistic

        if np.isfinite(correlation):
            spearman_scores.append(correlation)

        # Compare top five features.
        top_a = set(
            np.argsort(absolute_values[run_a])[-5:]
        )

        top_b = set(
            np.argsort(absolute_values[run_b])[-5:]
        )

        union = top_a | top_b
        intersection = top_a & top_b

        jaccard_scores.append(
            len(intersection) / len(union)
        )

    # Feature-wise standard deviation across 10 models.
    feature_std = np.std(sample_values, axis=0, ddof=0)

    spearman_rows.append({
        "Record_Index": sample_id,
        "Mean_Spearman": np.mean(spearman_scores),
        "Minimum_Spearman": np.min(spearman_scores),
        "Maximum_Spearman": np.max(spearman_scores)
    })

    jaccard_rows.append({
        "Record_Index": sample_id,
        "Mean_Top5_Jaccard": np.mean(jaccard_scores),
        "Minimum_Top5_Jaccard": np.min(jaccard_scores),
        "Maximum_Top5_Jaccard": np.max(jaccard_scores)
    })

    variation_rows.append({
        "Record_Index": sample_id,
        "Mean_SHAP_Std": np.mean(feature_std),
        "Maximum_SHAP_Std": np.max(feature_std)
    })

    if (sample_position + 1) % 10 == 0:
        print(
            f"Metrics calculated for "
            f"{sample_position + 1}/114 samples"
        )

# ============================================================
# 5. SAVE METRICS
# ============================================================

pd.DataFrame(spearman_rows).to_csv(
    OUTPUT / "individual_record_spearman_114.csv",
    index=False
)

pd.DataFrame(jaccard_rows).to_csv(
    OUTPUT / "individual_record_top5_jaccard_114.csv",
    index=False
)

pd.DataFrame(variation_rows).to_csv(
    OUTPUT / "individual_record_shap_variation_114.csv",
    index=False
)

# ============================================================
# 6. SAVE MEAN SHAP EXPLANATIONS
# ============================================================

mean_shap = np.mean(shap_array, axis=0)

mean_shap_df = pd.DataFrame(
    mean_shap,
    index=X_test.index,
    columns=X_test.columns
)

mean_shap_df.index.name = "Record_Index"

mean_shap_df.to_csv(
    OUTPUT / "mean_shap_values_114.csv"
)

# ============================================================
# 7. SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("EXPANDED SHAP EXPERIMENT COMPLETED")
print("=" * 55)

print("SHAP array:", shap_array.shape)
print("Samples:", len(X_test))
print("Models:", len(models))
print("Features:", len(X_test.columns))

print("\nMean Spearman:",
      np.mean([r["Mean_Spearman"] for r in spearman_rows]))

print("Mean Top-5 Jaccard:",
      np.mean([r["Mean_Top5_Jaccard"] for r in jaccard_rows]))

print("Mean SHAP variation:",
      np.mean([r["Mean_SHAP_Std"] for r in variation_rows]))

print("\nOutput directory:", OUTPUT)
print("All files saved successfully.")
