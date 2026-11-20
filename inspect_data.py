
import os
import pandas as pd
import joblib
import numpy as np

# --------------------------------------------------
# 1. SET PROJECT PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("ONCOEXPLAIN - EXPERIMENT DATA INSPECTION")
print("=" * 60)


# --------------------------------------------------
# 2. INSPECT SHAP FEATURE IMPORTANCE
# --------------------------------------------------

feature_file = os.path.join(
    BASE_DIR,
    "results",
    "shap_feature_importance_pilot.csv"
)

print("\n\nSHAP FEATURE IMPORTANCE")

if os.path.exists(feature_file):

    feature_df = pd.read_csv(feature_file)

    print("Shape:", feature_df.shape)
    print("Columns:", feature_df.columns.tolist())
    print(feature_df.head(10).to_string(index=False))

else:
    print("File not found:", feature_file)


# --------------------------------------------------
# 3. INSPECT PILOT EXPLANATION RECORDS
# --------------------------------------------------

pilot_file = os.path.join(
    BASE_DIR,
    "results",
    "pilot_explanation_records.csv"
)

print("\n\n" + "=" * 60)
print("PILOT EXPLANATION RECORDS")

if os.path.exists(pilot_file):

    pilot_df = pd.read_csv(pilot_file)

    print("Shape:", pilot_df.shape)
    print("Columns:", pilot_df.columns.tolist())
    print(pilot_df.head().to_string(index=False))

else:
    print("File not found:", pilot_file)


# --------------------------------------------------
# 4. INSPECT INDIVIDUAL RECORD SPEARMAN STABILITY
# --------------------------------------------------

spearman_file = os.path.join(
    BASE_DIR,
    "results",
    "stability_experiment",
    "individual_record_spearman_305.csv"
)

print("\n\n" + "=" * 60)
print("INDIVIDUAL RECORD SPEARMAN STABILITY")

if os.path.exists(spearman_file):

    spearman_df = pd.read_csv(spearman_file)

    print("Shape:", spearman_df.shape)
    print("Columns:", spearman_df.columns.tolist())
    print(spearman_df.to_string(index=False))

else:
    print("File not found:", spearman_file)


# --------------------------------------------------
# 5. INSPECT INDIVIDUAL RECORD TOP-5 JACCARD
# --------------------------------------------------

jaccard_file = os.path.join(
    BASE_DIR,
    "results",
    "stability_experiment",
    "individual_record_top5_jaccard_305.csv"
)

print("\n\n" + "=" * 60)
print("INDIVIDUAL RECORD TOP-5 JACCARD STABILITY")

if os.path.exists(jaccard_file):

    jaccard_df = pd.read_csv(jaccard_file)

    print("Shape:", jaccard_df.shape)
    print("Columns:", jaccard_df.columns.tolist())
    print(jaccard_df.to_string(index=False))

else:
    print("File not found:", jaccard_file)


# --------------------------------------------------
# 6. INSPECT SHAP ARRAY FILE FORMAT
# --------------------------------------------------


# --------------------------------------------------
# 6. LOAD AND INSPECT SHAP ARRAY USING JOBLIB
# --------------------------------------------------

shap_file = os.path.join(
    BASE_DIR,
    "results",
    "stability_experiment",
    "shap_array_305.pkl"
)

print("\n\n" + "=" * 60)
print("SHAP ARRAY INSPECTION")

if os.path.exists(shap_file):

    try:
        shap_data = joblib.load(shap_file)

        shap_array = np.asarray(shap_data)

        print("File loaded successfully!")
        print("Original data type:", type(shap_data))
        print("Array shape:", shap_array.shape)
        print("Array data type:", shap_array.dtype)
        print("Number of dimensions:", shap_array.ndim)

    except Exception as error:
        print("Error loading SHAP array:", error)

else:
    print("File not found:", shap_file)


# --------------------------------------------------
# 7. INSPECTION COMPLETED
# --------------------------------------------------

print("\n\n" + "=" * 60)
print("DATA INSPECTION COMPLETED")
print("=" * 60)
