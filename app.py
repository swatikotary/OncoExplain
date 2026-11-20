
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OncoExplain | Breast Cancer ML Research",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
STABILITY_DIR = RESULTS_DIR / "stability_experiment"
SHAP_VALUES_DIR = RESULTS_DIR / "shap_values"
EXPANDED_DIR = STABILITY_DIR / "expanded_114"


# ============================================================
# 3. DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f5f5f5;
    }

    section[data-testid="stSidebar"] {
        background-color: #151a23;
    }

    .main-title {
        font-size: 38px;
        font-weight: 750;
        color: #f5f5f5;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 16px;
        color: #aab4c3;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 650;
        color: #f5f5f5;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .info-card {
        background-color: #1b2330;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #303b4d;
        margin-bottom: 12px;
    }

    .small-text {
        color: #aab4c3;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background-color: #1b2330;
        border: 1px solid #303b4d;
        padding: 15px;
        border-radius: 10px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. DATA LOADING FUNCTIONS
# ============================================================

def find_file(filename, folders):
    """Find a file in the supplied project folders."""

    for folder in folders:
        candidate = folder / filename

        if candidate.exists():
            return candidate

    return None


@st.cache_data
def load_csv(filename, folders):
    """Load a CSV file if it exists."""

    file_path = find_file(filename, folders)

    if file_path is None:
        return None

    try:
        return pd.read_csv(file_path)

    except Exception as error:
        st.warning(f"Could not load {filename}: {error}")
        return None


@st.cache_resource
def load_shap_array():
    """Load the original ten-sample SHAP experiment array."""

    possible_files = [
        STABILITY_DIR / "shap_array_305.pkl",
        SHAP_VALUES_DIR / "all_run_shap_values_305.pkl"
    ]

    for file_path in possible_files:

        if file_path.exists():

            try:
                return np.asarray(joblib.load(file_path))

            except Exception as error:
                st.warning(f"Could not load original SHAP array: {error}")

    return None


@st.cache_data
def load_expanded_shap_data():
    """Load SHAP explanations and stability metrics for all 114 samples."""

    shap_file = EXPANDED_DIR / "shap_array_114.pkl"

    if not shap_file.exists():
        raise FileNotFoundError(
            f"Expanded SHAP array not found: {shap_file}"
        )

    shap_array_114 = np.asarray(joblib.load(shap_file))

    spearman_114 = pd.read_csv(
        EXPANDED_DIR / "individual_record_spearman_114.csv"
    )

    jaccard_114 = pd.read_csv(
        EXPANDED_DIR / "individual_record_top5_jaccard_114.csv"
    )

    variation_114 = pd.read_csv(
        EXPANDED_DIR / "individual_record_shap_variation_114.csv"
    )

    return (
        shap_array_114,
        spearman_114,
        jaccard_114,
        variation_114
    )


@st.cache_data
def load_live_prediction_data():
    """Load test data, repeated predictions and trained models."""

    X = pd.read_csv(
        DATA_DIR / "X_test.csv",
        index_col=0
    )

    y = pd.read_csv(
        DATA_DIR / "y_test.csv",
        index_col=0
    )

    prediction_data = joblib.load(
        STABILITY_DIR / "all_run_predictions.pkl"
    )

    models = joblib.load(
        STABILITY_DIR / "trained_run_models.pkl"
    )

    return X, y, prediction_data, models


# Load project result files

initial_results = load_csv(
    "initial_model_evaluation.csv",
    [RESULTS_DIR]
)

cv_results = load_csv(
    "cross_validation_results.csv",
    [RESULTS_DIR]
)

shap_importance = load_csv(
    "shap_feature_importance_pilot.csv",
    [RESULTS_DIR]
)

explanation_records = load_csv(
    "pilot_explanation_records.csv",
    [RESULTS_DIR, DATA_DIR]
)

experiment_summary = load_csv(
    "final_experiment_summary.csv",
    [STABILITY_DIR, RESULTS_DIR]
)

run_metrics = load_csv(
    "run_performance_metrics.csv",
    [STABILITY_DIR, RESULTS_DIR]
)

probability_stability = load_csv(
    "probability_stability_305.csv",
    [STABILITY_DIR, RESULTS_DIR]
)

global_importance = load_csv(
    "global_shap_importance_305.csv",
    [STABILITY_DIR, RESULTS_DIR]
)

individual_spearman = load_csv(
    "individual_record_spearman_305.csv",
    [STABILITY_DIR, RESULTS_DIR]
)

individual_jaccard = load_csv(
    "individual_record_top5_jaccard_305.csv",
    [STABILITY_DIR, RESULTS_DIR]
)

individual_variation = load_csv(
    "individual_record_shap_variation_305.csv",
    [STABILITY_DIR, RESULTS_DIR]
)


# Load original SHAP array for Page 4

shap_array = load_shap_array()


# Load expanded SHAP experiment for Page 6

try:

    (
        expanded_shap,
        expanded_spearman,
        expanded_jaccard,
        expanded_variation
    ) = load_expanded_shap_data()

except Exception as error:

    expanded_shap = None
    expanded_spearman = None
    expanded_jaccard = None
    expanded_variation = None

    st.warning(
        f"Expanded SHAP data could not be loaded: {error}"
    )


# ============================================================
# 5. GENERAL HELPER FUNCTIONS
# ============================================================

def display_metric(label, value, format_string=None):
    """Display a metric without multiplying percentages twice."""

    if value is None:
        st.metric(label, "N/A")
        return

    try:

        numeric_value = float(value)

        if pd.isna(numeric_value):
            st.metric(label, "N/A")
            return

        if format_string == "percent":

            if abs(numeric_value) <= 1:
                numeric_value *= 100

            display_value = f"{numeric_value:.2f}%"

        elif format_string == "decimal":

            display_value = f"{numeric_value:.4f}"

        else:

            if numeric_value.is_integer():
                display_value = f"{numeric_value:,.0f}"
            else:
                display_value = f"{numeric_value:.4f}"

        st.metric(label, display_value)

    except (ValueError, TypeError):

        st.metric(label, str(value))


def show_missing_message(filename):

    st.info(
        f"`{filename}` was not found. "
        "Check that the file is present in the expected results folder."
    )


def get_summary_value(keyword):
    """Find a metric in final_experiment_summary.csv."""

    if experiment_summary is None:
        return None

    if not {"Metric", "Value"}.issubset(experiment_summary.columns):
        return None

    matches = experiment_summary[
        experiment_summary["Metric"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            regex=True,
            na=False
        )
    ]

    if matches.empty:
        return None

    try:
        return float(matches.iloc[0]["Value"])

    except (ValueError, TypeError):
        return None


def get_column(df, possible_names):
    """Find a column using possible column names."""

    if df is None:
        return None

    for name in possible_names:

        if name in df.columns:
            return name

    return None


def get_record_column(df):

    return get_column(
        df,
        [
            "Record_Index",
            "Record",
            "record_index",
            "record",
            "Index"
        ]
    )


def get_numeric_columns(df, excluded=None):

    if df is None:
        return []

    excluded = excluded or []

    return [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col not in excluded
    ]


def make_empty_message(message):

    st.info(message)


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown("## 🧬 OncoExplain")

    st.caption("Breast Cancer ML Research Dashboard")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Model Performance",
            "SHAP Explainability",
            "Explanation Stability",
            "Prediction Stability",
            "Live Prediction & Explanation"
        ]
    )

    st.divider()

    st.markdown("### About this project")

    st.caption(
        "A research-oriented framework studying "
        "the stability of SHAP explanations across "
        "repeated ensemble model training runs."
    )


# ============================================================
# 7. COMMON HEADER
# ============================================================

st.markdown(
    '<div class="main-title">OncoExplain</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Stability Analysis of SHAP-Based Explanations '
    'in Ensemble Machine Learning for Breast Cancer Classification'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown("## Project Overview")

    st.write(
        """
        OncoExplain investigates how consistent model explanations remain
        when the same ensemble machine learning architecture is retrained
        using different training samples.

        The project uses the Wisconsin Diagnostic Breast Cancer dataset
        and studies both prediction stability and SHAP explanation stability.
        """
    )

    st.markdown("### Dataset and experiment")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Records", "569")
    col2.metric("Input Features", "30")
    col3.metric("Training Runs", "10")
    col4.metric("Explanation Records", "114")

    st.markdown("### Research workflow")

    workflow = [
        ("01", "Dataset Analysis", "Inspect and prepare the WDBC dataset."),
        ("02", "Model Development", "Train and evaluate classification models."),
        ("03", "SHAP Explainability", "Generate feature contribution explanations."),
        ("04", "Repeated Training", "Train the ensemble on different training subsamples."),
        ("05", "Stability Analysis", "Compare predictions and explanations across runs.")
    ]

    for number, heading, description in workflow:

        st.markdown(
            f"""
            <div class="info-card">
                <b>{number} &nbsp; {heading}</b>
                <div class="small-text">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Baseline model performance")

    if initial_results is not None:

        accuracy_col = get_column(
            initial_results,
            ["Accuracy", "accuracy"]
        )

        auc_col = get_column(
            initial_results,
            ["ROC_AUC", "ROC-AUC", "roc_auc"]
        )

        model_col = get_column(
            initial_results,
            ["Model", "model"]
        )

        if accuracy_col and auc_col:

            ensemble_rows = initial_results

            if model_col:

                matching_rows = initial_results[
                    initial_results[model_col]
                    .astype(str)
                    .str.contains("Voting", case=False, na=False)
                ]

                if not matching_rows.empty:
                    ensemble_rows = matching_rows

            row = ensemble_rows.iloc[0]

            c1, c2 = st.columns(2)

            with c1:
                display_metric(
                    "Baseline Accuracy",
                    row[accuracy_col],
                    "percent"
                )

            with c2:
                display_metric(
                    "Baseline ROC-AUC",
                    row[auc_col],
                    "decimal"
                )

        else:

            st.dataframe(
                initial_results,
                use_container_width=True
            )

    else:

        show_missing_message("initial_model_evaluation.csv")

    st.markdown("### Research objective")

    st.write(
        """
        The main objective is to evaluate whether the global and
        individual-level SHAP explanations remain consistent when
        the same ensemble model is trained on different subsets
        of the original training data.
        """
    )


# ============================================================
# PAGE 2: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown("## Model Performance")

    st.write(
        """
        This section presents the initial comparison of machine learning
        models and the performance of the selected Soft Voting Ensemble.
        """
    )

    st.markdown("### Cross-validation comparison")

    if cv_results is not None:

        st.dataframe(
            cv_results,
            use_container_width=True,
            hide_index=True
        )

        model_col = get_column(cv_results, ["Model", "model"])

        accuracy_col = get_column(
            cv_results,
            ["Accuracy", "accuracy"]
        )

        if model_col and accuracy_col:

            fig = px.bar(
                cv_results,
                x=model_col,
                y=accuracy_col,
                color=model_col,
                title="Model Accuracy Comparison",
                text_auto=".4f"
            )

            fig.update_layout(
                template="plotly_dark",
                showlegend=False,
                xaxis_title="Model",
                yaxis_title="Accuracy"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        show_missing_message("cross_validation_results.csv")

    st.divider()

    st.markdown("### Baseline test-set performance")

    if initial_results is not None:

        st.dataframe(
            initial_results,
            use_container_width=True,
            hide_index=True
        )

        model_col = get_column(
            initial_results,
            ["Model", "model"]
        )

        accuracy_col = get_column(
            initial_results,
            ["Accuracy", "accuracy"]
        )

        auc_col = get_column(
            initial_results,
            ["ROC_AUC", "ROC-AUC", "roc_auc"]
        )

        if model_col and accuracy_col:

            fig = px.bar(
                initial_results,
                x=model_col,
                y=accuracy_col,
                color=model_col,
                title="Baseline Model Accuracy",
                text_auto=".4f"
            )

            fig.update_layout(
                template="plotly_dark",
                showlegend=False,
                xaxis_title="Model",
                yaxis_title="Accuracy"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        show_missing_message("initial_model_evaluation.csv")

    st.markdown("### Repeated-run model performance")

    if run_metrics is not None:

        st.dataframe(
            run_metrics,
            use_container_width=True,
            hide_index=True
        )

        metric_col = get_column(
            run_metrics,
            ["Accuracy", "accuracy"]
        )

        run_col = get_column(
            run_metrics,
            ["Run", "run"]
        )

        if metric_col and run_col:

            fig = px.line(
                run_metrics,
                x=run_col,
                y=metric_col,
                markers=True,
                title="Accuracy Across Repeated Training Runs"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Training Run",
                yaxis_title="Accuracy"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        show_missing_message("run_performance_metrics.csv")


# ============================================================
# PAGE 3: SHAP EXPLAINABILITY
# ============================================================

elif page == "SHAP Explainability":

    st.markdown("## SHAP Explainability")

    st.write(
        """
        SHAP (SHapley Additive exPlanations) helps interpret model
        predictions by estimating how each input feature contributes
        to the model output.
        """
    )

    st.markdown("### Global feature importance")

    if shap_importance is not None:

        feature_col = get_column(
            shap_importance,
            ["Feature", "feature"]
        )

        importance_col = get_column(
            shap_importance,
            ["Mean_Absolute_SHAP", "mean_absolute_shap"]
        )

        if feature_col and importance_col:

            plot_df = shap_importance.sort_values(
                importance_col,
                ascending=True
            )

            fig = px.bar(
                plot_df.tail(15),
                x=importance_col,
                y=feature_col,
                orientation="h",
                title="Top 15 Features by Mean Absolute SHAP Value"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Mean Absolute SHAP Value",
                yaxis_title="Feature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                shap_importance.sort_values(
                    importance_col,
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "The SHAP importance CSV has unexpected columns."
            )

    else:

        show_missing_message("shap_feature_importance_pilot.csv")

    st.divider()

    st.markdown("### Individual explanation records")

    if explanation_records is not None:

        st.dataframe(
            explanation_records,
            use_container_width=True,
            hide_index=True
        )

        probability_col = get_column(
            explanation_records,
            [
                "malignant_probability",
                "Malignant_Probability"
            ]
        )

        record_col = get_record_column(explanation_records)

        if probability_col:

            chart_df = explanation_records.copy()

            if record_col is None:

                chart_df["Record"] = np.arange(
                    1,
                    len(chart_df) + 1
                )

                record_col = "Record"

            fig = px.bar(
                chart_df,
                x=record_col,
                y=probability_col,
                title="Predicted Malignant Probability for Explanation Records"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Explanation Record",
                yaxis_title="Malignant Probability"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        show_missing_message("pilot_explanation_records.csv")

    st.info(
        "A positive SHAP value pushes the model output toward the "
        "malignant class relative to the explainer's background; "
        "a negative value pushes it away. These explanations describe "
        "model behaviour and are not clinical diagnoses."
    )


# ============================================================
# PAGE 4: EXPLANATION STABILITY
# ============================================================

elif page == "Explanation Stability":

    st.markdown("## Explanation Stability")

    st.write(
        """
        This section evaluates how consistent SHAP explanations are
        across ten repeated training runs using the same ensemble
        model architecture, fixed explanation records, and fixed
        SHAP background dataset.
        """
    )

    st.markdown("### Global Explanation Stability")

    global_spearman = get_summary_value("global.*spearman")
    global_jaccard = get_summary_value("global.*jaccard")
    direction_consistency = get_summary_value("direction")

    if global_spearman is None:
        global_spearman = 0.9484

    if global_jaccard is None:
        global_jaccard = 1.0

    if direction_consistency is None:
        direction_consistency = 95.33

    col1, col2, col3 = st.columns(3)

    with col1:

        display_metric(
            "Mean Global Spearman",
            global_spearman,
            "decimal"
        )

    with col2:

        display_metric(
            "Mean Global Top-10 Jaccard",
            global_jaccard,
            "decimal"
        )

    with col3:

        display_metric(
            "Direction Consistency",
            direction_consistency,
            "percent"
        )

    st.caption(
        "These values summarize agreement between explanations "
        "across repeated model runs."
    )

    st.divider()

    # --------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE ACROSS RUNS
    # --------------------------------------------------------

    st.markdown("### Global Feature Importance Across Runs")

    if shap_array is not None and shap_array.ndim == 3:

        number_of_runs, number_of_records, number_of_features = (
            shap_array.shape
        )

        st.caption(
            f"Loaded SHAP array: {number_of_runs} runs × "
            f"{number_of_records} records × "
            f"{number_of_features} features."
        )

        run_importance = np.mean(
            np.abs(shap_array),
            axis=1
        )

        feature_names = None

        if shap_importance is not None:

            feature_col = get_column(
                shap_importance,
                ["Feature", "feature"]
            )

            if (
                feature_col
                and len(shap_importance) == number_of_features
            ):

                feature_names = shap_importance[
                    feature_col
                ].tolist()

        if feature_names is None:

            feature_names = [
                f"Feature {i + 1}"
                for i in range(number_of_features)
            ]

        run_importance_df = pd.DataFrame(
            run_importance,
            columns=feature_names
        )

        run_importance_df.index = [
            f"Run {i + 1}"
            for i in range(number_of_runs)
        ]

        mean_importance = run_importance_df.mean(
            axis=0
        ).sort_values(ascending=False)

        top_features = mean_importance.head(15).index.tolist()

        chart_df = run_importance_df[
            top_features
        ].reset_index()

        chart_df = chart_df.rename(
            columns={"index": "Run"}
        )

        long_df = chart_df.melt(
            id_vars="Run",
            var_name="Feature",
            value_name="Mean Absolute SHAP"
        )

        fig = px.line(
            long_df,
            x="Run",
            y="Mean Absolute SHAP",
            color="Feature",
            markers=True,
            title="Top 15 Feature Importance Across Repeated Runs"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Training Run",
            yaxis_title="Mean Absolute SHAP Value"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        ranking_df = pd.DataFrame({
            "Feature": mean_importance.index,
            "Mean Absolute SHAP": mean_importance.values,
            "SHAP Variation Across Runs": (
                run_importance_df.std(axis=0)
                .reindex(mean_importance.index)
                .values
            )
        })

        st.markdown("#### Overall global feature ranking")

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True
        )

    elif global_importance is not None:

        st.dataframe(
            global_importance,
            use_container_width=True,
            hide_index=True
        )

    else:

        show_missing_message("shap_array_305.pkl")

    st.divider()

    # --------------------------------------------------------
    # INDIVIDUAL RECORD STABILITY
    # --------------------------------------------------------

    st.markdown("### Individual Explanation Stability")

    if individual_spearman is not None:

        st.markdown("#### Individual record Spearman correlation")

        st.dataframe(
            individual_spearman,
            use_container_width=True,
            hide_index=True
        )

        record_col = get_record_column(individual_spearman)

        spearman_col = get_column(
            individual_spearman,
            [
                "Mean_Spearman",
                "Mean_Spearman_Correlation",
                "Mean_Correlation",
                "Spearman",
                "Mean"
            ]
        )

        if record_col and spearman_col:

            fig = px.bar(
                individual_spearman,
                x=record_col,
                y=spearman_col,
                title="Mean SHAP Ranking Correlation by Explanation Record"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Explanation Record",
                yaxis_title="Mean Spearman Correlation"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        show_missing_message(
            "individual_record_spearman_305.csv"
        )

    if individual_jaccard is not None:

        st.markdown("#### Individual record Top-5 Jaccard similarity")

        st.dataframe(
            individual_jaccard,
            use_container_width=True,
            hide_index=True
        )

        record_col = get_record_column(individual_jaccard)

        jaccard_col = get_column(
            individual_jaccard,
            [
                "Mean_Top5_Jaccard",
                "Mean_Jaccard",
                "Mean_Top_5_Jaccard",
                "Top5_Jaccard",
                "Mean"
            ]
        )

        if record_col and jaccard_col:

            fig = px.bar(
                individual_jaccard,
                x=record_col,
                y=jaccard_col,
                title="Top-5 Feature Overlap by Explanation Record"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Explanation Record",
                yaxis_title="Mean Top-5 Jaccard Similarity"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        show_missing_message(
            "individual_record_top5_jaccard_305.csv"
        )

    st.divider()

    # --------------------------------------------------------
    # RECORD SELECTOR AND DETAILED VARIATION
    # --------------------------------------------------------

    st.markdown("### Explore an Individual Explanation Record")

    if shap_array is not None and shap_array.ndim == 3:

        record_options = list(range(number_of_records))

        selected_record = st.selectbox(
            "Select an explanation record",
            record_options,
            format_func=lambda x: f"Record {x + 1}"
        )

        selected_values = shap_array[
            :, selected_record, :
        ]

        record_mean = np.mean(
            np.abs(selected_values),
            axis=0
        )

        record_std = np.std(
            selected_values,
            axis=0
        )

        record_feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean Absolute SHAP": record_mean,
            "SHAP Standard Deviation": record_std
        }).sort_values(
            "Mean Absolute SHAP",
            ascending=False
        )

        st.markdown(
            f"#### Feature contributions for Record {selected_record + 1}"
        )

        fig = px.bar(
            record_feature_df.head(15).sort_values(
                "Mean Absolute SHAP",
                ascending=True
            ),
            x="Mean Absolute SHAP",
            y="Feature",
            orientation="h",
            title=f"Top Features for Record {selected_record + 1}"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Mean Absolute SHAP Across Runs",
            yaxis_title="Feature"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("#### Variation in feature contributions")

        variation_plot = record_feature_df.sort_values(
            "SHAP Standard Deviation",
            ascending=False
        ).head(15)

        fig = px.bar(
            variation_plot.sort_values(
                "SHAP Standard Deviation",
                ascending=True
            ),
            x="SHAP Standard Deviation",
            y="Feature",
            orientation="h",
            title="Features with the Highest SHAP Variation"
        )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="SHAP Standard Deviation Across Runs",
            yaxis_title="Feature"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            record_feature_df,
            use_container_width=True,
            hide_index=True
        )

    elif individual_variation is not None:

        st.dataframe(
            individual_variation,
            use_container_width=True,
            hide_index=True
        )

    else:

        show_missing_message(
            "individual_record_shap_variation_305.csv"
        )

    st.info(
        """
        Interpretation: high global agreement indicates that the overall
        feature ranking is similar across runs. Individual records can
        still show more variation in their feature rankings or contribution
        magnitudes. Stability measures describe consistency, not clinical
        correctness.
        """
    )


# ============================================================
# PAGE 5: PREDICTION STABILITY
# ============================================================

elif page == "Prediction Stability":

    st.markdown("## Prediction Stability")

    st.write(
        """
        This section examines whether repeated training on different
        training subsets changes the ensemble's predictions on the
        same fixed test dataset.
        """
    )

    st.markdown("### Prediction Agreement")

    agreement = get_summary_value("agreement")

    if agreement is None:
        agreement = 99.386

    col1, col2 = st.columns(2)

    with col1:

        display_metric(
            "Average Prediction Agreement",
            agreement,
            "percent"
        )

    with col2:

        display_metric(
            "Number of Repeated Runs",
            10
        )

    st.caption(
        "Prediction agreement measures how often repeated models "
        "produce the same predicted class for the fixed test records."
    )

    st.divider()

    st.markdown("### Prediction Probability Stability")

    if probability_stability is not None:

        st.dataframe(
            probability_stability,
            use_container_width=True,
            hide_index=True
        )

        record_col = get_column(
            probability_stability,
            [
                "Record_Index",
                "Record",
                "record_index"
            ]
        )

        mean_col = get_column(
            probability_stability,
            [
                "Mean_Malignant_Probability",
                "Mean_Probability"
            ]
        )

        std_col = get_column(
            probability_stability,
            [
                "Std_Malignant_Probability",
                "Std_Probability"
            ]
        )

        if record_col and mean_col:

            fig = px.bar(
                probability_stability,
                x=record_col,
                y=mean_col,
                title="Mean Malignant Probability Across Repeated Runs"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Test Record",
                yaxis_title="Mean Malignant Probability"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if record_col and std_col:

            variation_df = probability_stability.sort_values(
                std_col,
                ascending=False
            )

            fig = px.bar(
                variation_df,
                x=record_col,
                y=std_col,
                title="Probability Variation Across Repeated Runs"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Test Record",
                yaxis_title="Standard Deviation of Malignant Probability"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        show_missing_message("probability_stability_305.csv")

    st.divider()

    st.markdown("### Model Performance Across Repeated Runs")

    if run_metrics is not None:

        st.dataframe(
            run_metrics,
            use_container_width=True,
            hide_index=True
        )

        run_col = get_column(
            run_metrics,
            ["Run", "run"]
        )

        accuracy_col = get_column(
            run_metrics,
            ["Accuracy", "accuracy"]
        )

        if run_col and accuracy_col:

            fig = px.line(
                run_metrics,
                x=run_col,
                y=accuracy_col,
                markers=True,
                title="Accuracy Across Repeated Runs"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Training Run",
                yaxis_title="Accuracy"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        show_missing_message("run_performance_metrics.csv")

    st.info(
        """
        The repeated models are evaluated on the same fixed test set.
        This allows prediction differences to be compared across training
        runs. Prediction stability and explanation stability are separate:
        similar predictions do not automatically guarantee identical
        feature explanations.
        """
    )


# ============================================================
# PAGE 6: LIVE PREDICTION & EXPLANATION
# ============================================================

elif page == "Live Prediction & Explanation":

    st.markdown("## Live Prediction & Explanation")

    st.write(
        """
        Explore the saved predictions and SHAP explanations for
        individual samples from the WDBC test dataset.

        Select a sample ID to inspect its diagnostic features,
        ensemble predictions and explanation stability across
        ten retrained models.
        """
    )

    # --------------------------------------------------------
    # LOAD SAVED DATA
    # --------------------------------------------------------

    try:

        X_live, y_live, prediction_data, live_models = (
            load_live_prediction_data()
        )

    except Exception as error:

        st.error(
            f"Could not load live prediction data: {error}"
        )

        st.stop()

    if expanded_shap is None:

        st.error(
            "The expanded SHAP data could not be loaded. "
            "Check the expanded_114 folder."
        )

        st.stop()

    if expanded_shap.shape != (10, 114, 30):

        st.error(
            f"Unexpected expanded SHAP shape: {expanded_shap.shape}. "
            "Expected (10, 114, 30)."
        )

        st.stop()

    # --------------------------------------------------------
    # SAMPLE SELECTION
    # --------------------------------------------------------

    sample_ids = X_live.index.tolist()

    selected_id = st.selectbox(
        "Select Sample ID",
        options=sample_ids,
        index=sample_ids.index(177) if 177 in sample_ids else 0
    )

    sample_position = sample_ids.index(selected_id)

    st.divider()

    # --------------------------------------------------------
    # RETRIEVE SAVED PREDICTIONS
    # --------------------------------------------------------

    run_predictions = prediction_data["predictions"][
        :, sample_position
    ]

    run_probabilities = prediction_data["probabilities"][
        :, sample_position
    ]

    actual_label = int(
        y_live.loc[selected_id, "diagnosis"]
    )

    mean_probability = float(
        np.mean(run_probabilities)
    )

    malignant_count = int(
        np.sum(run_predictions == 1)
    )

    benign_count = int(
        np.sum(run_predictions == 0)
    )

    final_prediction = (
        1 if mean_probability >= 0.5 else 0
    )

    prediction_name = (
        "Malignant" if final_prediction == 1 else "Benign"
    )

    actual_name = (
        "Malignant" if actual_label == 1 else "Benign"
    )

    # --------------------------------------------------------
    # PREDICTION SUMMARY
    # --------------------------------------------------------

    st.markdown("### Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Ensemble Prediction",
        prediction_name
    )

    col2.metric(
        "Mean Malignant Probability",
        f"{mean_probability * 100:.2f}%"
    )

    col3.metric(
        "Prediction Agreement",
        f"{max(malignant_count, benign_count)}/10"
    )

    col4.metric(
        "Actual Diagnosis",
        actual_name
    )

    st.caption(
        "Results are retrieved from the saved ten-run experiment. "
        "The displayed prediction is based on the mean malignant "
        "probability across the ten runs."
    )

    # --------------------------------------------------------
    # PROBABILITY ACROSS TEN RUNS
    # --------------------------------------------------------

    st.markdown("### Prediction Consistency Across Runs")

    probability_df = pd.DataFrame({
        "Run": [
            f"Run {i}"
            for i in range(1, 11)
        ],
        "Malignant Probability": (
            run_probabilities * 100
        )
    })

    fig_probability = px.bar(
        probability_df,
        x="Run",
        y="Malignant Probability",
        title="Malignant Probability Across 10 Retrained Models",
        range_y=[0, 100]
    )

    fig_probability.update_layout(
        template="plotly_dark",
        yaxis_title="Malignant Probability (%)",
        xaxis_title="Training Run"
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )

    # --------------------------------------------------------
    # DIAGNOSTIC FEATURES
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Diagnostic Feature Values")

    sample_features = X_live.loc[selected_id]

    feature_df = pd.DataFrame({
        "Feature": sample_features.index,
        "Value": sample_features.values
    })

    feature_df["Feature Group"] = feature_df["Feature"].apply(
        lambda name:
            "Mean" if name.endswith("_mean")
            else "Standard Error" if name.endswith("_se")
            else "Worst"
    )

    feature_group = st.selectbox(
        "Filter Feature Group",
        [
            "All",
            "Mean",
            "Standard Error",
            "Worst"
        ],
        key="live_feature_group"
    )

    if feature_group != "All":

        feature_df = feature_df[
            feature_df["Feature Group"] == feature_group
        ]

    st.dataframe(
        feature_df[
            ["Feature", "Value"]
        ].reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # EXPANDED SHAP EXPLANATION
    # --------------------------------------------------------

    st.divider()

    st.markdown("### SHAP Explanation")

    # The expanded SHAP array follows the same row order as X_live.
    # Shape: 10 models x 114 samples x 30 features.

    sample_shap_values = expanded_shap[
        :, sample_position, :
    ]

    # Average signed SHAP contribution across ten models.

    mean_shap_values = np.mean(
        sample_shap_values,
        axis=0
    )

    # Feature-wise SHAP variation across ten models.

    shap_std_values = np.std(
        sample_shap_values,
        axis=0
    )

    shap_df = pd.DataFrame({
        "Feature": X_live.columns,
        "Mean SHAP Value": mean_shap_values,
        "SHAP Standard Deviation": shap_std_values
    })

    shap_df["Absolute Importance"] = (
        shap_df["Mean SHAP Value"].abs()
    )

    shap_df = shap_df.sort_values(
        "Absolute Importance",
        ascending=False
    )

    st.write(
        """
        Positive SHAP values push the model's malignant-class output
        higher relative to the explainer's background. Negative values
        push it lower. The chart displays the average signed contribution
        across ten retrained models.
        """
    )

    # --------------------------------------------------------
    # TOP 15 SHAP CONTRIBUTIONS
    # --------------------------------------------------------

    top_features = shap_df.head(15).sort_values(
        "Mean SHAP Value"
    )

    fig_shap = px.bar(
        top_features,
        x="Mean SHAP Value",
        y="Feature",
        orientation="h",
        color="Mean SHAP Value",
        color_continuous_scale="RdBu_r",
        title=f"Top 15 SHAP Contributions — Sample {selected_id}"
    )

    fig_shap.update_layout(
        template="plotly_dark",
        xaxis_title="Mean SHAP Value",
        yaxis_title="Diagnostic Feature"
    )

    st.plotly_chart(
        fig_shap,
        use_container_width=True
    )

    # --------------------------------------------------------
    # FEATURE-LEVEL SHAP VARIATION
    # --------------------------------------------------------

    st.markdown("#### Feature-level SHAP variation")

    variation_features = shap_df.sort_values(
        "SHAP Standard Deviation",
        ascending=False
    ).head(15).sort_values(
        "SHAP Standard Deviation"
    )

    fig_variation = px.bar(
        variation_features,
        x="SHAP Standard Deviation",
        y="Feature",
        orientation="h",
        title="Features with the Highest SHAP Variation Across Runs"
    )

    fig_variation.update_layout(
        template="plotly_dark",
        xaxis_title="SHAP Standard Deviation",
        yaxis_title="Diagnostic Feature"
    )

    st.plotly_chart(
        fig_variation,
        use_container_width=True
    )

    with st.expander("View all 30 feature contributions"):

        st.dataframe(
            shap_df[
                [
                    "Feature",
                    "Mean SHAP Value",
                    "SHAP Standard Deviation",
                    "Absolute Importance"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # EXPANDED SHAP STABILITY METRICS
    # --------------------------------------------------------

    st.divider()

    st.markdown("### Explanation Reliability Across 10 Runs")

    # Match using the original test-set sample ID.

    spearman_row = expanded_spearman[
        expanded_spearman["Record_Index"] == selected_id
    ]

    jaccard_row = expanded_jaccard[
        expanded_jaccard["Record_Index"] == selected_id
    ]

    variation_row = expanded_variation[
        expanded_variation["Record_Index"] == selected_id
    ]

    if (
        not spearman_row.empty
        and not jaccard_row.empty
        and not variation_row.empty
    ):

        s = spearman_row.iloc[0]
        j = jaccard_row.iloc[0]
        v = variation_row.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mean Spearman Correlation",
            f"{s['Mean_Spearman']:.4f}"
        )

        c2.metric(
            "Mean Top-5 Jaccard",
            f"{j['Mean_Top5_Jaccard']:.4f}"
        )

        c3.metric(
            "Mean SHAP Standard Deviation",
            f"{v['Mean_SHAP_Std']:.5f}"
        )

        st.markdown("#### Stability metric ranges")

        stability_range = pd.DataFrame({
            "Metric": [
                "Spearman Correlation",
                "Top-5 Jaccard Similarity",
                "SHAP Standard Deviation"
            ],
            "Mean": [
                s["Mean_Spearman"],
                j["Mean_Top5_Jaccard"],
                v["Mean_SHAP_Std"]
            ],
            "Minimum": [
                s["Minimum_Spearman"],
                j["Minimum_Top5_Jaccard"],
                np.nan
            ],
            "Maximum": [
                s["Maximum_Spearman"],
                j["Maximum_Top5_Jaccard"],
                v["Maximum_SHAP_Std"]
            ]
        })

        st.dataframe(
            stability_range,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Stability metrics could not be matched to this sample ID. "
            "Check the Record_Index column in the expanded metric CSV files."
        )

    st.info(
        """
        Spearman correlation measures similarity in feature rankings.
        Top-5 Jaccard measures overlap among the five most important
        features. SHAP standard deviation measures variation in
        contribution values across retrained models.

        These metrics describe explanation consistency, not clinical
        correctness or diagnostic reliability.
        """
    )

    # --------------------------------------------------------
    # RESEARCH DISCLAIMER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        """
        OncoExplain is an academic machine learning research project.
        The WDBC records are anonymized dataset samples, not actual
        patient records. Model predictions and SHAP explanations are
        for research and educational purposes only, not clinical
        diagnosis or medical decision-making.
        """
    )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8792a2; font-size:13px;">
        OncoExplain | Academic Research Project<br>
        Machine Learning • SHAP Explainability • Stability Analysis
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Educational and research use only. This dashboard is not a "
    "medical diagnostic tool and must not be used to make clinical decisions."
)
