import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from sklearn.inspection import permutation_importance


def plot_LR_coefficients(model, X_train: pd.DataFrame) -> plt.Figure:
    """
    Plot the coefficients of a linear regression model.

    Args:
        model: The trained linear regression model.
        X_train (pd.DataFrame): Training feature data.

    Returns:
        plt.Figure: The matplotlib figure containing the plot.
    """
    coefficients = {}
    for i in range(len(X_train.columns)):
        coef_value = model.coef_[0][i].round(5)
        coefficients[X_train.columns[i]] = coef_value

    coef_df = pd.DataFrame(
        list(coefficients.items()), columns=["Feature", "Coefficient"]
    )
    coef_df["Color"] = np.where(coef_df["Coefficient"] > 0, "blue", "red")
    sorted_coef_df = coef_df.reindex(
        coef_df["Coefficient"].abs().sort_values(ascending=False).index
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 12))
    sns.barplot(
        data=sorted_coef_df,
        x=sorted_coef_df["Coefficient"].abs(),  # Take absolute value for the x-axis
        y="Feature",
        hue="Color",  # Use the Color column for hue
        palette={"blue": "blue", "red": "red"},  # Specify the colors for the hue
        alpha=0.7,
        legend=False,
    )

    plt.title("Linear Regression Coefficients (Sorted by Absolute Value)", fontsize=16)
    plt.xlabel("Coefficient Value (Absolute)", fontsize=14)
    plt.ylabel("Features", fontsize=14)

    plt.tight_layout()
    plt.show()

    return plt.gcf()


def feature_ranking_PER(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10
) -> pd.DataFrame:
    """
    Compute feature importance using permutation importance.

    Args:
        model: The trained model.
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        n_repeats (int): Number of times to permute a feature.

    Returns:
        pd.DataFrame: A DataFrame containing features and their permutation importance scores.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=2, n_jobs=2
    )
    feature_names = X.columns
    importances_PER = pd.Series(result.importances_mean, index=feature_names)
    importances_PER = importances_PER.sort_values(ascending=False)
    importances_PER = importances_PER.reset_index()
    importances_PER.columns = ["Feature", "PER"]
    return importances_PER


def feature_ranking_SHAP(model_name: str, model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature importance using SHAP values.

    Args:
        model_name (str): The name/type of the model (used to determine SHAP explainer).
        model: The trained model.
        X (pd.DataFrame): Feature data.

    Returns:
        pd.DataFrame: A DataFrame containing features and their SHAP importance scores.
    """
    explainer = None
    if "NeuralNetwork" in model_name:
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        explainer = shap.DeepExplainer(model, X_tensor)
        shap_values = explainer(X_tensor)
    else:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)

    feature_names = X.columns
    aggregated_shap_values = abs(shap_values.values).mean(
        axis=2
    )  # Mean across the targets
    feature_importances = aggregated_shap_values.mean(axis=0)  # Mean across samples
    importances_SHAP = pd.Series(feature_importances, index=feature_names).sort_values(
        ascending=False
    )
    importances_SHAP = importances_SHAP.reset_index()
    importances_SHAP.columns = ["Feature", "SHAP"]
    return importances_SHAP


def plot_PER_and_SHAP(
    feature_rank_PER: pd.DataFrame = None,
    feature_rank_SHAP: pd.DataFrame = None,
    model_name: str = "model",
) -> plt.Figure:
    """
    Plot feature importance from permutation importance and SHAP values.

    Args:
        feature_rank_PER (pd.DataFrame, optional): Feature importance from permutation importance.
        feature_rank_SHAP (pd.DataFrame, optional): Feature importance from SHAP values.
        model_name (str): The name of the model.

    Returns:
        plt.Figure: The matplotlib figure containing the plot.
    """
    if isinstance(feature_rank_PER, pd.DataFrame) and isinstance(
        feature_rank_SHAP, pd.DataFrame
    ):
        features_df = pd.merge(
            feature_rank_PER, feature_rank_SHAP, on="Feature", how="left"
        )
    elif isinstance(feature_rank_PER, pd.DataFrame):
        features_df = feature_rank_PER
    elif isinstance(feature_rank_SHAP, pd.DataFrame):
        features_df = feature_rank_SHAP
    else:
        raise ValueError("Both feature_rank_PER and feature_rank_SHAP cannot be None.")

    sort_by = "SHAP" if "SHAP" in features_df.columns else "PER"
    sorted_feature_ranks = features_df.sort_values(by=sort_by, ascending=False)

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 15))

    if "SHAP" in sorted_feature_ranks.columns:
        sns.barplot(
            data=sorted_feature_ranks,
            x="SHAP",
            y="Feature",
            color="skyblue",
            label="SHAP",
            alpha=0.7,
        )
    if "PER" in sorted_feature_ranks.columns:
        sns.barplot(
            data=sorted_feature_ranks,
            x="PER",
            y="Feature",
            color="orange",
            label="PER",
            alpha=0.7,
        )

    # Set plot details
    plt.title(f"Feature Importance for {model_name}", fontsize=16)
    plt.xlabel("Importance Score", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.legend(title="Importance Type")
    plt.xlim(left=0)
    plt.tight_layout()
    plt.show()

    return plt.gcf()
