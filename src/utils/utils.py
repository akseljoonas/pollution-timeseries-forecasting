import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from joblib import load
from mlflow.models import infer_signature
from sklearn.metrics import root_mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error


def inverse_scaling(preds: np.ndarray, particle: str, scaler_path: str) -> np.ndarray:
    """
    Inverse scale the predictions using the provided scaler.

    Args:
        preds (np.ndarray): The scaled predictions.
        particle (str): The name of the particle for which predictions are made.
        scaler_path (str): Path to the scaler used for inverse transformation.

    Returns:
        np.ndarray: The original predictions for the specified particle.
    """
    # Assume we have scaled predictions for 'NO2', 'O3', 'wind_speed'
    scaled_predictions = np.array([[row[0]] for row in preds])

    # Continuous columns in the original dataset
    continuous_columns = [
        "NO2",
        "O3",
        "wind_speed",
        "mean_temp",
        "global_radiation",
        "percipitation",
        "pressure",
        "minimum_visibility",
        "humidity",
    ]

    # Load the scaler
    scaler = load(scaler_path)

    # Inverse transform only the relevant features ('NO2', 'O3', 'wind_speed')
    scaled_subset_columns = [particle]
    subset_indices = [continuous_columns.index(col) for col in scaled_subset_columns]

    # Create an empty array for the full feature set (filled with zeros or NaNs)
    full_feature_shape = (scaled_predictions.shape[0], len(continuous_columns))
    scaled_full_array = np.zeros(full_feature_shape)

    # Fill the relevant columns with the scaled predictions
    scaled_full_array[:, subset_indices] = scaled_predictions

    # Inverse transform the entire array and extract the relevant features
    original_full_values = scaler.inverse_transform(scaled_full_array)
    original_predictions = original_full_values[:, subset_indices]

    return original_predictions


def log_mlflow_metrics_and_model(
    rmse: float,
    mape: float,
    model,
    X_train: pd.DataFrame,
    true_vs_pred_plot: plt.Figure,
    fig_3_days: plt.Figure,
    name: str,
    params: dict | None = None,
    losses_plot: plt.Figure | None = None,
    feature_rankings_plot: plt.Figure | None = None,
):
    """
    Log metrics and model to MLflow.

    Args:
        rmse (float): Root Mean Squared Error of the model.
        mape (float): Mean Absolute Percentage Error of the model.
        model: The trained model.
        X_train (pd.DataFrame): Training feature data.
        true_vs_pred_plot (plt.Figure): Figure of true vs predicted values.
        fig_3_days (plt.Figure): Figure of RMSE and MAPE for the next three days.
        name (str): Name to log the model under.
        params (dict, optional): Model parameters to log. Defaults to None.
        losses_plot (plt.Figure, optional): Figure of training and validation losses. Defaults to None.
        feature_rankings_plot (plt.Figure, optional): Figure of feature rankings. Defaults to None.

    Returns:
        mlflow.models.ModelInfo: Information about the logged model.
    """
    # End any existing runs
    run = mlflow.active_run()
    if run:
        mlflow.end_run()
        print("Existing run ended.")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5555")

    # Start a new run
    mlflow.start_run()

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)

    # Log parameters
    if params is not None:
        for key, value in params.items():
            mlflow.log_param(key, value)

    # Set tags
    mlflow.set_tag("Scores", name)

    # Log the model
    if "AirPollutionNet" in name:
        with torch.no_grad():
            X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            preds = model(X_tensor).numpy()
        signature = infer_signature(X_train, preds)
        model_info = mlflow.pytorch.log_model(
            model,
            artifact_path=name,
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name=name,
        )
    elif "XGBoost" in name:
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path=name,
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name=name,
        )
    else:
        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=name,
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name=name,
        )

    # Log figures
    mlflow.log_figure(true_vs_pred_plot, "predictions_plot.png")
    mlflow.log_figure(fig_3_days, "rmse_mape_3_days.png")
    if losses_plot:
        mlflow.log_figure(losses_plot, "train_val_loss.png")
    if feature_rankings_plot:
        mlflow.log_figure(feature_rankings_plot, "feature_rankings.png")

    # End the run
    if mlflow.active_run():
        mlflow.end_run()
        print("Run ended.")

    return model_info


def load_model_from_mlflow(model_uri: str):
    """
    Load a model from MLflow given the full model URI.

    Args:
        model_uri (str): The full URI to the model in MLflow.

    Returns:
        model: The loaded model.
    """
    try:
        # Set the MLflow tracking URI
        mlflow.set_tracking_uri("http://localhost:5555")

        # Decide which flavor to use for loading
        if "XGBoost" in model_uri:
            model = mlflow.xgboost.load_model(model_uri)
        elif "AirPollutionNet" in model_uri or "pytorch" in model_uri:
            model = mlflow.pytorch.load_model(model_uri)
        else:
            model = mlflow.pyfunc.load_model(model_uri)

        return model

    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return None


def evaluate_predictions_3_days(
    preds: np.ndarray, y_test: np.ndarray
) -> tuple[list[float], list[float], plt.Figure]:
    """
    Evaluate predictions for the next three days and plot the results.

    Args:
        preds (np.ndarray): Predicted values.
        y_test (np.ndarray): True values.

    Returns:
        tuple: Lists of RMSE and MAPE values, and the figure of the plots.
    """
    # Initialize lists to store RMSE and MAPE for each day
    rmse_values = []
    mape_values = []

    # Calculate RMSE and MAPE for each day (1st, 2nd, and 3rd)
    for i in range(3):
        rmse = root_mean_squared_error(y_test[:, i], preds[:, i])
        mape = mean_absolute_percentage_error(y_test[:, i], preds[:, i], symmetric=True)
        rmse_values.append(rmse)
        mape_values.append(mape)

    days = ["Day 1", "Day 2", "Day 3"]

    # RMSE Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(days, rmse_values, color="blue")
    plt.title("RMSE for Predictions Over 3 Days")
    plt.ylabel("RMSE")
    plt.grid(axis="y")

    # MAPE Plot
    plt.subplot(1, 2, 2)
    plt.bar(days, mape_values, color="orange")
    plt.title("MAPE for Predictions Over 3 Days")
    plt.ylabel("MAPE (%)")
    plt.grid(axis="y")

    plt.tight_layout()

    return rmse_values, mape_values, plt.gcf()


def plot_predictions_vs_true(preds: np.ndarray, y_test: np.ndarray) -> plt.Figure:
    """
    Plot predictions versus true values for the next three days.

    Args:
        preds (np.ndarray): Predicted values.
        y_test (np.ndarray): True values.

    Returns:
        plt.Figure: The matplotlib figure containing the plot.
    """
    # Create subplots for each day
    days = ["Day 1", "Day 2", "Day 3"]

    plt.figure(figsize=(15, 10))

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(y_test[:, i], label="True Values", color="blue", marker="o")
        plt.plot(preds[:, i], label="Predictions", color="orange", marker="x")
        plt.title(f"Predictions vs True Values for {days[i]}")
        plt.xlabel("Time Index")
        plt.ylabel("Particle Levels")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    return plt.gcf()


def plot_losses(total_train_loss, total_val_loss, interval=10):
    """
    Plots the training and validation losses over epochs.

    Parameters:
    - total_train_loss (list): List of average training losses recorded at each interval.
    - total_val_loss (list): List of average validation losses recorded at each interval.
    - interval (int, optional): The number of epochs between each recorded loss. Default is 10.
    """
    epochs = [i * interval for i in range(1, len(total_train_loss) + 1)]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, total_train_loss, label="Training Loss")
    plt.plot(epochs, total_val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    return plt.gcf()
