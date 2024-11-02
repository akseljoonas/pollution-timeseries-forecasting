import os
import warnings

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from src.past_data_api_calls import get_past_combined_data

warnings.filterwarnings("ignore")

load_dotenv()
login(token=os.getenv("HUGGINGFACE_DOWNLOAD_TOKEN"))


def create_features(
    data: pd.DataFrame,
    target_particle: str,  # Added this parameter
    lag_days: int = 7,
    sma_days: int = 7,
) -> pd.DataFrame:
    """
    Create features for predicting air quality particles (NO2 or O3) based on historical weather data.

    This function performs several feature engineering tasks, including:
    - Creating lagged features for specified pollutants.
    - Calculating rolling mean (SMA) features.
    - Adding sine and cosine transformations of the weekday and month.
    - Incorporating historical data for the same date in the previous year.

    Parameters:
    ----------
    data : pd.DataFrame
        A DataFrame containing historical weather and air quality data with a 'date' column.

    target_particle : str
        The target particle for prediction, must be either 'O3' or 'NO2'.

    lag_days : int, optional
        The number of days for which lagged features will be created. Default is 7.

    sma_days : int, optional
        The window size for calculating the simple moving average (SMA). Default is 7.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the transformed features, ready for modeling.

    Raises:
    ------
    ValueError
        If target_particle is not 'O3' or 'NO2'.
    """
    lag_features = [
        "NO2",
        "O3",
        "wind_speed",
        "mean_temp",
        "global_radiation",
        "minimum_visibility",
        "humidity",
    ]
    if target_particle == "NO2":
        lag_features = lag_features + ["percipitation", "pressure"]

    if target_particle not in ["O3", "NO2"]:
        raise ValueError("target_particle must be 'O3' or 'NO2'")

    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    # Extract 'weekday' and 'month' from 'date' if not present
    if "weekday" not in data.columns or data["weekday"].dtype == object:
        data["weekday"] = data["date"].dt.weekday  # Monday=0, Sunday=6
    if "month" not in data.columns:
        data["month"] = data["date"].dt.month  # 1 to 12

    # Create sine and cosine transformations for 'weekday' and 'month'
    data["weekday_sin"] = np.sin(2 * np.pi * data["weekday"] / 7)
    data["weekday_cos"] = np.cos(2 * np.pi * data["weekday"] / 7)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12)

    # Create lagged features for the specified lag days
    for feature in lag_features:
        for lag in range(1, lag_days + 1):
            data[f"{feature}_lag_{lag}"] = data[feature].shift(lag)

    # Create SMA features
    for feature in lag_features:
        data[f"{feature}_sma_{sma_days}"] = (
            data[feature].rolling(window=sma_days).mean()
        )

    # Create particle data (NO2 and O3) from the same time last year
    past_data = get_past_combined_data()

    # Today last year
    data["O3_last_year"] = past_data["O3"].iloc[-4]
    data["NO2_last_year"] = past_data["NO2"].iloc[-4]

    # 7 days before today last year
    for i in range(1, lag_days + 1):
        data[f"O3_last_year_{i}_days_before"] = past_data["O3"].iloc[i - 1]
        data[f"NO2_last_year_{i}_days_before"] = past_data["NO2"].iloc[i - 1]

    # 3 days after today last year
    data["O3_last_year_3_days_after"] = past_data["O3"].iloc[-1]
    data["NO2_last_year_3_days_after"] = past_data["NO2"].iloc[-1]

    # Drop missing values
    rows_before = data.shape[0]
    data = data.dropna().reset_index(drop=True)
    rows_after = data.shape[0]
    rows_dropped = rows_before - rows_after
    print(f"Number of rows with missing values dropped: {rows_dropped}/{rows_before}")
    print(data)

    # Ensure the data is sorted by date in ascending order
    data = data.sort_values("date").reset_index(drop=True)

    # Define feature columns
    exclude_cols = ["date", "weekday", "month"]
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    # Split features and targets
    x = data[feature_cols]

    # Scale
    repo_id = f"elisaklunder/Utrecht-{target_particle}-Forecasting-Model"
    file_name = f"feature_scaler_{target_particle}.joblib"
    path = hf_hub_download(repo_id=repo_id, filename=file_name)
    feature_scaler = joblib.load(path)
    X_scaled = feature_scaler.transform(x)

    # Convert scaled data back to DataFrame for consistency
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=x.index)

    return X_scaled
