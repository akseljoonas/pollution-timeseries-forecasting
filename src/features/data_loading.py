import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def create_features_and_targets(
    data: pd.DataFrame,
    target_particle: str,
    lag_days: int = 7,
    sma_days: int = 7,
    days_ahead: int = 3,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Creates lagged features, Simple Moving Average (SMA) features, last year's particle data for specific days,
    sine and cosine transformations for 'weekday' and 'month', and target variables for the specified
    particle ('O3' or 'NO2') for the next 'days_ahead' days. Scales features and targets without
    disregarding outliers and saves the scalers for inverse scaling. Splits the data into train,
    validation, and test sets using the most recent dates. Prints the number of rows with missing
    values dropped from the dataset.

    Args:
        data (pd.DataFrame): The input time-series dataset.
        target_particle (str): The target particle ('O3' or 'NO2') for which targets are created.
        lag_days (int): Number of lag days to create features for (default is 7).
        sma_days (int): Window size for Simple Moving Average (default is 7).
        days_ahead (int): Number of days ahead to create target variables for (default is 3).

    Returns:
        tuple: A tuple containing scaled training features, scaled training targets,
               scaled validation features, scaled validation targets,
               scaled test features, and scaled test targets.
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
        lag_features.append("percipitation")
        lag_features.append("pressure")

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
    data["month_sin"] = np.sin(
        2 * np.pi * (data["month"] - 1) / 12
    )  # Adjust month to 0-11
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
    data["O3_last_year"] = data["O3"].shift(365)
    data["NO2_last_year"] = data["NO2"].shift(365)

    # 7 days before today last year
    for i in range(1, lag_days + 1):
        data[f"O3_last_year_{i}_days_before"] = data["O3"].shift(365 + i)
        data[f"NO2_last_year_{i}_days_before"] = data["NO2"].shift(365 + i)

    # 3 days after today last year
    data["O3_last_year_3_days_after"] = data["O3"].shift(365 - 3)
    data["NO2_last_year_3_days_after"] = data["NO2"].shift(365 - 3)

    # Create targets only for the specified particle for the next 'days_ahead' days
    for day in range(1, days_ahead + 1):
        data[f"{target_particle}_plus_{day}_day"] = data[target_particle].shift(-day)

    # Calculate the number of rows before dropping missing values
    rows_before = data.shape[0]

    # Drop missing values
    data = data.dropna().reset_index(drop=True)

    # Calculate the number of rows after dropping missing values
    rows_after = data.shape[0]

    # Calculate and print the number of rows dropped
    rows_dropped = rows_before - rows_after
    print(f"Number of rows with missing values dropped: {rows_dropped}")

    # Now, split data into train, validation, and test sets using the most recent dates
    total_days = data.shape[0]
    test_size = 365
    val_size = 365

    if total_days < test_size + val_size:
        raise ValueError(
            "Not enough data to create validation and test sets of 365 days each."
        )

    # Ensure the data is sorted by date in ascending order
    data = data.sort_values("date").reset_index(drop=True)

    # Split data
    train_data = data.iloc[: -(val_size + test_size)]
    val_data = data.iloc[-(val_size + test_size) : -test_size]
    test_data = data.iloc[-test_size:]

    # Define target columns for the specified particle
    target_cols = [
        f"{target_particle}_plus_{day}_day" for day in range(1, days_ahead + 1)
    ]

    # Define feature columns
    exclude_cols = ["date", "weekday", "month"] + target_cols
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    # Split features and targets
    X_train = train_data[feature_cols]
    y_train = train_data[target_cols]

    X_val = val_data[feature_cols]
    y_val = val_data[target_cols]

    X_test = test_data[feature_cols]
    y_test = test_data[target_cols]

    # Initialize scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit the scalers on the training data
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)

    # Apply the scalers to validation and test data
    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)

    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)

    # Convert scaled data back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=feature_cols, index=X_train.index
    )
    y_train_scaled = pd.DataFrame(
        y_train_scaled, columns=target_cols, index=y_train.index
    )

    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
    y_val_scaled = pd.DataFrame(y_val_scaled, columns=target_cols, index=y_val.index)

    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=feature_cols, index=X_test.index
    )
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=target_cols, index=y_test.index)

    joblib.dump(feature_scaler, f"feature_scaler_{target_particle}.joblib")
    joblib.dump(target_scaler, f"target_scaler_{target_particle}.joblib")

    return (
        X_train_scaled,
        y_train_scaled,
        X_val_scaled,
        y_val_scaled,
        X_test_scaled,
        y_test_scaled,
    )
