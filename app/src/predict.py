import os
from datetime import date, datetime, timedelta

import joblib
import pandas as pd
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from src.data_api_calls import (
    get_combined_data,
    update_pollution_data,
    update_weather_data,
)
from src.features_pipeline import create_features

load_dotenv()
login(token=os.getenv("HUGGINGFACE_DOWNLOAD_TOKEN"))


def load_nn() -> torch.nn.Module:
    """
    Loads the neural network model for air pollution forecasting.

    Returns:
        torch.nn.Module: The loaded neural network model.
    """
    import torch.nn as nn
    from huggingface_hub import PyTorchModelHubMixin

    class AirPollutionNet(nn.Module, PyTorchModelHubMixin):
        def __init__(self, input_size: int, layers: list[int], dropout_rate: float):
            super(AirPollutionNet, self).__init__()
            self.layers_list = nn.ModuleList()
            in_features = input_size

            for units in layers:
                self.layers_list.append(nn.Linear(in_features, units))
                self.layers_list.append(nn.ReLU())
                self.layers_list.append(nn.Dropout(p=dropout_rate))
                in_features = units

            self.output = nn.Linear(in_features, 3)  # Output size is 3 for next 3 days

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the neural network.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the network.
            """
            for layer in self.layers_list:
                x = layer(x)
            x = self.output(x)
            return x

    model = AirPollutionNet.from_pretrained(
        "akseljoonas/Utrecht_pollution_forecasting_NO2"
    )
    return model


def load_model(particle: str) -> object:
    """
    Loads the forecasting model based on the specified particle.

    Args:
        particle (str): The type of particle ("O3" or "NO2").

    Returns:
        object: The loaded model (either a neural network or a support vector regression model).
    """
    repo_id = f"elisaklunder/Utrecht-{particle}-Forecasting-Model"
    if particle == "O3":
        file_name = "O3_svr_model.pkl"
        model_path = hf_hub_download(repo_id=repo_id, filename=file_name)
        model = joblib.load(model_path)
    else:
        model = load_nn()

    return model


def run_model(particle: str, data: pd.DataFrame) -> list:
    """
    Runs the model for the specified particle and makes predictions based on the input data.

    Args:
        particle (str): The type of particle ("O3" or "NO2").
        data (pd.DataFrame): The input data for making predictions.

    Returns:
        list: The predictions for the specified particle.
    """
    input_data = create_features(data=data, target_particle=particle)
    model = load_model(particle)

    if particle == "NO2":
        with torch.no_grad():
            prediction = model(torch.tensor(input_data.values, dtype=torch.float32))
        repo_id = "akseljoonas/Utrecht_pollution_forecasting_NO2"
        file_name = "target_scaler_NO2.joblib"
        path = hf_hub_download(repo_id=repo_id, filename=file_name)
    else:
        prediction = model.predict(input_data)

        repo_id = f"elisaklunder/Utrecht-{particle}-Forecasting-Model"
        file_name = f"target_scaler_{particle}.joblib"
        path = hf_hub_download(repo_id=repo_id, filename=file_name)

    target_scaler = joblib.load(path)
    prediction = target_scaler.inverse_transform(prediction)

    return prediction


def update_data_and_predictions() -> None:
    """
    Updates the weather and pollution data, makes predictions for O3 and NO2,
    and stores them in a CSV file.
    """
    update_weather_data()
    update_pollution_data()

    week_data = get_combined_data()

    o3_predictions = run_model("O3", data=week_data)
    no2_predictions = run_model("NO2", data=week_data)

    prediction_data = []
    for i in range(3):
        prediction_data.append(
            {
                "pollutant": "O3",
                "date_predicted": date.today(),
                "date": date.today() + timedelta(days=i + 1),
                "prediction_value": o3_predictions[0][i],
            }
        )
        prediction_data.append(
            {
                "pollutant": "NO2",
                "date_predicted": date.today(),
                "date": date.today() + timedelta(days=i + 1),
                "prediction_value": no2_predictions[0][i],
            }
        )

    predictions_df = pd.DataFrame(prediction_data)

    PREDICTIONS_FILE = "predictions_history.csv"

    if os.path.exists(PREDICTIONS_FILE):
        existing_data = pd.read_csv(PREDICTIONS_FILE)
        # Filter out predictions made today to avoid duplicates
        existing_data = existing_data[
            ~(existing_data["date_predicted"] == str(date.today()))
        ]
        combined_data = pd.concat([existing_data, predictions_df])
        combined_data.drop_duplicates()
    else:
        combined_data = predictions_df

    combined_data.to_csv(PREDICTIONS_FILE, index=False)


def get_data_and_predictions() -> tuple[pd.DataFrame, list, list]:
    """
    Retrieves combined data and today's predictions for O3 and NO2.

    Returns:
        tuple: A tuple containing:
            - week_data (pd.DataFrame): The combined data for the week.
            - list: Predictions for O3.
            - list: Predictions for NO2.
    """
    week_data = get_combined_data()

    PREDICTIONS_FILE = "predictions_history.csv"
    data = pd.read_csv(PREDICTIONS_FILE)

    today = datetime.today().strftime("%Y-%m-%d")
    today_predictions = data[(data["date_predicted"] == today)]

    # Extract predictions for O3 and NO2
    o3_predictions = today_predictions[today_predictions["pollutant"] == "O3"][
        "prediction_value"
    ].values
    no2_predictions = today_predictions[today_predictions["pollutant"] == "NO2"][
        "prediction_value"
    ].values

    return week_data, [o3_predictions], [no2_predictions]


if __name__ == "__main__":
    update_data_and_predictions()
