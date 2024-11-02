import mlflow
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from utils import load_model_from_mlflow


class EnsembleModel:
    def __init__(self, particle: str):
        self.model = None
        self.particle = particle
        self.base_models = {}

    def load_base_models(self):
        if self.particle == "O3":
            self.base_models["lr"] = load_model_from_mlflow(
                "mlflow-artifacts:/0/208c8e05b24d42beaf64efd31ed348da/artifacts/LR_O3"
            )
            self.base_models["xgboost"] = load_model_from_mlflow(
                "mlflow-artifacts:/0/65b1d3c935e24d499cc5f4511d13d780/artifacts/XGBoost_O3"
            )
            self.base_models["svr"] = load_model_from_mlflow(
                "mlflow-artifacts:/0/ef9774c0aa0146af9fc11528897b82e2/artifacts/SVR_O3"
            )
            self.base_models["nn"] = load_model_from_mlflow(
                "mlflow-artifacts:/0/56aebbf53b3e4312acfdf3f66e3f3d49/artifacts/AirPollutionNet_O3"
            )
        else:
            self.base_models["lr"] = load_model_from_mlflow(
                "mlflow-artifacts:/0/9649a825898d435a913b6b127eb03e98/artifacts/LR_NO2"
            )
            self.base_models["xgboost"] = load_model_from_mlflow(
                "mlflow-artifacts:/0/99db3f49541d431788afc2635971bfad/artifacts/XGBoost_NO2"
            )
            self.base_models["svr"] = load_model_from_mlflow(
                "mlflow-artifacts:/0/326f74ae1888404a88c856705ef7b928/artifacts/SVR_NO2"
            )
            self.base_models["nn"] = load_model_from_mlflow(
                "mlflow-artifacts:/0/d099f400845944009d2f2a04462728fa/artifacts/AirPollutionNet_NO2"
            )

    def train(self, X_train, y_train):
        lr_preds = self.base_models["lr"].predict(X_train)
        xgboost_preds = self.base_models["xgboost"].predict(X_train)
        svr_preds = self.base_models["svr"].predict(X_train)
        print(X_train.shape)
        with torch.no_grad():
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            nn_preds = self.base_models["nn"](X_train_tensor).numpy()

        meta_features = np.concatenate(
            (lr_preds, xgboost_preds, svr_preds, nn_preds), axis=1
        )  # Shape: (3554, 9)

        self.models = []  # store models for n+1, n+2, n+3
        for i in range(y_train.shape[1]):
            y_train_single = y_train.iloc[:, i].values  # Shape: (3554,)

            # Train a linear regression model as the meta-learner for the current time point
            model = LinearRegression()
            model.fit(meta_features, y_train_single)
            self.models.append(model)

        with mlflow.start_run():
            for idx, model in enumerate(self.models):
                mlflow.sklearn.log_model(model, f"ensemble_model_timepoint_{idx}")

    def load_model(self):
        self.model = load_model_from_mlflow(f"ensemble_model_{self.particle}")

    def predict(self, X_test):
        lr_preds = self.base_models["lr"].predict(X_test)
        xgboost_preds = self.base_models["xgboost"].predict(X_test)
        svr_preds = self.base_models["svr"].predict(X_test)

        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            nn_preds = self.base_models["nn"](X_test_tensor).numpy()

        meta_features = np.concatenate(
            (lr_preds, xgboost_preds, svr_preds, nn_preds), axis=1
        )  # Shape: (n_samples, 9)

        predictions = []
        for model in self.models:
            predictions.append(model.predict(meta_features))

        return np.column_stack(predictions)  # Shape: (n_samples, 3)
