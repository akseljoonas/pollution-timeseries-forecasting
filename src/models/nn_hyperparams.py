import copy
import os
import sys
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Push the best model to Hugging Face
from dotenv import load_dotenv
from sklearn.metrics import root_mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure you have your custom imports available
sys.path.append(os.path.abspath("../features"))
import joblib
from features.data_loading import create_features_and_targets

sys.path.append(os.path.abspath("../utils"))

# For pushing models to Hugging Face
from huggingface_hub import PyTorchModelHubMixin

# Set device
load_dotenv()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data(
    particle: str,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load and preprocess data for a specific particle.

    Args:
        particle (str): The name of the particle for which to load data.

    Returns:
        tuple: A tuple containing training, testing, and validation data.
    """
    data_path = "/Users/akseljoonas/Documents/Kool/ML4I/group-15-project/data/processed/combined_data.csv"
    data = pd.read_csv(data_path)
    X_train, y_train, X_test, y_test, X_val, y_val = create_features_and_targets(
        data, particle
    )
    return X_train, y_train, X_test, y_test, X_val, y_val


class AirPollutionDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Initialize the dataset.

        Args:
            X (pd.DataFrame): Feature data.
            y (pd.Series): Target data.
        """
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the feature and target tensors.
        """
        return self.X[idx], self.y[idx]


class AirPollutionNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, input_size: int, layers: list[int], dropout_rate: float) -> None:
        """
        Initialize the neural network model.

        Args:
            input_size (int): Number of input features.
            layers (list[int]): List of integers representing the number of units in each layer.
            dropout_rate (float): Dropout rate for regularization.
        """
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
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after forward pass.
        """
        for layer in self.layers_list:
            x = layer(x)
        x = self.output(x)
        return x


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    patience: int,
) -> tuple[nn.Module, float, int, list[float], list[float]]:
    """
    Train the model with early stopping.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): Dataloader for training data.
        val_dataloader (DataLoader): Dataloader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        tuple: The best model, best validation loss, best epoch, training loss history, and validation loss history.
    """
    best_loss = float("inf")
    stopping_counter = 0
    best_model = None
    best_epoch = 0
    total_train_loss = []
    total_val_loss = []
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for inputs, targets in train_dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = torch.sqrt(criterion(outputs, targets))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = torch.sqrt(criterion(outputs, targets))
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        total_train_loss.append(np.mean(train_losses))
        total_val_loss.append(val_loss)

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            stopping_counter = 0
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return best_model, best_loss, best_epoch, total_train_loss, total_val_loss


def evaluate_model(
    model: nn.Module, X_test: pd.DataFrame, y_test: pd.Series, scaler
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The trained model.
        X_test (pd.DataFrame): Feature data for testing.
        y_test (pd.Series): Target data for testing.
        scaler: Scaler used to inverse transform the predictions.

    Returns:
        tuple: Predicted values and actual values.
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        preds = model(X_test_tensor).numpy()
    preds = scaler.inverse_transform(preds)
    y_test = scaler.inverse_transform(y_test)
    return preds, y_test


def grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: dict,
    num_epochs: int = 1000,
    patience: int = 20,
    particle: str = "particle",
) -> tuple[nn.Module, dict, float, np.ndarray, np.ndarray, list[float], list[float]]:
    """
    Perform grid search to find the best hyperparameters for the model.

    Args:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target data.
        X_val (pd.DataFrame): Validation feature data.
        y_val (pd.Series): Validation target data.
        X_test (pd.DataFrame): Testing feature data.
        y_test (pd.Series): Testing target data.
        param_grid (dict): Dictionary of hyperparameters to search over.
        num_epochs (int): Number of epochs to train.
        patience (int): Number of epochs for early stopping.
        particle (str): The name of the particle being processed.

    Returns:
        tuple: The best model, best hyperparameters, best validation loss, predictions, true values, training loss history, and validation loss history.
    """
    input_size = X_train.shape[1]

    # Hyperparameter grid
    keys = param_grid.keys()
    best_model = None
    best_loss = float("inf")
    best_params = None
    best_train_loss = None
    best_val_loss = None

    total_combinations = len(list(product(*param_grid.values())))

    for values in tqdm(
        product(*param_grid.values()), total=total_combinations, desc="Grid Search"
    ):
        params = dict(zip(keys, values))
        print(f"\nTraining with params: {params}")

        # Prepare datasets and dataloaders with current batch_size
        batch_size = params["batch_size"]
        train_dataset = AirPollutionDataset(X_train, y_train)
        val_dataset = AirPollutionDataset(X_val, y_val)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        model = AirPollutionNet(
            input_size=input_size,
            layers=params["layers"],
            dropout_rate=params["dropout_rate"],
        )

        # Criterion and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        # Train the model
        trained_model, val_loss, best_epoch, total_train_loss, total_val_loss = (
            train_model(
                model,
                train_dataloader,
                val_dataloader,
                criterion,
                optimizer,
                num_epochs,
                patience,
            )
        )

        print(
            f"Params: {params}, Best Val Loss: {val_loss:.4f} at epoch {best_epoch+1}"
        )

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = trained_model
            best_params = params
            best_train_loss = total_train_loss
            best_val_loss = total_val_loss

    # Load the target scaler

    loaded_scaler = joblib.load(f"target_scaler_{particle}.joblib")
    preds, y_test = evaluate_model(best_model, X_test, y_test, loaded_scaler)

    return (
        best_model,
        best_params,
        best_loss,
        preds,
        y_test,
        best_train_loss,
        best_val_loss,
    )


def main():
    particles = ["O3"]  # , "NO2"]
    for particle in particles:
        print(f"\nProcessing particle: {particle}")

        X_train, y_train, X_test, y_test, X_val, y_val = load_data(particle)

        # Hyperparameter grid
        param_grid = {
            "layers": [
                [512, 1028, 512, 64],
                [512, 512, 254, 128],
                [254, 254, 128, 64],
            ],
            "dropout_rate": [0.2],
            "batch_size": [1],
            "learning_rate": [5e-05, 1e-05, 5e-06],
        }

        # Perform grid search
        best_model, best_params, best_loss, preds, y_test, train_loss, val_loss = (
            grid_search(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                param_grid,
                num_epochs=1000,
                patience=20,
                particle=particle,
            )
        )

        print(f"Best Hyperparameters for {particle}: {best_params}")
        print(f"Best Validation Loss for {particle}: {best_loss:.4f}")

        rmse = root_mean_squared_error(y_test, preds)
        smape = mean_absolute_percentage_error(preds, y_test, symmetric=True)
        print(f"Test RMSE for {particle}: {rmse:.4f}")
        print(f"Test SMAPE for {particle}: {smape:.4f}")

        if best_model:
            best_model.push_to_hub(
                f"akseljoonas/Utrecht_pollution_forecasting_{particle}",
                token=os.getenv("HF_WRITE"),
            )

        # Optionally, save metrics to a file or print them
        with open(f"metrics_{particle}.txt", "w") as f:
            f.write(f"Best Hyperparameters: {best_params}\n")
            f.write(f"Best Validation Loss: {best_loss:.4f}\n")
            f.write(f"Test RMSE: {rmse:.4f}\n")
            f.write(f"Test SMAPE: {smape:.4f}\n")

        print(f"Model for {particle} saved to Hugging Face successfully.")


if __name__ == "__main__":
    main()
