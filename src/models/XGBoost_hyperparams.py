import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


def grid_search(
    model: XGBRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators_range: list[int],
    learning_rate_range: list[float],
) -> XGBRegressor:
    """
    Perform a grid search for hyperparameter tuning.

    Args:
        model (XGBRegressor): The XGBoost model to tune.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        n_estimators_range (list[int]): Range of n_estimators values to try.
        learning_rate_range (list[float]): Range of learning rates to try.

    Returns:
        XGBRegressor: The best estimator found during the grid search.
    """
    parameters = {
        "n_estimators": n_estimators_range,
        "learning_rate": learning_rate_range,
    }

    clf = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="neg_root_mean_squared_error",
        verbose=2,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    print("Best Parameters:", clf.best_params_)
    print("Best Score:", clf.best_score_)
    return clf.best_estimator_


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    static_n_estimators: int,
    static_learning_rate: float,
) -> float:
    """
    Objective function for Optuna to optimize hyperparameters.

    Args:
        trial (optuna.Trial): The current trial instance.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        X_val (np.ndarray): Validation feature data.
        y_val (np.ndarray): Validation target data.
        static_n_estimators (int): Static number of estimators.
        static_learning_rate (float): Static learning rate.

    Returns:
        float: The root mean squared error of the model predictions on the validation set.
    """
    model = XGBRegressor(
        learning_rate=static_learning_rate,
        n_estimators=static_n_estimators,
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 5),
        subsample=trial.suggest_float("subsample", 0.5, 0.9),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
        gamma=trial.suggest_float("gamma", 0, 0.4),
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mse = np.sqrt(mean_squared_error(y_val, preds))
    return mse


def bayesian_search(
    model: XGBRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    static_n_estimators: int,
    static_learning_rate: float,
) -> optuna.Trial:
    """
    Perform a Bayesian optimization search for hyperparameter tuning using Optuna.

    Args:
        model (XGBRegressor): The XGBoost model to tune.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        static_n_estimators (int): Static number of estimators.
        static_learning_rate (float): Static learning rate.

    Returns:
        optuna.Trial: The best trial found during the optimization.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            X_train,
            y_train,
            X_val,
            y_val,
            static_n_estimators,
            static_learning_rate,
        ),
        n_trials=100,
    )

    # Print the best parameters
    print("Best Parameters from Optuna:")
    print(study.best_params)
    return study.best_trial
