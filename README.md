# Industry Project

This repository contains data and notebooks for preprocessing and analyzing air quality data specifically focusing on nitrogen dioxide (NO2) and ozone (O3) concentrations alongside meteorological data. This repository is currently still in development with the purpose of it extending over time, eventually leading to a fully trained and deployed machine learning algorithm for O3 and NO2 prediction. This README will be updated periodically alongside with the project.

## Directory Structure

### `app`
This directory contains the files concering the deployed Hugging Face Spaces application and necessary data fetching and processing for real-time prediction.

#### `app/pages`
Contains the additional `admin.py` streamlit page file for the admin panel.

#### `app/src`
Contains the `.py` files for feature pipelines, API calls and various helper functions that are used through the `app.py` application.

### `data`
This directory is divided into two subdirectories: `processed` and `raw`.

#### `data/raw`
Contains raw data files for NO2, O3, and weather:
- `NO2` and `O3`: Yearly and monthly data files from 2010 to mid-2024.
- `weather`: Meteorological data relevant to the study period.

#### `data/processed`
Includes cleaned and combined datasets ready for analysis:
- `clean_weather_data.csv`: Cleaned meteorological data.
- `combined_dataset.csv`: Data combined from weather, NO2, and O3 sources.
- Files named with `NO2_` and `O3_`: Cleaned and segregated files for daily averages, station-specific, final merged, and split datasets (train and test).

### `mlartifacts`
Contains machine learning model artifacts, including:
- Trained models (SVR, XGBoost, Linear Regression, Neural Networks)
- Model performance plots and metrics
- Serialized model files for future use

### `src`
Contains Jupyter notebooks and model files organized into subdirectories for data manipulation, feature engineering, and model development.

#### `src/data`
Scripts and utilities for data pipeline and preprocessing:
- `data_pipeline.py`: Handles data loading and processing.
- `data_preprocessing_utils.py`: Utility functions for data cleaning and preparation.

#### `src/features`
Notebooks and files for feature engineering and preprocessing:
- `data_loading.py`: Functions for loading datasets.

#### `src/models`
Notebooks and scripts for model development and training:
- `ensemble.ipynb`: Ensemble model training pipeline implementation.
- `ensemble.py`: Ensemble model implementation as a class.
- `linear_regression.ipynb`: Linear regression model implementation.
- `svr.ipynb`: Support Vector Regression model implementation.
- `nn.ipynb`: Neural network model training pipeline implementation.
- `nn_hyperparams.py`: Grid search for neural network hyperparameters.
- `XGBoost.ipynb`: XGBoost model implementation and training.
- `XGBoost_hyperparams.py`: Grid search for XGBoost hyperparameters.
- `feature_scaler_NO2.joblib` & `feature_scaler_O3.joblib`: Saved feature scaler files to allow inverse scaling at prediction time.
- `target_scaler_NO2.joblib` & `target_scaler_O3.joblib`: Saved target scaler files to allow inverse scaling at prediction time.

#### `src/utils`
Utility functions for data loading and processing:
- `feature_rankers.py`: Tools for ranking and selecting features.
- `utils.py`: General utility functions.

## Getting Started
To use this repository, clone it locally and navigate to the desired notebooks in the `src` directory. Ensure that Python 3.x is installed along with Jupyter Notebook to run the `.ipynb` files. Install necessary libraries by running `pip install -r requirements.txt` in your local environment.

## Usage
The notebooks and scripts in `src/data` and `src/features` are used for initial data cleaning, inspection, and feature engineering. Once the data is prepared, the notebooks in `src/models` can be used to train and evaluate different machine learning models for NO2 and O3 prediction. `app` contains a Streamlit application that runs real-time in Hugging Face Spaces. The application can be run locally from the `app` directory with `streamlit run app.py`. The application can be also accessed here: https://huggingface.co/spaces/Mihkelmj/utrecht-pollution-prediction

## Contributions
This repository was developed as part of a joint project by Elisa Klunder, Mika Ernesto Umana Lemus, Aksel Joonas Reedi and Mihkel Mariusz Jezierski. Thank you for visiting this repository, and we hope it aids in your air quality analysis efforts.
