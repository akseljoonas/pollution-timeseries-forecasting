import os
from datetime import datetime

import numpy as np
import pandas as pd

particles = ["O3", "NO2"]


def to_datetime(s: str) -> datetime:
    """
    Converts a string to a datetime object.

    Args:
        s (str): The string to convert.

    Returns:
        datetime: The converted datetime object.
    """
    try:
        time_info = datetime.strptime(s, "%Y%m%d %H:%M")
    except ValueError:
        time_info = datetime.fromisoformat(s).replace(tzinfo=None)

    return time_info


def load_data_particle_one_year(file_path: str, file_identifier: str) -> pd.DataFrame:
    """
    Loads particle data from a specified CSV file and processes it.

    Args:
        file_path (str): The path to the CSV file.
        file_identifier (str): Identifier for logging purposes (e.g., year/month).

    Returns:
        pd.DataFrame: A DataFrame containing the processed particle data.
    """
    data = pd.read_csv(file_path, sep=";", encoding="ISO-8859-1")
    required_columns = ["Unnamed: 3", "NL10636", "NL10639", "NL10643"]
    available_columns = [col for col in required_columns if col in data.columns]
    missing_columns = [col for col in required_columns if col not in available_columns]

    if missing_columns:
        print(
            f"Year/Month {file_identifier}: Missing columns - {', '.join(missing_columns)}"
        )

    data = data[available_columns]
    data.rename(columns={"Unnamed: 3": "time"}, inplace=True)
    data = data[9:-1]

    if "time" in data.columns:
        data["time"] = data["time"].map(to_datetime)  # Coerce errors to NaT
    else:
        print(f"Year/Month {file_identifier}: 'time' column is missing.")

    return data


def aggregate_particle_data(particle: str, oldest_year: int) -> pd.DataFrame:
    """
    Aggregates particle data across multiple years.

    Args:
        particle (str): The name of the particle (e.g., "O3", "NO2").
        oldest_year (int): The oldest year of data to include.

    Returns:
        pd.DataFrame: A DataFrame containing all aggregated particle data.
    """
    print(f"\n Processing particle {particle}")
    all_data = pd.DataFrame()
    file_identifiers = []
    while oldest_year != 2024:
        file_identifiers.append(oldest_year)
        oldest_year += 1
    file_identifiers += [
        "2024_01",
        "2024_02",
        "2024_03",
        "2024_04",
        "2024_05",
        "2024_06",
    ]

    for i in file_identifiers:
        input_file_path = os.path.join(
            os.path.dirname(__file__), f"../../data/raw/{particle}/{i}_{particle}.csv"
        )
        data = load_data_particle_one_year(input_file_path, i)
        all_data = pd.concat([all_data, data], ignore_index=True)

    # reset index after concatenation
    all_data.reset_index(drop=True, inplace=True)

    return all_data


def average_by_day(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Averages hourly data per day.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing hourly data.

    Returns:
        pd.DataFrame: A DataFrame with daily averages.
    """
    data = []

    for i in range(0, len(dataframe), 24):
        slice = dataframe.iloc[i : i + 24]
        time = slice["time"].iloc[0].date()

        NL10636 = NL10639 = NL10643 = np.nan

        if "NL10636" in slice.columns:
            NL10636 = (
                slice["NL10636"]
                .map(lambda n: float(n) if pd.notnull(n) else np.nan)
                .mean()
            )
        if "NL10639" in slice.columns:
            NL10639 = (
                slice["NL10639"]
                .map(lambda n: float(n) if pd.notnull(n) else np.nan)
                .mean()
            )
        if "NL10643" in slice.columns:
            NL10643 = (
                slice["NL10643"]
                .map(lambda n: float(n) if pd.notnull(n) else np.nan)
                .mean()
            )

        data.append([time, NL10636, NL10639, NL10643])

    return pd.DataFrame(data, columns=["time", "NL10636", "NL10639", "NL10643"])


def load_and_average_across_stations(data: pd.DataFrame, particle: str) -> pd.DataFrame:
    """
    Averages particle data across multiple stations.

    Args:
        data (pd.DataFrame): A DataFrame containing the data for the stations.
        particle (str): The name of the particle to average.

    Returns:
        pd.DataFrame: A DataFrame containing the averaged particle concentration.
    """
    df = data
    station_columns = [col for col in df.columns if col not in ["time", "Unnamed: 0"]]
    df[station_columns] = df[station_columns].apply(pd.to_numeric, errors="coerce")
    df[particle] = df[station_columns].mean(axis=1, skipna=True)

    return df[["time", particle]].rename(columns={"time": "date"})


def filter_data_by_date(
    data: pd.DataFrame, start_date: str = None, end_date: str = None
) -> pd.DataFrame:
    """
    Filters the data by a specified date range.

    Args:
        data (pd.DataFrame): The DataFrame to filter.
        start_date (str, optional): The start date for filtering (format: "YYYY-MM-DD").
        end_date (str, optional): The end date for filtering (format: "YYYY-MM-DD").

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if start_date is not None:
        start_date = pd.to_datetime(start_date, format="%Y-%m-%d")
    if end_date is not None:
        end_date = pd.to_datetime(end_date, format="%Y-%m-%d")

    # Filter based on the date range
    if start_date and end_date:
        return data.loc[start_date:end_date]
    elif start_date:
        return data.loc[start_date:]
    elif end_date:
        return data.loc[:end_date]
    else:
        return data


def clean_weather_data(
    data_path: str, start_date: str = "2011-01-01", end_date: str = "2024-06-30"
) -> pd.DataFrame:
    """
    Cleans weather data from a specified CSV file.

    Args:
        data_path (str): The path to the weather data CSV file.
        start_date (str, optional): The start date for filtering the data (default: "2011-01-01").
        end_date (str, optional): The end date for filtering the data (default: "2024-06-30").

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned weather data.
    """
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), data_path))
    data = data.drop("STN", axis=1)  # Dropping the weather station code
    data = data.replace(
        {pd.NA: None, "": None, pd.NA: None, pd.NaT: None, float("nan"): None}
    )

    # Changing relevant columns to type Int64
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").astype("Int64")

    # Convert the 'YYYYMMDD' column to datetime and set as index
    data["date"] = pd.to_datetime(data["YYYYMMDD"].astype(str), format="%Y%m%d")
    data["weekday"] = data["date"].dt.day_name()
    data = data.set_index("date")
    data = data.drop("YYYYMMDD", axis=1)

    weather_data = filter_data_by_date(data, start_date, end_date)
    return weather_data
