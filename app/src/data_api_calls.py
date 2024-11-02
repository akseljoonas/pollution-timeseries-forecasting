import codecs
import csv
import http.client
import os
import re
import sys
import urllib.request
from datetime import date, timedelta
from io import StringIO

import pandas as pd

WEATHER_DATA_FILE = "weather_data.csv"
POLLUTION_DATA_FILE = "pollution_data.csv"


def update_weather_data() -> None:
    """
    Updates weather data by fetching data.
    If the data file exists, it appends new data. If not, it creates a new file.
    """
    today = date.today().isoformat()

    if os.path.exists(WEATHER_DATA_FILE):
        df = pd.read_csv(WEATHER_DATA_FILE)
        last_date = pd.to_datetime(df["date"]).max()
        start_date = (last_date + timedelta(1)).isoformat()
    else:
        df = pd.DataFrame()
        start_date = (date.today() - timedelta(7)).isoformat()

    try:
        ResultBytes = urllib.request.urlopen(
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Utrecht/{start_date}/{today}?unitGroup=metric&elements=datetime%2Cwindspeed%2Ctemp%2Csolarradiation%2Cprecip%2Cpressure%2Cvisibility%2Chumidity&include=days&key=7Y6AY56M6RWVNHQ3SAVHNJWFS&maxStations=1&contentType=csv"
        )
        CSVText = csv.reader(codecs.iterdecode(ResultBytes, "utf-8"))

        new_data = pd.DataFrame(list(CSVText))
        new_data.columns = new_data.iloc[0]
        new_data = new_data[1:]
        new_data = new_data.rename(columns={"datetime": "date"})

        updated_df = pd.concat([df, new_data], ignore_index=True)
        updated_df.drop_duplicates(subset="date", keep="last", inplace=True)
        updated_df.to_csv(WEATHER_DATA_FILE, index=False)

    except urllib.error.HTTPError as e:
        ErrorInfo = e.read().decode()
        print("Error code: ", e.code, ErrorInfo)
        sys.exit()
    except urllib.error.URLError as e:
        ErrorInfo = e.read().decode()
        print("Error code: ", e.code, ErrorInfo)
        sys.exit()


def update_pollution_data() -> None:
    """
    Updates pollution data for NO2 and O3.
    The new data is appended to the existing pollution data file.
    """
    O3 = []
    NO2 = []
    particles = ["NO2", "O3"]
    stations = ["NL10636", "NL10639", "NL10643"]
    all_dataframes = []
    today = date.today().isoformat() + "T09:00:00Z"
    yesterday = (date.today() - timedelta(1)).isoformat() + "T09:00:00Z"

    if os.path.exists(POLLUTION_DATA_FILE):
        existing_data = pd.read_csv(POLLUTION_DATA_FILE)
        last_date = pd.to_datetime(existing_data["date"]).max()
        if last_date >= pd.Timestamp(date.today()):
            print("Data is already up to date.")
            return

    # Only pull data for today if not already updated
    for particle in particles:
        for station in stations:
            conn = http.client.HTTPSConnection("api.luchtmeetnet.nl")
            payload = ""
            headers = {}
            conn.request(
                "GET",
                f"/open_api/measurements?station_number={station}&formula={particle}&page=1&order_by=timestamp_measured&order_direction=desc&end={today}&start={yesterday}",
                payload,
                headers,
            )
            res = conn.getresponse()
            data = res.read()
            decoded_data = data.decode("utf-8")
            df = pd.read_csv(StringIO(decoded_data))
            df = df.filter(like="value")
            all_dataframes.append(df)
        combined_data = pd.concat(all_dataframes, ignore_index=True)
        values = []

        for row in combined_data:
            cleaned_value = re.findall(r"[-+]?\d*\.\d+|\d+", row)
            if cleaned_value:
                values.append(float(cleaned_value[0]))

        if values:
            avg = sum(values) / len(values)
            if particle == "NO2":
                NO2.append(avg)
            else:
                O3.append(avg)

    new_data = pd.DataFrame(
        {
            "date": [date.today()],
            "NO2": NO2,
            "O3": O3,
        }
    )

    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.drop_duplicates(subset="date", keep="last", inplace=True)

    updated_data.to_csv(POLLUTION_DATA_FILE, index=False)


def get_combined_data() -> pd.DataFrame:
    """
    Combines weather and pollution data for the last 7 days.

    Returns:
        pd.DataFrame: A DataFrame containing the combined weather and pollution data.
    """
    weather_df = pd.read_csv(WEATHER_DATA_FILE)

    today = pd.Timestamp.now().normalize()
    seven_days_ago = today - pd.Timedelta(days=7)
    weather_df["date"] = pd.to_datetime(weather_df["date"])
    weather_df = weather_df[
        (weather_df["date"] >= seven_days_ago) & (weather_df["date"] <= today)
    ]

    weather_df.insert(1, "NO2", None)
    weather_df.insert(2, "O3", None)
    weather_df.insert(10, "weekday", None)
    columns = list(weather_df.columns)
    columns.insert(3, columns.pop(6))
    weather_df = weather_df[columns]
    columns.insert(5, columns.pop(9))
    weather_df = weather_df[columns]
    columns.insert(9, columns.pop(6))
    weather_df = weather_df[columns]

    combined_df = weather_df

    # Apply scaling and renaming similar to the scale function from previous code
    combined_df = combined_df.rename(
        columns={
            "date": "date",
            "windspeed": "wind_speed",
            "temp": "mean_temp",
            "solarradiation": "global_radiation",
            "precip": "percipitation",
            "sealevelpressure": "pressure",
            "visibility": "minimum_visibility",
        }
    )

    combined_df["date"] = pd.to_datetime(combined_df["date"])
    combined_df["weekday"] = combined_df["date"].dt.day_name()

    combined_df["wind_speed"] = (combined_df["wind_speed"] / 3.6) * 10
    combined_df["mean_temp"] = combined_df["mean_temp"] * 10
    combined_df["minimum_visibility"] = combined_df["minimum_visibility"] * 10
    combined_df["percipitation"] = combined_df["percipitation"] * 10
    combined_df["pressure"] = combined_df["pressure"] * 10

    combined_df["wind_speed"] = combined_df["wind_speed"].astype(int)
    combined_df["mean_temp"] = combined_df["mean_temp"].astype(int)
    combined_df["minimum_visibility"] = combined_df["minimum_visibility"].astype(int)
    combined_df["percipitation"] = combined_df["percipitation"].astype(int)
    combined_df["pressure"] = combined_df["pressure"].astype(int)
    combined_df["humidity"] = combined_df["humidity"].astype(int)
    combined_df["global_radiation"] = combined_df["global_radiation"].astype(int)

    pollution_df = pd.read_csv(POLLUTION_DATA_FILE)

    pollution_df["date"] = pd.to_datetime(pollution_df["date"])
    pollution_df = pollution_df[
        (pollution_df["date"] >= seven_days_ago) & (pollution_df["date"] <= today)
    ]

    combined_df["NO2"] = pollution_df["NO2"]
    combined_df["O3"] = pollution_df["O3"]

    return combined_df
