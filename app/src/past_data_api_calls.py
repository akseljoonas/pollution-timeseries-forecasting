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

PAST_WEATHER_DATA_FILE = "past_weather_data.csv"
PAST_POLLUTION_DATA_FILE = "past_pollution_data.csv"


def update_past_weather_data() -> None:
    """
    Updates past weather data.
    The data is saved to a CSV file. If the file already exists, new data is appended.
    """
    last_year_date = date.today() - timedelta(days=365)

    if os.path.exists(PAST_WEATHER_DATA_FILE):
        df = pd.read_csv(PAST_WEATHER_DATA_FILE)
        start_date = pd.to_datetime(df["date"]).max().date().isoformat()
        end_date = (last_year_date + timedelta(days=2)).isoformat()
    else:
        df = pd.DataFrame()
        start_date = (last_year_date - timedelta(days=8)).isoformat()
        end_date = (last_year_date + timedelta(days=2)).isoformat()

    try:
        ResultBytes = urllib.request.urlopen(
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Utrecht/{start_date}/{end_date}?unitGroup=metric&elements=datetime%2Cwindspeed%2Ctemp%2Csolarradiation%2Cprecip%2Cpressure%2Cvisibility%2Chumidity&include=days&key=7Y6AY56M6RWVNHQ3SAVHNJWFS&maxStations=1&contentType=csv"
        )
        CSVText = csv.reader(codecs.iterdecode(ResultBytes, "utf-8"))

        data = pd.DataFrame(list(CSVText))
        data.columns = data.iloc[0]
        data = data[1:]
        data = data.rename(columns={"datetime": "date"})

        updated_df = pd.concat([df, data], ignore_index=True)
        updated_df.drop_duplicates(subset="date", keep="last", inplace=True)
        updated_df.to_csv(PAST_WEATHER_DATA_FILE, index=False)

    except urllib.error.HTTPError as e:
        ErrorInfo = e.read().decode()
        print("Error code: ", e.code, ErrorInfo)
        sys.exit()
    except urllib.error.URLError as e:
        ErrorInfo = e.read().decode()
        print("Error code: ", e.code, ErrorInfo)
        sys.exit()


def update_past_pollution_data() -> tuple[list[float], list[float]]:
    """
    Updates past pollution data for NO2 and O3.

    Returns:
        tuple: A tuple containing two lists with NO2 and O3 average values.
    """
    O3 = []
    NO2 = []
    particles = ["NO2", "O3"]
    stations = ["NL10636", "NL10639", "NL10643"]
    all_dataframes = []

    last_year_date = date.today() - timedelta(days=365)

    if os.path.exists(PAST_POLLUTION_DATA_FILE):
        existing_data = pd.read_csv(PAST_POLLUTION_DATA_FILE)
        last_date = pd.to_datetime(existing_data["date"]).max()
        if last_date >= pd.to_datetime(last_year_date):
            print("Data is already up to date.")
            return [], []
        else:
            start_date = last_date.date()
            end_date = last_year_date + timedelta(days=3)
    else:
        existing_data = pd.DataFrame()
        start_date = last_year_date - timedelta(days=7)
        end_date = last_year_date + timedelta(days=3)

    date_list = [
        start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
    ]
    for current_date in date_list:
        today = current_date.isoformat() + "T09:00:00Z"
        yesterday = (current_date - timedelta(1)).isoformat() + "T09:00:00Z"
        for particle in particles:
            all_dataframes = []  # Reset for each particle
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
            "date": date_list,
            "NO2": NO2,
            "O3": O3,
        }
    )

    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    updated_data.drop_duplicates(subset="date", keep="last", inplace=True)

    updated_data.to_csv(PAST_POLLUTION_DATA_FILE, index=False)

    return NO2, O3


def get_past_combined_data() -> pd.DataFrame:
    """
    Retrieves and combines past weather and pollution data.

    Returns:
        pd.DataFrame: A DataFrame containing the combined past weather and pollution data.
    """
    update_past_weather_data()
    update_past_pollution_data()

    combined_df = pd.read_csv(PAST_WEATHER_DATA_FILE)
    pollution_data = pd.read_csv(PAST_POLLUTION_DATA_FILE)

    combined_df = combined_df.merge(pollution_data, on="date", how="inner")
    combined_df = combined_df.tail(11)

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

    combined_df["wind_speed"] = combined_df["wind_speed"].astype(float)
    combined_df["mean_temp"] = combined_df["mean_temp"].astype(float)
    combined_df["minimum_visibility"] = combined_df["minimum_visibility"].astype(float)
    combined_df["percipitation"] = combined_df["percipitation"].astype(float)
    combined_df["pressure"] = combined_df["pressure"].astype(float).round()
    combined_df["humidity"] = combined_df["humidity"].astype(float).round()
    combined_df["global_radiation"] = combined_df["global_radiation"].astype(float)

    combined_df["wind_speed"] = (combined_df["wind_speed"] / 3.6) * 10
    combined_df["mean_temp"] = combined_df["mean_temp"] * 10
    combined_df["minimum_visibility"] = combined_df["minimum_visibility"] * 10
    combined_df["percipitation"] = combined_df["percipitation"] * 10
    combined_df["pressure"] = combined_df["pressure"] * 10

    combined_df["wind_speed"] = (
        combined_df["wind_speed"].astype(float).round().astype(int)
    )
    combined_df["mean_temp"] = (
        combined_df["mean_temp"].astype(float).round().astype(int)
    )
    combined_df["minimum_visibility"] = (
        combined_df["minimum_visibility"].astype(float).round().astype(int)
    )
    combined_df["percipitation"] = (
        combined_df["percipitation"].astype(float).round().astype(int)
    )
    combined_df["pressure"] = combined_df["pressure"].astype(float).round().astype(int)
    combined_df["humidity"] = combined_df["humidity"].astype(float).round().astype(int)
    combined_df["global_radiation"] = (
        combined_df["global_radiation"].astype(float).round().astype(int)
    )

    return combined_df
