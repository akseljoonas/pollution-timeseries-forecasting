import os
import warnings

import pandas as pd
from data_pipeline_utils import (
    aggregate_particle_data,
    average_by_day,
    clean_weather_data,
    load_and_average_across_stations,
)

warnings.filterwarnings("ignore")


class DataProcessor:
    """
    A class to process pollution data for specified particles and weather data.
    """

    def __init__(self, particles: list[str], start_year: int):
        """
        Initializes the DataProcessor with the specified particles and start year.

        Args:
            particles (list[str]): List of particle names to process.
            start_year (int): The starting year for data aggregation.
        """
        self.particles = particles
        self.start_year = start_year
        self.particle_data_frames = {particle: pd.DataFrame() for particle in particles}

    def process_particle_data(self) -> None:
        """
        Processes each particle's data by aggregating, averaging, and saving to CSV files.
        """
        for particle in self.particles:
            particle_data = aggregate_particle_data(
                particle, oldest_year=self.start_year
            )
            averaged_by_day = average_by_day(particle_data)
            averaged_by_day.to_csv(
                f"data/processed/{particle}_by_station.csv", index=False
            )

            averaged_by_stations = load_and_average_across_stations(
                data=averaged_by_day, particle=particle
            )
            averaged_by_stations.to_csv(
                f"data/processed/{particle}_across_stations.csv", index=False
            )

            self.particle_data_frames[particle] = averaged_by_stations

    def load_weather_data(self, data_path: str) -> pd.DataFrame:
        """
        Loads and cleans weather data from the specified path.

        Args:
            data_path (str): The path to the weather data CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing the cleaned weather data.
        """
        return clean_weather_data(data_path=data_path)

    def merge_data(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merges particle data with weather data.

        Args:
            weather_data (pd.DataFrame): The cleaned weather data.

        Returns:
            pd.DataFrame: A DataFrame containing merged pollution and weather data.
        """
        weather_data = weather_data.reset_index()
        weather_data["date"] = pd.to_datetime(weather_data["date"])

        for particle in self.particles:
            self.particle_data_frames[particle]["date"] = pd.to_datetime(
                self.particle_data_frames[particle]["date"]
            )

        particles_data = pd.merge(
            self.particle_data_frames["NO2"],
            self.particle_data_frames["O3"],
            on="date",
            how="outer",
        )
        merged_data = pd.merge(particles_data, weather_data, on="date", how="outer")

        return merged_data

    def save_combined_data(self, merged_data: pd.DataFrame, output_path: str) -> None:
        """
        Saves the combined data to a specified CSV file.

        Args:
            merged_data (pd.DataFrame): The DataFrame containing the combined data.
            output_path (str): The file path where the combined data should be saved.
        """
        merged_data.to_csv(output_path, index=False)


def main() -> None:
    """
    Main function to execute the pollution data processing.
    """
    particles = ["O3", "NO2"]
    start_year = 2011

    # Create an instance of DataProcessor
    processor = DataProcessor(particles, start_year)

    # Process particle data
    processor.process_particle_data()

    # Load weather data
    data_path = "../../data/raw/weather/weather_data.csv"
    weather_data = processor.load_weather_data(data_path)

    # Merge weather data and particles
    merged_data = processor.merge_data(weather_data)

    # Save combined data
    output_path = os.path.join(
        os.path.dirname(__file__), "../../data/processed/combined_data.csv"
    )
    processor.save_combined_data(merged_data, output_path)


if __name__ == "__main__":
    main()
