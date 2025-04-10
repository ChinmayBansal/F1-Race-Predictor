import os
import fastf1
import pandas as pd
from fastf1.api import driver_info
import numpy as np

# Ensure cache directory exists
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")


def get_race_data(year: int, round_number: int) -> pd.DataFrame:
    try:
        session = fastf1.get_session(year, round_number, 'R')
        session.load()

        laps = session.laps
        results = session.results
        drivers = session.drivers

        # Load driver info via API
        driver_metadata = driver_info(session.api_path)

        race_data = []

        for driver in drivers:
            driver_laps = laps.pick_driver(driver)
            if driver_laps.empty:
                continue

            best_lap = driver_laps["LapTime"].min().total_seconds()
            avg_lap = driver_laps["LapTime"].dt.total_seconds().mean()
            pit_stops = driver_laps["PitInTime"].notnull().sum()
            quali_time = get_qualifying_time(year, round_number, driver)

            full_name = driver_metadata.get(driver, {}).get("FullName", "Unknown")
           

            race_data.append({
                "Driver": driver,
                "Name": full_name,
                "Round": round_number,
                "BestLapTime": best_lap,
                "AvgLapTime": avg_lap,
                "PitStops": pit_stops,
                "QualifyingTime": quali_time,
                
            })

        # Now convert to DataFrame
        df = pd.DataFrame(race_data)

        if df.empty:
            return df

        # Fix: Ensure RacePosition is numeric (convert None to NaN)
        # df["RacePosition"] = pd.to_numeric(df["RacePosition"], errors="coerce")

        # Add team info
        df["Team"] = df["Driver"].map(lambda code: driver_metadata.get(code, {}).get("TeamName", "Unknown"))

        # Calculate team-level averages
        team_avg = df.groupby("Team").agg({
            "BestLapTime": "mean",
            "AvgLapTime": "mean",
            "QualifyingTime": "mean",
           
        }).rename(columns={
            "BestLapTime": "TeamAvgBestLapTime",
            "AvgLapTime": "TeamAvgLapTime",
            "QualifyingTime": "TeamAvgQualiTime",
        }).reset_index()

        # Merge back per driver
        df = df.merge(team_avg, on="Team", how="left")

        return df

    except Exception as e:
        print(f"⚠️ Failed to load race session {year} Round {round_number}: {e}")
        return pd.DataFrame()


def get_qualifying_time(year: int, round_number: int, driver: str):
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
        laps = quali.laps.pick_driver(driver)
        return laps["LapTime"].min().total_seconds()
    except Exception as e:
        print(f"⚠️ Quali session missing or failed for Round {round_number}, Driver {driver}: {e}")
        return None