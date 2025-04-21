import os
import fastf1
import pandas as pd
from fastf1.api import driver_info
from fastf1.ergast import Ergast
import numpy as np

# Ensure cache dir
os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

ergast = Ergast()

def get_race_data(year: int, round_number: int) -> pd.DataFrame:
    try:
        # Load race session
        session = fastf1.get_session(year, round_number, 'R')
        session.load()
        laps = session.laps
        drivers = session.drivers

        # Load driver metadata
        metadata = driver_info(session.api_path)

        # Load standings from Ergast
        driver_standings = ergast.get_driver_standings(season=year, round=round_number).content[0]
        constructor_standings = ergast.get_constructor_standings(season=year, round=round_number).content[0]

        # Create mapping dicts for quick access
        driver_points = {
            standing["Driver"]["code"]: int(standing["points"])
            for standing in driver_standings
            if "code" in standing["Driver"]
        }

        team_points = {
            standing["Constructor"]["name"]: int(standing["points"])
            for standing in constructor_standings
        }

        # Start collecting driver data
        race_data = []

        for driver in drivers:
            driver_laps = laps.pick_driver(driver)
            if driver_laps.empty:
                continue

            # Compute core stats
            best_lap = driver_laps["LapTime"].min()
            avg_lap = driver_laps["LapTime"].mean()

            if pd.isna(best_lap) or pd.isna(avg_lap):
                continue

            best_lap_sec = best_lap.total_seconds()
            avg_lap_sec = avg_lap.total_seconds()

            quali_time = get_qualifying_time(year, round_number, driver)
            if quali_time is None:
                continue

            driver_info = metadata.get(driver, {})
            full_name = driver_info.get("FullName", "Unknown")
            team_name = driver_info.get("TeamName", "Unknown")
            driver_code = driver_info.get("Tla", driver)  # fallback: 3-letter code

            driver_points_val = driver_points.get(driver_code, 0)
            team_points_val = team_points.get(team_name, 0)

            race_data.append({
                "Driver": driver,
                "Name": full_name,
                "DriverCode": driver_code,
                "Round": round_number,
                "Team": team_name,
                "BestLapTime": best_lap_sec,
                "AvgLapTime": avg_lap_sec,
                "QualifyingTime": quali_time,
                "DriverPoints": driver_points_val,
                "TeamPoints": team_points_val
            })

        return pd.DataFrame(race_data)

    except Exception as e:
        print(f"⚠️ Failed to load race session {year} Round {round_number}: {e}")
        return pd.DataFrame()

def get_qualifying_time(year: int, round_number: int, driver: str):
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
        laps = quali.laps.pick_driver(driver)
        best_quali = laps["LapTime"].min()
        return best_quali.total_seconds() if not pd.isna(best_quali) else None
    except Exception as e:
        print(f"⚠️ Qualifying session missing or failed for {driver} in Round {round_number}: {e}")
        return None