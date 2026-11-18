from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


DYNAMIC_COLUMNS = ["lat", "lon", "speed", "lon_acc", "lat_acc", "time"]


def read_pneuma_wide(path: str | Path) -> pd.DataFrame:
    """
    Parse a raw pNEUMA trajectory CSV.

    pNEUMA stores one vehicle per CSV row. The first four fields are:
        track_id, type, traveled_d, avg_speed

    After those, observations are stored as repeated six-field blocks:
        lat, lon, speed, lon_acc, lat_acc, time

    Rows have different lengths, so pandas.read_csv() cannot be used
    to load the raw file as a rectangular dataframe.
    """
    path = Path(path)
    observations = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter=";")

        # Skip header
        try:
            next(reader)
        except StopIteration:
            raise ValueError(f"The file is empty: {path}")

        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue

            # Remove trailing empty fields
            while row and not row[-1].strip():
                row.pop()

            if len(row) < 10:
                continue

            # Fixed vehicle information
            track_id_raw = row[0].strip()
            if not track_id_raw:
                continue

            try:
                track_id = int(float(track_id_raw))
            except ValueError:
                print(f"Warning: invalid track_id at line {row_number}: {track_id_raw!r}")
                continue

            vehicle_type = row[1].strip()
            traveled_d = pd.to_numeric(row[2], errors="coerce")
            avg_speed = pd.to_numeric(row[3], errors="coerce")

            # Repeated trajectory observations
            trajectory_values = row[4:]
            number_of_observations = len(trajectory_values) // 6

            for observation_index in range(number_of_observations):
                start = observation_index * 6
                block = trajectory_values[start:start + 6]

                if len(block) != 6:
                    continue

                lat_raw, lon_raw, speed_raw, lon_acc_raw, lat_acc_raw, time_raw = block

                lat = pd.to_numeric(lat_raw, errors="coerce")
                lon = pd.to_numeric(lon_raw, errors="coerce")
                speed = pd.to_numeric(speed_raw, errors="coerce")
                lon_acc = pd.to_numeric(lon_acc_raw, errors="coerce")
                lat_acc = pd.to_numeric(lat_acc_raw, errors="coerce")
                time = pd.to_numeric(time_raw, errors="coerce")

                # Position and time are required
                if pd.isna(lat) or pd.isna(lon) or pd.isna(time):
                    continue

                observations.append({
                    "track_id": track_id,
                    "type": vehicle_type,
                    "traveled_d": traveled_d,
                    "avg_speed": avg_speed,
                    "lat": lat,
                    "lon": lon,
                    "speed": speed,
                    "lon_acc": lon_acc,
                    "lat_acc": lat_acc,
                    "time": time,
                    "observation_index": observation_index,
                    "source_file": path.name,
                })

    if not observations:
        raise ValueError(f"No trajectory observations were parsed from {path}")

    df = pd.DataFrame(observations)
    df = df.sort_values(["track_id", "time"]).reset_index(drop=True)
    return df


def add_local_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert latitude/longitude to approximate local Cartesian
    coordinates in metres: x_m = east/west, y_m = north/south.
    """
    out = df.copy()
    earth_radius = 6_371_000.0

    lat0 = np.deg2rad(out["lat"].mean())
    lon0 = np.deg2rad(out["lon"].mean())

    lat = np.deg2rad(out["lat"].to_numpy())
    lon = np.deg2rad(out["lon"].to_numpy())

    out["x_m"] = earth_radius * (lon - lon0) * np.cos(lat0)
    out["y_m"] = earth_radius * (lat - np.deg2rad(out["lat"].mean()))

    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["track_id", "time"]).reset_index(drop=True)

    # Speed: pNEUMA speed is km/h
    out["speed_mps"] = out["speed"] / 3.6

    # Heading
    dx = out.groupby("track_id")["x_m"].diff()
    dy = out.groupby("track_id")["y_m"].diff()

    # Causal heading: use only the previous recorded position.
    # The first observation of each track has no previous position, so use
    # a neutral zero heading instead of interpolating from future observations.
    out["heading_rad"] = np.arctan2(dy, dx).fillna(0.0)

    # Angular representation avoids the +pi / -pi discontinuity
    out["heading_sin"] = np.sin(out["heading_rad"])
    out["heading_cos"] = np.cos(out["heading_rad"])

    # Acceleration magnitude
    out["accel_mag"] = np.sqrt(out["lon_acc"] ** 2 + out["lat_acc"] ** 2)

    return out


def preprocess_file(input_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    print(f"Reading pNEUMA file:\n{input_csv}")

    df = read_pneuma_wide(input_csv)
    print(f"Parsed {len(df):,} trajectory observations")
    print(f"Found {df['track_id'].nunique():,} vehicles")

    df = add_local_coordinates(df)
    df = add_derived_features(df)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert a raw pNEUMA trajectory CSV from wide to long format."
    )
    parser.add_argument("--input", required=True, help="Raw pNEUMA CSV")
    parser.add_argument("--output", required=True, help="Output long-format CSV")
    args = parser.parse_args()

    df = preprocess_file(args.input, args.output)

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Observations : {len(df):,}")
    print(f"Vehicles     : {df['track_id'].nunique():,}")
    print(f"Time range   : {df['time'].min():.2f} -> {df['time'].max():.2f} s")
    print(f"Output       : {args.output}")


if __name__ == "__main__":
    main()