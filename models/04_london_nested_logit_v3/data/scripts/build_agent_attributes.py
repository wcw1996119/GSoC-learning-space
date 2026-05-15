"""Build agent-level attributes (household, car ownership) per origin grid.

Sources:
  - Nomis TS003 (NM_2023_1): Household composition, LAD level (33 London boroughs)
  - Nomis TS045 (NM_2063_1): Car or van availability, LAD level

Fetch the raw CSVs (one-time; data/raw/ is gitignored):

    codes=$(printf "E0900%04d," $(seq 1 33) | sed 's/,$//')

    curl -sSL --fail \\
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2023_1.data.csv?\\
date=latest&geography=${codes}&measures=20100&\\
select=date_name,geography_code,geography_name,c2021_hhcomp_15_name,c2021_hhcomp_15,obs_value" \\
        -o data/raw/ts003_household_lad_london.csv

    curl -sSL --fail \\
        "https://www.nomisweb.co.uk/api/v01/dataset/NM_2063_1.data.csv?\\
date=latest&geography=${codes}&measures=20100&\\
select=date_name,geography_code,geography_name,c2021_cars_5_name,c2021_cars_5,obs_value" \\
        -o data/raw/ts045_cars_lad_london.csv

Derived features per origin grid i:
  - pct_with_kids_i  = P(household has dependent children at i's borough)
  - mean_cars_i      = expected number of cars per household at i's borough
                      (0·P(0) + 1·P(1) + 2·P(2) + 3.5·P(3+))

Then expand 33-borough values to 1725 grid via grid_borough_idx mapping.
Output: extra fields appended to data/processed/paperA_v3_aux_cervero.npz

Beijing portability:
  - household: 七普 户均人口 + 15岁以下人口比例 → pct_with_kids by 街道
  - cars: 北京统计年鉴《千户家庭机动车保有量》by 区 → mean_cars
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

V3_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = V3_ROOT.parent / "03_london_full_model_v2"


def read_nomis_csv(path: Path, cat_col: str) -> dict[str, dict[str, float]]:
    """Return {lad_code: {cat_code: obs_value}} from Nomis CSV."""
    out: dict[str, dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lad = row["GEOGRAPHY_CODE"]
            cat = row[cat_col]
            val = float(row["OBS_VALUE"])
            if lad not in out:
                out[lad] = {}
            out[lad][cat] = val
    return out


def derive_pct_with_kids(ts003: dict[str, dict[str, float]]) -> dict[str, float]:
    """TS003 categories with dependent children: 5, 8, 10, 13. Total = 0."""
    KIDS_CATS = {"5", "8", "10", "13"}
    TOTAL_CAT = "0"
    out = {}
    for lad, cells in ts003.items():
        total = cells.get(TOTAL_CAT, 0.0)
        with_kids = sum(cells.get(c, 0.0) for c in KIDS_CATS)
        out[lad] = with_kids / max(total, 1.0)
    return out


def derive_mean_cars(ts045: dict[str, dict[str, float]]) -> dict[str, float]:
    """TS045: cat 0=Total, 1=no car, 2=1 car, 3=2 cars, 4=3+ cars (use 3.5 for 3+)."""
    TOTAL_CAT = "0"
    counts = {"1": 0.0, "2": 1.0, "3": 2.0, "4": 3.5}
    out = {}
    for lad, cells in ts045.items():
        total = cells.get(TOTAL_CAT, 0.0)
        if total <= 0:
            out[lad] = 0.0
            continue
        weighted = sum(counts[c] * cells.get(c, 0.0) for c in counts)
        out[lad] = weighted / total
    return out


def main():
    ts003 = read_nomis_csv(V3_ROOT / "data" / "raw" / "ts003_household_lad_london.csv",
                           cat_col="C2021_HHCOMP_15")
    ts045 = read_nomis_csv(V3_ROOT / "data" / "raw" / "ts045_cars_lad_london.csv",
                           cat_col="C2021_CARS_5")
    print(f"TS003 loaded: {len(ts003)} LADs")
    print(f"TS045 loaded: {len(ts045)} LADs")

    pct_kids = derive_pct_with_kids(ts003)
    mean_cars = derive_mean_cars(ts045)

    # Print summary
    print(f"\n=== Household composition (% with dependent children) ===")
    sorted_lads = sorted(pct_kids.items(), key=lambda x: x[1])
    for lad, v in sorted_lads[:3] + sorted_lads[-3:]:
        # Get name from any row
        cells = ts003.get(lad, {})
        print(f"  {lad}: pct_with_kids = {v:.4f}")
    print(f"\n  range: [{min(pct_kids.values()):.3f}, {max(pct_kids.values()):.3f}]")

    print(f"\n=== Car ownership (mean cars per household) ===")
    sorted_cars = sorted(mean_cars.items(), key=lambda x: x[1])
    for lad, v in sorted_cars[:3] + sorted_cars[-3:]:
        print(f"  {lad}: mean_cars = {v:.4f}")
    print(f"\n  range: [{min(mean_cars.values()):.3f}, {max(mean_cars.values()):.3f}]")

    # Expand to 1725 grid using grid_borough_idx from v2 demo_cache
    cache = np.load(V2_ROOT / "data" / "processed" / "demo_cache.npz")
    grid_borough_idx = cache["grid_borough_idx"]   # (1725,) borough index 0-32
    boroughs = cache["boroughs"]                    # (33,) borough names

    # Map: borough name → LAD code
    BOROUGH_NAME_TO_LAD = {
        "City of London": "E09000001", "Barking and Dagenham": "E09000002",
        "Barnet": "E09000003", "Bexley": "E09000004",
        "Brent": "E09000005", "Bromley": "E09000006",
        "Camden": "E09000007", "Croydon": "E09000008",
        "Ealing": "E09000009", "Enfield": "E09000010",
        "Greenwich": "E09000011", "Hackney": "E09000012",
        "Hammersmith and Fulham": "E09000013", "Haringey": "E09000014",
        "Harrow": "E09000015", "Havering": "E09000016",
        "Hillingdon": "E09000017", "Hounslow": "E09000018",
        "Islington": "E09000019", "Kensington and Chelsea": "E09000020",
        "Kingston upon Thames": "E09000021", "Lambeth": "E09000022",
        "Lewisham": "E09000023", "Merton": "E09000024",
        "Newham": "E09000025", "Redbridge": "E09000026",
        "Richmond upon Thames": "E09000027", "Southwark": "E09000028",
        "Sutton": "E09000029", "Tower Hamlets": "E09000030",
        "Waltham Forest": "E09000031", "Wandsworth": "E09000032",
        "Westminster": "E09000033",
    }

    # 1) Build LAD per grid
    grid_lad_codes = []
    n_missing = 0
    for borough_i in grid_borough_idx:
        bname = str(boroughs[borough_i])
        lad_code = BOROUGH_NAME_TO_LAD.get(bname)
        if lad_code is None:
            print(f"  WARN: borough name not in lookup: '{bname}'")
            n_missing += 1
            lad_code = "E09000001"  # fallback to City of London
        grid_lad_codes.append(lad_code)

    if n_missing:
        print(f"  WARN: {n_missing} grids could not be mapped to LAD codes (used fallback)")

    pct_kids_per_grid = np.array([pct_kids[c] for c in grid_lad_codes], dtype=np.float64)
    mean_cars_per_grid = np.array([mean_cars[c] for c in grid_lad_codes], dtype=np.float64)

    print(f"\n=== Expanded to 1725 grid ===")
    print(f"pct_with_kids:   range [{pct_kids_per_grid.min():.4f}, {pct_kids_per_grid.max():.4f}], "
          f"mean {pct_kids_per_grid.mean():.4f}, std {pct_kids_per_grid.std():.4f}")
    print(f"mean_cars:       range [{mean_cars_per_grid.min():.4f}, {mean_cars_per_grid.max():.4f}], "
          f"mean {mean_cars_per_grid.mean():.4f}, std {mean_cars_per_grid.std():.4f}")

    # z-score
    pct_kids_z = (pct_kids_per_grid - pct_kids_per_grid.mean()) / max(pct_kids_per_grid.std(), 1e-9)
    mean_cars_z = (mean_cars_per_grid - mean_cars_per_grid.mean()) / max(mean_cars_per_grid.std(), 1e-9)

    # Save: append to existing v3 aux file
    aux_path = V3_ROOT / "data" / "processed" / "paperA_v3_aux_cervero.npz"
    aux = dict(np.load(aux_path).items())
    aux["pct_with_kids"] = pct_kids_per_grid
    aux["mean_cars"] = mean_cars_per_grid
    aux["pct_with_kids_z"] = pct_kids_z
    aux["mean_cars_z"] = mean_cars_z
    aux["pct_with_kids_mean"] = np.float64(pct_kids_per_grid.mean())
    aux["pct_with_kids_std"] = np.float64(pct_kids_per_grid.std())
    aux["mean_cars_mean"] = np.float64(mean_cars_per_grid.mean())
    aux["mean_cars_std_"] = np.float64(mean_cars_per_grid.std())

    np.savez(aux_path, **aux)
    print(f"\n[saved] {aux_path}  ({aux_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
