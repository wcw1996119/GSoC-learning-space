"""Build empirical SOC9 × 8-sector cross-tab (ε bridge) from ONS EMP08 adhoc 2978.

Source: ONS adhoc dataset 2978 (4-digit SOC2020 × 2-digit SIC2007, 2022-2024).

Output:
  data/processed/empirical_soc_sic_bridge.npz
    epsilon         (9, 8)   — P(SOC_k=k | SIC_c=c), columns sum to 1
    epsilon_joint   (9, 8)   — joint counts (unnormalized weighted employment)
    soc_labels      (9,)     — SOC9 major group labels
    sec_labels      (8,)     — v2 8-sector labels
    source          str      — provenance
    sic2_to_sec     dict     — mapping used

Uses JD23 (2023, mid-year) as primary; JD22/JD24 averaged in for stability check.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import openpyxl

V3_ROOT = Path(__file__).resolve().parents[2]
XLSX = V3_ROOT / "data" / "raw" / "ons_emp08_soc4_sic2_2022_2024.xlsx"
OUT_DIR = V3_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# v2's 8-sector schema (must match models_lib/occupation_match.py SECTORS)
SECTORS = [
    "sec1_primary", "sec2_manufacturing", "sec3_construction",
    "sec4_retail", "sec5_fnb", "sec6_info_finance",
    "sec7_public", "sec8_other",
]

# Map 2-digit SIC2007 division code → v2 sector schema
# SIC2007 sections A-U → 8-sector grouping per v2 wage_priors.py convention
SIC2_TO_SEC = {
    # A: Agriculture / forestry / fishing → primary
    1: "sec1_primary", 2: "sec1_primary", 3: "sec1_primary",
    # B: Mining → primary
    5: "sec1_primary", 6: "sec1_primary", 7: "sec1_primary",
    8: "sec1_primary", 9: "sec1_primary",
    # C: Manufacturing 10-33 → manufacturing
    **{c: "sec2_manufacturing" for c in range(10, 34)},
    # D: Electricity/gas (35) → other (utility)
    35: "sec8_other",
    # E: Water/sewerage 36-39 → other
    36: "sec8_other", 37: "sec8_other", 38: "sec8_other", 39: "sec8_other",
    # F: Construction 41-43 → construction
    41: "sec3_construction", 42: "sec3_construction", 43: "sec3_construction",
    # G: Wholesale/retail 45-47 → retail
    45: "sec4_retail", 46: "sec4_retail", 47: "sec4_retail",
    # H: Transport 49-53 → other (transport in sec8 per wage_priors)
    49: "sec8_other", 50: "sec8_other", 51: "sec8_other",
    52: "sec8_other", 53: "sec8_other",
    # I: Accommodation/food 55-56 → fnb
    55: "sec5_fnb", 56: "sec5_fnb",
    # J: Info/comm 58-63 → info_finance
    **{c: "sec6_info_finance" for c in range(58, 64)},
    # K: Financial 64-66 → info_finance
    64: "sec6_info_finance", 65: "sec6_info_finance", 66: "sec6_info_finance",
    # L: Real estate 68 → other
    68: "sec8_other",
    # M: Professional 69-75 → other
    **{c: "sec8_other" for c in range(69, 76)},
    # N: Admin support 77-82 → other
    **{c: "sec8_other" for c in range(77, 83)},
    # O: Public admin 84 → public
    84: "sec7_public",
    # P: Education 85 → public
    85: "sec7_public",
    # Q: Health/social 86-88 → public
    86: "sec7_public", 87: "sec7_public", 88: "sec7_public",
    # R: Arts 90-93 → other
    90: "sec8_other", 91: "sec8_other", 92: "sec8_other", 93: "sec8_other",
    # S: Other services 94-96 → other
    94: "sec8_other", 95: "sec8_other", 96: "sec8_other",
    # T: Households as employers 97-98 → other
    97: "sec8_other", 98: "sec8_other",
    # U: Extraterritorial 99 → other
    99: "sec8_other",
}

SOC9_LABELS = [
    "soc1_managers",
    "soc2_professional",
    "soc3_assoc_prof",
    "soc4_admin_secret",
    "soc5_skilled_trades",
    "soc6_caring_leisure",
    "soc7_sales_customer",
    "soc8_plant_operatives",
    "soc9_elementary",
]


def parse_value(v):
    """Convert cell value to float. '*' (disclosure ctrl) and '-' (zero) → 0.0."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s in ("*", "-", ""):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_sic_header(header: str) -> int | None:
    """Extract 2-digit SIC code from column header like '1 01  Crop, animal production, hunting'."""
    if not header:
        return None
    m = re.match(r"^\s*\d+\s+(\d{2})\s+", str(header))
    if m:
        return int(m.group(1))
    return None


def parse_soc4_to_soc1(label: str) -> int | None:
    """Extract first digit of SOC4 code → SOC9 major group (1-9)."""
    if not label:
        return None
    m = re.match(r"^\s*(\d)(\d{3})\s+", str(label))
    if m:
        return int(m.group(1))
    return None


def parse_sheet(ws) -> np.ndarray:
    """Return (9, 8) weighted employment counts aggregated to SOC9 × 8-sector."""
    counts = np.zeros((9, 8), dtype=np.float64)
    sec_idx = {s: i for i, s in enumerate(SECTORS)}

    # Row 7 = SIC headers; rows 8+ = data
    rows = list(ws.iter_rows(values_only=True))
    sic_header_row = rows[7]
    n_data_cols = len(sic_header_row) - 1  # last col is "Total"

    # Build column-to-sector index list (None for unmapped columns)
    col_sec = []
    for c in range(2, n_data_cols):  # skip col 0 (label), col 1 (SOC label)
        sic2 = parse_sic_header(sic_header_row[c])
        sec_name = SIC2_TO_SEC.get(sic2) if sic2 is not None else None
        col_sec.append(sec_idx[sec_name] if sec_name else None)

    # Data rows
    for r_idx, row in enumerate(rows[8:], start=8):
        soc_label = row[1]
        soc1 = parse_soc4_to_soc1(soc_label)
        if soc1 is None:
            continue
        if not 1 <= soc1 <= 9:
            continue
        for c, sec_i in enumerate(col_sec, start=2):
            if sec_i is None:
                continue
            v = parse_value(row[c])
            counts[soc1 - 1, sec_i] += v
    return counts


def main():
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)

    counts_per_year = {}
    for sheet_name in ("JD22", "JD23", "JD24"):
        ws = wb[sheet_name]
        counts_per_year[sheet_name] = parse_sheet(ws)
        print(f"\n=== {sheet_name} aggregated SOC9 × 8-sector counts (weighted employment) ===")
        print(f"  total employment = {counts_per_year[sheet_name].sum():.0f}")
        for k, lbl in enumerate(SOC9_LABELS):
            row_total = counts_per_year[sheet_name][k].sum()
            print(f"  {lbl:30s}: {row_total:12.0f}")

    # Average across 3 years for stability
    counts_avg = np.mean([counts_per_year[s] for s in counts_per_year], axis=0)

    header_label = "SOC / sec"
    print(f"\n=== 3-year averaged SOC9 × 8-sector matrix ===")
    print(f"{header_label:30s}" + "".join(f"{s[:11]:>13s}" for s in SECTORS))
    for k, lbl in enumerate(SOC9_LABELS):
        print(f"  {lbl:28s}" + "".join(f"{counts_avg[k, c]:>13.0f}" for c in range(8)))

    # ε[k, c] = P(SOC=k | SIC=c) — normalize each column to sum=1
    col_totals = counts_avg.sum(axis=0, keepdims=True)
    col_totals = np.where(col_totals > 0, col_totals, 1.0)
    epsilon = counts_avg / col_totals

    print(f"\n=== ε[k, c] = P(SOC=k | SIC=c) (columns sum to 1) ===")
    print(f"{header_label:30s}" + "".join(f"{s[:11]:>11s}" for s in SECTORS))
    for k, lbl in enumerate(SOC9_LABELS):
        print(f"  {lbl:28s}" + "".join(f"{epsilon[k, c]:>11.3f}" for c in range(8)))
    print(f"  {'col_sum (check=1.0)':28s}" + "".join(f"{epsilon[:, c].sum():>11.3f}" for c in range(8)))

    out = OUT_DIR / "empirical_soc_sic_bridge.npz"
    np.savez(
        out,
        epsilon=epsilon.astype(np.float64),
        epsilon_joint=counts_avg.astype(np.float64),
        soc_labels=np.array(SOC9_LABELS),
        sec_labels=np.array(SECTORS),
        source="ONS adhoc 2978 (APS 2022-2024 avg, SOC2020 × SIC2007 div) — built from 4-digit SOC × 2-digit SIC",
    )
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
