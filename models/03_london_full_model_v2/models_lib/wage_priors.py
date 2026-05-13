"""ASHE-anchored wage priors for London v2.3.

Sources (rounded median weekly £, full-time, place-of-work, ASHE 2023):
- ASHE Table 14.7a — workplace earnings by 1-digit SOC, London region
- ASHE Table 7.1 — earnings by SIC section, London region

These are constants here for prototype simplicity; a production version would
download the latest ASHE release. Beijing migration: replace with
"职工平均工资" from Beijing Statistical Yearbook by industry.
"""
import numpy as np

# Median weekly wage £ by 1-digit SOC, full-time, London workplace (ASHE 2023 ~)
SOC_WEEKLY_WAGE_LONDON = {
    "soc1": 1200,   # Managers, directors and senior officials
    "soc2": 1000,   # Professional occupations
    "soc3":  800,   # Associate professional and technical
    "soc4":  550,   # Administrative and secretarial
    "soc5":  600,   # Skilled trades
    "soc6":  450,   # Caring, leisure and other service
    "soc7":  450,   # Sales and customer service
    "soc8":  550,   # Process, plant and machine operatives
    "soc9":  450,   # Elementary occupations
}

# Within-SOC log-normal sigma (from ASHE distribution dispersion)
SOC_WAGE_SIGMA = 0.35

# Median weekly wage £ by 8-sector schema, London workplace
SECTOR_WEEKLY_WAGE_LONDON = {
    "sec1_primary":      600,   # Agriculture / mining (small in London)
    "sec2_manufacturing": 700,
    "sec3_construction": 750,
    "sec4_retail":       550,   # Wholesale + retail
    "sec5_fnb":          450,   # Food + accommodation
    "sec6_info_finance":1100,   # ICT + Finance — London's premium
    "sec7_public":       750,   # Public admin + education + health
    "sec8_other":        850,   # Professional + admin services + transport
}

LONDON_MEDIAN_WAGE = 600       # for β(income) reference
GLOBAL_MEDIAN_WAGE = 600       # for log(wage_j / median) normalisation


def sample_agent_income(soc: str, rng: np.random.Generator) -> float:
    """Sample one agent's income (£/week) from log-normal centred on SOC median."""
    med = SOC_WEEKLY_WAGE_LONDON.get(soc, LONDON_MEDIAN_WAGE)
    log_med = np.log(med)
    sample = float(rng.lognormal(mean=log_med, sigma=SOC_WAGE_SIGMA))
    # Clamp to reasonable range (£200..£3500 weekly)
    return float(np.clip(sample, 200.0, 3500.0))


def grid_wage_from_sector_mix(sector_employment: dict) -> float:
    """Compute weighted-mean wage for a grid given its sector employment mix.

    sector_employment: {"sec1_primary": float, ..., "sec8_other": float}
    Returns: wage_j in £/week (weighted by sector employment)
    """
    total = sum(sector_employment.values())
    if total <= 0:
        return LONDON_MEDIAN_WAGE
    weighted = sum(
        SECTOR_WEEKLY_WAGE_LONDON.get(sec, LONDON_MEDIAN_WAGE) * count
        for sec, count in sector_employment.items()
    )
    return float(weighted / total)


def beta_for_income(income_real: float,
                      beta_0: float = 0.07,
                      median: float = LONDON_MEDIAN_WAGE,
                      delta: float = 0.5) -> float:
    """Continuous β as function of real income.

    β(y) = β_0 × (median / y)^δ
    Lower income → higher β; higher income → lower β.

    Examples (β_0=0.07, median=600, δ=0.5):
      income £400 → β = 0.086
      income £600 → β = 0.070
      income £1000 → β = 0.054
      income £1500 → β = 0.044
    """
    return float(beta_0 * (median / max(income_real, 1.0)) ** delta)
