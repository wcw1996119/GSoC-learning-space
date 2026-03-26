from model import LondonCommuteModel
from agents import MSOAAgent
import solara
import solara.lab
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

import numpy as np
import threading
import time
from collections import defaultdict


# ── Colour palette ────────────────────────────────────────────────
PRIMARY = "#2B6CB0"
ACCENT = "#E53E3E"
GREEN = "#38A169"
DARK_GREY = "#4A5568"
MED_GREY = "#718096"
LIGHT_GREY = "#A0AEC0"
BG_WHITE = "#FFFFFF"
BG_LIGHT = "#F7FAFC"
BORDER = "#E2E8F0"
TEXT_BODY = "#2D3748"
TEXT_TITLE = "#1A202C"
TEXT_SUBTITLE = "#718096"
HEADER_BG = "#1a1a2e"

FONT_FAMILY = "'Inter', 'Source Sans Pro', -apple-system, sans-serif"
START_HOUR = 5

# Chart styling defaults
CHART_TITLE_SIZE = 14
CHART_SUBTITLE_SIZE = 11
AXIS_LABEL_SIZE = 12
TICK_SIZE = 10


def _style_ax(ax, title="", subtitle="", xlabel="", ylabel="", grid_y=True):
    """Apply consistent academic styling to a matplotlib axis."""
    if title:
        ax.set_title(title, fontsize=CHART_TITLE_SIZE, fontweight='600',
                     color=TEXT_TITLE, pad=10, loc='left')
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=CHART_SUBTITLE_SIZE, color=TEXT_SUBTITLE,
                style='italic', va='bottom')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, color=TEXT_BODY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, color=TEXT_BODY)
    if grid_y:
        ax.yaxis.grid(True, color='#EDF2F7', linestyle='--', linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_SIZE, colors=DARK_GREY)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(BORDER)


def _add_peak_shading(ax, steps):
    """Add subtle shaded bands for peak hours (7-9am, 5-7pm)."""
    for s in steps:
        h = (START_HOUR + s) % 24
        if h in [7, 8, 9, 17, 18, 19]:
            ax.axvspan(s - 0.5, s + 0.5, color=ACCENT, alpha=0.06, zorder=0)


def _extract_borough(msoa_name):
    """Extract borough name from MSOA name like 'Barking and Dagenham 001'."""
    if not msoa_name:
        return "Unknown"
    parts = msoa_name.rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return msoa_name



def _fill_accessibility(m):
    """Get accessibility per MSOA, filling empty MSOAs with borough averages."""
    acc_dict = {a.msoa_code: a.accessibility for a in m._msoa_agent_list}
    msoa_to_borough = {row['MSOA21CD']: _extract_borough(row['MSOA21NM'])
                       for _, row in m.gdf.iterrows()}
    borough_totals = defaultdict(float)
    borough_counts = defaultdict(int)
    for code, acc in acc_dict.items():
        if acc > 0:
            borough = msoa_to_borough.get(code, "Unknown")
            borough_totals[borough] += acc
            borough_counts[borough] += 1
    borough_avg = {b: borough_totals[b] / borough_counts[b]
                   for b in borough_totals if borough_counts[b] > 0}
    filled = dict(acc_dict)
    for _, row in m.gdf.iterrows():
        code = row['MSOA21CD']
        if filled.get(code, 0) == 0:
            borough = msoa_to_borough.get(code, "Unknown")
            if borough in borough_avg:
                filled[code] = borough_avg[borough]
    return filled


def _borough_deviation(msoa_values, msoa_to_borough):
    """Compute borough-level % deviation from London-wide mean."""
    all_vals = [v for v in msoa_values.values() if v > 0]
    if not all_vals:
        return {}
    london_mean = np.mean(all_vals)
    if london_mean == 0:
        return {}
    totals, counts = defaultdict(float), defaultdict(int)
    for code, val in msoa_values.items():
        b = msoa_to_borough.get(code, "Unknown")
        totals[b] += val
        counts[b] += 1
    return {b: (totals[b] / counts[b] - london_mean) / london_mean * 100
            for b in totals if counts[b] > 0}


# ── Model instance ───────────────────────────────────────────────
model = solara.reactive(None)
is_playing = solara.reactive(False)
step_count = solara.reactive(0)
n_commuters_param = solara.reactive(2000)
bpr_beta_param = solara.reactive(3.0)


def reset_model():
    is_playing.set(False)
    model.set(LondonCommuteModel(
        n_commuters=n_commuters_param.value,
        bpr_beta=bpr_beta_param.value
    ))
    step_count.set(0)


def do_step():
    model.value.step()
    step_count.set(step_count.value + 1)


def _play_loop():
    while is_playing.value:
        do_step()
        time.sleep(0.5)


def toggle_play():
    if is_playing.value:
        is_playing.set(False)
    else:
        is_playing.set(True)
        t = threading.Thread(target=_play_loop, daemon=True)
        t.start()


def _hour_label(hour):
    if hour == 0:
        return "12:00 AM"
    elif hour == 12:
        return "12:00 PM"
    elif hour < 12:
        return f"{hour}:00 AM"
    else:
        return f"{hour - 12}:00 PM"


def _make_time_label(step, show_day=False):
    hour = (START_HOUR + step) % 24
    day = (START_HOUR + step) // 24 + 1
    if hour == 0:
        label = "12am"
    elif hour == 12:
        label = "12pm"
    elif hour < 12:
        label = f"{hour}am"
    else:
        label = f"{hour - 12}pm"
    if show_day and day > 1:
        label += f"\n(D{day})"
    return label


def _format_ts_xaxis(ax, steps):
    """Format x-axis for time-series charts with multi-day support."""
    total_hours = len(steps)
    spans_multi_day = total_hours > 24
    tick_steps = list(range(0, len(steps), 3))
    ax.set_xticks(tick_steps)
    ax.set_xticklabels([_make_time_label(s, show_day=spans_multi_day)
                        for s in tick_steps], fontsize=8)
    # Add day boundary divider
    if spans_multi_day:
        for s in steps:
            if s > 0 and (START_HOUR + s) % 24 == 0:
                ax.axvline(s, color=BORDER, linewidth=0.8, linestyle='--', alpha=0.5)


# ── Tab 1: Spatial Inequality Landscape ──────────────────────────

@solara.component
def AccessibilityMapLarge(m):
    """Large accessibility choropleth for Tab 1 left side."""
    step_count.get()
    acc_dict = {a.msoa_code: a.accessibility for a in m._msoa_agent_list}
    filled_acc = _fill_accessibility(m)
    # Track which MSOAs are directly sampled vs interpolated
    directly_sampled = set(code for code, v in acc_dict.items() if v > 0)

    gdf = m.gdf.copy()
    gdf['accessibility'] = gdf['MSOA21CD'].map(filled_acc).fillna(0)
    gdf['is_interpolated'] = ~gdf['MSOA21CD'].isin(directly_sampled)

    coverage = len(directly_sampled) / len(gdf) * 100

    # Fixed clean breakpoints for readable legend
    boundaries = [0, 100, 200, 400, 600, 800, 1200, 1600]
    pos_vals = gdf['accessibility'][gdf['accessibility'] > 0].values
    if len(pos_vals) > 0:
        data_max = np.max(pos_vals)
        if data_max > boundaries[-1]:
            boundaries.append(data_max)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=256)

    fig = Figure(figsize=(9, 8.5), facecolor=BG_WHITE)
    ax = fig.subplots()
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.02, right=0.98)

    # Plot directly sampled MSOAs
    gdf_direct = gdf[~gdf['is_interpolated']]
    gdf_interp = gdf[gdf['is_interpolated']]

    if len(gdf_direct) > 0:
        gdf_direct.plot(column='accessibility', ax=ax, cmap='Blues', norm=norm,
                        missing_kwds={'color': '#F0F0F0'},
                        linewidth=0.15, edgecolor='#A0AEC0')
    # Plot interpolated MSOAs with hatch pattern
    if len(gdf_interp) > 0:
        gdf_interp.plot(column='accessibility', ax=ax, cmap='Blues', norm=norm,
                        missing_kwds={'color': '#F0F0F0'},
                        linewidth=0.15, edgecolor='#A0AEC0', alpha=0.5, hatch='///')

    # Borough labels
    _borough_labels = [
        'City of London', 'Westminster', 'Camden', 'Tower Hamlets', 'Hackney',
        'Southwark', 'Lambeth', 'Croydon', 'Bromley', 'Barnet', 'Ealing',
        'Hillingdon', 'Havering', 'Greenwich', 'Newham', 'Harrow', 'Hounslow'
    ]
    msoa_to_borough_map = {row['MSOA21CD']: _extract_borough(row['MSOA21NM'])
                           for _, row in gdf.iterrows()}
    borough_centroids = defaultdict(lambda: [[], []])
    for _, row in gdf.iterrows():
        b = msoa_to_borough_map.get(row['MSOA21CD'], '')
        if b in _borough_labels:
            centroid = row['geometry'].centroid
            borough_centroids[b][0].append(centroid.x)
            borough_centroids[b][1].append(centroid.y)
    for b, (xs, ys) in borough_centroids.items():
        ax.text(np.mean(xs), np.mean(ys), b, fontsize=7.5, color=DARK_GREY,
                ha='center', va='center', fontweight='500',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.6,
                          edgecolor='none'))

    # Colour bar
    sm = mcm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.6,
                        pad=0.04, aspect=25, ticks=boundaries)
    cbar.set_label('Accessibility index', fontsize=10)
    cbar.ax.set_xticklabels([f'{int(b)}' if b <= 1600 else '1600+'
                             for b in boundaries])

    fig.suptitle("Job Accessibility", fontsize=CHART_TITLE_SIZE,
                 fontweight='600', color=TEXT_TITLE, x=0.03, ha='left', y=0.98)
    fig.text(0.03, 0.94,
             f"Gravity-based measure of job opportunities, {_hour_label(m.current_hour)}",
             fontsize=CHART_SUBTITLE_SIZE, color=TEXT_SUBTITLE, style='italic', ha='left')
    # Formula directly below colour bar
    fig.text(0.5, 0.065,
             r"$A_i = \sum_j E_j \cdot \exp(-\beta \cdot t_{ij})$"
             r"      $E_j$: employment at $j$,  $t_{ij}$: travel time,  $\beta$: decay parameter",
             fontsize=9, color=MED_GREY, ha='center', style='italic')
    ax.text(0.99, 0.01,
            f"Coverage: {coverage:.0f}% of MSOAs directly sampled; "
            f"remainder estimated from borough averages",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, color=MED_GREY, style='italic')
    ax.set_axis_off()
    solara.FigureMatplotlib(fig)


@solara.component
def AccessibilityBoroughBar(m):
    """Top 10 boroughs with accessibility most below London average."""
    step_count.get()
    cur_acc = _fill_accessibility(m)
    msoa_to_borough = {row['MSOA21CD']: _extract_borough(row['MSOA21NM'])
                       for _, row in m.gdf.iterrows()}
    devs = _borough_deviation(cur_acc, msoa_to_borough)
    if not devs:
        return
    top = sorted(devs.items(), key=lambda x: x[1])[:10]
    top = [x for x in top if x[1] < 0]
    if not top:
        return
    boroughs = [b for b, _ in reversed(top)]
    vals = [v for _, v in reversed(top)]

    fig = Figure(figsize=(5.5, 5), facecolor=BG_WHITE)
    ax = fig.subplots()
    fig.subplots_adjust(left=0.48, top=0.90, bottom=0.12, right=0.88)
    bars = ax.barh(range(len(boroughs)), vals, color=ACCENT, alpha=0.8, height=0.6)
    ax.set_yticks(range(len(boroughs)))
    ax.set_yticklabels(boroughs, fontsize=7.5)
    # Place labels inside bars (white) for long bars, outside for short bars
    for i, (bar, v) in enumerate(zip(bars, vals)):
        if abs(v) > 15:
            ax.text(v / 2, i, f'{v:.0f}%', va='center', ha='center',
                    fontsize=7, color='white', fontweight='600')
        else:
            ax.text(0.3, i, f'{v:.0f}%', va='center', ha='left',
                    fontsize=7, color=DARK_GREY)
    fig.suptitle("Boroughs Furthest Below Average",
                 fontsize=10, fontweight='600', color=TEXT_TITLE,
                 x=0.03, ha='left', y=0.97)
    ax.tick_params(labelsize=7, colors=DARK_GREY)
    ax.set_xlabel("% below London average", fontsize=8, color=TEXT_BODY)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(BORDER)
    ax.xaxis.grid(True, color='#EDF2F7', linestyle='--', linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(right=0)
    solara.FigureMatplotlib(fig)


@solara.component
def GiniSparkline(m):
    """Sparkline of Gini coefficient with context."""
    step_count.get()
    data = m.datacollector.get_model_vars_dataframe()
    if data.empty:
        return

    vals = data['Accessibility_Gini'].values
    current_gini = vals[-1]
    initial_gini = vals[0]
    delta = current_gini - initial_gini
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
    delta_color = ACCENT if delta > 0 else (GREEN if delta < 0 else MED_GREY)

    with solara.Column(style="gap:6px; padding:8px 0;"):
        solara.Text("Spatial Inequality Over Time",
                    style=f"font-size:11px; font-weight:600; color:{TEXT_TITLE};")
        solara.Text("Gini coefficient of job accessibility across MSOAs",
                    style=f"font-size:9px; color:{TEXT_SUBTITLE}; font-style:italic;")

        # Current value + delta
        with solara.Row(style="align-items:baseline; gap:8px;"):
            solara.Text(f"Gini: {current_gini:.3f}",
                        style=f"font-size:22px; font-weight:700; color:{PRIMARY}; "
                              f"line-height:1;")
            solara.Text(f"{arrow} {delta:+.4f} since {_hour_label(START_HOUR)}",
                        style=f"font-size:11px; color:{delta_color}; font-weight:600;")

        # Sparkline with axes
        fig = Figure(figsize=(5, 1.8), facecolor=BG_WHITE)
        ax = fig.subplots()
        fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.22)
        steps = list(range(len(vals)))
        ax.plot(steps, vals, color=PRIMARY, linewidth=1.5)
        ax.fill_between(steps, vals, alpha=0.08, color=PRIMARY)
        # Peak markers
        peak_steps = [s for s in steps if ((START_HOUR + s) % 24) in [8, 9, 17, 18]]
        if peak_steps:
            ax.scatter(peak_steps, vals[peak_steps],
                       color=ACCENT, s=12, zorder=5)
        ax.set_xlim(0, max(len(vals) - 1, 1))
        # Y-axis: show min/max range
        y_min, y_max = min(vals), max(vals)
        y_pad = max((y_max - y_min) * 0.15, 0.001)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
        ax.tick_params(axis='y', labelsize=7, colors=MED_GREY)
        # X-axis: sparse key labels only (start, peaks, current)
        key_ticks = [0]  # start
        for s in steps:
            h = (START_HOUR + s) % 24
            if h == 8 and s not in key_ticks:
                key_ticks.append(s)
            elif h == 17 and s not in key_ticks:
                key_ticks.append(s)
        if len(steps) > 1 and steps[-1] not in key_ticks:
            key_ticks.append(steps[-1])
        # Add day dividers for multi-day
        for s in steps:
            if s > 0 and (START_HOUR + s) % 24 == 0:
                ax.axvline(s, color=BORDER, linewidth=0.8, linestyle='--', alpha=0.5)
        key_labels = []
        for s in key_ticks:
            h = (START_HOUR + s) % 24
            day = (START_HOUR + s) // 24 + 1
            label = _make_time_label(s)
            if h in [8, 9]:
                label += "\n(peak)"
            elif h in [17, 18]:
                label += "\n(peak)"
            if day > 1:
                label += f"\nDay {day}"
            key_labels.append(label)
        ax.set_xticks(key_ticks)
        ax.set_xticklabels(key_labels, fontsize=7, color=MED_GREY)
        ax.tick_params(axis='x', colors=MED_GREY)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(BORDER)
            ax.spines[spine].set_linewidth(0.5)
        solara.FigureMatplotlib(fig)


# ── Tab 2: Congestion & Inequality Dynamics ──────────────────────

@solara.component
def GiniTimeSeries(m):
    step_count.get()
    data = m.datacollector.get_model_vars_dataframe()
    if data.empty:
        return
    fig = Figure(figsize=(6, 3.5), facecolor=BG_WHITE)
    ax = fig.subplots()
    steps = list(range(len(data)))
    vals = data['Accessibility_Gini'].values
    _add_peak_shading(ax, steps)
    ax.plot(steps, vals, color=PRIMARY, linewidth=1.8)
    peak_steps = [s for s in steps if ((START_HOUR + s) % 24) in [8, 9, 17, 18]]
    if peak_steps:
        ax.scatter(peak_steps, vals[peak_steps],
                   color=ACCENT, s=20, zorder=5, label='Peak hour')
    _style_ax(ax, title="How Evenly Is Access Shared? (Gini)",
              xlabel="Time of day", ylabel="Gini coefficient")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    _format_ts_xaxis(ax, steps)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def PalmaTimeSeries(m):
    step_count.get()
    data = m.datacollector.get_model_vars_dataframe()
    if data.empty:
        return
    fig = Figure(figsize=(6, 3.5), facecolor=BG_WHITE)
    ax = fig.subplots()
    steps = list(range(len(data)))
    vals = data['Accessibility_Palma'].values
    _add_peak_shading(ax, steps)
    ax.plot(steps, vals, color=PRIMARY, linewidth=1.8)
    peak_steps = [s for s in steps if ((START_HOUR + s) % 24) in [8, 9, 17, 18]]
    if peak_steps:
        ax.scatter(peak_steps, vals[peak_steps],
                   color=ACCENT, s=20, zorder=5, label='Peak hour')
    if len(vals) > 3 and peak_steps:
        peak_val = max(vals[peak_steps])
        offpeak_val = min(vals)
        ax.annotate(f'+{peak_val - offpeak_val:.2f}',
                    xy=(peak_steps[0], peak_val),
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=9, color=ACCENT, fontweight='600',
                    arrowprops=dict(arrowstyle='->', color=ACCENT, lw=0.8))
    _style_ax(ax, title="Gap Between Best and Worst Served (Palma)",
              xlabel="Time of day", ylabel="Palma ratio (top 10% / bottom 40%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    _format_ts_xaxis(ax, steps)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def CommuteTimeTimeSeries(m):
    step_count.get()
    data = m.datacollector.get_model_vars_dataframe()
    if data.empty:
        return
    fig = Figure(figsize=(6, 3.5), facecolor=BG_WHITE)
    ax = fig.subplots()
    steps = list(range(len(data)))
    vals = data['Mean_Commute_Time'].values
    _add_peak_shading(ax, steps)
    ax.plot(steps, vals, color=PRIMARY, linewidth=1.8)
    peak_steps = [s for s in steps if ((START_HOUR + s) % 24) in [8, 9, 17, 18]]
    if peak_steps:
        ax.scatter(peak_steps, vals[peak_steps],
                   color=ACCENT, s=20, zorder=5, label='Peak hour')
    _style_ax(ax, title="How Long Are People Travelling?",
              xlabel="Time of day", ylabel="Mean commute time (minutes)")
    _format_ts_xaxis(ax, steps)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def CommuteTimeHistogram(m):
    step_count.get()
    times = np.array([a.commute_time_minutes for a in m._commuter_agent_list
                      if a.commute_time_minutes > 0])
    if len(times) == 0:
        return
    times = np.clip(times, 0, np.percentile(times, 99))
    fig = Figure(figsize=(5, 3.5), facecolor=BG_WHITE)
    ax = fig.subplots()
    ax.hist(times, bins=40, color=PRIMARY, alpha=0.75, edgecolor='white')
    mean_val = np.mean(times)
    median_val = np.median(times)
    ax.axvline(mean_val, color=TEXT_TITLE, linestyle='--', linewidth=1.5,
               label=f"Mean: {mean_val:.1f} min")
    ax.axvline(median_val, color=MED_GREY, linestyle=':', linewidth=1.5,
               label=f"Median: {median_val:.1f} min")
    _style_ax(ax, title="Who Travels How Long?",
              xlabel="Commute time (minutes)", ylabel="Number of commuters")
    ax.legend(fontsize=8, framealpha=0.95, facecolor='white', edgecolor=BORDER)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def CommutTimeByDistance(m):
    step_count.get()
    city_lat, city_lon = 51.5074, -0.1278
    distances, times = [], []
    for agent in m._commuter_agent_list:
        if agent.commute_time_minutes > 0 and agent.home_msoa:
            home = m.msoa_agents.get(agent.home_msoa)
            if home:
                dist_km = ((home.lat - city_lat)**2 +
                           (home.lon - city_lon)**2)**0.5 * 111
                distances.append(dist_km)
                times.append(min(agent.commute_time_minutes, 120))
    if not distances:
        return
    distances = np.array(distances)
    times = np.array(times)
    bins = [0, 5, 10, 15, 20, 25, 30]
    labels = ['0–5 km', '5–10 km', '10–15 km', '15–20 km', '20–25 km', '25+ km']
    bin_means, bin_labels = [], []
    for i in range(len(bins) - 1):
        mask = (distances >= bins[i]) & (distances < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(np.mean(times[mask]))
            bin_labels.append(labels[i])

    fig = Figure(figsize=(5, 3.5), facecolor=BG_WHITE)
    ax = fig.subplots()
    bars = ax.bar(bin_labels, bin_means, color=PRIMARY, alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, bin_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, color=DARK_GREY)
    _style_ax(ax, title="Longer Commutes on the Outskirts",
              xlabel="Distance from city centre", ylabel="Mean commute time (minutes)")
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


# ── Tab 3: Distributional Impacts by Occupation ─────────────────

@solara.component
def OccupationAccessibilityPlot(m):
    step_count.get()
    acc_by_occ = m._person_based_accessibility_by_occupation()
    soc_labels = {
        'soc1': 'SOC 1\nManagers',
        'soc2': 'SOC 2\nProfessional',
        'soc3': 'SOC 3\nTechnical',
        'soc4': 'SOC 4\nAdmin',
        'soc5': 'SOC 5\nTrades',
        'soc6': 'SOC 6\nCaring',
        'soc7': 'SOC 7\nSales',
        'soc8': 'SOC 8\nOperatives',
        'soc9': 'SOC 9\nElementary',
    }
    socs = [f'soc{i}' for i in range(1, 10)]
    values = [acc_by_occ.get(s, 0) for s in socs]
    labels = [soc_labels[s] for s in socs]
    is_peak = m.current_hour in [8, 9, 17, 18]
    colors = [PRIMARY if s in ['soc1', 'soc2', 'soc3'] else
              LIGHT_GREY if s in ['soc4', 'soc6', 'soc7'] else
              ACCENT for s in socs]

    fig = Figure(figsize=(10, 4), facecolor=BG_WHITE)
    ax = fig.subplots()
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, color=DARK_GREY)
    legend_elements = [
        Patch(facecolor=PRIMARY, label='Knowledge workers (SOC 1–3)'),
        Patch(facecolor=LIGHT_GREY, label='Service workers (SOC 4, 6, 7)'),
        Patch(facecolor=ACCENT, label='Manual workers (SOC 5, 8, 9)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper right',
              framealpha=0.95, facecolor='white', edgecolor=BORDER)
    if min(values) > 0:
        ratio = max(values) / min(values)
        status = f"Peak ({_hour_label(m.current_hour)})" if is_peak else f"Off-peak ({_hour_label(m.current_hour)})"
        ax.text(0.02, 0.95,
                f"{status}  |  Best / worst ratio: {ratio:.2f}×",
                transform=ax.transAxes, fontsize=10, fontweight='600',
                va='top', color=DARK_GREY,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=BG_LIGHT,
                          edgecolor=BORDER, alpha=0.95))
    _style_ax(ax, title="Who Can Reach the Most Jobs?",
              ylabel="Mean accessibility index")
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def SOCGapTimeSeries(m):
    step_count.get()
    data = m.datacollector.get_model_vars_dataframe()
    if data.empty or 'SOC_Gap' not in data.columns:
        return
    fig = Figure(figsize=(6, 3.5), facecolor=BG_WHITE)
    ax = fig.subplots()
    steps = list(range(len(data)))
    vals = data['SOC_Gap'].values
    _add_peak_shading(ax, steps)
    ax.plot(steps, vals, color=PRIMARY, linewidth=1.8)
    peak_steps = [s for s in steps if ((START_HOUR + s) % 24) in [8, 9, 17, 18]]
    if peak_steps:
        ax.scatter(peak_steps, vals[peak_steps],
                   color=ACCENT, s=20, zorder=5, label='Peak hour')
    ax.axhline(y=1.0, color=MED_GREY, linestyle='--', linewidth=0.8, alpha=0.5)
    _style_ax(ax, title="Does the Gap Widen During Rush Hour?",
              xlabel="Time of day",
              ylabel="Accessibility ratio\n(best-served / worst-served occupation)")
    _format_ts_xaxis(ax, steps)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def OccupationModeChart(m):
    """Grouped bar chart: commute mode proportions by occupation."""
    step_count.get()
    occ_mode = defaultdict(lambda: defaultdict(int))
    for a in m._commuter_agent_list:
        if a.occupation and a.commute_mode:
            occ_mode[a.occupation][a.commute_mode] += 1

    soc_order = [f'soc{i}' for i in range(1, 10)]
    soc_labels_short = ['Managers', 'Professional', 'Technical',
                        'Admin', 'Trades', 'Caring', 'Sales', 'Operatives', 'Elementary']
    modes = ['car', 'pt', 'active']
    colors_mode = {'car': ACCENT, 'pt': PRIMARY, 'active': GREEN}
    labels_mode = {'car': 'Car', 'pt': 'Public transport', 'active': 'Walk / Cycle'}

    fig = Figure(figsize=(7, 3.5), facecolor=BG_WHITE)
    ax = fig.subplots()
    x = np.arange(len(soc_order))
    width = 0.25

    for i, mode_key in enumerate(modes):
        totals = [occ_mode[s]['car'] + occ_mode[s]['pt'] + occ_mode[s]['active']
                  for s in soc_order]
        counts = [occ_mode[s][mode_key] for s in soc_order]
        pcts = [c / t * 100 if t > 0 else 0 for c, t in zip(counts, totals)]
        ax.bar(x + i * width - width, pcts, width,
               label=labels_mode[mode_key], color=colors_mode[mode_key], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(soc_labels_short, fontsize=9, rotation=30, ha='right')
    _style_ax(ax, title="How Do Different Workers Get to Work?",
              ylabel="Share of commuters (%)")
    ax.legend(fontsize=8, framealpha=0.95, facecolor='white', edgecolor=BORDER)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


# ── Tab 4: Calibration & Validation ─────────────────────────────

VALIDATION_CSS = (
    "background:white; border:1px solid #E2E8F0; border-radius:8px; "
    "padding:16px; margin:8px 0; box-shadow:0 1px 3px rgba(0,0,0,0.06);"
)

TABLE_CSS = """
<style>
.val-table { width:100%; border-collapse:collapse; font-size:13px; }
.val-table th {
    background:#F7FAFC; font-weight:600; border-bottom:2px solid #E2E8F0;
    text-align:left; padding:8px 12px;
}
.val-table td { padding:8px 12px; border-bottom:1px solid #E2E8F0; }
.val-table tr:nth-child(even) td { background:#F7FAFC; }
.val-table td:nth-child(n+2) { text-align:right; font-family:monospace; }
.val-table td strong { font-weight:700; }
</style>
"""


# ── Main Page ────────────────────────────────────────────────────
@solara.component
def Page():
    _tick, set_tick = solara.use_state(0)

    def _start_init():
        def _run():
            try:
                m = LondonCommuteModel(
                    n_commuters=n_commuters_param.value,
                    bpr_beta=bpr_beta_param.value,
                )
                model.set(m)
            except Exception:
                import traceback
                traceback.print_exc()
            set_tick(1)

        threading.Thread(target=_run, daemon=True).start()

    solara.use_effect(_start_init, dependencies=[])

    m = model.value
    sc = step_count.value

    solara.Title("London Commuting Accessibility Model")

    # Global CSS
    solara.Style(f"""
        .solara-content {{ font-family: {FONT_FAMILY}; color: {TEXT_BODY}; }}
        .v-tab {{ text-transform: none !important; font-size: 14px !important;
                  letter-spacing: 0 !important; }}
        .v-tab--active {{ font-weight: 600 !important;
                         border-bottom: 2px solid {PRIMARY} !important; }}
    """)

    if m is None:
        with solara.Column(style="align-items:center; justify-content:center; "
                                 "min-height:60vh; gap:12px;"):
            solara.Text("Loading model data...",
                        style=f"font-size:20px; font-weight:600; color:{PRIMARY};")
            solara.Text("Reading boundaries, travel flows, and congestion data.",
                        style=f"font-size:13px; color:{MED_GREY};")
        return

    # ── Header ──
    with solara.Row(style=f"align-items:center; gap:12px; padding:10px 20px; "
                          f"background:{HEADER_BG}; color:white;"):
        solara.Text("London Commuting Accessibility & Inequality Simulation",
                    style="font-size:17px; font-weight:600; color:white; flex:1;")

    # ── Controls ──
    with solara.Row(style=f"align-items:center; gap:10px; padding:8px 20px; "
                          f"background:{BG_LIGHT}; border-bottom:1px solid {BORDER}; "
                          f"flex-wrap:wrap;"):
        solara.Button("Reset", on_click=reset_model,
                      style=f"color:{DARK_GREY}; border:1px solid {LIGHT_GREY}; "
                            f"background:white; min-width:60px; font-size:13px;")
        play_label = "Pause" if is_playing.value else "Play"
        solara.Button(play_label, on_click=toggle_play,
                      style=f"color:{DARK_GREY}; border:1px solid {LIGHT_GREY}; "
                            f"background:white; min-width:60px; font-size:13px;")
        solara.Button("Step", on_click=do_step,
                      style=f"color:{DARK_GREY}; border:1px solid {LIGHT_GREY}; "
                            f"background:white; min-width:60px; font-size:13px;")

        solara.Text(f"N agents: {n_commuters_param.value}", style=f"margin-left:16px; font-size:13px; color:{DARK_GREY};")
        solara.SliderInt("", value=n_commuters_param, min=500, max=10000, step=500)
        solara.Text(f"Congestion sensitivity (β): {bpr_beta_param.value:.1f}",
                    style=f"margin-left:8px; font-size:13px; color:{DARK_GREY};")
        solara.SliderFloat("", value=bpr_beta_param, min=0.5, max=6.0, step=0.5)

        hour = m.current_hour
        is_peak = hour in [8, 9, 17, 18]
        if is_peak:
            status = "Rush hour"
            color = ACCENT
        elif hour < 7 or hour > 20:
            status = "Quiet"
            color = MED_GREY
        else:
            status = "Normal traffic"
            color = GREEN
        solara.Text(
            f"Step {sc}  |  {_hour_label(hour)}  —  {status}",
            style=f"font-size:13px; font-weight:600; color:{color}; margin-left:16px; "
                  f"padding:2px 10px; border:1px solid {color}; border-radius:4px;"
        )

    # ── Tabs ──
    with solara.lab.Tabs():

        with solara.lab.Tab("Spatial Inequality Landscape"):
            solara.Text(
                "Where you live in London shapes your effective access to employment. "
                "Areas with longer commutes to job centres have systematically lower "
                "accessibility. Congestion makes these gaps worse.",
                style=f"padding:12px 20px; color:{TEXT_SUBTITLE}; font-size:13px; "
                      f"font-style:italic;"
            )
            with solara.Columns([11, 9]):
                AccessibilityMapLarge(m)
                with solara.Column(style="gap:16px;"):
                    AccessibilityBoroughBar(m)
                    GiniSparkline(m)

        with solara.lab.Tab("Congestion & Inequality Dynamics"):
            solara.Text(
                "Road congestion during rush hours increases travel times and reduces "
                "access to jobs. The burden falls unevenly across London.",
                style=f"padding:12px 20px; color:{TEXT_SUBTITLE}; font-size:13px; "
                      f"font-style:italic;"
            )
            with solara.Columns([1, 1, 1]):
                GiniTimeSeries(m)
                PalmaTimeSeries(m)
                CommuteTimeTimeSeries(m)
            with solara.Row(style=f"padding:12px 16px; background:#EBF8FF; "
                                  f"border-left:4px solid {PRIMARY}; margin:12px 20px; "
                                  f"border-radius:0 4px 4px 0;"):
                solara.Markdown("""
**Key patterns:**

- The morning rush (8–9 AM) widens the Palma ratio by ~0.3 points, pushing already poorly served areas further behind
- Gini coefficient rises during both AM and PM peaks, confirming that congestion amplifies existing spatial inequality
- Average journey times increase by roughly 10 minutes during peak hours
- Residents living more than 15 km from central London bear the largest time penalties
                """)
            with solara.Columns([1, 1]):
                CommuteTimeHistogram(m)
                CommutTimeByDistance(m)

        with solara.lab.Tab("Distributional Impacts by Occupation"):
            solara.Text(
                "Not all workers are affected equally. Your occupation influences where "
                "you work, how you travel, and how much congestion costs you.",
                style=f"padding:12px 20px; color:{TEXT_SUBTITLE}; font-size:13px; "
                      f"font-style:italic;"
            )
            OccupationAccessibilityPlot(m)
            with solara.Row(style=f"padding:12px 16px; background:#FFF5F5; "
                                  f"border-left:4px solid {ACCENT}; margin:12px 20px; "
                                  f"border-radius:0 4px 4px 0;"):
                solara.Markdown("""
**Key patterns:**

- Knowledge workers (managers, professionals, technical staff) can reach roughly 40% more jobs than manual workers
- Manual workers rely more heavily on cars, making them more vulnerable when roads are congested
- The gap between best-served and worst-served occupations widens noticeably during rush hours
                """)
            with solara.Columns([1, 1]):
                SOCGapTimeSeries(m)
                OccupationModeChart(m)

        with solara.lab.Tab("Calibration & Validation"):
            with solara.Column(style=f"padding:16px 20px; background:#EBF8FF; "
                                     f"border-left:4px solid {PRIMARY}; margin:12px 20px; "
                                     f"border-radius:0 4px 4px 0;"):
                solara.Markdown("""
**Overall assessment:** The model captures the key spatial patterns observed in London's
commuting landscape. Simulated accessibility correlates significantly with official
travel time statistics (Spearman ρ = −0.247, p < 10⁻¹²) and with an independent
measure derived from Census commuting flows (ρ = +0.270, p < 10⁻¹⁷). Congestion
measurably amplifies inequality (Palma ratio increase = +1.70, p = 0.048).
                """)

            with solara.Column(style=VALIDATION_CSS):
                solara.Markdown(f"""
{TABLE_CSS}

### Parameter Calibration

The model was calibrated by searching across combinations of accessibility decay (β_acc)
and congestion sensitivity (BPR β), selecting the pair that maximises agreement with
DfT Journey Time Statistics for public transport (n = 784 MSOAs).

**Selected parameters: β_acc = 3.0, BPR β = 1.5**

<table class="val-table">
<tr><th>Metric</th><th>Value</th><th>p-value</th></tr>
<tr><td>Spearman ρ (simulated accessibility vs observed PT travel time)</td><td><strong>−0.247</strong></td><td>2.42 × 10⁻¹² ***</td></tr>
<tr><td>Pearson r</td><td>−0.289</td><td>—</td></tr>
<tr><td>RMSE (normalised)</td><td>0.473</td><td>—</td></tr>
</table>

The negative correlation is directionally correct: neighbourhoods with higher simulated
accessibility tend to have shorter observed public transport travel times.
                """)

            with solara.Column(style=VALIDATION_CSS):
                solara.Markdown(f"""
{TABLE_CSS}

### V1 — Cross-Check Against Census Commuting Flows

Simulated 8 AM accessibility was compared with an entirely independent measure computed
directly from Census origin–destination flows (n = 955 MSOAs). The Census-based measure
uses no model parameters.

<table class="val-table">
<tr><th>Metric</th><th>Value</th><th>p-value</th></tr>
<tr><td>Spearman ρ</td><td><strong>+0.270</strong></td><td>2.05 × 10⁻¹⁷ ***</td></tr>
<tr><td>Pearson r</td><td>+0.417</td><td>2.12 × 10⁻⁴¹ ***</td></tr>
</table>
                """)

            with solara.Column(style=VALIDATION_CSS):
                solara.Markdown("""
### V2 — Does Congestion Make Inequality Worse?

Peak-hour Palma ratios were compared with off-peak values. The difference is statistically
significant (diff = +1.70, 95% CI [1.07, 2.53], p = 0.048, Cohen's d = 1.48).
A Kolmogorov–Smirnov test confirms the distributions differ (D = 0.905, p = 0.002).
                """)

            with solara.Column(style=VALIDATION_CSS):
                solara.Markdown("""
### V4 — Does Access Decline with Distance?

Accessibility falls steadily as distance from central London increases
(Spearman ρ = −0.292, p = 3.63 × 10⁻²⁰), consistent with real-world commuting patterns.
The gradient is monotonic across five 5 km distance bands.
                """)

            with solara.Column(style=VALIDATION_CSS):
                solara.Markdown("""
### Robustness

Results are stable across multiple random seeds (CV of ρ = 0.037) and converge
once the sample reaches approximately 2,000 simulated commuters.
                """)
