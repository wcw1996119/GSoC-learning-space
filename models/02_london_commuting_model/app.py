from mesa.visualization import SolaraViz, make_plot_component
from model import LondonCommuteModel
from agents import MSOAAgent
import solara
import solara.lab
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter
import numpy as np
import threading
import time


# ── Model instance ──────────────────────────────────────────────
model = solara.reactive(LondonCommuteModel(n_commuters=5000))
is_playing = solara.reactive(False)
step_count = solara.reactive(0)

# Model params
n_commuters_param = solara.reactive(5000)
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


# ── Draw helpers ────────────────────────────────────────────────
def _hour_label(hour):
    period = "AM" if hour < 12 else "PM"
    h = hour if hour <= 12 else hour - 12
    return f"{h}:00 {period}"


def _make_time_label(step, steps_per_day=16, start_hour=6):
    day = step // steps_per_day + 1
    hour = start_hour + (step % steps_per_day)
    period = "AM" if hour < 12 else "PM"
    h = hour if hour <= 12 else hour - 12
    return f"D{day} {h}{period}"


# ── Map components ───────────────────────────────────────────────
@solara.component
def CommuteTimeMap(m):
    step_count.get()
    home_times, home_counts = {}, {}
    for agent in m._commuter_agent_list:
        if agent.commute_time_minutes > 0 and agent.home_msoa:
            home = agent.home_msoa
            home_times[home] = home_times.get(home, 0) + agent.commute_time_minutes
            home_counts[home] = home_counts.get(home, 0) + 1
    mean_times = {h: home_times[h] / home_counts[h] for h in home_times}
    gdf = m.gdf.copy()
    gdf['mean_commute_time'] = gdf['MSOA21CD'].map(mean_times)
    is_peak = m.current_hour in [8, 9, 17, 18]
    status = "PEAK HOUR" if is_peak else "Off-peak"
    fig = Figure(figsize=(10, 8))
    ax = fig.subplots()
    gdf.plot(column='mean_commute_time', ax=ax, cmap='RdYlBu_r',
             legend=True,
             legend_kwds={'label': 'Mean commute time (min)',
                          'orientation': 'vertical', 'shrink': 0.7},
             missing_kwds={'color': 'lightgrey'},
             linewidth=0.3, edgecolor='white', vmin=0, vmax=60)
    ax.set_title(f"Mean Commute Time — {status} ({_hour_label(m.current_hour)})",
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def AccessibilityMap(m):
    step_count.get()
    acc_dict = {a.msoa_code: a.accessibility for a in m._msoa_agent_list}
    gdf = m.gdf.copy()
    gdf['accessibility'] = gdf['MSOA21CD'].map(acc_dict).fillna(0)
    is_peak = m.current_hour in [8, 9, 17, 18]
    status = "PEAK HOUR" if is_peak else "Off-peak"
    vals = gdf['accessibility'][gdf['accessibility'] > 0]
    vmin = vals.quantile(0.05) if len(vals) > 0 else 0
    vmax = vals.quantile(0.95) if len(vals) > 0 else 1
    fig = Figure(figsize=(10, 8))
    ax = fig.subplots()
    gdf.plot(column='accessibility', ax=ax, cmap='RdYlBu_r',
             legend=True,
             legend_kwds={'label': 'A_i = Σ E_j·exp(−β·t_ij)',
                          'orientation': 'vertical', 'shrink': 0.7},
             missing_kwds={'color': 'lightgrey'},
             linewidth=0.3, edgecolor='white', vmin=vmin, vmax=vmax)
    ax.set_title(f"Employment Accessibility — {status} ({_hour_label(m.current_hour)})",
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


# ── Time series ──────────────────────────────────────────────────
@solara.component
def GiniTimeSeries(m):
    step_count.get()
    data = m.datacollector.get_model_vars_dataframe()
    if data.empty:
        return
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    steps = list(range(len(data)))
    vals = data['Accessibility_Gini'].values
    ax.plot(steps, vals, color='steelblue', linewidth=1.5)
    peak_steps = [s for s in steps if (s % 16) in [2, 3, 11, 12]]
    if peak_steps:
        ax.scatter(peak_steps, vals[peak_steps],
                   color='red', s=20, zorder=5, label='Peak hour')
    ax.set_xlabel("Time")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Accessibility Inequality (Gini)")
    tick_steps = list(range(0, len(steps), 16))
    ax.set_xticks(tick_steps)
    ax.set_xticklabels([_make_time_label(s) for s in tick_steps], fontsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def PalmaTimeSeries(m):
    step_count.get()
    data = m.datacollector.get_model_vars_dataframe()
    if data.empty:
        return
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    steps = list(range(len(data)))
    vals = data['Accessibility_Palma'].values
    ax.plot(steps, vals, color='purple', linewidth=1.5)
    peak_steps = [s for s in steps if (s % 16) in [2, 3, 11, 12]]
    if peak_steps:
        ax.scatter(peak_steps, vals[peak_steps],
                   color='red', s=20, zorder=5, label='Peak hour')
    ax.set_xlabel("Time")
    ax.set_ylabel("Palma ratio")
    ax.set_title("Palma Ratio\n(Top 10% / Bottom 40%)")
    tick_steps = list(range(0, len(steps), 16))
    ax.set_xticks(tick_steps)
    ax.set_xticklabels([_make_time_label(s) for s in tick_steps], fontsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def CommuteTimeTimeSeries(m):
    step_count.get()
    data = m.datacollector.get_model_vars_dataframe()
    if data.empty:
        return
    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    steps = list(range(len(data)))
    ax.plot(steps, data['Mean_Commute_Time'].values,
            color='darkorange', linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean commute time (min)")
    ax.set_title("Mean Commute Time")
    tick_steps = list(range(0, len(steps), 16))
    ax.set_xticks(tick_steps)
    ax.set_xticklabels([_make_time_label(s) for s in tick_steps], fontsize=7)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def CommuteTimeHistogram(m):
    step_count.get()
    times = np.array([a.commute_time_minutes for a in m._commuter_agent_list
                      if a.commute_time_minutes > 0])
    if len(times) == 0:
        return
    is_peak = m.current_hour in [8, 9, 17, 18]
    color = "tomato" if is_peak else "steelblue"
    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots()
    ax.hist(times, bins=40, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(times), color='black', linestyle='--', linewidth=1.5,
               label=f"Mean: {np.mean(times):.1f} min")
    ax.axvline(np.median(times), color='gray', linestyle=':', linewidth=1.5,
               label=f"Median: {np.median(times):.1f} min")
    ax.set_xlabel("Commute time (minutes)")
    ax.set_ylabel("Number of commuters")
    ax.set_title(f"Commute Time Distribution — "
                 f"{'Peak' if is_peak else 'Off-peak'} ({_hour_label(m.current_hour)})")
    ax.legend(fontsize=8)
    ax.set_xlim(0, np.percentile(times, 98))
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
                times.append(agent.commute_time_minutes)
    if not distances:
        return
    distances = np.array(distances)
    times = np.array(times)
    bins = [0, 5, 10, 15, 20, 25, 30]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25+']
    bin_means, bin_labels = [], []
    for i in range(len(bins)-1):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(np.mean(times[mask]))
            bin_labels.append(labels[i])
    is_peak = m.current_hour in [8, 9, 17, 18]
    color = "tomato" if is_peak else "steelblue"
    fig = Figure(figsize=(5, 3.5))
    ax = fig.subplots()
    bars = ax.bar(bin_labels, bin_means, color=color, alpha=0.7, edgecolor='white')
    ax.set_xlabel("Distance from city centre (km)")
    ax.set_ylabel("Mean commute time (min)")
    ax.set_title(f"Commute Time by Distance — "
                 f"{'Peak' if is_peak else 'Off-peak'} ({_hour_label(m.current_hour)})")
    for bar, val in zip(bars, bin_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def OutflowComparisonPlot(m):
    step_count.get()
    sim_outflow = {}
    for agent in m._commuter_agent_list:
        if agent.home_msoa:
            sim_outflow[agent.home_msoa] = sim_outflow.get(agent.home_msoa, 0) + 1
    real_outflow = m.od_df.groupby('MSOA21CD_home')['count'].sum().to_dict()
    common = sorted(set(sim_outflow.keys()) & set(real_outflow.keys()))
    if len(common) < 2:
        return
    sim_vals = np.array([sim_outflow[k] for k in common], dtype=float)
    real_vals = np.array([real_outflow[k] for k in common], dtype=float)
    sim_norm = (sim_vals - sim_vals.min()) / (sim_vals.max() - sim_vals.min() + 1e-10)
    real_norm = (real_vals - real_vals.min()) / (real_vals.max() - real_vals.min() + 1e-10)
    corr = np.corrcoef(sim_norm, real_norm)[0, 1]
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    ax.scatter(real_norm, sim_norm, alpha=0.4, s=12, color="darkorange")
    z = np.polyfit(real_norm, sim_norm, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Real outflow per home MSOA (normalised)")
    ax.set_ylabel("Simulated outflow (normalised)")
    ax.set_title("Validation: Simulated vs Real Commuting Outflow\n"
                 "(proxy: commuters sampled from real OD weights)")
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


@solara.component
def AccessibilityComparisonPlot(m):
    step_count.get()
    sim_acc = {a.msoa_code: a.accessibility for a in m._msoa_agent_list}
    real_acc = m.real_accessibility
    common = sorted(set(sim_acc.keys()) & set(real_acc.keys()))
    if len(common) < 2:
        return
    sim_vals = np.array([sim_acc[k] for k in common], dtype=float)
    real_vals = np.array([real_acc[k] for k in common], dtype=float)

    def percentile_norm(arr):
        p5, p95 = np.percentile(arr, [5, 95])
        return np.clip((arr - p5) / (p95 - p5 + 1e-10), 0, 1)

    sim_norm = percentile_norm(sim_vals)
    real_norm = percentile_norm(real_vals)
    corr = np.corrcoef(sim_norm, real_norm)[0, 1]
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    ax.scatter(real_norm, sim_norm, alpha=0.4, s=12, color="steelblue")
    z = np.polyfit(real_norm, sim_norm, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Real accessibility (normalised)")
    ax.set_ylabel("Simulated accessibility (normalised)")
    ax.set_title("Validation: Simulated vs Real Accessibility\n"
                 "(gravity model, free-flow baseline)")
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()
    solara.FigureMatplotlib(fig)


# ── Main Page ────────────────────────────────────────────────────
@solara.component
def Page():
    m = model.value
    sc = step_count.value  # trigger re-render on step

    solara.Title("London Commuting Accessibility Model")

    # ── Header bar ──
    with solara.Row(style="align-items:center; gap:16px; padding:8px 16px; "
                          "background:#1565C0; color:white;"):
        solara.Text("London Commuting Accessibility Model",
                    style="font-size:20px; font-weight:bold; color:white;")

    # ── Controls bar ──
    with solara.Row(style="align-items:center; gap:12px; padding:8px 16px; "
                          "background:#f5f5f5; border-bottom:1px solid #ddd;"):

        # Buttons
        solara.Button("⟳ Reset", on_click=reset_model,
                      style="background:#e53935; color:white;")
        play_label = "⏸ Pause" if is_playing.value else "▶ Play"
        solara.Button(play_label, on_click=toggle_play,
                      style="background:#1976D2; color:white;")
        solara.Button("⏭ Step", on_click=do_step,
                      style="background:#388E3C; color:white;")

        # Sliders
        solara.Text("Commuters:", style="margin-left:16px;")
        solara.SliderInt("", value=n_commuters_param, min=500, max=10000, step=500)
        solara.Text("BPR β:", style="margin-left:8px;")
        solara.SliderFloat("", value=bpr_beta_param, min=0.5, max=6.0, step=0.5)

        # Current time display
        hour = m.current_hour
        period = "AM" if hour < 12 else "PM"
        display_hour = hour if hour <= 12 else hour - 12
        is_peak = hour in [8, 9, 17, 18]
        status = "PEAK HOUR" if is_peak else "Off-peak"
        color = "#c62828" if is_peak else "#2e7d32"
        solara.Text(
            f"  |  Step {sc}  |  {display_hour}:00 {period}  —  {status}",
            style=f"font-size:15px; font-weight:bold; color:{color}; margin-left:16px;"
        )

    # ── Tabs ──
    with solara.lab.Tabs():

        with solara.lab.Tab("🗺 Maps"):
            with solara.Columns([1, 1]):
                CommuteTimeMap(m)
                AccessibilityMap(m)

        with solara.lab.Tab("📊 Inequality"):
            with solara.Columns([1, 1, 1]):
                GiniTimeSeries(m)
                PalmaTimeSeries(m)
                CommuteTimeTimeSeries(m)
            with solara.Columns([1, 1]):
                CommuteTimeHistogram(m)
                CommutTimeByDistance(m)

        with solara.lab.Tab("✅ Validation"):
            with solara.Columns([1, 1]):
                OutflowComparisonPlot(m)
                AccessibilityComparisonPlot(m)