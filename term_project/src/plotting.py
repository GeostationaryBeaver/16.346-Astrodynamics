import matplotlib.pyplot as plt
import numpy as np


def draw_earth(ax, R=6.371e6):
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='royalblue', alpha=0.12, linewidth=0.3)
    ax.plot_surface(x, y, z, color='steelblue', alpha=0.07)

def plot_orbital_elements(days_long, a_l, e_l, i_l, raan_l, argp_l, M_l, path=None):
    for vals, title in zip(
        [a_l, e_l, i_l, raan_l, argp_l, M_l],
        ["a (m)", "e", "i (deg)", "RAAN (deg)", "ω (deg)", "M (deg)"]
    ):
        fig, ax = plt.subplots()
        ax.plot(days_long, vals, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Days")
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        if path is not None:
            plt.savefig(f"{path}/{title}.png")


def plot_earth_frame(chief_eci, dep_eci_exag, snap_idx, colors, labels, EXAG, path=None):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    draw_earth(ax)
    ax.plot(chief_eci[:, 0], chief_eci[:, 1], chief_eci[:, 2],
            color='red', linewidth=1.2, label='Chief')

    for k, arr in enumerate(dep_eci_exag):
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2],
                color=colors[k], linewidth=0.9, alpha=0.85, label=labels[k])

    for idx in snap_idx:
        for k, arr in enumerate(dep_eci_exag):
            ax.plot([chief_eci[idx, 0], arr[idx, 0]],
                    [chief_eci[idx, 1], arr[idx, 1]],
                    [chief_eci[idx, 2], arr[idx, 2]],
                    color=colors[k], linewidth=0.5, alpha=0.2, linestyle='--')

    ax.scatter(0, 0, 0, color='blue', s=60, zorder=6, label='Earth centre')
    lim = 8e6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_title(f'Earth Frame (deputy offset ×{EXAG})', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    if path is not None:
        plt.savefig(f"{path}/Dep_Offsets.png")


def plot_hill_3d(rel_short, colors, labels, path=None):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='red', s=60, zorder=5, label='Chief')

    for k, arr in enumerate(rel_short):
        ax.plot(arr[:, 0]/1e3, arr[:, 1]/1e3, arr[:, 2]/1e3,
                color=colors[k], linewidth=0.9, alpha=0.85, label=labels[k])
        ax.scatter(*arr[0]/1e3, color=colors[k], s=30, zorder=5)

    ax.set_xlabel('Radial (km)', fontsize=8)
    ax.set_ylabel('In-track (km)', fontsize=8)
    ax.set_zlabel('Cross-track (km)', fontsize=8)
    ax.set_title('Hill Frame (LVLH) — 3D', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    if path is not None:
        plt.savefig(f"{path}/Hill_Traj.png")


def plot_radial_intrack(rel_short, colors, labels, path=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(0, 0, color='red', s=60, zorder=5, label='Chief')

    for k, arr in enumerate(rel_short):
        ax.plot(arr[:, 0]/1e3, arr[:, 1]/1e3, color=colors[k], linewidth=1.0, label=labels[k])
        ax.scatter(arr[0, 0]/1e3, arr[0, 1]/1e3, color=colors[k], s=30, zorder=5)

    ax.set_xlabel('Radial (km)')
    ax.set_ylabel('In-track (km)')
    ax.set_title('Hill Frame — Radial vs In-track', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if path is not None:
        plt.savefig(f"{path}/Radial_Intrack.png")


def plot_intrack_crosstrack(rel_short, colors, labels, path=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(0, 0, color='red', s=60, zorder=5, label='Chief')

    for k, arr in enumerate(rel_short):
        ax.plot(arr[:, 1]/1e3, arr[:, 2]/1e3, color=colors[k], linewidth=1.0, label=labels[k])
        ax.scatter(arr[0, 1]/1e3, arr[0, 2]/1e3, color=colors[k], s=30, zorder=5)

    ax.set_xlabel('In-track (km)')
    ax.set_ylabel('Cross-track (km)')
    ax.set_title('Hill Frame — In-track vs Cross-track', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=7)
    fig.tight_layout()
    if path is not None:
        plt.savefig(f"{path}/Intrack_Crosstrack.png")

def plot_mean_separation_with_exits(times_s, rel_list, labels, colors, box_side_km, max_dist,title="", save_path=None):
    days = np.asarray(times_s) / 86400.0

    dep_sep_km = [np.linalg.norm(r, axis=1) / 1000.0 for r in rel_list]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(y=max_dist, color = "grey", linestyle="--")
    for k, d in enumerate(dep_sep_km):
        ax.plot(days, d, color=colors[k], label=f"{labels[k]}-Chief")

        # box exit marker
        if np.isscalar(box_side_km):
            L = np.array([box_side_km, box_side_km, box_side_km]) * 1000.0
        else:
            L = np.array(box_side_km) * 1000.0
        half = 0.5 * L
        outside = np.any(np.abs(rel_list[k]) > half, axis=1)
        # # Temporary debug check: is the TOTAL distance > 5km?
        # dist = np.linalg.norm(rel_list[k], axis=1)
        # outside = dist > 5000.0
        if np.any(outside):
            idx = int(np.argmax(outside))
            x = days[idx]
            ax.axvline(x, color=colors[k], ls="--", alpha=0.8)
            ax.plot(x, d[idx], "o", color=colors[k], ms=5)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Separation [km]")
    ax.set_title(title if title else "Deputy-Chief Separation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        fig.savefig(f"{save_path}/separations.png", dpi=200, bbox_inches="tight")
    return fig, ax

def plot_solar_power(times_s, power_dict, colors, labels, path=None):
    """
    Plot instantaneous solar-panel power for the chief and all deputies.

    Produces two subplots stacked vertically:
      - **Top**: Individual power traces for every spacecraft vs. time.
        Eclipse entry is visible as a sharp drop to 0 W.
      - **Bottom**: Total fleet power (sum of all spacecraft) vs. time,
        giving a system-level view of the orbital data center's power budget.

    Parameters
    ----------
    times_s : array-like of float
        Time stamps [s] from epoch.  Converted to hours internally.
    power_dict : dict
        Power history dictionary as returned by ``run_propagation_dsst``::

            {
                "chief":    np.ndarray shape (N,),   # Watts
                "deputies": [np.ndarray, ...],       # Watts, per deputy
            }
    colors : list of str
        Matplotlib colour for each deputy trace (len = number of deputies).
    labels : list of str
        Legend label for each deputy (len = number of deputies).
    path : str or Path or None, optional
        If given, the figure is saved to ``{path}/Solar_Power.png``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of matplotlib.axes.Axes
        (ax_individual, ax_total)

    Notes
    -----
    - The chief is always plotted in black with the label "Chief".
    - Time axis is in hours for short propagations and in days for
      propagations longer than 48 hours (auto-detected).
    """
    hours = np.asarray(times_s) / 3600.0

    # auto-select x-axis units
    if hours[-1] > 48.0:
        x = np.asarray(times_s) / 86400.0
        x_label = "Time [days]"
    else:
        x = hours
        x_label = "Time [hours]"

    chief_p = power_dict["chief"]
    dep_p   = power_dict["deputies"]

    n_sc    = 1 + len(dep_p)
    total   = chief_p.copy()
    for dp in dep_p:
        total += dp

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---- Top panel: individual traces ----
    ax1.plot(x, chief_p, color="black", linewidth=1.0, label="Chief")
    for k, dp in enumerate(dep_p):
        ax1.plot(x, dp, color=colors[k], linewidth=1.0,
                 label=labels[k], alpha=0.85)

    ax1.set_ylabel("Power [W]")
    ax1.set_title("Instantaneous Solar-Panel Power per Spacecraft")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_ylim(bottom=-5)

    # ---- Bottom panel: fleet total ----
    ax2.plot(x, total, color="darkgreen", linewidth=1.2, label="Fleet Total")
    ax2.axhline(
        y=np.mean(total), color="grey", linestyle=":", linewidth=0.8,
        label=f"Mean = {np.mean(total):.1f} W"
    )
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Power [W]")
    ax2.set_title(f"Total Formation Power ({n_sc} spacecraft)")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_ylim(bottom=-5)

    fig.tight_layout()

    if path is not None:
        fig.savefig(f"{path}/Solar_Power.png", dpi=200, bbox_inches="tight")

    return fig, (ax1, ax2)
