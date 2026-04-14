import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import savefig


def draw_earth(ax, R=6.371e6):
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='royalblue', alpha=0.12, linewidth=0.3)
    ax.plot_surface(x, y, z, color='steelblue', alpha=0.07)

def plot_orbital_elements(days_long, a_l, e_l, i_l, raan_l, argp_l, M_l, path):
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
        savefig(f"{path}/{title}.png")


def plot_earth_frame(chief_eci, dep_eci_exag, snap_idx, colors, labels, EXAG, path):
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
    savefig(f"{path}/Dep_Offsets.png")


def plot_hill_3d(rel_short, colors, labels, path):
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
    savefig(f"{path}/Hill_Traj.png")


def plot_radial_intrack(rel_short, colors, labels, path):
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
    savefig(f"{path}/Radial_Intrack.png")


def plot_intrack_crosstrack(rel_short, colors, labels, path):
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
    savefig(f"{path}/Intrack_Crosstrack.png")

def plot_mean_separation_with_exits(times_s, rel_list, labels, colors, box_side_km, title="", save_path=None):
    days = np.asarray(times_s) / 86400.0

    dep_sep_km = [np.linalg.norm(r, axis=1) / 1000.0 for r in rel_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, d in enumerate(dep_sep_km):
        ax.plot(days, d, color=colors[k], label=f"{labels[k]}-Chief")

        # box exit marker
        if np.isscalar(box_side_km):
            L = np.array([box_side_km, box_side_km, box_side_km]) * 1000.0
        else:
            L = np.array(box_side_km) * 1000.0
        half = 0.5 * L
        outside = np.any(np.abs(rel_list[k]) > half, axis=1)
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
