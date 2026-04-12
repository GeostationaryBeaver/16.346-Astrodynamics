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

def plot_orbital_elements(days_long, a_l, e_l, i_l, raan_l, argp_l, M_l):
    for vals, title in zip(
        [a_l, e_l, i_l, raan_l, argp_l, M_l],
        ["a (m)", "e", "i (deg)", "RAAN (deg)", "ω (deg)", "M (deg)"]
    ):
        fig, ax = plt.subplots()
        ax.plot(days_long, vals, linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Days")
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.suptitle("Chief Orbital Elements — 6 Months (J_2)", fontsize=13)
        plt.tight_layout()
        savefig(".")


def plot_earth_frame(chief_eci, dep_eci_exag, snap_idx, colors, labels, EXAG):
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
    fig.suptitle(
        f'Formation Flying — Earth Frame (5 orbits)\nDeputy offsets ×{EXAG} in ECI',
        fontsize=11
    )
    fig.tight_layout()
    savefig("figures/Dep_Offsets.png")


def plot_hill_3d(rel_short, colors, labels):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='red', s=60, zorder=5, label='Chief')

    for k, arr in enumerate(rel_short):
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2],
                color=colors[k], linewidth=0.9, alpha=0.85, label=labels[k])
        ax.scatter(*arr[0], color=colors[k], s=30, zorder=5)

    ax.set_xlabel('Radial (m)', fontsize=8)
    ax.set_ylabel('In-track (m)', fontsize=8)
    ax.set_zlabel('Cross-track (m)', fontsize=8)
    ax.set_title('Hill Frame (LVLH) — 3D', fontsize=11)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_box_aspect([1, 1, 1])
    ax.tick_params(labelsize=7)
    fig.suptitle('Formation Flying — Hill Frame 3D (5 orbits)', fontsize=11)
    fig.tight_layout()
    savefig("figures/Hill_Traj.png")


def plot_radial_intrack(rel_short, colors, labels):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(0, 0, color='red', s=60, zorder=5, label='Chief')

    for k, arr in enumerate(rel_short):
        ax.plot(arr[:, 0], arr[:, 1], color=colors[k], linewidth=1.0, label=labels[k])
        ax.scatter(arr[0, 0], arr[0, 1], color=colors[k], s=30, zorder=5)

    ax.set_xlabel('Radial (m)')
    ax.set_ylabel('In-track (m)')
    ax.set_title('Hill Frame — Radial vs In-track', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=7)
    fig.suptitle('Formation Flying — Radial vs In-track (5 orbits)', fontsize=11)
    fig.tight_layout()
    savefig("figures/Radial_Intrack.png")


def plot_intrack_crosstrack(rel_short, colors, labels):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(0, 0, color='red', s=60, zorder=5, label='Chief')

    for k, arr in enumerate(rel_short):
        ax.plot(arr[:, 1], arr[:, 2], color=colors[k], linewidth=1.0, label=labels[k])
        ax.scatter(arr[0, 1], arr[0, 2], color=colors[k], s=30, zorder=5)

    ax.set_xlabel('In-track (m)')
    ax.set_ylabel('Cross-track (m)')
    ax.set_title('Hill Frame — In-track vs Cross-track', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=7)
    fig.suptitle('Formation Flying — In-track vs Cross-track (5 orbits)', fontsize=11)
    fig.tight_layout()
    savefig("figures/Intrack_Crosstrack.png")