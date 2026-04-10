import orekit
orekit.initVM()
import os
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

if not os.path.exists("orekit-data.zip"):
    download_orekit_data_curdir()   # downloads orekit-data.zip to cwd

setup_orekit_curdir()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from org.orekit.utils import PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

# ------------------------------------
# 1. SETUP
# ------------------------------------
utc = TimeScalesFactory.getUTC()
initial_date = AbsoluteDate(2026, 1, 1, 12, 0, 0.0, utc)

eci = FramesFactory.getEME2000()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
mu = Constants.EIGEN5C_EARTH_MU

# Earth gravity field (includes J2, J3, ...)
gravity_provider = GravityFieldFactory.getNormalizedProvider(10, 10)
gravity_model = HolmesFeatherstoneAttractionModel(itrf, gravity_provider)


def create_propagator(initial_orbit):
    min_step = 0.1
    max_step = 300.0

    # State is 7D in numerical propagator (x,y,z,vx,vy,vz,mass)
    abs_tol = [10.0] * 7
    rel_tol = [1e-9] * 7

    integrator = DormandPrince853Integrator(min_step, max_step, abs_tol, rel_tol)
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)

    initial_state = SpacecraftState(initial_orbit)
    propagator.setInitialState(initial_state)
    propagator.addForceModel(gravity_model)
    return propagator


def to_keplerian(orbit):
    """Safely convert any Orekit orbit representation to KeplerianOrbit."""
    return KeplerianOrbit.cast_(OrbitType.KEPLERIAN.convertType(orbit))


# ------------------------------------
# 2. CHIEF ORBIT
# ------------------------------------
chief_orbit = KeplerianOrbit(
    float(7000e3),           # a (m)
    float(0.0),              # e
    float(np.radians(98.0)), # i
    float(0.0),              # argument of perigee
    float(0.0),              # RAAN
    float(0.0),              # mean anomaly
    PositionAngleType.MEAN,
    eci,
    initial_date,
    mu
)

# ------------------------------------
# 3. ROE -> ABSOLUTE KEPLERIAN ELEMENTS
# ------------------------------------
def apply_ROE(chief_orbit, roe):
    """
    Quasi-nonsingular ROE convention:
      da  = delta(a)/a
      dl  = delta(M + omega)
      dex = delta(e*cos(omega))
      dey = delta(e*sin(omega))
      dix = delta(i)
      diy = delta(RAAN)*sin(i)
    """
    a_c = float(chief_orbit.getA())
    e_c = float(chief_orbit.getE())
    i_c = float(chief_orbit.getI())
    raan_c = float(chief_orbit.getRightAscensionOfAscendingNode())
    argp_c = float(chief_orbit.getPerigeeArgument())
    M_c = float(chief_orbit.getMeanAnomaly())

    ex_c = e_c * np.cos(argp_c)
    ey_c = e_c * np.sin(argp_c)

    a_d = float(a_c * (1.0 + roe["da"]))

    ex_d = ex_c + roe["dex"]
    ey_d = ey_c + roe["dey"]
    e_d = float(np.sqrt(ex_d**2 + ey_d**2))
    argp_d = float(np.arctan2(ey_d, ex_d))

    i_d = float(i_c + roe["dix"])
    raan_d = float(raan_c + roe["diy"] / np.sin(i_c))

    d_argp = argp_d - argp_c
    M_d = float(M_c + roe["dl"] - d_argp)

    return KeplerianOrbit(
        a_d, e_d, i_d, argp_d, raan_d, M_d,
        PositionAngleType.MEAN,
        chief_orbit.getFrame(),
        chief_orbit.getDate(),
        mu
    )


# ------------------------------------
# 4. DEPUTY ROEs (2 deputies only)
# ------------------------------------
rho = 1.2e-4   # ~840 m amplitude (a * rho)

deputy_roes = [
    dict(da=0, dl=0,      dex=rho,  dey=0.0,  dix=rho, diy=rho),    # Deputy 1
    dict(da=0, dl=0,      dex=0.0,  dey=rho,  dix=rho, diy=-rho),   # Deputy 2
]

labels = [
    "Deputy 1 (PCO, φ=0°)",
    "Deputy 2 (PCO, φ=90°)",
]
colors = ["steelblue", "tomato"]

deputy_orbits = [apply_ROE(chief_orbit, r) for r in deputy_roes]

# ------------------------------------
# 5. TIME ARRAYS
# ------------------------------------
T_orb = 2 * np.pi * float(np.sqrt((7000e3)**3 / mu))
T_short = 5 * T_orb
T_long = 6 * 30 * 24 * 3600

N_short = 500
N_long = 1500

times_short = np.linspace(0, T_short, N_short)
times_long = np.linspace(0, T_long, N_long)


# ------------------------------------
# 6. PROPAGATION HELPER
# ------------------------------------
def run_propagation(times):
    chief_prop = create_propagator(chief_orbit)
    dep_props = [create_propagator(o) for o in deputy_orbits]

    a_v, e_v, i_v, raan_v, argp_v, M_v, alt_v = [], [], [], [], [], [], []
    rel = [[] for _ in deputy_orbits]
    dist = [[] for _ in deputy_orbits]

    for t in times:
        date_t = initial_date.shiftedBy(float(t))

        # Chief
        chief_state = chief_prop.propagate(date_t)
        chief_kep = to_keplerian(chief_state.getOrbit())

        pv_c = chief_state.getPVCoordinates(eci)
        p_c = pv_c.getPosition()
        v_c = pv_c.getVelocity()

        a_v.append(chief_kep.getA())
        e_v.append(chief_kep.getE())
        i_v.append(np.degrees(chief_kep.getI()))
        raan_v.append(np.degrees(chief_kep.getRightAscensionOfAscendingNode()))
        argp_v.append(np.degrees(chief_kep.getPerigeeArgument()))
        M_v.append(np.degrees(chief_kep.getMeanAnomaly()) % 360)

        tr = eci.getTransformTo(itrf, date_t)
        p_itrf = tr.transformPVCoordinates(PVCoordinates(p_c, Vector3D.ZERO)).getPosition()
        alt_v.append(p_itrf.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS)
        alt_v.append(p_itrf.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS)

        # LVLH basis
        r_vec = np.array([p_c.getX(), p_c.getY(), p_c.getZ()])
        v_vec = np.array([v_c.getX(), v_c.getY(), v_c.getZ()])
        r_hat = r_vec / np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_hat = h_vec / np.linalg.norm(h_vec)
        t_hat = np.cross(h_hat, r_hat)

        # Deputies
        for k in range(len(deputy_orbits)):
            dep_state = dep_props[k].propagate(date_t)
            pv_d = dep_state.getPVCoordinates(eci)
            p_d = pv_d.getPosition()

            dr = np.array([
                p_d.getX() - p_c.getX(),
                p_d.getY() - p_c.getY(),
                p_d.getZ() - p_c.getZ()
            ])

            rel[k].append([
                float(np.dot(dr, r_hat)),
                float(np.dot(dr, t_hat)),
                float(np.dot(dr, h_hat))
            ])
            dist[k].append(float(np.linalg.norm(dr)))

    return (
        a_v, e_v, i_v, raan_v, argp_v, M_v, alt_v,
        [np.array(rel[k]) for k in range(len(deputy_orbits))],
        [np.array(dist[k]) for k in range(len(deputy_orbits))]
    )


def get_eci_trajectories(times):
    chief_prop = create_propagator(chief_orbit)
    dep_props = [create_propagator(o) for o in deputy_orbits]

    chief_pts = []
    dep_pts = [[] for _ in deputy_orbits]

    for t in times:
        date_t = initial_date.shiftedBy(float(t))

        c_state = chief_prop.propagate(date_t)
        c_pos = c_state.getPVCoordinates(eci).getPosition()
        chief_pts.append([c_pos.getX(), c_pos.getY(), c_pos.getZ()])

        for k, prop in enumerate(dep_props):
            d_state = prop.propagate(date_t)
            d_pos = d_state.getPVCoordinates(eci).getPosition()
            dep_pts[k].append([d_pos.getX(), d_pos.getY(), d_pos.getZ()])

    chief_pts = np.array(chief_pts)
    dep_pts = [np.array(arr) for arr in dep_pts]
    return chief_pts, dep_pts


# Run both time spans
(a_s, e_s, i_s, raan_s, argp_s, M_s, alt_s, rel_short, dist_short) = run_propagation(times_short)
(a_l, e_l, i_l, raan_l, argp_l, M_l, alt_l, rel_long, dist_long) = run_propagation(times_long)

days_short = times_short / 86400
days_long = times_long / 86400

# ECI trajectories for plotting
chief_eci, dep_eci = get_eci_trajectories(times_long)

EXAG = 75 #deputies exaggerated 75x to improve readability
dep_eci_exag = [chief_eci + EXAG * (arr - chief_eci) for arr in dep_eci]
n_snaps = 16
snap_idx = np.linspace(0, len(times_long) - 1, n_snaps, dtype=int)


def draw_earth(ax, R=6.371e6):
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color='royalblue', alpha=0.12, linewidth=0.3)
    ax.plot_surface(x, y, z, color='steelblue', alpha=0.07)


# ------------------------------------
# 7. PLOTS
# ------------------------------------
chief_color = "red"
colors = ["blue", "gold"]   # deputies: blue, yellow
# Plot 1: Orbital elements
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, vals, title in zip(
    axes.flat,
    [a_l, e_l, i_l, raan_l, argp_l, M_l],
    ["a (m)", "e", "i (deg)", "RAAN (deg)", "ω (deg)", "M (deg)"]
):
    ax.plot(days_long, vals, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Days")
    ax.grid(True, linestyle='--', alpha=0.4)



plt.suptitle("Chief Orbital Elements — 6 Months (Numerical, J2..J10)", fontsize=13)
plt.tight_layout()
plt.show()


# Plot 2: Formation
fig = plt.figure(figsize=(14, 12))

ax1 = fig.add_subplot(221, projection='3d')
draw_earth(ax1)
ax1.plot(chief_eci[:, 0], chief_eci[:, 1], chief_eci[:, 2],
         color='red', linewidth=1.2, label='Chief')

# plot all deputies
for k, arr in enumerate(dep_eci_exag):
    ax1.plot(arr[:, 0], arr[:, 1], arr[:, 2],
             color=colors[k], linewidth=0.9, alpha=0.85, label=labels[k])

for idx in snap_idx:
    for k, arr in enumerate(dep_eci_exag):
        ax1.plot([chief_eci[idx, 0], arr[idx, 0]],
                 [chief_eci[idx, 1], arr[idx, 1]],
                 [chief_eci[idx, 2], arr[idx, 2]],
                 color=colors[k], linewidth=0.5, alpha=0.2, linestyle='--')

ax1.scatter(0, 0, 0, color='blue', s=60, zorder=6, label='Earth centre')
lim = 8e6
ax1.set_xlim(-lim, lim)
ax1.set_ylim(-lim, lim)
ax1.set_zlim(-lim, lim)
ax1.set_title(f'Earth frame (deputy offset ×{EXAG})', fontsize=10)
ax1.legend(fontsize=7, loc='upper left')
ax1.set_box_aspect([1, 1, 1])
ax1.tick_params(labelsize=7)

ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(0, 0, 0, color='red', s=60, zorder=5, label='Chief')
for k, arr in enumerate(rel_short):
    ax2.plot(arr[:, 0], arr[:, 1], arr[:, 2],
             color=colors[k], linewidth=0.9, alpha=0.85, label=labels[k])
    ax2.scatter(*arr[0], color=colors[k], s=30, zorder=5)
ax2.set_xlabel('Radial (m)', fontsize=8)
ax2.set_ylabel('In-track (m)', fontsize=8)
ax2.set_zlabel('Cross-track (m)', fontsize=8)
ax2.set_title('Hill frame (LVLH) — 3D', fontsize=10)
ax2.legend(fontsize=7, loc='upper left')
ax2.set_box_aspect([1, 1, 1])
ax2.tick_params(labelsize=7)

ax3 = fig.add_subplot(223)
ax3.scatter(0, 0, color='red', s=60, zorder=5, label='Chief')
for k, arr in enumerate(rel_short):
    ax3.plot(arr[:, 0], arr[:, 1], color=colors[k], linewidth=1.0, label=labels[k])
    ax3.scatter(arr[0, 0], arr[0, 1], color=colors[k], s=30, zorder=5)
ax3.set_xlabel('Radial (m)')
ax3.set_ylabel('In-track (m)')
ax3.set_title('Hill frame — Radial vs In-track', fontsize=10)
ax3.set_aspect('equal')
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend(fontsize=7)

ax4 = fig.add_subplot(224)
ax4.scatter(0, 0, color='red', s=60, zorder=5, label='Chief')
for k, arr in enumerate(rel_short):
    ax4.plot(arr[:, 1], arr[:, 2], color=colors[k], linewidth=1.0, label=labels[k])
    ax4.scatter(arr[0, 1], arr[0, 2], color=colors[k], s=30, zorder=5)
ax4.set_xlabel('In-track (m)')
ax4.set_ylabel('Cross-track (m)')
ax4.set_title('Hill frame — In-track vs Cross-track', fontsize=10)
ax4.set_aspect('equal')
ax4.grid(True, linestyle='--', alpha=0.4)
ax4.legend(fontsize=7)

plt.suptitle(
    f'Formation Flying — Earth Frame + Hill Frame (5 orbits)\nDeputy offsets ×{EXAG} in ECI plots',
    fontsize=12
)
plt.tight_layout()
plt.show()
print("Saved: plot2_formation.png")
