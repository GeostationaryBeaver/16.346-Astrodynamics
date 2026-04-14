import plotting
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from refactored_ideal_funcs import AbsoluteDate, TimeScalesFactory
from refactored_ideal_funcs import FramesFactory
from refactored_ideal_funcs import Constants, IERSConventions
from refactored_ideal_funcs import HolmesFeatherstoneAttractionModel
from refactored_ideal_funcs import GravityFieldFactory
from refactored_ideal_funcs import KeplerianOrbit, PositionAngleType, make_along_track_deputies


from refactored_ideal_funcs import run_propagation, get_eci_trajectories, apply_ROE


#save directory
THIS_DIR = Path(__file__).resolve().parent
path = THIS_DIR.parent / "figs" / "ideal_force_model"


# ------------------------------------
# 1. SETUP
# ------------------------------------
utc = TimeScalesFactory.getUTC()
initial_date = AbsoluteDate(2026, 1, 1, 12, 0, 0.0, utc)

eci = FramesFactory.getEME2000()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
mu = Constants.EIGEN5C_EARTH_MU

side_km = 1.5

# Earth gravity field (includes J2 perturbations)
gravity_provider = GravityFieldFactory.getNormalizedProvider(2, 0)
j2 = HolmesFeatherstoneAttractionModel(itrf, gravity_provider)

perturbs = [j2]


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
#DEPUTY ROEs (2 deputies only)
# ------------------------------------
rho = 1.2e-4   # ~840 m amplitude (a * rho)

# deputy_roes = [
#     dict(da=0, dl=0,      dex=rho,  dey=0.0,  dix=rho, diy=rho),    # Deputy 1
#     dict(da=0, dl=0,      dex=0.0,  dey=rho,  dix=rho, diy=-rho),   # Deputy 2
# ]

deputy_roes = make_along_track_deputies(chief_orbit, 0.5 * 1e3)

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






(a, e, i, raan, argp, M, alt, rel, dist) = run_propagation(times_short, initial_date,
                                                           perturbs, chief_orbit,
                                                           deputy_orbits)

days_short = times_short / 86400
days_long = times_long / 86400

# ECI trajectories for plotting
#chief_eci, dep_eci = get_eci_trajectories(times_long)

# EXAG = 75 #deputies exaggerated 75x to improve readability
# dep_eci_exag = [chief_eci + EXAG * (arr - chief_eci) for arr in dep_eci]
# n_snaps = 16
# snap_idx = np.linspace(0, len(times_long) - 1, n_snaps, dtype=int)

# Plot 1: Orbital elements
plotting.plot_orbital_elements(days_short, a, e, i, raan, argp, M, path)
# plotting.plot_earth_frame(chief_eci, dep_eci_exag, snap_idx, colors, labels, EXAG, path)
plotting.plot_hill_3d(rel, colors, labels, path)
plotting.plot_radial_intrack(rel, colors, labels, path)
plotting.plot_intrack_crosstrack(rel, colors, labels, path)
plotting.plot_mean_separation_with_exits(times_short, rel, labels, colors, side_km, "", path)

plt.show()
