import plotting
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from propagate import AbsoluteDate, TimeScalesFactory
from propagate import FramesFactory
from propagate import Constants, IERSConventions
from propagate import HolmesFeatherstoneAttractionModel
from propagate import GravityFieldFactory
from propagate import KeplerianOrbit, PositionAngleType, make_along_track_deputies
from propagate import run_propagation, get_eci_trajectories, apply_ROE


#save directory
THIS_DIR = Path(__file__).resolve().parent
path = THIS_DIR.parent / "figs" / "ideal_force_model" / "j2_resistant" / "design_B"


# ------------------------------------
# 1. SETUP
# ------------------------------------
utc = TimeScalesFactory.getUTC()
initial_date = AbsoluteDate(2026, 1, 1, 12, 0, 0.0, utc)

eci = FramesFactory.getEME2000()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
mu = Constants.EIGEN5C_EARTH_MU

side_km = 5

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

#initialize the deputies without accounting for J2
# deputy_roes = [
#     dict(da=rho, dl=0,      dex=rho,  dey=0.0,  dix=rho, diy=rho),    # Deputy 1
#     dict(da=-rho, dl=0,      dex=0.0,  dey=rho,  dix=rho, diy=-rho),   # Deputy 2
# ]

#initialize deputies on the same track as the chief
init_sep = 2000 #(m)
deputy_roes = make_along_track_deputies(chief_orbit, init_sep)

labels = [
    "Deputy 1",
    "Deputy 2",
]

colors = ["steelblue", "tomato"]

deputy_orbits = [apply_ROE(chief_orbit, r) for r in deputy_roes]

# ------------------------------------
# 5. TIME ARRAYS
# ------------------------------------
T_orb = 2 * np.pi * float(np.sqrt((7000e3)**3 / mu))
#T_span = 5 * T_orb
T_span = 1 * 30 * 24 * 3600

#N = 500
N = 1500

times = np.linspace(0, T_span, N)

(a, e, i, raan, argp, M, alt, rel, dist) = run_propagation(times, initial_date,
                                                           perturbs, chief_orbit,
                                                           deputy_orbits)

days = times / 86400


# Plot 1: Orbital elements
plotting.plot_orbital_elements(days, a, e, i, raan, argp, M, path)
plotting.plot_hill_3d(rel, colors, labels, path)
plotting.plot_radial_intrack(rel, colors, labels, path)
plotting.plot_intrack_crosstrack(rel, colors, labels, path)
plotting.plot_mean_separation_with_exits(times, rel, labels, colors, side_km, "", path)

plt.show()
