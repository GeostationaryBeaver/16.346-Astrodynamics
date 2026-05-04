"""
design_A.py
===========
Design A: Helix formation with 0.5–1.5 km inter-satellite distances.
Force model: J2 zonal harmonics only (ideal/simplified).
Includes solar-power tracking for all three spacecraft.
"""

import orekit
orekit.initVM()
from orekit.pyhelpers import setup_orekit_curdir
setup_orekit_curdir("term_project/src/orekit-data.zip")


import plotting
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from propagate import (
    AbsoluteDate, TimeScalesFactory,
    FramesFactory,
    Constants, IERSConventions,
    GravityFieldFactory,
    KeplerianOrbit, PositionAngleType,
    init_close_helix_deputies,
    apply_ROE, run_propagation_dsst,
    CelestialBodyFactory, get_i,
    DSSTZonal,
    build_full_force_model,
    init_string_of_pearls,
)

# ── Save directory ──
THIS_DIR = Path(__file__).resolve().parent
path = THIS_DIR.parent / "figs" / "full_force_model" / "design_A" / "pearls"

# ══════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════
utc = TimeScalesFactory.getUTC()
initial_date = AbsoluteDate(2026, 1, 1, 6, 0, 0.0, utc)

eci  = FramesFactory.getEME2000()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
mu   = Constants.EIGEN5C_EARTH_MU

# ── Sun (for RAAN alignment AND power tracking) ──
sun    = CelestialBodyFactory.getSun()
sun_pv = sun.getPVCoordinates(initial_date, eci).getPosition()
sun_ra = np.arctan2(sun_pv.getY(), sun_pv.getX())

# ── Force model: J2 only ──
# perturbs = build_full_force_model(gravity_degree=2, gravity_order=0,
#                                   include_drag=False,include_srp=False,
#                                   include_third_body=False)
perturbs = build_full_force_model()

max_dist = 1.5 #(km)

# ══════════════════════════════════════════════════════════════════════
# CHIEF ORBIT
# ══════════════════════════════════════════════════════════════════════
a_c  = float(1000e3 + Constants.WGS84_EARTH_EQUATORIAL_RADIUS)
e_c  = float(0.0)
i_c  = get_i(a_c, e_c)
arg_peri       = float(0.0)
chief_init_raan = float(sun_ra - np.pi / 2.0)   # dawn-dusk alignment
M_0  = float(0.0)

chief_orbit = KeplerianOrbit(
    a_c, e_c, i_c, arg_peri, chief_init_raan, M_0,
    PositionAngleType.MEAN, eci, initial_date, mu
)

# ══════════════════════════════════════════════════════════════════════
# DEPUTY INITIALIZATION — HELIX
# ══════════════════════════════════════════════════════════════════════
deputy_roes   = init_close_helix_deputies(chief_orbit, helix_radius_m=300)
deputy_orbits = [apply_ROE(chief_orbit, r) for r in deputy_roes]

# ══════════════════════════════════════════════════════════════════════
# TIME GRID
# ══════════════════════════════════════════════════════════════════════
# T_orb  = 2 * np.pi * float(np.sqrt(a_c**3 / mu))
# T_span = 5 * T_orb
T_span = 6 * 30 * 24 * 3600
N      = 1500
times  = np.linspace(0, T_span, N)
days   = times / 86400

# ══════════════════════════════════════════════════════════════════════
# PROPAGATION (with solar-power tracking via sun= argument)
# ══════════════════════════════════════════════════════════════════════
(a, e, i, raan, argp, M, alt, rel, dist, power) = run_propagation_dsst(
    times, initial_date, perturbs, chief_orbit, deputy_orbits, sun=sun
)

# ══════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════
colors = ["steelblue", "tomato"]
labels = ["Deputy 1", "Deputy 2"]

plotting.plot_hill_3d(rel, colors, labels, path=path)
plotting.plot_radial_intrack(rel, colors, labels, path=path)
plotting.plot_intrack_crosstrack(rel, colors, labels, path=path)
plotting.plot_solar_power(times, power, colors, labels, path=path)
plotting.plot_mean_separation_with_exits(times, rel, labels, colors, max_dist, "", path=path)
plotting.plot_asymmetric_sep(times,rel, path=path)

plt.show()

