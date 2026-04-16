import plotting
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from propagate import AbsoluteDate, TimeScalesFactory
from propagate import FramesFactory
from propagate import Constants, IERSConventions
from propagate import HolmesFeatherstoneAttractionModel
from propagate import GravityFieldFactory
from propagate import KeplerianOrbit, PositionAngleType, init_close_helix_deputies
from propagate import apply_ROE, run_propagation_dsst
from propagate import CelestialBodyFactory, get_i

from propagate import DSSTZonal

#save directory
THIS_DIR = Path(__file__).resolve().parent
path = THIS_DIR.parent / "figs" / "ideal_force_model" / "j2_resistant" / "design_A"


# ------------------------------------
# SETUP
# ------------------------------------
utc = TimeScalesFactory.getUTC()
initial_date = AbsoluteDate(2026, 1, 1, 6, 0, 0.0, utc)

eci = FramesFactory.getEME2000()
itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
mu = Constants.EIGEN5C_EARTH_MU

sun = CelestialBodyFactory.getSun()
sun_pv = sun.getPVCoordinates(initial_date, eci).getPosition()
sun_ra = np.arctan2(sun_pv.getY(), sun_pv.getX())

# Earth gravity field (includes J2 perturbations) -- OG FORMULATION
# gravity_provider = GravityFieldFactory.getNormalizedProvider(2, 0)
# j2 = HolmesFeatherstoneAttractionModel(itrf, gravity_provider)
#
# perturbs = [j2]

gravity_provider = GravityFieldFactory.getUnnormalizedProvider(2,0)
j2 = DSSTZonal(gravity_provider)
perturbs = [j2]

#chief initialization
a_c = float(7000e3)
e_c = float(0.0)
i_c = get_i(a_c, e_c)
arg_peri = float(0.0)
chief_init_raan = float(sun_ra - np.pi/2.0)
M_0 = float(0.0)

chief_orbit = KeplerianOrbit(
    a_c,                     # a (m)
    e_c,                     # e
    i_c,                     # i
    arg_peri,                # argument of perigee
    chief_init_raan,         # RAAN
    M_0,                     # mean anomaly
    PositionAngleType.MEAN,
    eci,
    initial_date,
    mu
)


#string of pearls (one deputy ahead and another behind the chief)
rho = 1.2e-4   # ~840 m amplitude (a * rho)

deputy_roes = init_close_helix_deputies(chief_orbit, 100)

deputy_orbits = [apply_ROE(chief_orbit, r) for r in deputy_roes]

#Time Arrays
T_orb = 2 * np.pi * float(np.sqrt((7000e3)**3 / mu))
T_span = 5 * T_orb
# T_span = 6 * 30 * 24 * 3600
N = 1500
times = np.linspace(0, T_span, N)

side_km = 10
max_dist = 1.5 #(km)

(a, e, i, raan, argp, M, alt, rel, dist) = run_propagation_dsst(times, initial_date,
                                                           perturbs, chief_orbit,
                                                           deputy_orbits)

days = times / 86400


# Plot 1: Orbital elements
colors = ["steelblue", "tomato"]
labels = ["Deputy 1", "Deputy 2"]

#OG plotting
# plotting.plot_orbital_elements(days, a, e, i, raan, argp, M, path)
# plotting.plot_hill_3d(rel, colors, labels, path)
# plotting.plot_radial_intrack(rel, colors, labels, path)
# plotting.plot_intrack_crosstrack(rel, colors, labels, path)
# plotting.plot_mean_separation_with_exits(times, rel, labels, colors, side_km, max_dist, "", path)

plotting.plot_hill_3d(rel, colors, labels)

plt.show()
