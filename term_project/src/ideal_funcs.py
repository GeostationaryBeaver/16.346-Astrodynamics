import orekit
orekit.initVM()
import os
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

if not os.path.exists("orekit-data.zip"):
    download_orekit_data_curdir()

setup_orekit_curdir()

import numpy as np
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

# Earth gravity field (includes J2 perturbations)
gravity_provider = GravityFieldFactory.getNormalizedProvider(2, 0)
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