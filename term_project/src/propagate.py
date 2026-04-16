import orekit
orekit.initVM()
import os
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

if not os.path.exists("orekit-data.zip"):
    download_orekit_data_curdir()

setup_orekit_curdir()

import numpy as np
from scipy.optimize import newton

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from org.orekit.utils import PVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.semianalytical.dsst import DSSTPropagator
from org.orekit.propagation.semianalytical.dsst.forces import DSSTZonal, DSSTTesseral
from org.orekit.propagation import PropagationType
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator



def create_dsst_propagator(initial_orbit, forces):
    # DSST usually works better with slightly larger steps than numerical
    min_step = 1.0
    max_step = 86400.0  # DSST can take massive steps (even a full day!)

    # State is 6D for DSST (no mass usually needed for J2 studies)
    abs_tol = [10.0] * 7
    rel_tol = [1e-9] * 7
    integrator = DormandPrince853Integrator(min_step, max_step, abs_tol, rel_tol)

    # Initialize DSST Propagator
    # We tell it to propagate the MEAN elements
    propagator = DSSTPropagator(integrator, PropagationType.MEAN)

    # 1. Convert standard gravity provider to DSST Zonal Force (J2, J3, etc.)
    # This is the 'averaged' gravity logic
    # zonal_force = DSSTZonal(gravity_provider)
    # propagator.addForceModel(zonal_force)
    for force in forces:
        propagator.addForceModel(force)

    #add the wobbles back in
    for force_model in propagator.getAllForceModels():
        force_model.registerAttitudeProvider(propagator.getAttitudeProvider())

    # 2. Set Initial State
    # CRITICAL: We explicitly tell Orekit the input orbit IS MEAN.
    # This prevents the 'initial snap' that causes drift.
    initial_state = SpacecraftState(initial_orbit)
    propagator.setInitialState(initial_state, PropagationType.MEAN)

    return propagator


def to_keplerian(orbit):
    """Safely convert any Orekit orbit representation to KeplerianOrbit."""
    return KeplerianOrbit.cast_(OrbitType.KEPLERIAN.convertType(orbit))

def get_i(a, e):
    """Initialize inclination appropriate for the chosen a and e parameters"""
    provider = GravityFieldFactory.getUnnormalizedProvider(2, 0)
    coeffs = provider.onDate(provider.getReferenceDate())
    J2 = -coeffs.getUnnormalizedCnm(2, 0)

    mu = Constants.EIGEN5C_EARTH_MU
    R_e = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

    p = a * (1-e**2)
    n = np.sqrt(mu/a**3)
    raan_prec_rate = 2*np.pi / (365.24219*86400)

    def f(i):
        return raan_prec_rate + 3/2 * n * J2 * (R_e/p)**2 * np.cos(i)

    return float(newton(f, np.radians(98.0)))


def init_string_of_pearls(
    chief_orbit,
    separation_m: float = 1000.0,
    osc_fraction: float = 0.1,
) -> list:
    """
    Place two deputies along the along-track axis at ±separation_m from
    the chief at epoch, with small controlled oscillation amplitudes. Is intended
    to be used when initializing deputies for design B as the constraints are
    looser.
    """
    a  = float(chief_orbit.getA())
    u0 = (float(chief_orbit.getPerigeeArgument())
          + float(chief_orbit.getMeanAnomaly()))

    # oscillation amplitudes — small fraction of desired separation
    de_mag = osc_fraction * separation_m / a
    di_mag = osc_fraction * separation_m / a

    # Step 3: R₀=0 → cos(u₀ − φ_e) = 0 → φ_e = u₀ + π/2
    # This gives dex·cos(u₀) + dey·sin(u₀) = 0 ✓
    phi_e = u0 + np.pi / 2.0
    dex   = de_mag * np.cos(phi_e)
    dey   = de_mag * np.sin(phi_e)

    # Step 5: N₀=0 → φ_i = u₀ exactly
    # dix·sin(u₀) − diy·cos(u₀) = 0 ✓
    dix = di_mag * np.cos(u0)
    diy = di_mag * np.sin(u0)

    roes = []
    for sign in (1.0, -1.0):
        # Enforce equidistant deputies to the chief
        curr_dex = sign * dex
        curr_dey = sign * dey
        curr_dix = sign * dix
        curr_diy = sign * diy

        T0 = sign * separation_m
        dl = (T0 / a) + 2.0 * curr_dex * np.sin(u0) - 2.0 * curr_dey * np.cos(u0)
        roes.append(dict(
            da=0.0,
            dl=float(dl),
            dex=float(curr_dex),
            dey=float(curr_dey),
            dix=float(curr_dix),
            diy=float(curr_diy),
        ))

    return roes


def init_close_helix_deputies(
        chief_orbit,
        helix_radius_m: float = 500.0,
        mean_dist_m: float = 0.0
) -> list:
    """
    Initializes deputies in a 'Helix' formation with passive safety
    via parallel e/i-vector separation.

    Parameters
    ----------
    chief_orbit  : KeplerianOrbit
    helix_radius_m : The 'radius' of the relative motion ellipse (m).
                     Determines cross-track and radial swing.
    mean_dist_m    : Mean along-track offset from chief (m).
                     Set to 0 to circle exactly around the chief.
    """
    a = float(chief_orbit.getA())
    # u0 is the Argument of Latitude (perigee + mean anomaly)
    u0 = float(chief_orbit.getPerigeeArgument()) + float(chief_orbit.getMeanAnomaly())

    # Magnitudes for ROEs
    # For a quasi-circular relative orbit:
    # Radial swing is a*de, Cross-track swing is a*di
    de_mag = helix_radius_m / a
    di_mag = (2.0 * helix_radius_m) / a  # Standard scaling for 'circular' look

    # Passive Safety Phase: Parallel Vectors
    # We want the phase of the e-vector (phi) and i-vector (theta)
    # to be aligned with the current position u0.
    phi = u0  # phase of eccentricity vector
    theta = u0  # phase of inclination vector

    # Decompose into ROE components
    dex = de_mag * np.cos(phi)
    dey = de_mag * np.sin(phi)

    dix = di_mag * np.cos(theta)
    diy = di_mag * np.sin(theta)

    roes = []
    # Create two deputies on opposite sides of the helix
    for sign in (1.0, -1.0):
        # The mean distance (dl) is modified by the eccentricity to
        # keep the 'instantaneous' T0 centered at mean_dist_m
        T_offset = sign * mean_dist_m
        dl = (T_offset / a) + 2.0 * (sign * dex) * np.sin(u0) - 2.0 * (sign * dey) * np.cos(u0)

        roes.append(dict(
            da= 0.0,
            dl=float(dl),
            dex=float(sign * dex),
            dey=float(sign * dey),
            dix=float(sign * dix),
            diy=float(sign * diy),
        ))

    return roes

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
        Constants.EIGEN5C_EARTH_MU #Earth's mu
    )



# ------------------------------------
# 6. PROPAGATION HELPER
# ------------------------------------
def get_eci_trajectories(times, init_date, chief_orbit, deputy_orbits):
    chief_prop = create_propagator(chief_orbit)
    dep_props = [create_propagator(o) for o in deputy_orbits]

    chief_pts = []
    dep_pts = [[] for _ in deputy_orbits]

    eci = FramesFactory.getEME2000()

    for t in times:
        date_t = init_date.shiftedBy(float(t))

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


def run_propagation_dsst(times, init_date, forces, chief_orbit, deputy_orbits):
    chief_prop = create_dsst_propagator(chief_orbit, forces)
    dep_props = [create_dsst_propagator(o, forces) for o in deputy_orbits]

    a_v, e_v, i_v, raan_v, argp_v, M_v, alt_v = [], [], [], [], [], [], []
    rel = [[] for _ in deputy_orbits]
    dist = [[] for _ in deputy_orbits]

    eci = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    for t in times:
        date_t = init_date.shiftedBy(float(t))

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
