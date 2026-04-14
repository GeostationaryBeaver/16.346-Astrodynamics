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


def create_propagator(initial_orbit, forces):
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
    for force in forces:
        propagator.addForceModel(force)
    return propagator


def to_keplerian(orbit):
    """Safely convert any Orekit orbit representation to KeplerianOrbit."""
    return KeplerianOrbit.cast_(OrbitType.KEPLERIAN.convertType(orbit))


def make_along_track_deputies(
    chief_orbit,
    separation_m: float = 1000.0,
    osc_fraction: float = 0.1,
) -> list:
    """
    Place two deputies along the along-track axis at ±separation_m from
    the chief at epoch, with small controlled oscillation amplitudes.

    Parameters
    ----------
    chief_orbit   : KeplerianOrbit
    separation_m  : desired initial T-separation in metres (default 1 km)
    osc_fraction  : oscillation amplitude as fraction of separation (default 0.1)
                    → deputies drift at most osc_fraction*separation_m per orbit
                    before secular J2 terms act

    Logic
    -----
    R₀ = 0, N₀ = 0 for both deputies.
    T₀ = +separation_m (Deputy 1), −separation_m (Deputy 2).

    From linearized RTN at epoch u₀:
      (1) R₀/a =  dex·cos(u₀) + dey·sin(u₀)       → sets eccentricity vector phase
      (2) T₀/a =  dl − 2·dex·sin(u₀) + 2·dey·cos(u₀) → sets dl
      (3) N₀/a =  dix·sin(u₀) − diy·cos(u₀)        → sets inclination vector phase
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
        T0 = sign * separation_m

        # Step 4: solve for dl from equation (2)
        dl = (T0 / a) + 2.0 * dex * np.sin(u0) - 2.0 * dey * np.cos(u0)

        roes.append(dict(
            da  = 0.0,
            dl  = float(dl),
            dex = float(dex),
            dey = float(dey),
            dix = float(dix),
            diy = float(diy),
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
def run_propagation(times, init_date, forces, chief_orbit, deputy_orbits):
    chief_prop = create_propagator(chief_orbit, forces)
    dep_props = [create_propagator(o, forces) for o in deputy_orbits]

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