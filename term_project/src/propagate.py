import orekit

# orekit.initVM()
# from orekit.pyhelpers import setup_orekit_curdir

# setup_orekit_curdir("term_project/src/orekit-data.zip")

import numpy as np
from scipy.optimize import newton
from java.util import ArrayList


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
from org.orekit.propagation.semianalytical.dsst.forces import (
    DSSTZonal, DSSTTesseral, DSSTThirdBody, DSSTNewtonianAttraction,
    DSSTSolarRadiationPressure, DSSTAtmosphericDrag
)
from org.orekit.propagation.conversion.osc2mean import FixedPointConverter
from org.orekit.propagation.conversion.osc2mean import DSSTTheory
from org.orekit.propagation import PropagationType
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
from org.orekit.forces.drag import IsotropicDrag
from org.orekit.models.earth.atmosphere import HarrisPriester
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

from org.orekit.models.earth.atmosphere import NRLMSISE00
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.forces.drag import IsotropicDrag

# =============================================================================
# SPACECRAFT PHYSICAL CONSTANTS (ISARA-class 3U CubeSat)
# =============================================================================
ISARA_MASS_KG = 5.0               # Spacecraft wet mass [kg]
ISARA_PANEL_AREA_M2 = 0.18        # Total deployable solar panel area [m²]
ISARA_CROSS_SECTION_M2 = 0.06     # Drag cross-section (3U long-axis forward) [m²]
ISARA_SRP_AREA_M2 = 0.18          # SRP effective area (panels facing sun) [m²]
ISARA_CR = 1.5                    # Radiation pressure coefficient [-]
ISARA_CD = 2.2                    # Drag coefficient [-]
SOLAR_CELL_EFFICIENCY = 0.30      # Triple-junction GaAs cell efficiency [-]
SOLAR_FLUX_W_M2 = 1361.0          # Solar constant at 1 AU [W/m²]


def create_earth_body():
    """
    Create an Orekit OneAxisEllipsoid representing Earth using WGS84 parameters.

    Returns
    -------
    OneAxisEllipsoid
        Earth body model with equatorial radius and flattening defined by WGS84,
        attached to the ITRF frame.
    """
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    return OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf
    )


def build_full_force_model(gravity_degree=12, gravity_order=12,
                           include_srp=True, include_drag=True,
                           include_third_body=True,
                           cross_section=ISARA_CROSS_SECTION_M2,
                           srp_area=ISARA_SRP_AREA_M2,
                           Cr=ISARA_CR, Cd=ISARA_CD,
                           mass_kg=ISARA_MASS_KG):
    """
    Build a comprehensive list of DSST-compatible force models for high-fidelity propagation.

    This function assembles a realistic perturbation environment for LEO formation
    flying, including:

    1. **Zonal harmonics** (J2, J3, ... Jn): Axially symmetric gravitational
       perturbations from Earth's oblateness. These produce secular drift in Ω,
       ω, and M. For SSO, J2 is the dominant perturbation.

    2. **Tesseral harmonics**: Non-axially-symmetric gravitational perturbations
       (longitude-dependent). These cause short-period and resonant effects,
       particularly near repeat-groundtrack orbits. Included up to the specified
       degree and order.

    3. **Solar Radiation Pressure (SRP)**: Photon momentum transfer from sunlight.
       Uses a cannonball (isotropic) model with a single reflectivity coefficient Cr.
       The DSST implementation includes Earth shadow transitions. At 630 km altitude,
       SRP acceleration is ~10⁻⁸ m/s², small but non-negligible over months.

    4. **Atmospheric Drag**: Aerodynamic force from residual atmosphere. Uses the
       Harris-Priester density model, which is a static empirical model providing
       density as a function of altitude, latitude, and solar activity via the
       diurnal bulge direction. At 630 km, drag is weak (~10⁻⁹ m/s²) but causes
       irreversible semi-major axis decay.

    5. **Third-body perturbations**: Gravitational attraction from the Sun and Moon.
       These cause long-period oscillations in eccentricity and inclination.
       Solar third-body effects also couple with the SSO constraint since the
       orbit plane is deliberately oriented relative to the Sun.

    Parameters
    ----------
    gravity_degree : int, optional
        Maximum degree of spherical harmonic expansion. Default: 12.
        Degree 12 captures all significant spatial variations for LEO.
    gravity_order : int, optional
        Maximum order of spherical harmonic expansion. Default: 12.
        Order = degree gives the full field at that resolution.
    include_srp : bool, optional
        Whether to include solar radiation pressure. Default: True.
    include_drag : bool, optional
        Whether to include atmospheric drag. Default: True.
    include_third_body : bool, optional
        Whether to include Sun and Moon third-body gravity. Default: True.
    cross_section : float, optional
        Spacecraft drag cross-sectional area [m²]. Default: ISARA 0.06 m².
    srp_area : float, optional
        Spacecraft area exposed to solar radiation [m²]. Default: ISARA 0.3 m².
    Cr : float, optional
        Radiation pressure coefficient [-]. Default: 1.5.
        (1.0 = perfect absorption, 2.0 = perfect specular reflection)
    Cd : float, optional
        Aerodynamic drag coefficient [-]. Default: 2.2.
        (standard for LEO CubeSats in free molecular flow)
    mass_kg : float, optional
        Spacecraft mass [kg]. Default: 4.0 (ISARA wet mass).

    Returns
    -------
    list
        List of DSST force model objects ready to pass to ``create_dsst_propagator``.

    Notes
    -----
    - The gravity field data comes from Orekit's bundled EGM model (loaded via
      ``GravityFieldFactory``). Both unnormalized (for zonal) and normalized
      (for tesseral) providers are created from the same underlying model.
    - The Harris-Priester atmosphere model requires a ``OneAxisEllipsoid`` for
      Earth and the Sun body for diurnal bulge orientation.
    - The DSST tesseral force requires Earth's body frame and rotation rate
      for proper resonance handling.
    """
    forces = []

    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    gravity_provider = GravityFieldFactory.getUnnormalizedProvider(gravity_degree, gravity_order)

    # --- 1. Zonal harmonics (axisymmetric gravity) ---
    forces.append(DSSTZonal(itrf, gravity_provider))

    # --- 2. Tesseral harmonics (longitude-dependent gravity) ---
    if gravity_order > 0:
        earth_rot = Constants.WGS84_EARTH_ANGULAR_VELOCITY
        forces.append(
            DSSTTesseral(itrf, earth_rot, gravity_provider)
        )

    # --- 3. Solar Radiation Pressure ---
    if include_srp:
        sun = CelestialBodyFactory.getSun()
        earth = create_earth_body()
        # Isotropic (cannonball) radiation model: force depends on Cr, area, mass
        radiation_sc = IsotropicRadiationSingleCoefficient(
            float(srp_area), float(Cr)
        )
        # DSST SRP uses the averaged force including shadow entry/exit
        # Parameters: equatorial radius, Cr*A/m via the spacecraft model,
        # sun body, earth ellipsoid
        srp_force = DSSTSolarRadiationPressure(
            sun,
            earth,
            radiation_sc,
            float(Constants.EIGEN5C_EARTH_MU)
        )
        forces.append(srp_force)

    # --- 4. Atmospheric Drag ---
    if include_drag:
        sun = CelestialBodyFactory.getSun()
        earth = create_earth_body()
        
        # This object looks inside your orekit-data.zip for files matching 'M_S_A_F_E.txt'
        # It provides the F10.7 and Ap indices for the atmospheric density calculation.
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
        )
        
        # Initialize the MSIS model. 
        # It uses 'msafe' to know the solar weather at date_t.
        atmosphere = NRLMSISE00(msafe, sun, earth)
        
        # Define the spacecraft drag properties
        drag_sc = IsotropicDrag(float(cross_section), float(Cd))
        
        # Wrap for DSST. This performs the Gaussian averaging required 
        # for semi-analytical propagation.
        drag_force = DSSTAtmosphericDrag(atmosphere, drag_sc, float(mass_kg))
        
        forces.append(drag_force)

    # --- 5. Third-body perturbations (Sun and Moon) ---
    if include_third_body:
        sun = CelestialBodyFactory.getSun()
        moon = CelestialBodyFactory.getMoon()
        # DSSTThirdBody computes the doubly-averaged disturbing function
        # from a point-mass third body
        forces.append(DSSTThirdBody(sun, float(Constants.EIGEN5C_EARTH_MU)))
        forces.append(DSSTThirdBody(moon, float(Constants.EIGEN5C_EARTH_MU)))

    return forces


def create_dsst_propagator(initial_orbit, forces):
    """
    Create and configure a DSST (Double-averaged Semi-analytical) propagator.

    The DSST propagator averages short-period oscillations out of the equations
    of motion, propagating only the mean orbital elements. This makes it highly
    efficient for long-duration propagations (days to months) while still
    capturing secular and long-period perturbation effects.

    The integrator used is Dormand-Prince 8(5,3), a variable-step explicit
    Runge-Kutta method well suited to the smooth mean-element ODEs.

    Parameters
    ----------
    initial_orbit : Orbit
        The initial orbit in mean-element space. Must be defined at a specific
        epoch in an inertial frame with an associated gravitational parameter.
    forces : list
        List of DSST-compatible force models (e.g., DSSTZonal, DSSTThirdBody,
        DSSTSolarRadiationPressure, DSSTAtmosphericDrag).

    Returns
    -------
    DSSTPropagator
        Fully configured DSST propagator ready for .propagate() calls.

    Notes
    -----
    - The initial state is set explicitly as MEAN elements to prevent the
      'initial snap' artifact that causes secular drift when osculating
      elements are mistakenly treated as mean elements.
    - Tolerance vectors are 7-dimensional (6 orbital elements + mass).
    """
    # DSST usually works better with slightly larger steps than numerical
    min_step = 1.0
    max_step = 86400.0  # DSST can take massive steps (even a full day!)

    # State is 6D for DSST (no mass usually needed for J2 studies)
    abs_tol = [1e-3] * 7
    rel_tol = [1e-9] * 7
    integrator = DormandPrince853Integrator(min_step, max_step, abs_tol, rel_tol)

    # Initialize DSST Propagator with mean elements in mind
    propagator = DSSTPropagator(integrator, PropagationType.MEAN)

    for force in forces:
        propagator.addForceModel(force)

    #add the wobbles back in
    for force_model in propagator.getAllForceModels():
        force_model.registerAttitudeProvider(propagator.getAttitudeProvider())

    #The propagator automatically accounts for Newtonian attractions, the converter doesn't
    #Mean convsersion should only worry about conservative forces (ie Zonal/Tesseral harmonics)
    converter_forces = ArrayList()
    converter_forces.add(DSSTNewtonianAttraction(Constants.WGS84_EARTH_MU))
    for elt in forces:
        if isinstance(elt, DSSTAtmosphericDrag) or isinstance(elt, DSSTSolarRadiationPressure):
            continue
        converter_forces.add(elt)
    osculating_state = SpacecraftState(initial_orbit)
    #Numerically convert osculating into mean state (iterative methods must be used)
    mean_state = propagator.computeMeanState(osculating_state,
                                            None,               #Attitude provider --> use default
                                            converter_forces,
                                            1e-11,              #Precision of conversion
                                            100                 #Max iterations
                                            )

    # Initialize with mean elements
    propagator.setInitialState(mean_state, PropagationType.MEAN)

    return propagator


def to_keplerian(orbit):
    """
    Convert any Orekit orbit representation to a KeplerianOrbit.

    Orekit internally stores orbits in various representations (Cartesian,
    equinoctial, circular, Keplerian). This utility ensures we always get
    classical Keplerian elements for analysis.

    Parameters
    ----------
    orbit : Orbit
        Any Orekit Orbit subclass instance.

    Returns
    -------
    KeplerianOrbit
        The same orbit expressed in classical Keplerian elements
        (a, e, i, ω, Ω, M or ν).
    """
    return KeplerianOrbit.cast_(OrbitType.KEPLERIAN.convertType(orbit))


def get_i(a, e):
    """
    Compute the sun-synchronous inclination for a given semi-major axis and eccentricity.

    For a sun-synchronous orbit, the RAAN precession rate must equal Earth's
    mean motion around the Sun (~0.9856°/day or 2π/year). This function uses
    Newton's method to solve the implicit relationship:

        0 = Ω̇_SSO + (3/2) * n * J2 * (R_e/p)² * cos(i)

    where Ω̇_SSO = 2π / (365.24219 * 86400) rad/s.

    Parameters
    ----------
    a : float
        Semi-major axis [m].
    e : float
        Eccentricity [-].

    Returns
    -------
    float
        Sun-synchronous inclination [rad]. Typically ~97°-99° for LEO.

    Notes
    -----
    - Uses the unnormalized J2 coefficient from Orekit's gravity field.
    - Initial guess of 98° is appropriate for LEO SSO altitudes.
    - Convergence is typically achieved in 3-5 iterations.
    """
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


def is_in_sunlight(r_sc_eci, r_sun_eci, R_earth=Constants.WGS84_EARTH_EQUATORIAL_RADIUS):
    """
    Determine whether a spacecraft is illuminated by the Sun using a
    cylindrical shadow model.

    The cylindrical model assumes Earth casts a shadow cylinder of radius
    R_earth in the anti-sun direction. A spacecraft is eclipsed if:
      1. It is on the anti-sun side of Earth (negative projection onto sun vector)
      2. Its perpendicular distance from the Earth-Sun line is less than R_earth

    This is a first-order approximation that neglects:
      - Penumbra (partial shadow from finite solar disk)
      - Earth's oblateness
      - Atmospheric refraction

    Parameters
    ----------
    r_sc_eci : np.ndarray, shape (3,)
        Spacecraft position in ECI frame [m].
    r_sun_eci : np.ndarray, shape (3,)
        Sun position in ECI frame [m].
    R_earth : float, optional
        Earth's equatorial radius [m]. Default is WGS84 value.

    Returns
    -------
    bool
        True if spacecraft is in sunlight, False if eclipsed.
    """
    sun_dir = r_sun_eci / np.linalg.norm(r_sun_eci)

    # Projection of spacecraft position onto sun direction
    proj = np.dot(r_sc_eci, sun_dir)

    if proj > 0:
        # Spacecraft is on the sun-side of Earth -> always illuminated
        return True

    # Spacecraft is behind Earth relative to sun; check shadow cylinder
    perp_vec = r_sc_eci - proj * sun_dir
    perp_dist = np.linalg.norm(perp_vec)

    return perp_dist > R_earth


def get_sun_position_eci(sun, date_t, eci):
    """
    Extract the Sun's position vector in ECI coordinates at a given epoch.

    Parameters
    ----------
    sun : CelestialBody
        Orekit Sun body from ``CelestialBodyFactory.getSun()``.
    date_t : AbsoluteDate
        Epoch at which to evaluate the Sun's position.
    eci : Frame
        ECI reference frame (typically EME2000).

    Returns
    -------
    np.ndarray, shape (3,)
        Sun position [m] in the ECI frame.
    """
    sun_pos = sun.getPVCoordinates(date_t, eci).getPosition()
    return np.array([sun_pos.getX(), sun_pos.getY(), sun_pos.getZ()])


def compute_panel_normal_nadir_pointing(r_sc_eci, v_sc_eci):
    """
    Compute the solar-panel outward normal for a nadir-pointing spacecraft.

    For a nadir-pointing CubeSat with body-fixed deployable panels, the panel
    normal is perpendicular to both the velocity vector and the nadir direction.
    Specifically, for panels deployed along the orbit-normal (±h-hat) axis —
    which is standard for dawn-dusk SSO missions to maximise solar exposure —
    the panel normal points along the orbit-normal direction.

    However, many CubeSat configurations (including ISARA) deploy panels that
    lie in the orbit plane and rotate with the body.  For a nadir-pointing
    body, the panel normal is approximated as the **anti-nadir** direction
    (pointing away from Earth, i.e., +r̂).  This is the "worst case" for a
    single-axis deployment and gives a realistic cos(θ) variation.

    A more accurate model would track panel gimbal angles or use the body
    y-axis for side-deployed panels.  Here we use the orbit-normal (+ĥ)
    convention, which is the most favourable fixed-body orientation in a
    dawn-dusk SSO (panels face the Sun most of the time).

    Parameters
    ----------
    r_sc_eci : np.ndarray, shape (3,)
        Spacecraft position in ECI [m].
    v_sc_eci : np.ndarray, shape (3,)
        Spacecraft velocity in ECI [m/s].

    Returns
    -------
    np.ndarray, shape (3,)
        Unit vector of the panel outward normal in ECI.

    Notes
    -----
    In a dawn-dusk SSO the orbit normal is roughly perpendicular to the
    Sun direction, so panels deployed along ±ĥ receive near-optimal
    illumination with cos(θ) ≈ 0.7–1.0 through most of the orbit.
    """
    h_vec = np.cross(r_sc_eci, v_sc_eci)
    h_hat = h_vec / np.linalg.norm(h_vec)
    return h_hat


def compute_solar_flux_at_distance(r_sc_eci, r_sun_eci,
                                   S0=SOLAR_FLUX_W_M2):
    """
    Compute solar flux at the spacecraft accounting for Earth-Sun distance variation.

    The solar "constant" S0 = 1361 W/m² is defined at 1 AU.  Earth's orbital
    eccentricity causes the actual flux to vary from ~1321 W/m² (aphelion,
    July) to ~1413 W/m² (perihelion, January) — a ±3.3% swing.

    The corrected flux follows the inverse-square law:

        S(r) = S0 · (1 AU / |r_sun - r_sc|)²

    In practice for LEO, |r_sun - r_sc| ≈ |r_sun| since the spacecraft
    altitude (~630 km) is negligible compared to 1 AU (~1.496 × 10⁸ km).

    Parameters
    ----------
    r_sc_eci : np.ndarray, shape (3,)
        Spacecraft position in ECI [m].
    r_sun_eci : np.ndarray, shape (3,)
        Sun position in ECI [m].
    S0 : float, optional
        Solar constant at 1 AU [W/m²]. Default: 1361.0.

    Returns
    -------
    float
        Solar flux at spacecraft distance [W/m²].
    """
    AU_m = 1.496e11  # 1 AU in metres
    # Vector from spacecraft to Sun
    r_to_sun = r_sun_eci - r_sc_eci
    dist = np.linalg.norm(r_to_sun)
    return S0 * (AU_m / dist) ** 2


def compute_solar_power(r_sc_eci, r_sun_eci,
                        v_sc_eci=None,
                        panel_area=ISARA_PANEL_AREA_M2,
                        efficiency=SOLAR_CELL_EFFICIENCY,
                        solar_flux=SOLAR_FLUX_W_M2):
    """
    Compute instantaneous electrical power from solar panels on a nadir-pointing
    spacecraft, including eclipse, sun-incidence angle, and distance corrections.

    The power model is:

        P = η · A · S(r) · max(cos θ, 0)     if in sunlight
        P = 0                                  if eclipsed

    where:
        η     = solar cell conversion efficiency
        A     = total panel area
        S(r)  = solar flux corrected for Earth–Sun distance (1/r² from 1 AU)
        θ     = angle between panel outward normal and spacecraft-to-Sun vector

    The cos(θ) term is the key physical addition.  For a nadir-pointing
    CubeSat in a dawn-dusk SSO with panels deployed along the orbit normal,
    θ varies as the spacecraft orbits:

      - Near the terminator crossing (ascending/descending node): θ ≈ 0°,
        cos θ ≈ 1 → maximum power.
      - Over the poles: θ increases, cos θ drops → reduced power.
      - The variation is quasi-sinusoidal with orbital period (~97 min).

    Crucially, because the three formation spacecraft are at slightly different
    orbital positions (different true anomaly, RAAN, inclination), their θ
    values differ at each time step, producing distinguishable power traces.

    Parameters
    ----------
    r_sc_eci : np.ndarray, shape (3,)
        Spacecraft position in ECI [m].
    r_sun_eci : np.ndarray, shape (3,)
        Sun position in ECI [m].
    v_sc_eci : np.ndarray or None, optional
        Spacecraft velocity in ECI [m/s].  Required for panel-normal
        computation.  If None, falls back to the old binary (on/off)
        model with ideal sun-pointing (backward compatible).
    panel_area : float, optional
        Total solar panel area [m²]. Default: ISARA 0.3 m².
    efficiency : float, optional
        Solar cell conversion efficiency [-]. Default: 0.283.
    solar_flux : float, optional
        Solar constant at 1 AU [W/m²]. Default: 1361.0.
        Only used as fallback when v_sc_eci is None.

    Returns
    -------
    float
        Instantaneous electrical power [W]. Zero during eclipse.
    """
    # ── Eclipse check (cylindrical shadow) ──
    if not is_in_sunlight(r_sc_eci, r_sun_eci):
        return 0.0

    # ── If velocity not provided, fall back to simple binary model ──
    if v_sc_eci is None:
        return efficiency * panel_area * solar_flux

    # ── Distance-corrected solar flux ──
    S = compute_solar_flux_at_distance(r_sc_eci, r_sun_eci, S0=solar_flux)

    # ── Panel normal (orbit-normal for nadir-pointing SC with h-hat panels) ──
    n_hat = compute_panel_normal_nadir_pointing(r_sc_eci, v_sc_eci)

    # ── Sun direction from spacecraft ──
    to_sun = r_sun_eci - r_sc_eci
    sun_hat = to_sun / np.linalg.norm(to_sun)

    # ── Cosine of incidence angle ──
    cos_theta = np.dot(n_hat, sun_hat)

    # Panels can be illuminated from either side in some configurations,
    # but we assume single-sided panels: only the outward face generates power.
    # If cos_theta < 0 the Sun is behind the panel → use |cos_theta| if
    # panels are double-sided, or take max(cos_theta, 0) for single-sided.
    # For orbit-normal panels the Sun can be on either side of the orbit
    # plane, so we use abs() to capture both halves of the orbit.
    cos_incidence = abs(cos_theta)

    return efficiency * panel_area * S * cos_incidence


def init_string_of_pearls(
    chief_orbit,
    separation_m: float = 1000.0,
    osc_fraction: float = 0.1,
) -> list:
    """
    Initialize two deputies in a 'string of pearls' (along-track) formation.

    Deputies are placed symmetrically ahead of and behind the chief in the
    along-track direction. Small eccentricity and inclination offsets are
    added to create bounded oscillatory motion that prevents collision while
    maintaining the along-track separation.

    The ROE initialization enforces:
      - R₀ = 0: Zero initial radial offset (via eccentricity vector phasing)
      - N₀ = 0: Zero initial cross-track offset (via inclination vector phasing)
      - T₀ = ±separation_m: Symmetric along-track placement

    Parameters
    ----------
    chief_orbit : KeplerianOrbit
        Reference orbit of the chief spacecraft.
    separation_m : float, optional
        Desired along-track separation from chief [m]. Default: 1000 m.
        Each deputy is placed at +separation_m and -separation_m.
    osc_fraction : float, optional
        Fraction of separation used as oscillation amplitude [-]. Default: 0.1.
        Controls the 'breathing' of the formation to avoid rectilinear drift.

    Returns
    -------
    list of dict
        Two ROE dictionaries (one per deputy), each containing:
        {'da', 'dl', 'dex', 'dey', 'dix', 'diy'}.

    Notes
    -----
    This configuration is inherently J2-robust because:
    - da = 0 prevents differential mean motion (no along-track drift)
    - The symmetric placement naturally satisfies δ(Ṁ + ω̇) ≈ 0
    - Small de/di ensure the formation stays bounded

    Best suited for Design B where 2-5 km separations are acceptable.
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

'''
def init_close_helix_deputies(
        chief_orbit,
        helix_radius_m: float = 500.0,
        mean_dist_m: float = 0.0
) -> list:
    """
    Initialize two deputies in a 'helix' formation with passive safety.

    The helix formation places deputies on opposite sides of a relative
    motion ellipse that wraps around the chief. Passive safety is achieved
    by enforcing parallel eccentricity/inclination vector separation
    (Spurmann & D'Amico, 2011), ensuring the deputies never cross through
    the chief's position even without active control.

    The geometry produces:
      - Radial oscillation amplitude: a * δe (= helix_radius_m)
      - Cross-track oscillation amplitude: a * δi (= 2 * helix_radius_m)
      - The 2:1 scaling creates a quasi-circular cross-section in the
        radial-crosstrack plane

    Parameters
    ----------
    chief_orbit : KeplerianOrbit
        Reference orbit of the chief spacecraft.
    helix_radius_m : float, optional
        Radial swing amplitude of the relative motion [m]
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
'''

def init_close_helix_deputies(
        chief_orbit,
        helix_radius_m: float = 500.0,
        mean_dist_m: float = 0.0
) -> list:
    """
    Initialize two deputies in a helix formation WITH J2-invariant conditions.

    Unlike the naive helix initializer that sets δa = 0, this function
    computes the required δa from the J2-invariant conditions:

        δΩ̇ = 0   →   constrains δa as a function of δi
        δ(Ṁ + ω̇) = 0   →   provides second constraint relating δa, δe, δi

    The resulting formation has a small but non-zero semi-major axis offset
    that exactly cancels the differential J2 precession caused by δe and δi.
    This prevents the secular along-track drift that otherwise dominates
    relative motion on timescales of weeks to months.

    For a near-circular chief (e_c ≈ 0), the J2-invariant conditions reduce to:

        Condition 1 (δΩ̇ = 0):
            δa/a = -(2/7) × tan(i_c) × δi_x

        Condition 2 (δ(Ṁ + ω̇) = 0):
            δa/a = [4η(5cos²i - 1)/(2η+1)(7(1-3sin²i/2)η + 4(5cos²i-1))] 
                   × sin(2i)/(1 - 5cos²i) × ... (complex)

    In practice for SSO inclinations (~98°), both conditions are approximately
    satisfied by the same δa when δe is small relative to δi, because the
    eccentricity coupling terms are suppressed by the near-circular geometry.

    The implementation uses the full Schaub & Junkins formulation (§13.6)
    for the combined secular rate matching.

    Parameters
    ----------
    chief_orbit : KeplerianOrbit
        Reference orbit of the chief spacecraft (near-circular SSO).
    helix_radius_m : float, optional
        Radial swing amplitude [m]. Default: 500 m.
        Cross-track amplitude will be 2× this value.
    mean_dist_m : float, optional
        Mean along-track offset [m]. Default: 0.

    Returns
    -------
    list of dict
        Two ROE dictionaries with J2-invariant δa values.
        Keys: {'da', 'dl', 'dex', 'dey', 'dix', 'diy'}
    """
    a = float(chief_orbit.getA())
    e_c = float(chief_orbit.getE())
    i_c = float(chief_orbit.getI())
    u0 = float(chief_orbit.getPerigeeArgument()) + float(chief_orbit.getMeanAnomaly())

    provider = GravityFieldFactory.getUnnormalizedProvider(2, 0)
    coeffs = provider.onDate(provider.getReferenceDate())
    J2 = -coeffs.getUnnormalizedCnm(2, 0)
    R_e = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
    mu = Constants.EIGEN5C_EARTH_MU
    n = np.sqrt(mu / a**3)

    de_mag = helix_radius_m / a
    di_mag = (2.0 * helix_radius_m) / a


    # phi = u0
    # theta = u0

    # dex = de_mag * np.cos(phi)
    # dey = de_mag * np.sin(phi)
    # dix = di_mag * np.cos(theta)
    # diy = di_mag * np.sin(theta)

    # eta = np.sqrt(1.0 - e_c**2)
    # p = a * (1.0 - e_c**2)
    # sin_i = np.sin(i_c)
    # cos_i = np.cos(i_c)
    # tan_i = sin_i / cos_i  # Note: negative for i > 90° (retrograde SSO)

    # # Gamma factor: (Re/p)²
    # gamma = (R_e / p)**2


    # da_over_a_from_raan = -(2.0 / 7.0) * tan_i * di_mag


    # da_over_a = da_over_a_from_raan

    phi   =  np.pi / 2   # relative eccentricity vector phase
    theta = -np.pi / 2   # relative inclination vector phase (anti-parallel: phi = -theta)

    dex = de_mag * np.cos(phi)    #  = 0
    dey = de_mag * np.sin(phi)    #  = +de_mag
    dix = di_mag * np.cos(theta)  #  = 0       → Condition A: da = 0
    diy = di_mag * np.sin(theta)  #  = -di_mag

    roes = []
    for sign in (1.0, -1.0):
        curr_dex = sign * dex   # sign flips dey: ±de_mag
        curr_dey = sign * dey
        curr_dix = sign * dix   # always 0 — no inclination component
        curr_diy = sign * diy   # sign flips diy: ∓di_mag

        # dix = 0 → both J2 conditions satisfied at da = 0 (Eq. 10, 16)
        curr_da_over_a = 0.0

        # Along-track offset: curr_dex = 0 simplifies this to:
        # dl = T_offset/a - 2*curr_dey*cos(u0)
        T_offset = sign * mean_dist_m
        dl = (T_offset / a) + 2.0 * curr_dex * np.sin(u0) - 2.0 * curr_dey * np.cos(u0)

        roes.append(dict(
            da=float(curr_da_over_a),
            dl=float(dl),
            dex=float(curr_dex),
            dey=float(curr_dey),
            dix=float(curr_dix),
            diy=float(curr_diy),
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


def initialize_formation_propagators(chief_orbit, deputy_orbits, forces, init_date):
    """
    Initialize chief + deputy propagators with consistent mean elements.
    Deputies are initialized from the chief's mean state to preserve ROEs.
    """
    eci = FramesFactory.getEME2000()
    
    # Chief: standard osc→mean conversion
    chief_prop = create_dsst_propagator(chief_orbit, forces)
    chief_mean_kep = to_keplerian(chief_prop.propagate(init_date).getOrbit())
    
    # Deputies: apply ROEs to chief's mean state (no independent conversion)
    a_c = float(chief_orbit.getA())
    dep_props = []
    for dep_orbit in deputy_orbits:
        roe = extract_roe_dict(chief_orbit, dep_orbit)  # extract as dict
        dep_mean = apply_ROE(chief_mean_kep, roe)
        dep_props.append(_create_dsst_from_mean_orbit(dep_mean, forces))
    
    return chief_prop, dep_props


def extract_roe_dict(chief_orbit, deputy_orbit):
    """Extract ROE as a dictionary compatible with apply_ROE()."""
    a_c = float(chief_orbit.getA())
    e_c = float(chief_orbit.getE())
    i_c = float(chief_orbit.getI())
    raan_c = float(chief_orbit.getRightAscensionOfAscendingNode())
    argp_c = float(chief_orbit.getPerigeeArgument())
    M_c = float(chief_orbit.getMeanAnomaly())
    
    a_d = float(deputy_orbit.getA())
    e_d = float(deputy_orbit.getE())
    i_d = float(deputy_orbit.getI())
    raan_d = float(deputy_orbit.getRightAscensionOfAscendingNode())
    argp_d = float(deputy_orbit.getPerigeeArgument())
    
    return dict(
        da  = (a_d - a_c) / a_c,
        dl  = float(((argp_d + float(deputy_orbit.getMeanAnomaly())) 
                     - (argp_c + M_c) 
                     + (raan_d - raan_c) * np.cos(i_c))),
        dex = e_d * np.cos(argp_d) - e_c * np.cos(argp_c),
        dey = e_d * np.sin(argp_d) - e_c * np.sin(argp_c),
        dix = i_d - i_c,
        diy = (raan_d - raan_c) * np.sin(i_c),
    )


def run_propagation_dsst(times, init_date, forces, chief_orbit, deputy_orbits,
                         sun=None):
    """
    Propagate a chief + N deputies using DSST and collect time histories.

    At each output time step this function:
      1. Propagates the chief and every deputy to the current epoch.
      2. Extracts the chief's mean Keplerian elements.
      3. Builds a radial / in-track / cross-track (LVLH) frame centred on
         the chief and projects each deputy's relative position into it.
      4. (Optional) If a ``sun`` CelestialBody is provided, evaluates the
         cylindrical-shadow eclipse model, computes the sun-incidence angle
         on orbit-normal-deployed panels, applies the 1/r² flux correction,
         and records instantaneous solar power for every spacecraft.

    The LVLH basis vectors are constructed as:
      - r̂  = r / |r|                        (radial, away from Earth)
      - ĥ  = (r × v) / |r × v|             (orbit-normal)
      - t̂  = ĥ × r̂                         (along-track / in-track, ~velocity)

    Parameters
    ----------
    times : array-like of float
        Output time stamps measured from ``init_date`` [s].
    init_date : AbsoluteDate
        Orekit epoch corresponding to times[0] = 0.
    forces : list
        DSST-compatible force models passed to ``create_dsst_propagator``.
    chief_orbit : KeplerianOrbit
        Chief's initial mean-element orbit.
    deputy_orbits : list of KeplerianOrbit
        Each deputy's initial mean-element orbit.
    sun : CelestialBody or None, optional
        Orekit Sun body (from ``CelestialBodyFactory.getSun()``).  When
        provided, the function tracks eclipse state and solar-panel power
        for every spacecraft at every time step.  Default: None (no power
        tracking).

    Returns
    -------
    tuple of length 10
        ``(a, e, i, raan, argp, M, alt, rel, dist, power)``

        - **a** : list of float — chief semi-major axis [m]
        - **e** : list of float — chief eccentricity [-]
        - **i, raan, argp, M** : list of float — chief angles [deg]
        - **alt** : list of float — chief geodetic altitude [m]
        - **rel** : list of np.ndarray, shape (N, 3) — deputy LVLH
          positions [m] (radial, in-track, cross-track)
        - **dist** : list of np.ndarray, shape (N,) — deputy-chief
          Euclidean distances [m]
        - **power** : dict or None — If ``sun`` was provided::

              {
                  "chief":    np.ndarray shape (N,),  # [W]
                  "deputies": [np.ndarray, ...],      # [W] per deputy
              }

            If ``sun`` was ``None``, returns ``None``.
    """
    # ------------------------------------------------------------------ setup
    chief_prop, dep_props = initialize_formation_propagators(chief_orbit, deputy_orbits, forces, init_date)


    eci  = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    a_v, e_v, i_v, raan_v, argp_v, M_v, alt_v = [], [], [], [], [], [], []
    rel  = [[] for _ in deputy_orbits]
    dist = [[] for _ in deputy_orbits]

    track_power = sun is not None
    if track_power:
        chief_power = []
        dep_power   = [[] for _ in deputy_orbits]

    # --------------------------------------------------------------- main loop
    for t in times:
        date_t = init_date.shiftedBy(float(t))

        # ── Chief state ──
        chief_state = chief_prop.propagate(date_t)
        chief_kep   = to_keplerian(chief_state.getOrbit())

        pv_c = chief_state.getPVCoordinates(eci)
        p_c  = pv_c.getPosition()
        v_c  = pv_c.getVelocity()

        a_v.append(chief_kep.getA())
        e_v.append(chief_kep.getE())
        i_v.append(np.degrees(chief_kep.getI()))
        raan_v.append(np.degrees(
            chief_kep.getRightAscensionOfAscendingNode()))
        argp_v.append(np.degrees(chief_kep.getPerigeeArgument()))
        M_v.append(np.degrees(chief_kep.getMeanAnomaly()) % 360)

        # Geodetic altitude
        tr     = eci.getTransformTo(itrf, date_t)
        p_itrf = tr.transformPVCoordinates(
            PVCoordinates(p_c, Vector3D.ZERO)).getPosition()
        alt_v.append(
            p_itrf.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS)

        # ── LVLH basis ──
        r_vec = np.array([p_c.getX(), p_c.getY(), p_c.getZ()])
        v_vec = np.array([v_c.getX(), v_c.getY(), v_c.getZ()])
        r_hat = r_vec / np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_hat = h_vec / np.linalg.norm(h_vec)
        t_hat = np.cross(h_hat, r_hat)

        # ── Sun vector (once per time step) ──
        if track_power:
            r_sun = get_sun_position_eci(sun, date_t, eci)
            # Chief power — pass velocity for incidence angle computation
            chief_power.append(
                compute_solar_power(r_vec, r_sun, v_sc_eci=v_vec)
            )

        # ── Deputies ──
        for k in range(len(deputy_orbits)):
            dep_state = dep_props[k].propagate(date_t)
            pv_d      = dep_state.getPVCoordinates(eci)
            p_d       = pv_d.getPosition()
            v_d       = pv_d.getVelocity()

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

            if track_power:
                r_dep = r_vec + dr
                v_dep = np.array([v_d.getX(), v_d.getY(), v_d.getZ()])
                dep_power[k].append(
                    compute_solar_power(r_dep, r_sun, v_sc_eci=v_dep)
                )

    rel  = [np.array(rel[k])  for k in range(len(deputy_orbits))]
    dist = [np.array(dist[k]) for k in range(len(deputy_orbits))]

    if track_power:
        power = {
            "chief":    np.array(chief_power),
            "deputies": [np.array(dep_power[k])
                         for k in range(len(deputy_orbits))]
        }
    else:
        power = None

    return (a_v, e_v, i_v, raan_v, argp_v, M_v, alt_v, rel, dist, power)



def compute_stm(dt, a, e, i, J2, R_e, mu):
    """
    Compute the 7x7 state transition matrix Φ(t, t0) from Eq. (6)
    of Gaias & Ardaens 2018.

    The ROE state vector ordering is:
        x = [aδa, aδȧ, aδλ, aδix, aδiy, aδex, aδey]
              0     1    2    3     4     5     6

    Parameters
    ----------
    dt  : float     — propagation interval [s]
    a   : float     — chief semi-major axis [m]
    e   : float     — chief eccentricity
    i   : float     — chief inclination [rad]
    J2  : float     — J2 coefficient (~1.08263e-3)
    R_e : float     — Earth equatorial radius [m]
    mu  : float     — Earth gravitational parameter [m^3/s^2]

    Returns
    -------
    Phi : np.ndarray, shape (7, 7)
    """
    eta = np.sqrt(1.0 - e**2)
    n   = np.sqrt(mu / a**3)
    gam = (J2 / 2.0) * (R_e / (a * eta**2))**2   # γ

    sin_i  = np.sin(i)
    cos_i  = np.cos(i)
    cos2_i = cos_i**2

    # --- Scalar rate constants (Eq. 6) ---
    nu      = -(3.0 / 2.0) * n
    mu_A    = -(21.0 / 4.0) * n * gam * (3.0 * cos2_i - 1.0) * (eta + 1.0)
    mu_I    = -(3.0 / 2.0) * n * gam * np.sin(2.0 * i) * (3.0 * eta + 4.0)
    phi_dot =  (3.0 / 2.0) * n * gam * (5.0 * cos2_i - 1.0)   # ė-vector rotation
    lam_A   =  (21.0 / 4.0) * n * gam * np.sin(2.0 * i)
    lam_I   =  3.0 * n * gam * sin_i**2

    nu_plus = nu + mu_A   # combined Keplerian + drag secular rate on δλ

    # --- Eccentricity vector rotation angle ---
    phi_rot = phi_dot * dt
    c_phi   = np.cos(phi_rot)
    s_phi   = np.sin(phi_rot)

    # --- Assemble Φ (7×7) ---
    Phi = np.zeros((7, 7))

    # Row 0: aδa — constant (only maneuvers or drag change it)
    Phi[0, 0] = 1.0

    # Row 1: aδȧ — drag rate, constant between replans
    Phi[1, 0] = dt          # δa grows linearly from current δȧ
    Phi[1, 1] = 1.0

    # Row 2: aδλ — along-track, most coupled element
    Phi[2, 0] = 0.5 * nu_plus * dt**2
    Phi[2, 1] = nu_plus * dt
    Phi[2, 2] = 1.0
    Phi[2, 3] = mu_I * dt   # J2 coupling from inclination difference

    # Row 3: aδix — inclination x-component, no secular coupling
    Phi[3, 3] = 1.0

    # Row 4: aδiy — inclination y-component, secular drift from J2 on δix
    Phi[4, 0] = 0.5 * lam_A * dt**2
    Phi[4, 1] = lam_A * dt
    Phi[4, 3] = lam_I * dt
    Phi[4, 4] = 1.0

    # Rows 5-6: aδe vector — rigid rotation at rate φ̇ (NOT linearized)
    Phi[5, 5] =  c_phi
    Phi[5, 6] = -s_phi
    Phi[6, 5] =  s_phi
    Phi[6, 6] =  c_phi

    return Phi


def extract_roe(chief_orbit, deputy_orbit):
    """
    Compute the quasi-nonsingular ROE vector from two KeplerianOrbits.
    Returns the dimensional state vector x = a_c * [δa, δȧ, δλ, δix, δiy, δex, δey]
    where δȧ = 0 at initialization (estimated by filter in flight).

    Returns
    -------
    x : np.ndarray, shape (7,)   [m]
    """
    a_c    = float(chief_orbit.getA())
    e_c    = float(chief_orbit.getE())
    i_c    = float(chief_orbit.getI())
    raan_c = float(chief_orbit.getRightAscensionOfAscendingNode())
    argp_c = float(chief_orbit.getPerigeeArgument())
    M_c    = float(chief_orbit.getMeanAnomaly())
    u_c    = argp_c + M_c   # mean argument of latitude

    a_d    = float(deputy_orbit.getA())
    e_d    = float(deputy_orbit.getE())
    i_d    = float(deputy_orbit.getI())
    raan_d = float(deputy_orbit.getRightAscensionOfAscendingNode())
    argp_d = float(deputy_orbit.getPerigeeArgument())
    M_d    = float(deputy_orbit.getMeanAnomaly())

    # Eccentricity vector components
    ex_c = e_c * np.cos(argp_c);  ey_c = e_c * np.sin(argp_c)
    ex_d = e_d * np.cos(argp_d);  ey_d = e_d * np.sin(argp_d)

    da   = (a_d - a_c) / a_c
    dex  = ex_d - ex_c
    dey  = ey_d - ey_c
    dix  = i_d - i_c
    diy  = (raan_d - raan_c) * np.sin(i_c)

    # Mean longitude difference — wrap to [-π, π]
    u_d  = argp_d + M_d
    dl   = (u_d - u_c) + (raan_d - raan_c) * np.cos(i_c)
    dl   = (dl + np.pi) % (2 * np.pi) - np.pi

    # Dimensional state vector, δȧ = 0 at epoch
    x = a_c * np.array([da, 0.0, dl, dix, diy, dex, dey])
    return x


def passive_safety_margin(x_roe):
    """
    Compute the minimum cross-track (RN-plane) separation δr_RN^min
    from Eq. (9) of Gaias & Ardaens 2018.

    Uses numerical minimization over one orbit (3600 samples in
    true anomaly) for robustness.  This is the collision-safety metric:
    the formation is considered safe if δr_RN^min exceeds a threshold.

    The RN-plane distance as a function of argument of latitude u is:

        ρ²(u) = (aδi · sin(u - θ))² + (aδa - aδe · cos(u - θ - ϕ'))²

    where θ = atan2(aδiy, aδix) is the phase of the inclination vector
    and ϕ' = atan2(aδey, aδex) - θ is the relative phase between the
    eccentricity and inclination vectors.

    Parameters
    ----------
    x_roe : np.ndarray (7,)
        Dimensional ROE state [m]:
        [aδa, aδȧ, aδλ, aδix, aδiy, aδex, aδey]

    Returns
    -------
    drmin : float
        Minimum RN-plane separation [m].  Always ≥ 0.
    """
    ada  = x_roe[0]
    adix = x_roe[3]
    adiy = x_roe[4]
    adex = x_roe[5]
    adey = x_roe[6]

    adi = np.sqrt(adix**2 + adiy**2)
    ade = np.sqrt(adex**2 + adey**2)

    phi_e = np.arctan2(adey, adex)  # eccentricity vector phase
    phi_i = np.arctan2(adiy, adix)  # inclination vector phase

    # Numerical minimization over one full orbit
    u_test = np.linspace(0, 2 * np.pi, 3600)
    rn_sq = (adi * np.sin(u_test - phi_i))**2 + \
            (ada - ade * np.cos(u_test - phi_e))**2

    return float(np.sqrt(np.min(rn_sq)))


def compute_b_matrix(u_m, n, a):
    """
    Control input matrix B(t_m) from Eq. (B2): maps a 3-vector
    [δvR, δvT, δvN] in RTN frame to a ROE jump Δx = a·B·δv.

    The ordering matches x = [aδa, aδȧ, aδλ, aδix, aδiy, aδex, aδey].
    Note: the δȧ row is zero — impulsive maneuvers don't change the
    differential drag rate (that's a slow atmospheric effect).
    """
    sin_u = np.sin(u_m)
    cos_u = np.cos(u_m)

    # Eq. (B2), scaled by 1/n so that Δx = a · B · δv [m·m/s → m]
    B = np.array([
        [ 0.0,          2.0,      0.0    ],   # δa
        [ 0.0,          0.0,      0.0    ],   # δȧ (unaffected by impulse)
        [-2.0,          0.0,      0.0    ],   # δλ (radial → longitude coupling)
        [ 0.0,          0.0,      cos_u  ],   # δix
        [ 0.0,          0.0,      sin_u  ],   # δiy
        [ sin_u,        2*cos_u,  0.0    ],   # δex
        [-cos_u,        2*sin_u,  0.0    ],   # δey
    ]) / n

    return B   # units: [s], so that a·B·δv [m/s] → ROE [m]


def plan_station_keeping(x0, x_target, t0, t_F, maneuver_times,
                         chief_orbit, J2, R_e, mu):
    """
    Solve the minimum-ΔV station-keeping problem using the analytical
    approach of Gaias & Ardaens 2018, Appendix A & B.

    Split into three decoupled correction maneuvers:
      1. Out-of-plane (δix, δiy): single normal burn at optimal u
      2. Eccentricity vector (δex, δey): paired tangential burns
      3. Along-track (δa + δλ): tangential burns that establish a
         temporary δa offset to create the needed δλ drift, then
         remove it at the end of the duty cycle.

    The key insight for along-track control: δλ cannot be corrected
    cheaply with radial burns. Instead, we establish a temporary
    semi-major axis offset Δδa_temp that drifts at rate:

        δλ̇ = ν · δa   where ν = -(3/2)n (Keplerian) + J2 terms

    Over the duty cycle Δt:
        Δδλ = ν · Δδa_temp · Δt

    Solving:
        Δδa_temp = Δδλ_needed / (ν · Δt)

    This requires two tangential burns: one to establish Δδa at the
    start, one to remove it at the end.

    Parameters
    ----------
    x0 : np.ndarray (7,)
        Current dimensional ROE state [m].
        Order: [aδa, aδȧ, aδλ, aδix, aδiy, aδex, aδey]
    x_target : np.ndarray (7,)
        Target dimensional ROE state [m].
    t0 : float
        Current epoch offset [s].
    t_F : float
        End of duty cycle [s].
    maneuver_times : list of float
        Candidate burn times [s from init_date].
    chief_orbit : KeplerianOrbit
        Chief's current mean orbit.
    J2, R_e, mu : float
        Gravity constants.

    Returns
    -------
    dv_schedule : list of (float, np.ndarray)
        Each entry is (burn_time [s], delta_v_RTN [m/s]).
        Sorted by burn time. Only non-negligible burns included.
    """
    a = float(chief_orbit.getA())
    e = float(chief_orbit.getE())
    i = float(chief_orbit.getI())
    n = np.sqrt(mu / a**3)

    eta    = np.sqrt(1.0 - e**2)
    gam    = (J2 / 2.0) * (R_e / (a * eta**2))**2
    phi_dot = (3.0 / 2.0) * n * gam * (5.0 * np.cos(i)**2 - 1.0)

    dt = t_F - t0
    if dt <= 0:
        return []

    # ── Natural propagation: what correction is needed? ──
    Phi_F0 = compute_stm(dt, a, e, i, J2, R_e, mu)
    b0 = x_target - Phi_F0 @ x0   # dimensional correction [m]

    # Convert to dimensionless for burn computation
    b0_dimless = b0 / a

    # ════════════════════════════════════════════════════════════════
    # SUBPROBLEM 1: OUT-OF-PLANE (δix, δiy)
    # Single normal burn at argument of latitude u_N
    # From B matrix: δix = cos(u)·δvN/n,  δiy = sin(u)·δvN/n
    # ════════════════════════════════════════════════════════════════
    b_dix = b0_dimless[3]
    b_diy = b0_dimless[4]
    di_mag = np.sqrt(b_dix**2 + b_diy**2)

    oop_burns = []
    if di_mag > 1e-8:
        # Optimal burn latitude: u_N = atan2(δiy_needed, δix_needed)
        u_N = np.arctan2(b_diy, b_dix)
        # Magnitude: |δvN| = n · |δi|  [m/s, since δi is in radians]
        dv_N_mag = n * di_mag

        t_N, u_actual = _find_best_burn_slot(
            u_N, maneuver_times, t0, chief_orbit, mu)

        # Compute signed δvN to achieve the desired (δix, δiy) at u_actual
        cos_ua = np.cos(u_actual)
        sin_ua = np.sin(u_actual)
        # From B: δix = cos(u)·δvN/n, δiy = sin(u)·δvN/n
        # We need: b_dix = cos(u_actual)·δvN/n, b_diy = sin(u_actual)·δvN/n
        # Use the component with larger leverage for numerical stability
        if abs(cos_ua) > abs(sin_ua):
            dv_N_signed = n * b_dix / cos_ua
        else:
            dv_N_signed = n * b_diy / sin_ua

        oop_burns.append((t_N, np.array([0.0, 0.0, dv_N_signed])))

    # ════════════════════════════════════════════════════════════════
    # SUBPROBLEM 2: ECCENTRICITY VECTOR (δex, δey)
    # Paired tangential burns at phase angle of Δδe
    # From B: δex = (sin u · δvR + 2cos u · δvT) / n
    #         δey = (-cos u · δvR + 2sin u · δvT) / n
    # Optimal 2-burn: burns at u_bar and u_bar + π
    # ════════════════════════════════════════════════════════════════
    b_dex = b0_dimless[5]
    b_dey = b0_dimless[6]
    delta_e_mag = np.sqrt(b_dex**2 + b_dey**2)

    ecc_burns = []
    if delta_e_mag > 1e-8:
        # Phase angle of required eccentricity change
        u_bar = np.arctan2(b_dey, b_dex)

        # Two tangential burns: at u_bar → +δvT, at u_bar+π → -δvT
        # Each contributes δe_component = 2·δvT/n in the u_bar direction
        # Total |Δδe| = 4·|δvT|/n  →  |δvT| = n·|Δδe|/4
        dv_e_mag = n * delta_e_mag / 4.0

        for k_offset in [0, 1]:
            u_burn = u_bar + k_offset * np.pi
            sign = 1.0 if k_offset == 0 else -1.0
            t_best, _ = _find_best_burn_slot(
                u_burn, maneuver_times, t0, chief_orbit, mu)
            ecc_burns.append((t_best, np.array([0.0, sign * dv_e_mag, 0.0])))

    # ════════════════════════════════════════════════════════════════
    # SUBPROBLEM 3: ALONG-TRACK (δa and δλ)
    #
    # Strategy: combined δa + δλ correction using drift.
    #
    # The along-track drift rate is:
    #   δλ̇ = ν_eff · δa/a
    # where ν_eff = -(3/2)n + J2 correction (from STM row 2, col 1)
    #
    # We need to correct:
    #   b_da  = target δa/a - propagated δa/a  (SMA ratio error)
    #   b_dl  = target δλ - propagated δλ      (along-track error)
    #
    # Approach:
    #   1. Compute total δa change needed: δa_total = δa_permanent + δa_drift
    #      - δa_permanent = b_da (to fix the SMA offset)
    #      - δa_drift = b_dl / (ν_eff · Δt/2) (temporary offset for drift)
    #        The /2 accounts for the drift acting over half the cycle on average
    #   2. Apply as two tangential burns:
    #      - First burn: establish δa_total at start of cycle
    #      - Second burn: remove δa_drift at end (keep only δa_permanent)
    #
    # From B matrix row 0: δ(a/a) = (2/n)·δvT
    # So: δvT = n·δ(a/a)/2
    # ════════════════════════════════════════════════════════════════
    b_da = b0_dimless[0]   # needed δ(a/a) correction
    b_dl = b0_dimless[2]   # needed δλ correction [rad]

    # Effective along-track drift rate from the STM
    # ν_eff = Φ[2,1] / dt  (the δλ sensitivity to δa over time)
    # From compute_stm: Phi[2,1] = nu_plus * dt where nu_plus = ν + μ_A
    nu_eff = -(3.0/2.0) * n + (-(21.0/4.0) * n * gam *
              (3.0 * np.cos(i)**2 - 1.0) * (eta + 1.0))

    at_burns = []

    if abs(b_da) > 1e-8 or abs(b_dl) > 1e-7:
        # Temporary δa offset to create the needed δλ drift
        if abs(nu_eff * dt) > 1e-10:
            da_drift = b_dl / (nu_eff * dt * 0.5)
        else:
            da_drift = 0.0

        # Total δa at start of cycle
        da_total_start = b_da + da_drift

        # Burns:
        # Burn 1 (start of cycle): establish da_total_start
        #   δvT_1 = n · da_total_start / 2
        # Burn 2 (end of cycle): remove drift component, keep permanent
        #   δvT_2 = -n · da_drift / 2
        dv_T1 = n * da_total_start / 2.0
        dv_T2 = -n * da_drift / 2.0

        # Find burn slots: first available and last available
        t_first = maneuver_times[0]   # earliest candidate
        t_last  = maneuver_times[-1]  # latest candidate

        if abs(dv_T1) > 1e-8:
            at_burns.append((t_first, np.array([0.0, dv_T1, 0.0])))
        if abs(dv_T2) > 1e-8:
            at_burns.append((t_last, np.array([0.0, dv_T2, 0.0])))

    # ════════════════════════════════════════════════════════════════
    # COMBINE AND SORT ALL BURNS
    # ════════════════════════════════════════════════════════════════
    all_burns = oop_burns + ecc_burns + at_burns
    all_burns.sort(key=lambda b: b[0])

    # Merge burns at same time slot (if two burns share a slot, sum them)
    merged = []
    for (t_b, dv) in all_burns:
        if merged and abs(merged[-1][0] - t_b) < 1.0:
            # Same slot → accumulate
            merged[-1] = (merged[-1][0], merged[-1][1] + dv)
        else:
            merged.append((t_b, dv.copy()))

    # Filter negligible burns
    min_threshold = 1e-4  # 0.1 mm/s
    merged = [(t_b, dv) for (t_b, dv) in merged
              if np.linalg.norm(dv) > min_threshold]

    # Safety cap: no single burn should exceed 1 m/s for station-keeping
    # If it does, something is wrong upstream → clamp and warn
    max_dv = 1.0  # m/s
    clamped = []
    for (t_b, dv) in merged:
        mag = np.linalg.norm(dv)
        if mag > max_dv:
            print(f"  [SK WARNING] Clamping burn at t={t_b:.0f}s from "
                  f"{mag*1000:.1f} mm/s to {max_dv*1000:.1f} mm/s")
            dv = dv * (max_dv / mag)
        clamped.append((t_b, dv))

    return clamped


def _find_best_burn_slot(target_u, maneuver_times, t0,
                         chief_orbit, mu):
    """
    From a list of candidate maneuver times, find the one whose mean
    argument of latitude is closest to target_u (mod π for paired burns).
    Returns (best_time, actual_u_at_that_time).
    """
    a  = float(chief_orbit.getA())
    n  = np.sqrt(mu / a**3)
    T  = 2.0 * np.pi / n   # orbital period [s]

    argp0 = float(chief_orbit.getPerigeeArgument())
    M0    = float(chief_orbit.getMeanAnomaly())
    u0    = argp0 + M0

    best_t   = maneuver_times[0]
    best_err = np.inf

    for t_m in maneuver_times:
        dt    = t_m - t0
        u_m   = (u0 + n * dt) % (2.0 * np.pi)
        err   = abs(((u_m - target_u + np.pi) % (2.0 * np.pi)) - np.pi)
        if err < best_err:
            best_err = err
            best_t   = t_m
            best_u   = u_m

    return best_t, best_u


def run_propagation_dsst_with_sk(times, init_date, forces, chief_orbit,
                                  deputy_orbits, J2, R_e, mu,
                                  target_roes,
                                  safety_threshold,
                                  duty_cycle_s,
                                  min_dv=1e-3,
                                  sun=None):
    """
    DSST propagation with direct-feedback ROE station keeping.

    At every time step, for each deputy:
      1. Extract current ROEs relative to the chief.
      2. Compute the ROE error (current − target).
      3. Invert the B matrix at the current argument of latitude to
         find the RTN delta-v that would zero the error.
      4. Apply a FRACTION (gain) of that correction as an impulse.
      5. Restart the deputy propagator from the corrected mean orbit.

    This is a proportional controller in ROE space. The gain controls
    the tradeoff between responsiveness and fuel cost:
      - gain = 1.0: correct all error immediately (expensive, aggressive)
      - gain = 0.01: correct 1% per step (cheap, smooth, slight lag)

    The controller naturally handles ALL perturbations because it
    corrects whatever error has accumulated, regardless of source.

    Parameters
    ----------
    times : array-like of float
        Output time stamps [s from init_date].
    init_date : AbsoluteDate
    forces : list
        DSST force models.
    chief_orbit : KeplerianOrbit
    deputy_orbits : list of KeplerianOrbit
    J2, R_e, mu : float
    target_roes : list of np.ndarray (7,)
        Target dimensional ROE [m] for each deputy.
    safety_threshold : float
        Minimum δr_RN^min [m] (used for deadband).
    duty_cycle_s : float
        Minimum time between corrections [s]. Acts as a control
        cadence to prevent excessive thruster cycling.
    min_dv : float
        Minimum burn magnitude [m/s]. Burns below this are skipped.
    sun : CelestialBody or None

    Returns
    -------
    tuple
        (a, e, i, raan, argp, M, alt, rel, dist, power, dv_log)
    """
    eci  = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    # Chief: normal initialization with osc→mean conversion
    chief_prop = create_dsst_propagator(chief_orbit, forces)

    # Extract the chief's converged mean Keplerian elements
    chief_mean_state = chief_prop.propagate(init_date)
    chief_mean_kep = to_keplerian(chief_mean_state.getOrbit())
    a_mean = float(chief_mean_kep.getA())

    # Reconstruct ROE dicts from the target_roes arrays
    a_c_orig = float(chief_orbit.getA())
    deputy_roes_dicts = []
    for tgt in target_roes:
        deputy_roes_dicts.append(dict(
            da  = float(tgt[0] / a_c_orig),
            dl  = float(tgt[2] / a_c_orig),
            dex = float(tgt[5] / a_c_orig),
            dey = float(tgt[6] / a_c_orig),
            dix = float(tgt[3] / a_c_orig),
            diy = float(tgt[4] / a_c_orig),
        ))

    # Apply ROEs to the chief's MEAN state → deputy mean orbits
    dep_mean_orbits = [apply_ROE(chief_mean_kep, roe) for roe in deputy_roes_dicts]

    # Initialize deputy propagators directly from mean elements (NO conversion)
    dep_props = [_create_dsst_from_mean_orbit(orb, forces) for orb in dep_mean_orbits]
    dep_orbits_current = list(dep_mean_orbits)

    # Recompute target_roes using the MEAN chief SMA for consistency
    target_roes = [
        a_mean * np.array([d['da'], 0.0, d['dl'],
                           d['dix'], d['diy'],
                           d['dex'], d['dey']])
        for d in deputy_roes_dicts
    ]

    # Print verification
    for k, dep_kep in enumerate(dep_mean_orbits):
        x0 = extract_roe(chief_mean_kep, dep_kep)
        print(f"  Deputy {k} initial ROE error: "
              f"|δa|={abs(x0[0]-target_roes[k][0]):.1f} m, "
              f"|δλ|={abs(x0[2]-target_roes[k][2]):.1f} m, "
              f"|δe|={np.sqrt((x0[5]-target_roes[k][5])**2+(x0[6]-target_roes[k][6])**2):.1f} m")


    a_v, e_v, i_v, raan_v, argp_v, M_v, alt_v = [], [], [], [], [], [], []
    rel  = [[] for _ in deputy_orbits]
    dist = [[] for _ in deputy_orbits]
    dv_log = {k: [] for k in range(len(deputy_orbits))}

    track_power = sun is not None
    if track_power:
        chief_power = []
        dep_power = [[] for _ in deputy_orbits]

    last_burn_time = {k: times[0] for k in range(len(deputy_orbits))}

    # ── Control gains ──
    # These determine how aggressively each ROE component is corrected.
    # Higher gain = faster correction but more fuel.
    # Lower gain = smoother but allows temporary drift.
    #
    # The gain per step should account for the time-step size:
    # effective_gain = gain_per_orbit × (dt_step / T_orbit)
    # This ensures the controller behaves consistently regardless of
    # output time resolution.
    dt_step = times[1] - times[0] if len(times) > 1 else 1.0
    T_orb   = 2.0 * np.pi * np.sqrt(float(chief_orbit.getA())**3 / mu)

    # Gain per orbit for each ROE component:
    # [δa, δȧ, δλ, δix, δiy, δex, δey]
    # δȧ (index 1) is never corrected by impulses → gain = 0
    # δλ needs the strongest correction (fastest drift)
    # δe, δi need moderate correction
    # δa needs careful correction (affects δλ drift rate)
    gain_per_orbit = np.array([
        0.5,   # δa:  moderate (affects drift)
        0.0,   # δȧ:  cannot correct with impulse
        0.8,   # δλ:  aggressive (main drift axis)
        0.3,   # δix: gentle (slow drift)
        0.3,   # δiy: gentle (slow drift)
        0.3,   # δex: gentle
        0.3,   # δey: gentle
    ])

    # Scale to per-step gain
    gain = gain_per_orbit * min(dt_step / T_orb, 1.0)

    for idx, t in enumerate(times):
        date_t = init_date.shiftedBy(float(t))

        # ══════════════════════════════════════════════════════════════
        # CHIEF
        # ══════════════════════════════════════════════════════════════
        chief_state = chief_prop.propagate(date_t)
        chief_kep   = to_keplerian(chief_state.getOrbit())
        pv_c = chief_state.getPVCoordinates(eci)
        p_c  = pv_c.getPosition()
        v_c  = pv_c.getVelocity()

        a_v.append(chief_kep.getA())
        e_v.append(chief_kep.getE())
        i_v.append(np.degrees(chief_kep.getI()))
        raan_v.append(np.degrees(
            chief_kep.getRightAscensionOfAscendingNode()))
        argp_v.append(np.degrees(chief_kep.getPerigeeArgument()))
        M_v.append(np.degrees(chief_kep.getMeanAnomaly()) % 360)

        tr     = eci.getTransformTo(itrf, date_t)
        p_itrf = tr.transformPVCoordinates(
            PVCoordinates(p_c, Vector3D.ZERO)).getPosition()
        alt_v.append(
            p_itrf.getNorm() - Constants.WGS84_EARTH_EQUATORIAL_RADIUS)

        r_vec = np.array([p_c.getX(), p_c.getY(), p_c.getZ()])
        v_vec = np.array([v_c.getX(), v_c.getY(), v_c.getZ()])
        r_hat = r_vec / np.linalg.norm(r_vec)
        h_vec = np.cross(r_vec, v_vec)
        h_hat = h_vec / np.linalg.norm(h_vec)
        t_hat = np.cross(h_hat, r_hat)

        if track_power:
            r_sun = get_sun_position_eci(sun, date_t, eci)
            chief_power.append(
                compute_solar_power(r_vec, r_sun, v_sc_eci=v_vec))

        # ══════════════════════════════════════════════════════════════
        # DEPUTIES
        # ══════════════════════════════════════════════════════════════
        for k in range(len(deputy_orbits)):

            # ── Propagate deputy ──
            dep_state = dep_props[k].propagate(date_t)
            dep_kep   = to_keplerian(dep_state.getOrbit())
            pv_d      = dep_state.getPVCoordinates(eci)
            p_d       = pv_d.getPosition()
            v_d       = pv_d.getVelocity()

            dr = np.array([
                p_d.getX() - p_c.getX(),
                p_d.getY() - p_c.getY(),
                p_d.getZ() - p_c.getZ()
            ])
            rel[k].append([float(np.dot(dr, r_hat)),
                           float(np.dot(dr, t_hat)),
                           float(np.dot(dr, h_hat))])
            dist[k].append(float(np.linalg.norm(dr)))

            # ──────────────────────────────────────────────────────────
            # STATION-KEEPING
            # ──────────────────────────────────────────────────────────
            time_since_burn = t - last_burn_time[k]

            if time_since_burn >= duty_cycle_s and idx > 0:
                # Measure current ROE state and error
                x_now = extract_roe(chief_kep, dep_kep)
                x_tgt = target_roes[k]
                a_dep = float(dep_kep.getA())
                n_dep = np.sqrt(mu / a_dep**3)

                # Dimensional errors [m]
                err = x_now - x_tgt
                err_da  = err[0]        # [m] = a × δ(a/a)
                err_dl  = err[2]        # [m] = a × δλ
                err_dix = err[3]        # [m] = a × δix
                err_diy = err[4]        # [m] = a × δiy
                err_dex = err[5]        # [m] = a × δex
                err_dey = err[6]        # [m] = a × δey

                # ════════════════════════════════════════════════════════
                # TANGENTIAL BURN: Controls δa (and δe as side-effect)
                #
                # Strategy: The DOMINANT error is δλ (along-track drift).
                # We correct it by establishing a temporary δa offset that
                # creates drift back toward the target.
                #
                # Required δa offset to zero δλ in time τ:
                #   Δ(δa/a) = -δλ_error / ((3/2) n × τ)
                #
                # Then the tangential burn to create this δa:
                #   δvT = (n/2) × Δ(δa/a) × a   ... NO!
                #   δvT = (n/2) × Δ(δa/a)       [m/s, since B gives δ(a/a) = 2δvT/n]
                # ════════════════════════════════════════════════════════

                # Time horizon to correct δλ — shorter = more aggressive
                tau = 1.0 * T_orb   # correct over 1 orbit

                # Convert errors to dimensionless
                err_da_dimless = err_da / a_dep      # δ(a/a)
                err_dl_dimless = err_dl / a_dep      # δλ [rad]

                # Desired δa correction = fix current δa error + extra to drift δλ back
                nu_drift = -(3.0 / 2.0) * n_dep     # along-track drift rate [rad/s per δa/a]
                
                if abs(nu_drift * tau) > 1e-12:
                    da_for_dl = -err_dl_dimless / (nu_drift * tau)
                else:
                    da_for_dl = 0.0

                # Total desired δ(a/a) change this step
                da_total_dimless = -err_da_dimless + da_for_dl

                # Apply gain (don't try to fix everything at once)
                gain_at = 0.5 * min(dt_step / T_orb, 1.0)
                da_cmd = gain_at * da_total_dimless

                # Convert to tangential burn: δ(a/a) = (2/n)×δvT → δvT = n×δ(a/a)/2
                dvT = n_dep * da_cmd / 2.0

                # ════════════════════════════════════════════════════════
                # NORMAL BURN: Controls δix, δiy
                #
                # From B matrix: δix = cos(u)×δvN/n, δiy = sin(u)×δvN/n
                # We want to reduce the inclination vector error.
                # At the current u, the achievable correction is:
                #   δvN → Δδix = cos(u)×δvN/n, Δδiy = sin(u)×δvN/n
                #
                # Optimal: project the error onto the achievable direction
                # ════════════════════════════════════════════════════════
                argp_d = float(dep_kep.getPerigeeArgument())
                M_d    = float(dep_kep.getMeanAnomaly())
                u_now  = argp_d + M_d

                sin_u = np.sin(u_now)
                cos_u = np.cos(u_now)

                err_dix_dimless = err_dix / a_dep
                err_diy_dimless = err_diy / a_dep

                # Project inclination error onto the achievable direction at this u
                # Achievable direction: [cos(u), sin(u)] in (δix, δiy) space
                # Projection: component of error along [cos(u), sin(u)]
                proj = err_dix_dimless * cos_u + err_diy_dimless * sin_u

                # Gain for normal direction
                gain_n = 0.3 * min(dt_step / T_orb, 1.0)
                dvN_needed = -gain_n * n_dep * proj  # δvN = n × δi_proj, with sign

                # ════════════════════════════════════════════════════════
                # ASSEMBLE AND APPLY
                # ════════════════════════════════════════════════════════
                dv_rtN = np.array([0.0, dvT, dvN_needed])
                dv_mag = np.linalg.norm(dv_rtN)

                # Skip negligible burns
                if dv_mag < min_dv:
                    if track_power:
                        r_dep = r_vec + dr
                        v_dep = np.array([v_d.getX(), v_d.getY(), v_d.getZ()])
                        dep_power[k].append(
                            compute_solar_power(r_dep, r_sun, v_sc_eci=v_dep))
                    continue

                # Safety cap
                max_dv_per_step = 0.5  # m/s
                if dv_mag > max_dv_per_step:
                    dv_rtN = dv_rtN * (max_dv_per_step / dv_mag)
                    dv_mag = max_dv_per_step

                # Apply burn
                try:
                    dep_orbits_current[k], dep_props[k] = \
                        _apply_impulsive_burn(
                            dep_props[k], dep_orbits_current[k],
                            dv_rtN, date_t, forces, eci, mu)
                    dv_log[k].append((t, dv_rtN.copy()))
                    last_burn_time[k] = t

                    if idx % 100 == 0:
                        print(f"  t={t/86400:.1f}d dep{k}: "
                              f"|dv|={dv_mag*1e3:.3f} mm/s "
                              f"(T={dvT*1e3:.3f}, N={dvN_needed*1e6:.1f}μm/s) "
                              f"|err_λ|={abs(err_dl)/1e3:.1f} km "
                              f"|err_a|={abs(err_da):.0f} m "
                              f"da_cmd={da_cmd:.2e}")

                except ValueError as err:
                    print(f"  [SK] Deputy {k} burn rejected at "
                          f"t={t/86400:.1f}d: {err}")

            # ── Power tracking ──
            if track_power:
                r_dep = r_vec + dr
                v_dep = np.array([v_d.getX(), v_d.getY(), v_d.getZ()])
                dep_power[k].append(
                    compute_solar_power(r_dep, r_sun, v_sc_eci=v_dep))

    # ══════════════════════════════════════════════════════════════════
    # PACKAGE
    # ══════════════════════════════════════════════════════════════════
    rel  = [np.array(rel[k])  for k in range(len(deputy_orbits))]
    dist = [np.array(dist[k]) for k in range(len(deputy_orbits))]

    if track_power:
        power = {
            "chief":    np.array(chief_power),
            "deputies": [np.array(dep_power[k])
                         for k in range(len(deputy_orbits))]
        }
    else:
        power = None

    # Summary
    for k in range(len(deputy_orbits)):
        total_dv = sum(np.linalg.norm(dv) for (_, dv) in dv_log[k])
        n_burns  = len(dv_log[k])
        print(f"  Deputy {k}: {n_burns} burns, "
              f"total ΔV = {total_dv:.4f} m/s "
              f"({total_dv*1000:.1f} mm/s)")

    return (a_v, e_v, i_v, raan_v, argp_v, M_v, alt_v,
            rel, dist, power, dv_log)

def _apply_impulsive_burn(dep_prop, dep_orbit_mean, dv_rtN,
                          date_t, forces, eci, mu):
    """
    Apply an instantaneous ΔV to a deputy by computing the ROE-space
    effect analytically and constructing a new mean-element orbit.

    The procedure:
      1. Extract mean Keplerian elements from the current DSST state.
      2. Compute the B matrix (Eq. B2, Gaias & Ardaens 2018) that maps
         RTN delta-v to dimensionless ROE changes.
      3. Multiply by a to get dimensional ROE changes [m].
      4. Apply the ROE changes to the mean elements.
      5. Validate the resulting orbit (periapsis check).
      6. Restart the DSST propagator from the new mean orbit.

    Parameters
    ----------
    dep_prop : DSSTPropagator
        Current deputy propagator.
    dep_orbit_mean : KeplerianOrbit
        Current deputy mean orbit (used as fallback reference).
    dv_rtN : np.ndarray, shape (3,)
        Impulsive delta-v in RTN frame [m/s]: [radial, tangential, normal].
    date_t : AbsoluteDate
        Epoch of the burn.
    forces : list
        DSST force models for the new propagator.
    eci : Frame
        ECI reference frame.
    mu : float
        Earth gravitational parameter [m³/s²].

    Returns
    -------
    new_mean_orbit : KeplerianOrbit
        Post-burn mean-element orbit.
    new_prop : DSSTPropagator
        Fresh propagator initialized at the post-burn state.

    Raises
    ------
    ValueError
        If the burn produces an unphysical orbit (SMA ratio > 2 or < 0.5,
        or periapsis below 100 km).
    """
    # ── Step 1: Extract current mean elements ──
    mean_state = dep_prop.propagate(date_t)
    mean_orbit = to_keplerian(mean_state.getOrbit())

    a    = float(mean_orbit.getA())
    e    = float(mean_orbit.getE())
    i    = float(mean_orbit.getI())
    raan = float(mean_orbit.getRightAscensionOfAscendingNode())
    argp = float(mean_orbit.getPerigeeArgument())
    M    = float(mean_orbit.getMeanAnomaly())
    n    = np.sqrt(mu / a**3)
    u_m  = argp + M  # mean argument of latitude at burn epoch

    # ── Step 2: B matrix (Eq. B2) — maps δv → DIMENSIONLESS ROE δα ──
    sin_u = np.sin(u_m)
    cos_u = np.cos(u_m)

    # B / n gives dimensionless ROE change per m/s of delta-v
    # Row order: [δ(a/a), δȧ, δλ, δix, δiy, δex, δey]
    B_dimless = np.array([
        [ 0.0,       2.0,      0.0    ],   # δ(a/a)  = (2/n)·δvT
        [ 0.0,       0.0,      0.0    ],   # δȧ (unaffected)
        [-2.0,       0.0,      0.0    ],   # δλ = -(2/n)·δvR
        [ 0.0,       0.0,      cos_u  ],   # δix = (cos u / n)·δvN
        [ 0.0,       0.0,      sin_u  ],   # δiy = (sin u / n)·δvN
        [ sin_u,     2*cos_u,  0.0    ],   # δex
        [-cos_u,     2*sin_u,  0.0    ],   # δey
    ]) / n  # units: [s] per [m/s] → dimensionless

    # ── Step 3: DIMENSIONLESS ROE change ──
    d_roe_dimless = B_dimless @ dv_rtN  # dimensionless δα vector

    # ── Step 4: Validate before applying ──
    new_a_over_a = 1.0 + d_roe_dimless[0]  # δ(a/a) is already a ratio
    if new_a_over_a <= 0.5 or new_a_over_a >= 2.0:
        raise ValueError(
            f"Burn produces unphysical SMA ratio {new_a_over_a:.6f}. "
            f"ΔV={dv_rtN} m/s at u={np.degrees(u_m):.1f}°."
        )

    # ── Step 5: Apply dimensionless ROE changes to mean elements ──
    # These are the DIMENSIONLESS quasi-nonsingular ROE offsets:
    #   d_roe_dimless[0] = δ(a)/a     (relative SMA change)
    #   d_roe_dimless[2] = δλ         (mean longitude change)
    #   d_roe_dimless[3] = δix        (inclination change) [rad]
    #   d_roe_dimless[4] = δiy        (RAAN-like change) [rad·sin(i)]
    #   d_roe_dimless[5] = δex        (eccentricity x-component change)
    #   d_roe_dimless[6] = δey        (eccentricity y-component change)

    a_new = a * new_a_over_a

    ex_c = e * np.cos(argp)
    ey_c = e * np.sin(argp)

    ex_new  = ex_c + d_roe_dimless[5]
    ey_new  = ey_c + d_roe_dimless[6]
    e_new   = float(np.sqrt(ex_new**2 + ey_new**2))
    argp_new = float(np.arctan2(ey_new, ex_new))

    i_new    = float(i + d_roe_dimless[3])
    raan_new = float(raan + d_roe_dimless[4] / np.sin(i))

    d_argp = argp_new - argp
    M_new  = float(M + d_roe_dimless[2] - d_argp)

    # ── Step 6: Periapsis safety check ──
    R_e = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
    periapsis_alt = a_new * (1.0 - e_new) - R_e
    if periapsis_alt < 100e3:
        raise ValueError(
            f"Post-burn periapsis altitude {periapsis_alt/1e3:.1f} km is "
            f"below 100 km. ΔV={dv_rtN} m/s, e_new={e_new:.6f}"
        )

    # ── Step 7: Construct new orbit and restart propagator ──
    new_mean_orbit = KeplerianOrbit(
        float(a_new), float(e_new), float(i_new),
        float(argp_new), float(raan_new), float(M_new),
        PositionAngleType.MEAN, eci, date_t, mu
    )

    new_prop = _create_dsst_from_mean_orbit(new_mean_orbit, forces)
    return new_mean_orbit, new_prop

def _create_dsst_from_mean_orbit(mean_orbit, forces):
    """
    Create a DSST propagator directly from an orbit already in mean-element
    space, bypassing the computeMeanState osculating→mean conversion.

    This is safe when the orbit was derived analytically from DSST mean
    element output (as in _apply_impulsive_burn above) rather than from
    osculating PV coordinates.

    The only difference from create_dsst_propagator is that we call
    setInitialState with the orbit directly, wrapped in SpacecraftState,
    without the computeMeanState round-trip.
    """
    min_step = 1.0
    max_step = 86400.0
    abs_tol  = [1e-3] * 7
    rel_tol  = [1e-9] * 7
    integrator = DormandPrince853Integrator(min_step, max_step, abs_tol, rel_tol)

    propagator = DSSTPropagator(integrator, PropagationType.MEAN)

    for force in forces:
        propagator.addForceModel(force)

    for force_model in propagator.getAllForceModels():
        force_model.registerAttitudeProvider(propagator.getAttitudeProvider())

    # Set initial state directly as mean elements — no conversion needed
    mean_spacecraft_state = SpacecraftState(mean_orbit)
    propagator.setInitialState(mean_spacecraft_state, PropagationType.MEAN)

    return propagator

