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





# =============================================================================
# SPACECRAFT PHYSICAL CONSTANTS (ISARA-class 3U CubeSat)
# =============================================================================
ISARA_MASS_KG = 4.0               # Spacecraft wet mass [kg]
ISARA_PANEL_AREA_M2 = 0.3         # Total deployable solar panel area [m²]
ISARA_CROSS_SECTION_M2 = 0.06     # Drag cross-section (3U long-axis forward) [m²]
ISARA_SRP_AREA_M2 = 0.3           # SRP effective area (panels facing sun) [m²]
ISARA_CR = 1.5                    # Radiation pressure coefficient [-]
ISARA_CD = 2.2                    # Drag coefficient [-]
SOLAR_CELL_EFFICIENCY = 0.283     # Triple-junction GaAs cell efficiency [-]
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
        # Harris-Priester: empirical static atmosphere, function of altitude
        # and solar activity proxy (embedded in model tables).
        # Requires Sun for diurnal bulge calculation.
        atmosphere = HarrisPriester(sun, earth)
        # Isotropic drag model: force = 0.5 * Cd * A * ρ * v²
        drag_sc = IsotropicDrag(float(cross_section), float(Cd))
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
    chief_prop = create_dsst_propagator(chief_orbit, forces)
    dep_props = [create_dsst_propagator(o, forces) for o in deputy_orbits]

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
