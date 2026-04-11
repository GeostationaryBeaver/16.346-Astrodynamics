import plotting
import numpy as np
from ideal_funcs import run_propagation, get_eci_trajectories, times_short, times_long, colors, labels

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


# ------------------------------------
# 7. PLOTS
# ------------------------------------

# Plot 1: Orbital elements
plotting.plot_orbital_elements(days_long, a_l, e_l, i_l, raan_l, argp_l, M_l)
plotting.plot_earth_frame(chief_eci, dep_eci_exag, snap_idx, colors, labels, EXAG)
plotting.plot_hill_3d(rel_short, colors, labels)
plotting.plot_radial_intrack(rel_short, colors, labels)
plotting.plot_intrack_crosstrack(rel_short, colors, labels)

plt.show()