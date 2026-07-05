import numpy as np
def no_perturbation(t):
    """Always zero — baseline control run."""
    return 0.0

def step_perturbation(t,
                      T_ON=1.0, 
                      T_OFF=2.0, 
                      W_FREEZE=1.0
                      ):
    """
    Rectangular pulse: freeze-on active between T_ON and T_OFF.
    Returns a uniform (scalar) w_basal across all x.
    """
    return W_FREEZE if T_ON <= t < T_OFF else 0.0

def localized_freeze_on_step(
        t, 
        T_ON=1.0, 
        T_OFF=2.0, 
        PERT_XLIM=(0.3, 0.7),
        W_FREEZE=1.0
        ):
    wb = np.zeros(150)
    if T_ON <= t < T_OFF:
        # Freeze-on active only in x = [0.3, 0.7]
        x = np.linspace(0, 1, 150)
        mask = (x >= PERT_XLIM[0]) & (x <= PERT_XLIM[1])
        wb[mask] = W_FREEZE
    return wb

def localized_freeze_on_gaussian(
        t, 
        T_ON=1.0, 
        T_OFF=2.0, 
        PERT_XLIM=(0.3, 0.7), 
        W_FREEZE=1.0
        ):
    wb = np.zeros(150)
    if T_ON <= t < T_OFF:
        # Freeze-on active
        x = np.linspace(0, 1, 150)
        mask = (x >= PERT_XLIM[0]) & (x <= PERT_XLIM[1])
        wb[mask] = W_FREEZE * np.exp(-((x[mask]-(PERT_XLIM[0]+PERT_XLIM[1])/2.0)/0.1)**2)
    return wb

def localized_freeze_on_smooth(
    t,
    nx          = 150,
    x_sigma     = 0.1,
    x_lo        = 0.3,
    x_hi        = 0.7,
    t_on        = 1.0,    # nondim time to switch on freeze-on
    t_off       = 2.0,    # nondim time to switch off freeze-on
    W_FREEZE    = 1.0,    # nondim peak basal upward
    t_to_peak   = 0.1,    # nondim time to ramp from 0 → W_FREEZE after t_on
    t_to_zero   = 0.1,    # nondim time to ramp from W_FREEZE → 0 after t_off
):
    """
    Spatially Gaussian, temporally smooth basal freeze-on perturbation.

    Spatial shape
    -------------
    Gaussian centred at x_center with std x_sigma, masked to [x_lo, x_hi].
    Peak amplitude = W_FREEZE.

    Temporal shape
    --------------
    Uses a smooth C-infinity ramp (based on the standard bump function)
    rather than a hard step, so:
      - ramps from 0 → W_FREEZE over [t_on,        t_on  + t_to_peak]
      - holds at W_FREEZE         over [t_on+t_to_peak, t_off]
      - ramps from W_FREEZE → 0  over [t_off,       t_off + t_to_zero]
      - zero outside that window

    Parameters
    ----------
    t           : float  — current nondim time
    nx          : int    — number of x grid points (must match model)
    x_center    : float  — Gaussian centre in nondim x
    x_sigma     : float  — Gaussian std in nondim x
    x_lo, x_hi  : float  — hard mask: Gaussian is zeroed outside this window
    t_to_peak   : float  — nondim ramp-up duration after t_on
    t_to_zero   : float  — nondim ramp-down duration after t_off

    Returns
    -------
    wb : ndarray (nx,)  — nondim basal upward velocity at each x
    """

    # ── Smooth ramp kernel: maps s ∈ [0,1] → [0,1] via bump function ─────────
    # Uses the C∞ sigmoid  φ(s) = 1 / (1 + exp(1/(s-1) - 1/s))
    # which is exactly 0 at s≤0, exactly 1 at s≥1, and infinitely smooth.
    def smooth_ramp(s):
        """Vectorisable C∞ ramp: 0→1 over s ∈ [0,1]."""
        s   = np.asarray(s, dtype=float)
        out = np.zeros_like(s)
        interior = (s > 0.0) & (s < 1.0)
        si = s[interior]
        # Numerically safe: clamp exponent argument
        exp_arg = np.clip(1.0 / (si - 1.0) - 1.0 / si, -500, 500)
        out[interior] = 1.0 / (1.0 + np.exp(exp_arg))
        out[s >= 1.0] = 1.0
        return out

    x_center = (x_lo + x_hi) / 2.0

    # ── Temporal amplitude ────────────────────────────────────────────────────
    if t < t_on or t >= t_off + t_to_zero:
        return np.zeros(nx)

    if t < t_on + t_to_peak:
        # Ramp up: s goes 0→1 as t goes t_on → t_on+t_to_peak
        s         = (t - t_on) / t_to_peak
        amplitude = smooth_ramp(np.array([s]))[0]

    elif t < t_off:
        # Plateau
        amplitude = 1.0

    else:
        # Ramp down: s goes 0→1 as t goes t_off → t_off+t_to_zero
        s         = (t - t_off) / t_to_zero
        amplitude = 1.0 - smooth_ramp(np.array([s]))[0]

    # ── Spatial Gaussian profile ──────────────────────────────────────────────
    x    = np.linspace(0, 1, nx)
    mask = (x >= x_lo) & (x <= x_hi)
    wb   = np.zeros(nx)
    wb[mask] = W_FREEZE * amplitude * np.exp(
        -((x[mask] - x_center) / x_sigma) ** 2
    )

    return wb

def localized_freeze_on_melt_smooth(
    t,
    nx          = 150,
    x_lo        = 0.3,
    x_hi        = 0.7,
    t_on        = 1.0,
    t_off       = 2.0,
    W_FREEZE    = 1.0,
    t_to_peak   = 0.1,
    t_to_zero   = 0.1,
):
    """
    Spatially sinusoidal (one full period), temporally smooth basal perturbation.

    Spatial shape
    -------------
    One full sine wave over [x_lo, x_hi]:
      - first half  [x_lo, x_mid]: freeze-on  (+W_FREEZE)
      - second half [x_mid, x_hi]: melt        (-W_FREEZE)

    Temporal shape
    --------------
    Same smooth C∞ bump-function envelope as localized_freeze_on_smooth:
      - ramps 0 → 1 over [t_on,          t_on + t_to_peak]
      - holds at 1   over [t_on+t_to_peak, t_off]
      - ramps 1 → 0  over [t_off,         t_off + t_to_zero]
      - zero outside that window
    """

    # ── Smooth ramp kernel ────────────────────────────────────────────────────
    def smooth_ramp(s):
        s   = np.asarray(s, dtype=float)
        out = np.zeros_like(s)
        interior = (s > 0.0) & (s < 1.0)
        si = s[interior]
        exp_arg = np.clip(1.0 / (si - 1.0) - 1.0 / si, -500, 500)
        out[interior] = 1.0 / (1.0 + np.exp(exp_arg))
        out[s >= 1.0] = 1.0
        return out

    # ── Temporal amplitude (identical to localized_freeze_on_smooth) ──────────
    if t < t_on or t >= t_off + t_to_zero:
        return np.zeros(nx)

    if t < t_on + t_to_peak:
        s         = (t - t_on) / t_to_peak
        amplitude = smooth_ramp(np.array([s]))[0]

    elif t < t_off:
        amplitude = 1.0

    else:
        s         = (t - t_off) / t_to_zero
        amplitude = 1.0 - smooth_ramp(np.array([s]))[0]

    # ── Spatial sine profile over [x_lo, x_hi] ───────────────────────────────
    x    = np.linspace(0, 1, nx)
    mask = (x >= x_lo) & (x <= x_hi)
    wb   = np.zeros(nx)

    # map x ∈ [x_lo, x_hi] → one full period of sine [0, 2π]
    x_norm      = (x[mask] - x_lo) / (x_hi - x_lo)   # 0 → 1
    wb[mask]    = W_FREEZE * amplitude * np.sin(2.0 * np.pi * x_norm)

    return wb
