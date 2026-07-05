def physical_to_nondim(
    u_surface_myr: float,
    H_m: float,
    a_myr: float,
    L_m: float        = None,
    w_basal_myr: float = 0.0,
    verbose: bool      = True,
) -> dict:
    """
    Translate physical glaciological quantities into the nondimensional
    parameters used by SigmaAgeModel.

    Nondimensionalisation recap
    ---------------------------
        x_tilde  = x / L
        zeta     = (z - b) / H          ∈ [0, 1]
        t_tilde  = t * a / H            (1 unit = one ice-column replacement)
        X_tilde  = X * a / H            (1 unit = H/a years)
        u_tilde  = u / (L * a / H)      → Pe = u_surface / (L * a / H)
        w_tilde  = w_sigma / (a / H)    → w_basal_tilde = w_basal / (a / H)

    Parameters
    ----------
    u_surface_myr : float
        Surface horizontal ice velocity  [m yr⁻¹]
    H_m : float
        Mean ice thickness               [m]
    a_myr : float
        Surface mass balance (accumulation rate)  [m yr⁻¹ ice equiv.]
    L_m : float, optional
        Horizontal domain length  [m].
        If None, estimated as the advective length scale L = u_surface * H / a
        (i.e. the distance ice travels in one residence time).
    w_basal_myr : float, optional
        Basal freeze-on / melt rate  [m yr⁻¹].
        Positive = freeze-on (upward sigma velocity).  Default 0.
    verbose : bool
        Print a formatted summary table.

    Returns
    -------
    dict with keys:
        Flush         – Flushing number (horizontal flux over vertical flux)
        W_basal       – nondim basal vertical velocity
        t_scale_yr    – physical duration of 1 nondim time unit  [yr]
        X_scale_yr    – physical age represented by X_tilde = 1  [yr]
        L_m           – domain length used  [m]
        residence_time_yr – H / a  [yr]  (= t_scale_yr)
    """

    # ── Derived scales ────────────────────────────────────────────────────────
    residence_time_yr = H_m / abs(a_myr)          # [yr]  one nondim time unit
    w_scale           = abs(a_myr) / H_m          # [yr⁻¹] vertical velocity scale

    # Domain length: use supplied value or default to advective scale
    if L_m is None:
        L_m = u_surface_myr * H_m / abs(a_myr)   # distance travelled in H/a years
        L_estimated = True
    else:
        L_estimated = False

    u_scale = L_m * abs(a_myr) / H_m             # [m yr⁻¹] horizontal velocity scale

    # ── Nondimensional numbers ────────────────────────────────────────────────
    Flush       = u_surface_myr / u_scale       # = u_s * H / (L * a)
    W_basal  = w_basal_myr   / (abs(a_myr) / H_m * H_m)   # = w_b / a
    # cleaner form: w_tilde = w_basal / a
    W_basal  = w_basal_myr / abs(a_myr)

    # ── Verbose summary ───────────────────────────────────────────────────────
    if verbose:
        sep = "─" * 54
        print(f"\n{sep}")
        print(f"  Physical → Nondimensional Parameter Translation")
        print(sep)
        print(f"  {'INPUT PHYSICAL QUANTITIES':}")
        print(f"    Surface velocity   u_s  = {u_surface_myr:>10.2f}  m yr⁻¹")
        print(f"    Ice thickness      H    = {H_m:>10.2f}  m")
        print(f"    Accumulation rate  a    = {a_myr:>10.4f}  m yr⁻¹")
        print(f"    Domain length      L    = {L_m:>10.2f}  m"
              + ("  (estimated)" if L_estimated else ""))
        print(f"    Basal velocity     w_b  = {w_basal_myr:>10.4f}  m yr⁻¹")
        print(f"\n  CHARACTERISTIC SCALES")
        print(f"    Time scale   H/a        = {residence_time_yr:>10.1f}  yr")
        print(f"    Horiz. vel. scale L·a/H = {u_scale:>10.2f}  m yr⁻¹")
        print(f"    Vert.  vel. scale  a/H  = {w_scale:>10.6f}  yr⁻¹  "
              f"({a_myr/H_m*1000:.4f} × 10⁻³ yr⁻¹)")
        print(f"\n  NONDIMENSIONAL PARAMETERS")
        print(f"    Flush   = u_s / (L·a/H) = {Flush:>8.3f}")
        print(f"    W_basal = w_b / a       = {W_basal:>8.3f}")
        print(f"\n  INTERPRETATION")
        print(f"    1 nondim time unit      = {residence_time_yr:.1f} yr")
        print(f"    X_tilde = 1  means age  = {residence_time_yr:.1f} yr")
        print(f"    X_tilde = {Flush:.1f}  means age  "
              f"= {Flush * residence_time_yr:.1f} yr  (≈ transit time)")
        if w_basal_myr != 0.0:
            print(f"    W_basal = {W_basal:.3f}  means basal freeze-on "
                  f"= {abs(w_basal_myr):.4f} m yr⁻¹")
            print(f"    → freeze-on is {abs(W_basal)*100:.1f}% of surface accumulation")
        print(sep + "\n")

    return dict(
        Flush             = Flush,
        W_basal           = W_basal,
        t_scale_yr        = residence_time_yr,
        X_scale_yr        = residence_time_yr,
        L_m               = L_m,
        residence_time_yr = residence_time_yr,
    )

def nondim_to_physical(
    Flush: float,
    W_basal: float,
    t_scale_yr: float,
    L_m: float,
    verbose: bool = True,
) -> dict:
    """
    Translate nondimensional parameters used by SigmaAgeModel into physical
    glaciological quantities.

    Parameters
    ----------
    Flush : float
        Flushing number (horizontal flux over vertical flux)
    W_basal : float
        Nondimensional basal vertical velocity
    t_scale_yr : float
        Physical duration of 1 nondim time unit  [yr]
    L_m : float
        Domain length used  [m]
    verbose : bool
        Print a formatted summary table.

    Returns
    -------
    dict with keys:
        u_surface_myr : Surface horizontal ice velocity  [m yr⁻¹]
        H_m           : Mean ice thickness               [m]
        a_myr         : Surface mass balance (accumulation rate)  [m yr⁻¹ ice equiv.]
        w_basal_myr   : Basal freeze-on / melt rate  [m yr⁻¹]
    """
    # ── Derived scales ────────────────────────────────────────────────────────
    H_m = t_scale_yr * abs(W_basal) / abs(W_basal)  # [m]  one nondim time unit
    a_myr = H_m / t_scale_yr                         # [m yr⁻¹] accumulation rate
    u_surface_myr = Flush * L_m * a_myr / H_m        # [m yr⁻¹] surface velocity
    w_basal_myr = W_basal * a_myr                    # [m yr⁻¹] basal velocity

    # ── Verbose summary ───────────────────────────────────────────────────────
    if verbose:
        sep = "─" * 54
        print(f"\n{sep}")
        print(f"  Nondimensional → Physical Parameter Translation")
        print(sep)
        print(f"  {'INPUT NONDIMENSIONAL PARAMETERS':}")
        print(f"    Flush   = {Flush:>8.3f}")
        print(f"    W_basal = {W_basal:>8.3f}")
        print(f"    t_scale_yr = {t_scale_yr:>10.1f} yr")
        print(f"    L_m       = {L_m:>10.2f} m")
        print(f"\n  DERIVED PHYSICAL QUANTITIES")
        print(f"    Surface velocity   u_s  = {u_surface_myr:>10.2f} m yr⁻¹")
        print(f"    Ice thickness      H    = {H_m:>10.2f} m")
        print(f"    Accumulation rate  a    = {a_myr:>10.4f} m yr⁻¹")
        print(f"    Basal velocity     w_b  = {w_basal_myr:>10.4f} m yr⁻¹")
        print(sep + "\n")

    return dict(
        u_surface_myr=u_surface_myr,
        H_m=H_m,
        a_myr=a_myr,
        w_basal_myr=w_basal_myr,
    )