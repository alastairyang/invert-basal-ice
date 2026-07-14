import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class SigmaAgeModel:
    """
    Nondimensional ice age solver in sigma (terrain-following) coordinates.

    PDE:  dX/dt + u_tilde * dX/dx + w_sigma * dX/dzeta = 1

    Nondimensionalisation
    ---------------------
    x_tilde  = x_phys / L
    zeta     = (z - b(x)) / H(x)        in [0, 1]
    t_tilde  = t_phys * a / H_mean
    X_tilde  = X_phys * a / H_mean
    u_tilde  = u_phys / (L * a / H_mean)
    w_sigma  = w_phys_sigma / (a / H_mean)

    Bed kinematic condition
    -----------------------
    In physical coords:
        w_phys(bed) = u_phys * db/dx + m_dot
    where m_dot is the basal melt rate (positive = melt, negative = freeze-on).

    In sigma coords the bed-slope term (u * db/dx) is already absorbed into
    the metric correction during the upward integration of incompressibility.
    Therefore w_basal passed to _build_velocity() carries ONLY the melt/freeze
    anomaly, nondimensionalised as:
        w_basal_nd = m_dot / a_rate

    BCs
    ---
        X = 0  at zeta = 1          (surface: fresh snow)
        X = 0  at x = 0  (Flush>0)  (inflow divide)
        dX/dx = 0 at x = 1 (Flush>0)(outflow: zero-gradient)

    Usage
    -----
    # From synthetic geometry (original behaviour):
        model = SigmaAgeModel(nx=150, nz=100, Flush=1.0)

    # From physical geometry arrays:
        model = SigmaAgeModel.from_geometry(
            x_phys, bed_phys, surface_phys,
            nx=150, nz=100,
            u_mean=100.0, a_rate=0.5, Flush=1.0
        )
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, nx=150, nz=100, Flush=1.0,
                 bed_nd=None, surface_nd=None,
                 bed_amp=0.0, cfl_safety=0.4, n=3.0, shape_factor=1.0):
        """
        Low-level constructor — works in nondimensional space.

        Parameters
        ----------
        nx, nz      : grid resolution
        Flush       : nondim horizontal flux scale (= Pe in some notations)
        bed_nd      : (nx,) nondim bed elevation.     None → synthetic sine.
        surface_nd  : (nx,) nondim surface elevation. None → 1 - 0.05*x.
        bed_amp     : amplitude for synthetic bed (ignored if bed_nd given).
        cfl_safety  : CFL safety factor.
        """
        self.nx         = nx
        self.nz         = nz
        self.Flush      = Flush
        self.cfl_safety = cfl_safety
        self.n = n # glen's flow exponent
        self.shape_factor = shape_factor

        # ── Grid ──────────────────────────────────────────────────────────────
        self.x    = np.linspace(0, 1, nx)
        self.zeta = np.linspace(0, 1, nz)
        self.dx   = self.x[1]    - self.x[0]
        self.dz   = self.zeta[1] - self.zeta[0]
        self.XX, self.ZZ = np.meshgrid(self.x, self.zeta)   # (nz, nx)

        # ── Geometry ──────────────────────────────────────────────────────────
        if bed_nd is not None:
            self.b_tilde = np.asarray(bed_nd,     dtype=float)
            self.s_tilde = np.asarray(surface_nd, dtype=float)
        else:
            self.b_tilde = bed_amp * np.sin(2 * np.pi * self.x)
            self.s_tilde = 1.0 - 0.05 * self.x

        self.H_tilde = self.s_tilde - self.b_tilde          # (nx,)
        self.dHdx    = np.gradient(self.H_tilde, self.x)
        self.dbdx    = np.gradient(self.b_tilde,  self.x)   # ← bed slope (nd)

        # ── Velocity (unperturbed baseline) ───────────────────────────────────
        self._build_velocity(w_basal=0.0)

        # ── Macro timestep ────────────────────────────────────────────────────
        self.dt_macro = self._cfl_dt()
        self.dt       = self.dt_macro
        print(f"  Baseline CFL dt_macro = {self.dt_macro:.5f}  "
              f"(u_max={np.abs(self.u).max():.3f}, "
              f"w_max={np.abs(self.w).max():.4f})")

        # ── Age field + diagnostics ───────────────────────────────────────────
        self.X             = np.zeros((nz, nx))
        self.time_series   = []
        self.age_snapshots = []

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def from_geometry(cls, x_phys, bed_phys, surface_phys,
                      nx=150, nz=100,
                      u_mean=100.0, a_rate=0.5, Flush=1.0,
                      cfl_safety=0.4, n=3.0, shape_factor = 1.0):
        """
        Construct a SigmaAgeModel from *physical* geometry arrays.

        The physical arrays are resampled to nx points and nondimensionalised
        before being passed to __init__.

        Parameters
        ----------
        x_phys       : (N,) physical along-flow distance [m], ascending.
        bed_phys     : (N,) bed elevation [m a.s.l.].
        surface_phys : (N,) surface elevation [m a.s.l.].
        nx           : number of horizontal grid points.
        nz           : number of vertical sigma levels.
        u_mean       : mean horizontal ice velocity [m/yr].
        a_rate       : surface accumulation rate [m ice/yr].
        Flush        : nondim flux scale  (= u_mean * H_mean / (L * a_rate)).
                       If None, computed automatically from the geometry.
        cfl_safety   : CFL safety factor.
        n            : Glen's flow exponent (default 3.0).
        shape_factor : shape factor for horizontal velocity profile

        Returns
        -------
        SigmaAgeModel instance with physical metadata stored as attributes.
        """
        x_phys       = np.asarray(x_phys,       dtype=float)
        bed_phys     = np.asarray(bed_phys,      dtype=float)
        surface_phys = np.asarray(surface_phys,  dtype=float)

        # ── Resample to nx points ─────────────────────────────────────────────
        xi = np.linspace(0, 1, nx)
        t  = np.linspace(0, 1, len(x_phys))

        x_rs  = interp1d(t, x_phys,       kind='linear')(xi)
        b_rs  = interp1d(t, bed_phys,     kind='linear')(xi)
        s_rs  = interp1d(t, surface_phys, kind='linear')(xi)

        # ── Physical scales ───────────────────────────────────────────────────
        L      = x_rs[-1] - x_rs[0]                    # domain length [m]
        H_mean = np.mean(s_rs - b_rs)                  # mean thickness [m]

        # Auto-compute Flush if not provided
        if Flush is None:
            Flush = u_mean * H_mean / (L * a_rate)
            print(f"  Auto Flush = {Flush:.4f}  "
                  f"(u={u_mean} m/yr, H={H_mean:.1f} m, "
                  f"L={L:.1f} m, a={a_rate} m/yr)")


        # ── Nondimensionalise geometry ────────────────────────────────────────
        # x_nd in [0,1], bed and surface normalised by H_mean
        # (so that H_nd ~ 1 on average, consistent with the PDE scaling)
        b_nd = b_rs / H_mean
        s_nd = s_rs / H_mean

        # ── Build model ───────────────────────────────────────────────────────
        obj = cls(nx=nx, nz=nz, Flush=Flush,
                  bed_nd=b_nd, surface_nd=s_nd,
                  cfl_safety=cfl_safety, n=n)

        # ── Store physical metadata for plotting / age conversion ─────────────
        obj.x_phys       = x_rs          # (nx,) m
        obj.bed_phys     = b_rs          # (nx,) m
        obj.surface_phys = s_rs          # (nx,) m
        obj.L            = L             # m
        obj.H_mean       = H_mean        # m
        obj.u_mean       = u_mean        # m/yr
        obj.a_rate       = a_rate        # m/yr
        obj.has_geometry = True
        obj.Flush         = Flush
        obj.n            = n
        obj.shape_factor  = shape_factor

        return obj

    # ─────────────────────────────────────────────────────────────────────────
    def _cfl_dt(self):
        u_max = np.abs(self.u).max()
        w_max = np.abs(self.w).max() + 1e-12
        return self.cfl_safety * min(self.dx / u_max, self.dz / w_max)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_velocity(self, w_basal=0.0):
        """
        Build u and w_sigma from the Raymond velocity profile +
        incompressibility, with an explicit bed kinematic condition.

        Bed kinematic condition (sigma coords)
        ---------------------------------------
        In physical coords:
            w_phys(bed) = u_phys * db_phys/dx  +  m_dot

        Transforming to sigma:
            w_sigma(zeta=0) = [ w_phys - (ZZ*dH/dx + db/dx)*u / H ] / H * H
                            = m_dot / H_nd   (nondim)

        The bed-slope term  u * db/dx  is handled by the metric correction
        during upward integration of incompressibility.  Therefore:

            w_basal (argument) = m_dot_nd = m_dot / a_rate

        Positive w_basal → freeze-on (upward sigma velocity at bed).
        Negative w_basal → melt      (downward sigma velocity at bed).

        Parameters
        ----------
        w_basal : float or (nx,) array
            Nondim basal melt/freeze rate.  Bed-slope contribution is NOT
            included here — it is already in the metric term.
        """
        ZZ = self.ZZ

        def raymond_profile_uniform(n, zeta):
            """
            Analytical Raymond profile for vertical velocity over depth
            """
            omega = ((n+2)/(n+1)) * (1 - (1-ZZ)**(n+1)) \
              - (1/(n+1))     * (1 - (1-ZZ)**(n+2))
            return omega / omega[-1, 0]
        
        def vertical_profile(n, zeta):
            """
            vertical velocity profile, accomodating depth-dependent 
            Glen's flow parameter (n)
            we have to solve numerically

            w_s - w_0 = -\frac{d u_s}{d x} \int_H^z [1 - (z/H)^{n+1}] dz


            """
            return

        # ── Raymond shape function (n=3) ──────────────────────────────────────
        omega = raymond_profile_uniform(self.n, self.zeta)
        self.omega = omega

        self.u = self.Flush * self.shape_factor * omega          # (nz, nx)

        # ── Incompressibility RHS in sigma coords ─────────────────────────────
        # Full form:
        #   dw/dzeta = -du/dx|_sigma + [(ZZ*dH/dx + db/dx)/H] * du/dzeta
        #
        # du/dx|_sigma = 0 under parallel-flow assumption
        dudz    = np.gradient(self.u, self.zeta, axis=0)        # du/dzeta
        dHdx_2d = self.dHdx[np.newaxis, :]                      # (1, nx)
        dbdx_2d = self.dbdx[np.newaxis, :]                      # (1, nx)
        H_2d    = self.H_tilde[np.newaxis, :]                   # (1, nx)

        # Metric: corrects for tilted sigma surfaces
        metric   = (ZZ * dHdx_2d + dbdx_2d) / H_2d * dudz      # (nz, nx)
        dwdz_rhs = metric                                        # du/dx|_sigma = 0

        # ── Bed kinematic BC ──────────────────────────────────────────────────
        # w_sigma(zeta=0) = w_basal_nd (melt/freeze only; slope already in metric)
        w_bed = np.broadcast_to(
            np.atleast_1d(np.asarray(w_basal, dtype=float)),
            (self.nx,)
        ).copy()

        # ── Upward integration from bed ───────────────────────────────────────
        self.w      = np.zeros_like(self.u)
        self.w[0,:] = w_bed

        for k in range(1, self.nz):
            self.w[k, :] = self.w[k-1, :] + self.dz * dwdz_rhs[k-1, :]

        # ── Surface kinematic BC correction ───────────────────────────────────
        # After integration, w[-1,:] will not exactly equal -1 due to
        # discretisation error.  Apply a linear-in-zeta correction that:
        #   - leaves w[0,:]  = w_bed  unchanged  (zeta=0, alpha=0)
        #   - forces w[-1,:] = -1               (zeta=1, alpha=1)
        w_top_raw = self.w[-1, :].copy()
        for k in range(self.nz):
            self.w[k, :] += self.zeta[k] * (-1.0 - w_top_raw)

        self.w[-1, :] = -1.0    # safety pin

    # ─────────────────────────────────────────────────────────────────────────
    # Everything below is unchanged from original
    # ─────────────────────────────────────────────────────────────────────────

    def _upwind_step(self, X):
        u_neg = np.minimum(self.u, 0.0)
        u_pos = np.maximum(self.u, 0.0)

        fwd_x         = np.zeros_like(X)
        fwd_x[:, :-1] = (X[:, 1:] - X[:, :-1]) / self.dx
        fwd_x[:, -1]  = 0.0

        bwd_x         = np.zeros_like(X)
        bwd_x[:, 1:]  = (X[:, 1:] - X[:, :-1]) / self.dx
        bwd_x[:, 0]   = 0.0

        dXdx = u_pos * bwd_x + u_neg * fwd_x

        w_neg = np.minimum(self.w, 0.0)
        w_pos = np.maximum(self.w, 0.0)

        fwd_z         = np.zeros_like(X)
        fwd_z[:-1, :] = (X[1:, :] - X[:-1, :]) / self.dz
        fwd_z[-1,  :] = 0.0

        bwd_z         = np.zeros_like(X)
        bwd_z[1:,  :] = (X[1:, :] - X[:-1, :]) / self.dz
        bwd_z[0,   :] = 0.0

        dXdz = w_neg * fwd_z + w_pos * bwd_z

        return -dXdx - dXdz + 1.0

    def _apply_bcs(self, X):
        X[-1, :] = 0.0
        if self.Flush > 0:
            X[:,  0] = 0.0
            X[:, -1] = X[:, -2]
        elif self.Flush < 0:
            X[:, -1] = 0.0
            X[:,  0] = X[:,  1]
        else:
            X[:,  0] = X[:,  1]
            X[:, -1] = X[:, -2]
        return np.maximum(X, 0.0)

    def _advance_one_macro_step(self, X, t, dt_macro, basal_perturbation):
        if basal_perturbation is not None:
            wb = basal_perturbation(t)
            self._build_velocity(w_basal=wb)

        dt_cfl   = self._cfl_dt()
        n_sub    = max(1, int(np.ceil(dt_macro / dt_cfl)))
        dt_micro = dt_macro / n_sub

        for _ in range(n_sub):
            dX = self._upwind_step(X)
            X  = X + dt_micro * dX
            X  = self._apply_bcs(X)

        return X, n_sub, dt_micro

    def run(self, n_residence_times=6, basal_perturbation=None,
            snapshot_interval=None, verbose=True):
        t_total  = float(n_residence_times)
        nt_macro = int(t_total / self.dt_macro)
        X        = self._apply_bcs(self.X.copy())

        if basal_perturbation is None:
            self._build_velocity(w_basal=0.0)

        print(f"  Running {nt_macro} macro-steps  "
              f"(t_total={t_total:.1f}, "
              f"perturbation={'ON' if basal_perturbation else 'OFF'})")

        self.time_series.clear()
        self.age_snapshots.clear()
        max_sub_seen = 1

        for n in range(nt_macro):
            t = n * self.dt_macro

            if basal_perturbation is not None:
                X, n_sub, dt_micro = self._advance_one_macro_step(
                    X, t, self.dt_macro, basal_perturbation)
                max_sub_seen = max(max_sub_seen, n_sub)
            else:
                dX       = self._upwind_step(X)
                X        = X + self.dt_macro * dX
                X        = self._apply_bcs(X)
                n_sub    = 1
                dt_micro = self.dt_macro

            if snapshot_interval and (n % snapshot_interval == 0):
                self.time_series.append(t)
                self.age_snapshots.append(X.copy())

            if verbose and (n % max(1, nt_macro // 8) == 0):
                wb_now = basal_perturbation(t) if basal_perturbation else 0.0
                wb_str = (f"{np.mean(wb_now):.3f}"
                          if hasattr(wb_now, '__len__') else f"{wb_now:.3f}")
                print(f"    macro {n:6d}/{nt_macro}  t={t:.3f}  "
                      f"w_basal={wb_str}  sub={n_sub}  "
                      f"dt_micro={dt_micro:.6f}  "
                      f"max(X)={X.max():.3f}  mean(X)={X.mean():.3f}")

        self.X = X
        self.time_series.append(nt_macro * self.dt_macro)
        self.age_snapshots.append(X.copy())
        print(f"  Done.  Max sub-steps: {max_sub_seen}  "
              f"(dt_macro={self.dt_macro:.5f})")
        return X

    # ─────────────────────────────────────────────────────────────────────────
    def age_years(self):
        """
        Convert nondim age field to physical years.
        Requires model to have been built via from_geometry().
        """
        if not getattr(self, 'has_geometry', False):
            raise AttributeError("age_years() requires from_geometry() construction.")
        return self.X * self.H_mean / self.a_rate    # (nz, nx) years

    # ─────────────────────────────────────────────────────────────────────────
    def plot(self, ax_row=None, fig=None, title_prefix=""):
        own_fig = ax_row is None
        if own_fig:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        else:
            axes = ax_row

        X = self.X

        ax   = axes[0]
        vmax = np.percentile(X, 98)
        cf   = ax.contourf(self.XX, self.ZZ, X,
                           levels=np.linspace(0, vmax, 40), cmap='plasma',
                           extend='max')
        cs   = ax.contour(self.XX, self.ZZ, X,
                          levels=np.linspace(0, vmax, 10),
                          colors='white', linewidths=0.6, alpha=0.7)
        ax.clabel(cs, fmt='%.1f', fontsize=15, colors='white')
        plt.colorbar(cf, ax=ax, label=r'$\tilde{X} = Xa/H$')
        bed_norm = self.b_tilde / self.H_tilde
        ax.fill_between(self.x, 0, np.clip(bed_norm, -0.05, 0.05),
                        color='saddlebrown', alpha=0.5)
        ax.set_xlabel(r'$\tilde{x}$', fontsize=15);  
        ax.set_ylabel(r'$\zeta$', fontsize=15)
        ax.set_title(f'{title_prefix}\nAge Field  (Flush={self.Flush:.1f})')
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1)

        ax  = axes[1]
        cf2 = ax.contourf(self.XX, self.ZZ, self.u, levels=40, cmap='viridis')
        plt.colorbar(cf2, ax=ax, label=r'$\tilde{u}$ (horiz)')
        skip = (slice(None, None, 8), slice(None, None, 10))
        ax.quiver(self.XX[skip], self.ZZ[skip],
                  self.u[skip], self.w[skip] * 3,
                  color='white', scale=30, width=0.003, alpha=0.8)
        ax.set_xlabel(r'$\tilde{x}$');  ax.set_ylabel(r'$\zeta$')
        ax.set_title('Velocity Field')
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1)

        ax   = axes[2]
        cols = ['royalblue', 'tomato', 'seagreen', 'darkorange']
        for col, pos in zip(cols, [0.25, 0.50, 0.75, 0.90]):
            ix = int(pos * self.nx)
            ax.plot(X[:, ix], self.zeta, color=col, lw=2,
                    label=fr'$\tilde{{x}}={pos}$')
        ax.set_xlabel(r'$\tilde{X} = Xa/H$', fontsize=15)
        ax.set_ylabel(r'$\zeta$  (0=bed, 1=surface)', fontsize=15)
        ax.set_title('Age–Depth Profiles', fontsize=15)
        ax.legend(fontsize=15);  ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        if own_fig:
            plt.tight_layout()
            plt.savefig('sigma_age_model.png', dpi=150, bbox_inches='tight')
            plt.show()
