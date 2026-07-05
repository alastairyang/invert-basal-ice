import numpy as np
import matplotlib.pyplot as plt


class SigmaAgeModel:
    """
    Nondimensional ice age solver in sigma (terrain-following) coordinates.

    PDE:  dX/dt + u_tilde * dX/dx + w_sigma * dX/dzeta = 1

    Nondimensionalisation:
        x_tilde = x / L
        zeta    = (z - b(x)) / H(x)   in [0,1]
        t_tilde = t * a / H
        X_tilde = X * a / H
        u_tilde = u / (L*a/H)  =>  u_tilde_max = Pe at surface
        w_tilde = w_sigma / (a/H)

    Basal freeze-on perturbation:
        w_basal(x, t) is a nondim upward velocity at zeta=0.
        Positive values represent basal freeze-on (upward sigma velocity).
        This is passed as a callable to run():
            basal_perturbation(t) -> scalar or 1-D array of length nx

    BCs:
        X = 0  at zeta = 1  (surface: fresh snow)
        X = 0  at x = 0     (inflow divide: ice enters young)
        dX/dx = 0 at x = 1  (outflow: zero-gradient, ice exits freely)

    CFL:
        dt is recomputed after every velocity rebuild (_build_velocity).
        The outer time loop advances by a fixed macro-step dt_macro, but
        each macro-step is sub-divided into as many CFL-safe micro-steps
        as required by the current velocity field.  This guarantees
        stability under arbitrarily large or time-varying basal forcing
        without requiring the caller to tune anything.
    """

    def __init__(self, nx=150, nz=100, Flush=1.0, bed_amp=0.00, cfl_safety=0.4):
        self.nx         = nx
        self.nz         = nz
        self.Flush      = Flush
        self.bed_amp    = bed_amp
        self.cfl_safety = cfl_safety

        # ── Grid ──────────────────────────────────────────────────────────────
        self.x    = np.linspace(0, 1, nx)
        self.zeta = np.linspace(0, 1, nz)
        self.dx   = self.x[1]    - self.x[0]
        self.dz   = self.zeta[1] - self.zeta[0]
        self.XX, self.ZZ = np.meshgrid(self.x, self.zeta)   # (nz, nx)

        # ── Geometry ──────────────────────────────────────────────────────────
        self.b_tilde = bed_amp * np.sin(2 * np.pi * self.x)
        self.s_tilde = 1.0 - 0.05 * self.x
        self.H_tilde = self.s_tilde - self.b_tilde
        self.dHdx    = np.gradient(self.H_tilde, self.x)
        self.dbdx    = np.gradient(self.b_tilde,  self.x)

        # ── Velocity (unperturbed baseline) ───────────────────────────────────
        self._build_velocity(w_basal=0.0)

        # ── Macro timestep: CFL from baseline velocity ────────────────────────
        # dt_macro is the fixed outer step size used to advance "model time".
        # It is set once from the unperturbed (smallest expected) flow so that
        # the snapshot/verbose cadence is predictable.  During integration,
        # each macro-step is sub-divided if the perturbed velocity demands it.
        self.dt_macro = self._cfl_dt()
        print(f"  Baseline CFL dt_macro = {self.dt_macro:.5f}  "
              f"(u_max={np.abs(self.u).max():.3f}, "
              f"w_max={np.abs(self.w).max():.4f})")

        # Convenience alias so external code that reads model.dt still works
        self.dt = self.dt_macro

        # ── Age field ─────────────────────────────────────────────────────────
        self.X = np.zeros((nz, nx))

        # ── Diagnostic storage ────────────────────────────────────────────────
        self.time_series   = []
        self.age_snapshots = []

    # ─────────────────────────────────────────────────────────────────────────
    def _cfl_dt(self):
        """
        Compute the CFL-safe timestep from the *current* velocity field.

        Returns
        -------
        float
            dt = cfl_safety * min(dx / u_max, dz / w_max)
        """
        u_max = np.abs(self.u).max()
        w_max = np.abs(self.w).max() + 1e-12   # guard against zero
        return self.cfl_safety * min(self.dx / u_max, self.dz / w_max)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_velocity(self, w_basal=0.0):
        """
        Lliboutry horizontal profile + incompressibility-derived sigma velocity.

        Parameters
        ----------
        w_basal : float or 1-D array of shape (nx,)
            Nondimensional basal vertical velocity at zeta=0.
            Positive = upward (freeze-on pushes ice upward in sigma coords).
            Default 0.0 = no basal melt/freeze.
        """
        ZZ = self.ZZ

        # ── Lliboutry shape (p=3), normalised so omega(1)=1 ──────────────────
        p      = 3.0
        omega  = ((p+2)/(p+1)) * (1 - (1-ZZ)**(p+1)) \
            - (1/(p+1))     * (1 - (1-ZZ)**(p+2))
        omega /= omega[-1, 0]

        # Horizontal velocity: u_tilde = Flush * omega(zeta)
        self.u = self.Flush * omega                              # (nz, nx)

        # du/dx = 0 (parallel-flow assumption)
        dudx_sigma = np.zeros_like(self.u)

        # Metric correction term
        dudz    = np.gradient(self.u, self.zeta, axis=0)
        dHdx_2d = self.dHdx[np.newaxis, :]
        dbdx_2d = self.dbdx[np.newaxis, :]
        H_2d    = self.H_tilde[np.newaxis, :]
        metric  = (self.ZZ * dHdx_2d + dbdx_2d) / H_2d * dudz

        # ── Integrate w upward from bed ───────────────────────────────────────
        w_bed = np.broadcast_to(
            np.atleast_1d(np.asarray(w_basal, dtype=float)),
            (self.nx,)
        ).copy()

        dwdz_rhs    = -dudx_sigma + metric
        self.w      = np.zeros_like(self.u)
        self.w[0,:] = w_bed

        # Step 1: full upward integration using incompressibility RHS
        for k in range(1, self.nz):
            self.w[k, :] = self.w[k-1, :] + self.dz * dwdz_rhs[k-1, :]

        # Step 2: capture raw integrated surface value AFTER the loop
        w_top_raw = self.w[-1, :].copy()   # shape (nx,)

        # Step 3: linear-in-zeta correction to honour both BCs exactly
        #   zeta=0 : alpha=0 → no change        → w[0,:] = w_bed   ✓
        #   zeta=1 : alpha=1 → adds(-1-w_top)   → w[-1,:]= -1      ✓
        for k in range(self.nz):
            self.w[k, :] += self.zeta[k] * (-1.0 - w_top_raw)

        # Explicit safety pin (should already be satisfied to machine precision)
        self.w[-1, :] = -1.0

    def _upwind_step(self, X):
        """
        First-order upwind advection + source term.
        Handles both positive and negative Flush (u can be negative).
        """
        # ── Horizontal: split by sign of u ───────────────────────────────────
        # u_neg uses forward  difference (upwind when flow is leftward)
        # u_pos uses backward difference (upwind when flow is rightward)
        u_neg = np.minimum(self.u, 0.0)
        u_pos = np.maximum(self.u, 0.0)

        fwd_x          = np.zeros_like(X)
        fwd_x[:, :-1]  = (X[:, 1:] - X[:, :-1]) / self.dx   # forward in x
        fwd_x[:, -1]   = 0.0

        bwd_x          = np.zeros_like(X)
        bwd_x[:, 1:]   = (X[:, 1:] - X[:, :-1]) / self.dx   # backward in x
        bwd_x[:, 0]    = 0.0

        dXdx = u_pos * bwd_x + u_neg * fwd_x

        # ── Vertical: split by sign of w ─────────────────────────────────────
        w_neg = np.minimum(self.w, 0.0)
        w_pos = np.maximum(self.w, 0.0)

        fwd_z          = np.zeros_like(X)
        fwd_z[:-1, :]  = (X[1:, :] - X[:-1, :]) / self.dz
        fwd_z[-1,  :]  = 0.0

        bwd_z          = np.zeros_like(X)
        bwd_z[1:,  :]  = (X[1:, :] - X[:-1, :]) / self.dz
        bwd_z[0,   :]  = 0.0

        dXdz = w_neg * fwd_z + w_pos * bwd_z

        return -dXdx - dXdz + 1.0


    def _apply_bcs(self, X):
        """
        Boundary conditions, aware of flow direction set by sign of Flush.

        Flush > 0  (flow left → right):
            inflow  at x = 0  → X = 0 (fresh ice enters)
            outflow at x = 1  → zero-gradient
        Flush < 0  (flow right → left):
            inflow  at x = 1  → X = 0 (fresh ice enters)
            outflow at x = 0  → zero-gradient
        Flush = 0:
            no horizontal transport; apply zero-gradient both sides
        """
        # Surface always fresh
        X[-1, :] = 0.0

        if self.Flush > 0:
            X[:,  0]  = 0.0          # inflow at left
            X[:, -1]  = X[:, -2]     # outflow at right
        elif self.Flush < 0:
            X[:, -1]  = 0.0          # inflow at right
            X[:,  0]  = X[:,  1]     # outflow at left
        else:
            X[:,  0]  = X[:,  1]     # no flow: zero-gradient both sides
            X[:, -1]  = X[:, -2]

        return np.maximum(X, 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    def _advance_one_macro_step(self, X, t, dt_macro, basal_perturbation):
        """
        Advance the age field by exactly dt_macro in physical time, using
        as many CFL-safe micro-steps as the current velocity field requires.

        The velocity is rebuilt once at the *start* of the macro-step and
        held fixed for all micro-steps within it (operator-splitting style).
        This is consistent with the assumption that w_basal varies on the
        macro timescale, not the micro timescale.

        Parameters
        ----------
        X : ndarray (nz, nx)
            Age field at time t.
        t : float
            Current nondim time (start of macro-step).
        dt_macro : float
            Desired macro time increment.
        basal_perturbation : callable or None

        Returns
        -------
        X : ndarray (nz, nx)
            Age field at time t + dt_macro.
        n_sub : int
            Number of micro-steps taken.
        dt_micro : float
            Micro-step size used.
        """
        # ── Rebuild velocity at start of macro-step ───────────────────────────
        if basal_perturbation is not None:
            wb = basal_perturbation(t)
            self._build_velocity(w_basal=wb)

        # ── Compute CFL-safe micro-step for this velocity field ───────────────
        dt_cfl  = self._cfl_dt()
        n_sub   = max(1, int(np.ceil(dt_macro / dt_cfl)))
        dt_micro = dt_macro / n_sub          # exact subdivision, sums to dt_macro

        # ── Sub-step loop ─────────────────────────────────────────────────────
        for _ in range(n_sub):
            dX = self._upwind_step(X)
            X  = X + dt_micro * dX
            X  = self._apply_bcs(X)

        return X, n_sub, dt_micro

    # ─────────────────────────────────────────────────────────────────────────
    def run(self, n_residence_times=6, basal_perturbation=None,
            snapshot_interval=None, verbose=True):
        """
        Integrate the age equation forward in time.

        The outer loop advances by dt_macro (set at init from the baseline
        CFL condition).  Each outer step calls _advance_one_macro_step(),
        which internally sub-steps as many times as needed to satisfy the
        CFL condition for the *current* (possibly perturbed) velocity field.

        Parameters
        ----------
        n_residence_times : float
            Total nondim integration time (in units of H/a).
        basal_perturbation : callable or None
            f(t) -> scalar or array(nx,)  giving w_basal at time t.
            If None, velocity is built once and never updated.
        snapshot_interval : int or None
            Save X every this many *macro* steps.
            None = save only the final state.
        verbose : bool
        """
        t_total  = float(n_residence_times)
        nt_macro = int(t_total / self.dt_macro)

        X = self._apply_bcs(self.X.copy())

        # For unperturbed case build velocity once up front
        if basal_perturbation is None:
            self._build_velocity(w_basal=0.0)

        print(f"  Running {nt_macro} macro-steps  "
              f"(t_total={t_total:.1f} residence times, "
              f"perturbation={'ON' if basal_perturbation else 'OFF'})")

        self.time_series.clear()
        self.age_snapshots.clear()

        max_sub_seen = 1   # track worst-case sub-stepping for diagnostics

        for n in range(nt_macro):
            t = n * self.dt_macro

            if basal_perturbation is not None:
                # Full adaptive sub-stepping path
                X, n_sub, dt_micro = self._advance_one_macro_step(
                    X, t, self.dt_macro, basal_perturbation
                )
                max_sub_seen = max(max_sub_seen, n_sub)
            else:
                # Unperturbed: velocity fixed, single micro-step = macro-step
                dX = self._upwind_step(X)
                X  = X + self.dt_macro * dX
                X  = self._apply_bcs(X)
                n_sub    = 1
                dt_micro = self.dt_macro

            # ── Snapshots ─────────────────────────────────────────────────────
            if snapshot_interval and (n % snapshot_interval == 0):
                self.time_series.append(t)
                self.age_snapshots.append(X.copy())

            if verbose and (n % max(1, nt_macro // 8) == 0):
                wb_now = basal_perturbation(t) if basal_perturbation else 0.0
                wb_str = (f"{np.mean(wb_now):.3f}"
                          if hasattr(wb_now, '__len__') else f"{wb_now:.3f}")
                print(f"    macro {n:6d}/{nt_macro}  t={t:.3f}  "
                      f"w_basal={wb_str}  sub-steps={n_sub}  "
                      f"dt_micro={dt_micro:.6f}  "
                      f"max(X)={X.max():.3f}  mean(X)={X.mean():.3f}")

        self.X = X
        self.time_series.append(nt_macro * self.dt_macro)
        self.age_snapshots.append(X.copy())

        print(f"  Done.  Max sub-steps used in any macro-step: {max_sub_seen}  "
              f"(dt_macro={self.dt_macro:.5f})")
        return X

    # ─────────────────────────────────────────────────────────────────────────
    def plot(self, ax_row=None, fig=None, title_prefix=""):
        """Plot age field, velocity field, and age-depth profiles."""
        own_fig = ax_row is None
        if own_fig:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        else:
            axes = ax_row

        X = self.X

        # ── Panel 1: Age field ────────────────────────────────────────────────
        ax   = axes[0]
        vmax = np.percentile(X, 98)
        cf   = ax.contourf(self.XX, self.ZZ, X,
                           levels=np.linspace(0, vmax, 40), cmap='plasma',
                           extend='max')
        cs   = ax.contour(self.XX, self.ZZ, X,
                          levels=np.linspace(0, vmax, 10),
                          colors='white', linewidths=0.6, alpha=0.7)
        ax.clabel(cs, fmt='%.1f', fontsize=7, colors='white')
        plt.colorbar(cf, ax=ax, label=r'$\tilde{X} = Xa/H$')
        bed_norm = self.b_tilde / self.H_tilde
        ax.fill_between(self.x, 0, np.clip(bed_norm, -0.05, 0.05),
                        color='saddlebrown', alpha=0.5)
        ax.set_xlabel(r'$\tilde{x}$');  ax.set_ylabel(r'$\zeta$')
        ax.set_title(f'{title_prefix}\nAge Field  (Flush={self.Flush:.1f})')
        ax.set_xlim(0, 1);  ax.set_ylim(0, 1)

        # ── Panel 2: Horizontal velocity + quiver ─────────────────────────────
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

        # ── Panel 3: Age-depth profiles ───────────────────────────────────────
        ax   = axes[2]
        cols = ['royalblue', 'tomato', 'seagreen', 'darkorange']
        for col, pos in zip(cols, [0.25, 0.50, 0.75, 0.90]):
            ix = int(pos * self.nx)
            ax.plot(X[:, ix], self.zeta, color=col, lw=2,
                    label=fr'$\tilde{{x}}={pos}$')
        ax.set_xlabel(r'$\tilde{X} = Xa/H$')
        ax.set_ylabel(r'$\zeta$  (0=bed, 1=surface)')
        ax.set_title('Age–Depth Profiles')
        ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        if own_fig:
            plt.tight_layout()
            plt.savefig('sigma_age_model.png', dpi=150, bbox_inches='tight')
            plt.show()
