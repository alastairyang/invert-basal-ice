import matplotlib.pyplot as plt
import numpy as np
from src.utilities import inverse_transform_geometry

class Plotter:
    def __init__(self, model, FLUSH, u_surf):
        self.model = model
        self.FLUSH = FLUSH # flushing parameter
        self.u_surf = u_surf # surface velocity (m/a)

        self._get_velocity_field()

        # vel field: none by default

    def get_physical_coordinates(self, s, b, distance):
        """
        Load physical coordinates for plotting in the along-flow direction
        and derive the geometry in the physical space.
        """
        # check that x, z, and distance are 1d
        if not (s.ndim == 1 and b.ndim == 1 and distance.ndim == 1):
            raise ValueError("s, b, and distance must be 1-dimensional arrays.")
        self.s = s
        self.b = b
        self.distance = distance
        self.L_hat    =  np.max(distance) - np.min(distance)  # horizontal length scale
        self.H_hat    = np.mean(s - b)  # characteristic ice thickness
        self.adot_hat = self.u_surf * self.H_hat / (self.L_hat * self.FLUSH)

        self.X_phys, self.Z_phys = inverse_transform_geometry(
            self.model.XX, self.model.ZZ, self.s, self.b, self.distance
        )
        return
    
    def physical_age_field(self):
        """
        Transform the non-dimensional age field to physical coordinates.
        """
        self.age_phys = self.model.X * self.H_hat / self.adot_hat
        return self.age_phys
    
    def _ice_mask_(self):
        """
        Create a mask for the ice domain
        """
        # if self.age_phys is None:
        # run the function first
        if not hasattr(self, 'age_phys'):
            self.physical_age_field()

        z_bed_phys     = self.Z_phys[0,  :]
        z_surface_phys = self.Z_phys[-1, :]
        age_phys_masked = np.ma.masked_where(
            (self.Z_phys < z_bed_phys[np.newaxis, :]) |
            (self.Z_phys > z_surface_phys[np.newaxis, :]),
            self.age_phys
        )
        self.age_mask = age_phys_masked
        return self.age_mask

    def load_forcing(self, forcing, time = None, n_sample = 5):
        """
        Load the forcing data for plotting.
        Forcing should be a complete spatiotemporal field (2D), or 
        a snapshot (1D).

        Parameters
        ----------
        forcing : 1D or 2D array
            The nondimensional forcing data to be loaded.
            If 2D, the first dimension is time and the second dimension is space.
        n_sample : int, optional
            Number of samples to take if forcing is 2D. Default is 5.
        """
        if forcing is None:
            self.forcing = np.zeros((self.model.nx,))  # default to zero forcing
            self.forcing_time = None
            print("No forcing data provided.")
            return

        if forcing.ndim == 2:
            # take n_sample evenly spaced snapshots along time dimension
            n_sample = min(n_sample, forcing.shape[0])
            print(f"Sampling {n_sample} snapshots from the forcing field.")
            indices = np.linspace(0, forcing.shape[0] - 1, n_sample, dtype=int)
            self.forcing = forcing[indices, :]
            self.forcing_time = time[indices] if time is not None else None
            # convert time to physical time
            self.forcing_time = self.forcing_time / (self.adot_hat / self.H_hat)

            print(f"Loaded a 2D forcing field with {forcing.shape[0]} time steps. ")
            print(f"Sampling {n_sample} snapshots at indices: {indices}.")
        else:
            self.forcing      = forcing
            self.forcing_time = None
            print(f"Loaded a 1D forcing field.")

        # forcing back to physical unit
        self.forcing = self.forcing * self.adot_hat
        return
    
    def _get_velocity_field(self):
        """
        Get velocity field for quiver plot later.
        """
        if hasattr(self, 'model'):
            self.u = self.model.u
            self.w = self.model.w
        return

        
    def plot_age(self, flip_lr=False, arrow_scale=40, overlay_x=None, overlay_y=None):
        if flip_lr:  # depend on the flow direction and visual consistency
            dist_raw     = self.distance
            dist_raw     = np.fliplr(dist_raw.reshape(1, -1)).ravel()
            X_phys       = np.fliplr(self.X_phys)
            x_ph_contour = self.X_phys[0, :]
            x_ph_contour = np.fliplr(x_ph_contour.reshape(1, -1)).ravel()
            u = -self.u
        else:
            dist_raw     = self.distance
            X_phys       = self.X_phys
            x_ph_contour = self.X_phys[0, :]
            u            = self.u

        if not hasattr(self, 'age_mask'):
            age_phys_masked = self._ice_mask_()
        else:
            age_phys_masked = self.age_mask

        # ── Colour scale ──────────────────────────────────────────────────────────────
        vmin_yr, vmax_yr = 0, np.percentile(age_phys_masked, 98)
        levels_yr = np.linspace(vmin_yr, vmax_yr, 30)

        # ── Figure layout ─────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(10, 7))
        gs  = fig.add_gridspec(
            2, 1,
            height_ratios=[1, 4],
            hspace=0.05,
            left=0.09, right=0.82, top=0.91, bottom=0.09
        )
        ax_wb = fig.add_subplot(gs[0, 0])
        ax_ph = fig.add_subplot(gs[1, 0])

        fig.suptitle(
            rf"Age Field  |  $\phi_x / \phi_z$ = {self.FLUSH:.2f}  |  ",
            fontsize=14, fontweight='bold'
        )

        # Rock below bed — brown fill between bottom of plot and bed profile
        ax_ph.fill_between(
            dist_raw,
            self.b,
            ax_ph.get_ylim()[0] if ax_ph.get_ylim()[0] < 0 else -50,
            color="#AA8B72", alpha=0.5, zorder=1
        )

        # Ice age contourf
        cf_ph = ax_ph.contourf(
            X_phys, self.Z_phys,
            age_phys_masked,
            cmap='plasma', alpha=0.7, zorder=2
        )

        ax_ph.plot(dist_raw, self.b,
                'k-',  lw=1.5, label='Bed',     zorder=4)
        ax_ph.plot(dist_raw, self.s,
                'k--', lw=1.5, label='Surface', zorder=4)

        # White above surface — fill between surface and top of plot
        ax_ph.fill_between(
            dist_raw,
            self.s,
            self.s.max() * 1.05,
            color='white', alpha=1.0, zorder=3
        )
        ax_ph.set_facecolor('#8B6343')
        ax_ph.set_xlabel('Distance (m)', fontsize=15)
        ax_ph.set_ylabel('Elevation (m)', fontsize=15)
        ax_ph.legend(loc='lower right', fontsize=15)

        x_min_f = x_ph_contour.min()
        x_max_f = x_ph_contour.max()
        y_min   = np.min(self.b) - 50
        y_max   = np.max(self.s) + 50

        ax_ph.set_xlim(x_min_f, x_max_f)
        ax_ph.set_ylim(y_min, y_max)
        ax_ph.fill_between(
            dist_raw,
            self.b,
            y_min,
            color="#AA8B72", alpha=0.7, zorder=1
        )

        cbar_ax = fig.add_axes([0.84, 0.09, 0.02, 0.60])
        fig.colorbar(cf_ph, cax=cbar_ax, label='Age (years)')

        # ── Quiver ────────────────────────────────────────────────────────────────────
        skip = (slice(None, None, 12), slice(None, None, 10))
        ax_ph.quiver(X_phys[skip], self.Z_phys[skip],
                    u[skip], self.w[skip] * 1,
                    color='white', scale=arrow_scale, width=0.005, alpha=0.8)
        ax_ph.set_xlabel(r'$x$ (m)', fontsize=15)
        ax_ph.set_ylabel(r'$z$ (m)', fontsize=15)

        # ── Overlay panel: leftmost 10% of ax_ph, in figure coordinates ───────────────
        if overlay_x is not None and overlay_y is not None:
            # Flush any pending layout so ax_ph.get_position() is accurate
            fig.canvas.draw()

            pos = ax_ph.get_position()          # Bbox in figure-fraction coords
            ov_left   = pos.x0
            ov_bottom = pos.y0
            ov_width  = pos.width * 0.10        # 10 % of the main panel width
            ov_height = pos.height

            ax_ov = fig.add_axes(
                [ov_left, ov_bottom, ov_width, ov_height],
                zorder=5                        # sit on top of ax_ph
            )

            ax_ov.plot(overlay_x, overlay_y, color='black', lw=1.5)

            # Transparent background so the age field shows through
            ax_ov.patch.set_alpha(0.8)
            ax_ov.set_facecolor('white')

            # Hide all ticks and tick labels
            ax_ov.set_xticks([])
            ax_ov.set_yticks([])

            # Black frame around the overlay panel
            for spine in ax_ov.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(0.8)

        # ── Top panel — basal perturbation ────────────────────────────────────────────
        if self.forcing.ndim == 1:
            n_forcing_tstep = 1
        else:
            n_forcing_tstep = self.forcing.shape[0]

        if n_forcing_tstep > 1:
            cmap_f = plt.cm.viridis
            colors = [cmap_f(i / (n_forcing_tstep - 1)) for i in range(n_forcing_tstep)]
            print("shape of x_ph_contour:", x_ph_contour.shape)
            print("shape of forcing:",      self.forcing.shape)
            for i in range(n_forcing_tstep):
                time = self.forcing_time[i]
                ax_wb.plot(
                    x_ph_contour,
                    self.forcing[i],
                    color=colors[i],
                    lw=1.2,
                    label=f't = Year {time:.1f}' if n_forcing_tstep <= 10 else None
                )
        else:
            ax_wb.plot(
                x_ph_contour,
                self.forcing,
                color='blue',
                lw=1.5,
                label='forcing'
            )

        ax_wb.axhline(0, color='k', lw=0.8)
        ax_wb.set_ylabel(r'$w_b$ (m/a)', fontsize=15)
        ax_wb.tick_params(labelsize=15)
        ax_wb.set_title('Basal perturbation', fontsize=12)
        plt.setp(ax_wb.get_xticklabels(), visible=False)




# # ─────────────────────────────────────────────────────────────────────────────
# # Figure 2: Difference map  ΔX = X_perturbed − X_control
# # ─────────────────────────────────────────────────────────────────────────────
# dX_diff = model_pert.X - model_ctrl.X
# vabs    = np.percentile(np.abs(dX_diff), 98)

# fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
# fig2.suptitle(r'Age Anomaly  $\Delta\tilde{X}$ = Perturbed $-$ Control',
#               fontsize=13, fontweight='bold')

# ax = axes2[0]
# cf = ax.contourf(model_ctrl.XX, model_ctrl.ZZ, dX_diff,
#                  levels=np.linspace(-vabs, vabs, 41), cmap='RdBu_r',
#                  extend='both')
# ax.contour(model_ctrl.XX, model_ctrl.ZZ, dX_diff,
#            levels=[0], colors='k', linewidths=1.2)
# plt.colorbar(cf, ax=ax, label=r'$\Delta\tilde{X}$')
# ax.set_xlabel(r'$\tilde{x}$');  ax.set_ylabel(r'$\zeta$')
# ax.set_title('Age Anomaly (2-D)')
# ax.set_xlim(0, 1);  ax.set_ylim(0, 1)

# # Age-depth profiles of anomaly at four x-positions
# ax   = axes2[1]
# cols = ['royalblue', 'tomato', 'seagreen', 'darkorange']
# for col, pos in zip(cols, [0.25, 0.50, 0.75, 0.90]):
#     ix = int(pos * model_ctrl.nx)
#     ax.plot(dX_diff[:, ix], model_ctrl.zeta, color=col, lw=2,
#             label=fr'$\tilde{{x}}={pos}$')
# ax.axvline(0, color='k', lw=0.8, ls='--')
# ax.set_xlabel(r'$\Delta\tilde{X}$')
# ax.set_ylabel(r'$\zeta$  (0=bed, 1=surface)')
# ax.set_title('Age Anomaly Profiles')
# ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
# ax.set_ylim(0, 1)

# plt.tight_layout()
# plt.savefig('../fig/basal_perturbation_anomaly.png', dpi=150, bbox_inches='tight')
# plt.show()