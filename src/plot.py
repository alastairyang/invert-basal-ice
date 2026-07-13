import matplotlib.pyplot as plt
import numpy as np
from src.utilities import inverse_transform_geometry

class Plotter:
    def __init__(self, model, FLUSH, u_surf):
        self.model = model
        self.FLUSH = FLUSH # flushing parameter
        self.u_surf = u_surf # surface velocity (m/a)

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
            The forcing data to be loaded.
            If 2D, the first dimension is time and the second dimension is space.
        n_sample : int, optional
            Number of samples to take if forcing is 2D. Default is 5.
        """
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
        return
    
    def plot_age(self, flip_lr=False):
        if flip_lr: # depend on the flow direction and visual consistency
            dist_raw     = self.distance                        
            dist_raw   = np.fliplr(dist_raw.reshape(1, -1)).ravel()       
            X_phys     = np.fliplr(self.X_phys)                                  
            x_ph_contour = self.X_phys[0, :]                                        
            x_ph_contour = np.fliplr(x_ph_contour.reshape(1, -1)).ravel()   

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
            f"Age Field  |  Flush={self.FLUSH}  |  ",
            fontsize=12, fontweight='bold'
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
        ax_ph.set_facecolor('#8B6343')   # rock color for any remaining gaps
        ax_ph.set_xlabel('Distance (m)')
        ax_ph.set_ylabel('Elevation (m)')
        ax_ph.legend(loc='lower right', fontsize=9)

        x_min_f = x_ph_contour.min()
        x_max_f = x_ph_contour.max()
        y_min   = np.min(self.b) - 50
        y_max   = np.max(self.s) + 50

        ax_ph.set_ylim(y_min, y_max)
        ax_ph.fill_between(
            dist_raw,
            self.b,
            y_min,
            color="#AA8B72", alpha=0.7, zorder=1
        )
        cbar_ax = fig.add_axes([0.84, 0.09, 0.02, 0.60])
        fig.colorbar(cf_ph, cax=cbar_ax, label='Age (years)')

        # ─────────────────────────────────────────────────────────────────────────────
        # Top — perturbation bar
        # ─────────────────────────────────────────────────────────────────────────────
        if self.forcing.ndim == 1:
            n_forcing_tstep = 1
        else:
            n_forcing_tstep = self.forcing.shape[0]
            
        if n_forcing_tstep > 1:
            # Use a colormap to assign sequential colors to each time step
            cmap = plt.cm.viridis
            colors = [cmap(i / (n_forcing_tstep - 1)) for i in range(n_forcing_tstep)]
            print("shape of x_ph_contour:", x_ph_contour.shape)
            print("shape of forcing:", self.forcing.shape)
            for i in range(n_forcing_tstep):
                time = self.forcing_time[i]
                ax_wb.plot(
                    x_ph_contour,
                    self.forcing[i],
                    color=colors[i],
                    lw=1.2,
                    label=f't = Year {time:.1f}' if n_forcing_tstep <= 10 else None  # avoid legend clutter
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
        ax_wb.set_ylabel(r'$\tilde{w}_b$', fontsize=9)
        ax_wb.tick_params(labelsize=8)

        # Only add legend + colorbar if step count is manageable
        if n_forcing_tstep <= 10:
            ax_wb.legend(fontsize=7, loc='upper right', framealpha=0.6)
        else:
            # Add a colorbar to represent time progression
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_forcing_tstep - 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_wb, orientation='vertical', pad=0.02)
            cbar.set_label('Time step', fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        ax_wb.set_title('Basal perturbation', fontsize=9)
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