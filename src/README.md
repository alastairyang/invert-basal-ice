# Aging equation
Ice sheet aging equation in the physical (x,z) is commonly written as

$$ \frac{\partial X}{\partial t} + u(x,z) \frac{\partial X}{\partial x} + w(x,z) \frac{\partial X}{\partial z} = 1 $$

Let the surface topography and bed topography be denoted as $s(x,z)$ and $b(x,z)$ respectively. The PDE is subject to the boundary condition at the ice surface

$$X(s,t) = 0$$

We assume the Raymond profile for the vertical velocity $w$

$$w(x,z) = w_s\left\{1 - \frac{z}{H(x)}\left[\frac{n+2}{n+1} - \frac{1}{n+1}\left(\frac{z}{H(x)} \right)^{n+1}  \right] \right\}$$
where $n$ is Glen's flow law exponent and $w_s$ is vertical velocity at the ice surface. The horizontal velocity is modeled with a simple scaling argument

$$u(x,z) = \frac{u_s H}{a L}w(x,z) $$

> Theoretically, in areas of large thickness change (e.g. Eqip glacier), the Raymond velocity profile is no longer valid. Assuming $\frac{\partial w}{\partial x} =0$, we model the velocity field explicitly (Nye):
> $$ u(x,z) = u_s(x,z) - \frac{2A(T,\phi)}{n+1}\left(\rho g \alpha \right)^n z^{n+1} $$
> $$ w(x,z) = \int_b^s \frac{\partial u}{\partial x}dz $$
> $$ w(x,s) = w_s $$
> where $A(T,\phi)$ is temperature- and impurity-dependent ice fluidity, $\alpha$ is surface slope. 

> As a matter of fact, when freeze-on or transient slip takes place, even $\frac{\partial w}{\partial x}=0$ is not a good assumption any more. One needs to solve the stokes-kind equation for the stress balance in the x-z plane: 
> $$\nabla \cdot \boldsymbol{\sigma} = \rho \boldsymbol{g}$$
> where $\boldsymbol{\sigma} = [\sigma_{xx},\sigma_{xz}; \sigma_{zx}, \sigma_{zz}]$ and $\sigma_{ij} = p\boldsymbol{I}-\tau_{ij}$ and $\tau_{ij} = (\mu\epsilon_{ij})^{1/n}$. The most assumption that one can make is $\sigma_{xx}=0$ if we assume $\epsilon_{xx} = \frac{\partial u}{\partial x}=0$, leading to Blatter-Pattyn approximation (HO). 


# Non-dimensionalization 
To non-dimensionalize the equation, we adopt the $\sigma$ (terrain-following) coordinate system (Hindmarsh, 2001; Vieli et al., 2007). We write using $\zeta \in [0,1]$ that

$$\zeta(x,z) = \frac{z-b(x,z)}{s(x,z)-b(x,z)}$$

In addition, we introduce the following nondimensional groups

$$\mathcal{F} = \frac{\hat{u}H}{\hat{a}L},$$

$$\tilde{u} = u\left(\frac{H}{\mathcal{F}\hat{a}L}\right), \,\,\tilde{w} = \frac{w}{\hat{a}},$$

$$\tilde{x} = \frac{x}{L},$$

$$\tilde{t} = t\frac{\hat{a}}{H}, \,\,\tilde{X} = X\frac{\hat{a}}{H}$$

where we call $\mathcal{F}$ flushing parameter, a ratio between the residence time in the vertical axis (accumulation/melting) over the residence time in the horizontal axis (horizontal ice flow). $\hat{a}$ is characteristic accumulation/melting rate, $\hat{u}$ is characteristic horizontal flow velocity, $H$ is characteristic ice thickness, and $L$ is characteristic horizontal length scale. 


# References
Vieli et al., 2007. Three-dimensional flow influences on radar layer stratigraphy. Journal of Glaciology

Hindmarsh, R.C.A. 2001. Notes on basic glaciological computational methods and algorithms.Continuum mechanics and
applications in geophysics and the environment. Berlin, etc.,
Springer-Verlag, 222–249