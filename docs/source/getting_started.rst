Getting Started
===============

qdiffusivity provides kernel density estimator (KDE) tools for transverse
number-density profiles of nanoconfined molecular dynamics simulations,
built on MDAnalysis.

Density profiles with the Epanechnikov KDE
------------------------------------------

The :class:`qdiffusivity.TransverseDensityKDE` analysis class pools
per-frame positions of an :class:`~MDAnalysis.core.groups.AtomGroup` along
the confined axis and evaluates an Epanechnikov-kernel KDE on a uniform
grid spanning the confined region.  Kernel mass that would leak beyond the
boundaries is folded back by mirror reflection, so the profile is
artefact-free at the walls.

.. code-block:: python

    import MDAnalysis as mda
    from qdiffusivity import TransverseDensityKDE

    u = mda.Universe("topology.data", "trajectory.xtc")
    ag = u.select_atoms("type 1 2")  # water atoms

    kde = TransverseDensityKDE(
        ag,
        dim=2,
        z_bot=10.0,
        z_top=90.0,
        n_points=400,
        grouping="residues",
        bandwidth="auto",
    )
    kde.run()

    # Number density (particles / Å^3):
    #   n(z) = (N_total / (n_frames_used * Lx * Ly)) * rho
    import numpy as np
    Lx, Ly = u.dimensions[:2]
    n_density = (kde.n_total /
                 (kde.n_frames_used * Lx * Ly)) * kde.rho

The class is an :class:`~MDAnalysis.analysis.base.AnalysisBase`
subclass, so the usual ``run(start, stop, step)`` interface applies.

Low-level KDE utilities
-----------------------

The building blocks are also exposed for direct use on pooled position
arrays:

.. code-block:: python

    from qdiffusivity import (
        epanechnikov_kernel,
        kde_1d,
        select_bandwidth,
    )

    z_pooled = np.array([...])  # pooled per-frame positions
    z_eval = np.linspace(z_bot, z_top, 400)
    h = select_bandwidth(z_pooled, z_bot, z_top, method="auto")
    rho, n_eff = kde_1d(z_pooled, z_eval, h, z_bot, z_top)

See the :doc:`api` for full reference.

Diffusivity profiles with the KDE local estimator
--------------------------------------------------

The :class:`qdiffusivity.TransverseDiffusivityKDE` analysis class
estimates the perpendicular (transverse) and parallel diffusivities as a
function of position along the confined axis.  It works in
*CDF-uniformised* u-space, where the equilibrium measure is uniform so a
single global bandwidth is appropriate across the whole gap (including
near the walls).  The perpendicular estimator uses the z-space local
estimator :math:`(\Delta z)^2/(2\Delta t)`, kernel-weighted in u-space;
the parallel estimator uses
:math:`(\Delta x^2+\Delta y^2)/(4\Delta t)`, kernel-weighted by the
starting position in u-space.  Kernel mass leaking beyond
:math:`u \in [0, 1]` is folded back by mirror reflection.

.. code-block:: python

    import MDAnalysis as mda
    from qdiffusivity import TransverseDiffusivityKDE

    u = mda.Universe("topology.data", "trajectory.xtc")
    ag = u.select_atoms("type 1 2")  # water atoms

    kde = TransverseDiffusivityKDE(
        ag,
        dim=2,
        n_points=200,
        bandwidth="auto",
        kernel="gaussian",
    )
    kde.run()

    # D_perp, D_para are in Å²/ps if the trajectory dt is in ps.
    # Mask poorly-sampled regions using the Kish effective sample size:
    valid = kde.n_eff_perp > 5

Both the Gaussian (infinite support, smooth) and Epanechnikov (compact
support, no leakage) kernels are available via ``kernel="gaussian"`` or
``kernel="epanechnikov"``.  The class is an
:class:`~MDAnalysis.analysis.base.AnalysisBase` subclass, so the usual
``run(start, stop, step)`` interface applies.

Itô correction
~~~~~~~~~~~~~~~

The perpendicular local estimator :math:`(\Delta z)^2/(2\Delta t)` carries
an :math:`O(\Delta t)` Itô bias
:math:`\frac{\Delta t}{2}\Phi(z)^2` where
:math:`\Phi = D(z)\,\rho'(z)/\rho(z)` in the isothermal
(Hänggi–Klimontovich) convention.  In wall-bound geometries with
adsorption layers this bias is *self-suppressing* (the :math:`D^2`
prefactor and the anti-correlation of :math:`D` with
:math:`|V'| = |\rho'/\rho|` make it small — a few % at the walls,
negligible in bulk), so it is **off by default**.  To subtract it
explicitly, pass ``ito_correction=True``:

.. code-block:: python

    kde = TransverseDiffusivityKDE(
        ag, dim=2, n_points=200, ito_correction=True,
    )
    kde.run()
    # kde.ito_bias holds the subtracted (Δt/2) Φ² array; kde.D_perp
    # is the bias-corrected perpendicular diffusivity (clipped >= 0).

The parallel estimator has **zero** Itô bias (no parallel drift) and is
unaffected by this option.