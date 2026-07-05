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