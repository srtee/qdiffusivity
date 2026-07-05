r"""Kernel density estimator for transverse number-density profiles.

This module provides an Epanechnikov-kernel 1-D KDE with mirror-reflection
boundary handling and a Sheather-Jones plug-in bandwidth, packaged as an
:class:`~MDAnalysis.analysis.base.AnalysisBase` class so it can be applied
to any :class:`~MDAnalysis.core.groups.AtomGroup` along the confined axis of
a nanoconfined simulation.

The reusable KDE machinery (bandwidth selection, kernel evaluation, boundary
mirroring, Kish effective sample size) is generalised from the per-project
script ``zn-el/analysis/dens_kde/kde_density.py``; species selection, residue
COM pooling, mass-density conversion and plotting glue are left to the
caller.

Theory
------
For evaluation point :math:`z_0` and pooled samples :math:`\{z_j\}_{j=1}^{N}`,

.. math::

   \hat\rho(z_0) = \frac{1}{N}\sum_j K_h(z_0 - z_j),

with the Epanechnikov kernel

.. math::

   K_h(x) = \frac{3}{4h}\Bigl(1 - (x/h)^2\Bigr)\;\; \text{for } |x| < h,
   \quad 0 \text{ otherwise}.

:math:`\hat\rho` integrates to 1 over the unbounded support.

* **Bandwidth.**  ``"auto"`` uses a Sheather-Jones plug-in (an Epanechnikov
  pilot density is built on a fine grid, its second derivative estimated by
  central differences, and the oracle bandwidth
  :math:`h^* = (\|K\|^2 / (N\,\mu_2(K)^2\,\widehat{\int[\hat f'']^2}))^{1/5}`
  evaluated with the Epanechnikov constants :math:`\|K\|^2 = 3/5`,
  :math:`\mu_2(K) = 1/5`); the Silverman rule of thumb is the fallback.
  ``"silverman"`` uses the rule of thumb directly, and a float fixes the
  bandwidth.

* **Boundary handling.**  Particles cannot cross the confined-region
  boundaries, so kernel mass that would leak beyond :math:`[z_{\mathrm{bot}},
  z_{\mathrm{top}}]` is mirrored back inside.  For each sample at :math:`z_j`,
  mirror copies at :math:`2z_{\mathrm{bot}} - z_j` and
  :math:`2z_{\mathrm{top}} - z_j` are added before kernel evaluation.

* **Effective sample size.**  The Kish effective sample size
  :math:`N_{\mathrm{eff}}(z_0) = (\sum_j K_h)^2 / \sum_j K_h^2` is returned at
  each grid point so unreliable regions (small local sample size) can be
  masked.

* **Evaluation grid.**  ``n_points`` cell-centred points uniformly across the
  confined region, :math:`z_m = z_{\mathrm{bot}} + (m + 0.5)(z_{\mathrm{top}} -
  z_{\mathrm{bot}})/M` for :math:`m = 0, \ldots, M-1`.

Converting the normalised KDE :math:`\hat\rho` (1/length, integrates to 1) to a
number density is the caller's responsibility; for a simulation with
cross-sectional area :math:`A` and :math:`N_f` analysis frames,

.. math::

   n(z) = \frac{N_{\mathrm{total}}}{N_f\,A}\,\hat\rho(z),

and a mass density follows by multiplying by :math:`M_{\mathrm{mol}}/N_A`
(with the appropriate :math:`10^{24}` factor for Å → cm).
"""

from __future__ import annotations

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase

# Epanechnikov kernel constants used by the Sheather-Jones plug-in.
#   ||K||^2 = int K(u)^2 du = 3/5
#   mu_2(K) = int u^2 K(u) du = 1/5
_EPA_KERNEL_NORM_SQ = 3.0 / 5.0
_EPA_KERNEL_MU2 = 1.0 / 5.0
_EPA_KERNEL_MU2_SQ = _EPA_KERNEL_MU2**2


def silverman_bw(z_data):
    r"""Silverman's rule of thumb.

    Returns :math:`h = 1.06\,\hat\sigma\,N^{-1/5}` with the robust scaled-IQR
    scale :math:`\hat\sigma = \min(\mathrm{std}, \mathrm{IQR}/1.34)` to guard
    against heavy-tailed distributions.  This is a Gaussian-reference rule of
    thumb used as a fallback for the plug-in bandwidth; the KDE itself uses
    the Epanechnikov kernel.

    Parameters
    ----------
    z_data : array_like
        1-D sample of positions.

    Returns
    -------
    float
        Bandwidth ``h`` (same units as ``z_data``).
    """
    z = np.asarray(z_data, dtype=float).ravel()
    n = z.size
    if n < 2:
        return 0.1
    std = float(np.std(z, ddof=1))
    q75, q25 = np.percentile(z, [75, 25])
    iqr = q75 - q25
    sigma_hat = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma_hat <= 0:
        sigma_hat = std if std > 0 else 1.0
    h = 1.06 * sigma_hat * n ** (-1.0 / 5.0)
    return float(h)


def sheather_jones_bw(z_data, z_lo, z_hi):
    r"""Simplified Sheather-Jones plug-in bandwidth for the Epanechnikov kernel.

    Stage 1 — *Pilot density.*  The z-data is histogrammed with ~50 bins over
    ``[z_lo, z_hi]`` and smoothed with an Epanechnikov kernel of Silverman
    bandwidth to obtain a pilot :math:`\hat f` on a fine grid.

    Stage 2 — *Second derivative.*  :math:`\hat f''` is estimated from the
    pilot by central finite differences and
    :math:`\widehat{\int[\hat f'']^2} = \sum \hat f''^2 \Delta z` computed.

    Stage 3 — *Oracle bandwidth.*

    .. math::

       h^* = \Bigl(\frac{\|K\|^2}{N\,\mu_2(K)^2\,
             \widehat{\int[\hat f'']^2}}\Bigr)^{1/5},

    with the Epanechnikov constants.  If the result is NaN, non-positive, or
    larger than the Silverman fallback, :func:`silverman_bw` is returned.

    Parameters
    ----------
    z_data : array_like
        1-D sample of positions.
    z_lo, z_hi : float
        Bounding interval used for the pilot histogram.

    Returns
    -------
    float
        Bandwidth ``h``.
    """
    z = np.asarray(z_data, dtype=float).ravel()
    n = z.size
    if n < 2:
        return 0.1
    h_silver = silverman_bw(z)

    n_hist = 50
    counts, edges = np.histogram(z, bins=n_hist, range=(z_lo, z_hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    f_pilot_raw = counts.astype(float) / (n * bin_width)

    n_fine = 400
    z_fine = np.linspace(z_lo, z_hi, n_fine)
    dz = z_fine[1] - z_fine[0]
    h0 = max(h_silver, dz)
    diff = z_fine[:, None] - centers[None, :]
    s = diff / h0
    kernel = np.where(np.abs(s) < 1.0, 0.75 * (1.0 - s * s) / h0, 0.0)
    f_pilot = kernel @ f_pilot_raw

    integral = np.trapezoid(f_pilot, z_fine)
    if integral > 0:
        f_pilot = f_pilot / integral

    f_pp = np.empty_like(f_pilot)
    f_pp[1:-1] = (f_pilot[2:] - 2.0 * f_pilot[1:-1] + f_pilot[:-2]) / (dz**2)
    f_pp[0] = f_pp[1]
    f_pp[-1] = f_pp[-2]

    int_fpp_sq = np.trapezoid(f_pp**2, z_fine)

    if int_fpp_sq <= 0 or not np.isfinite(int_fpp_sq):
        return h_silver
    h_sj = (_EPA_KERNEL_NORM_SQ / (n * _EPA_KERNEL_MU2_SQ * int_fpp_sq)) ** 0.2
    h_sj *= 1.05
    if not np.isfinite(h_sj) or h_sj <= 0 or h_sj > h_silver:
        return h_silver
    return float(h_sj)


def select_bandwidth(z_data, z_lo, z_hi, method="auto"):
    """Select the KDE bandwidth.

    Parameters
    ----------
    z_data : array_like
        1-D sample of positions.
    z_lo, z_hi : float
        Bounding interval (only used by the plug-in stage).
    method : {"auto", "silverman"} or float
        ``"auto"`` uses :func:`sheather_jones_bw` with a
        :func:`silverman_bw` fallback; ``"silverman"`` uses the rule of thumb
        directly; a float fixes the bandwidth.

    Returns
    -------
    float
        Bandwidth ``h``.
    """
    if method is None or method == "auto":
        h = sheather_jones_bw(z_data, z_lo, z_hi)
        if not np.isfinite(h) or h <= 0:
            h = silverman_bw(z_data)
        return float(h)
    if method == "silverman":
        return silverman_bw(z_data)
    return float(method)


def epanechnikov_kernel(x, h):
    r"""Epanechnikov kernel :math:`K_h(x) = \frac{3}{4h}(1 - (x/h)^2)` for
    :math:`|x| < h`, zero otherwise.

    Parameters
    ----------
    x : array_like
        Offsets (any shape).
    h : float
        Bandwidth (must be positive).

    Returns
    -------
    numpy.ndarray
        Kernel values, same shape as ``x``.
    """
    s = np.asarray(x, dtype=float) / h
    return np.where(np.abs(s) < 1.0, 0.75 * (1.0 - s * s) / h, 0.0)


def kde_1d(z_data, z_eval, h, z_bot, z_top, *, chunk_size=50_000):
    r"""1-D Epanechnikov KDE with mirror-reflection boundary handling.

    For each evaluation point :math:`z_0` in ``z_eval``,

    .. math::

       \hat\rho(z_0) = \frac{1}{N}\sum_j K_h(z_0 - z_j),

    with mirror copies at :math:`2z_{\mathrm{bot}} - z_j` and
    :math:`2z_{\mathrm{top}} - z_j` added to suppress leakage beyond the
    boundaries.  The :math:`O(N \cdot M)` kernel evaluation is performed in
    chunks of ``chunk_size`` rows to keep the broadcasting array
    memory-bounded.

    Parameters
    ----------
    z_data : array_like
        ``(N,)`` pooled sample positions.
    z_eval : array_like
        ``(M,)`` evaluation grid points.
    h : float
        Bandwidth (must be positive).
    z_bot, z_top : float
        Mirror-reflection boundaries of the confined region.
    chunk_size : int, optional
        Row-chunk size for the kernel matrix.  Default 50_000.

    Returns
    -------
    rho_hat : numpy.ndarray
        ``(M,)`` normalised KDE density (1/length, integrates to ~1 over the
        unbounded support).
    n_eff : numpy.ndarray
        ``(M,)`` Kish effective sample size
        :math:`(\sum_j K_h)^2 / \sum_j K_h^2`.
    """
    z = np.asarray(z_data, dtype=float).ravel()
    z_eval = np.asarray(z_eval, dtype=float).ravel()
    n = z.size
    M = z_eval.size

    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}")

    if n == 0 or M == 0:
        return np.zeros(M, dtype=np.float64), np.zeros(M, dtype=np.float64)

    z_aug = np.concatenate([z, 2.0 * z_bot - z, 2.0 * z_top - z])

    w_sum = np.zeros(M, dtype=np.float64)
    w2_sum = np.zeros(M, dtype=np.float64)

    n_aug = z_aug.size
    for i0 in range(0, n_aug, chunk_size):
        i1 = min(i0 + chunk_size, n_aug)
        z_chunk = z_aug[i0:i1][:, None]
        diff = z_eval[None, :] - z_chunk
        w = epanechnikov_kernel(diff, h)
        w_sum += w.sum(axis=0)
        w2_sum += (w * w).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        rho_hat = w_sum / n
        n_eff = np.where(w2_sum > 0, w_sum**2 / w2_sum, 0.0)
    return rho_hat, n_eff


class TransverseDensityQKDE(AnalysisBase):
    r"""Epanechnikov KDE transverse number-density profile.

    Pools per-frame positions of ``atomgroup`` along ``dim`` (the confined
    axis) across the analysis window and evaluates an Epanechnikov-kernel KDE
    on a uniform grid spanning the confined region ``[z_bot, z_top]``.  Kernel
    mass leaking beyond the boundaries is folded back by mirror reflection.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        Atoms whose positions are sampled (or, with ``grouping="residues"``,
        the atoms whose residue centre-of-mass positions are sampled).
    dim : int, optional
        Confined-axis index (0=x, 1=y, 2=z).  Default 2 (z).
    z_bot, z_top : float, optional
        Boundaries of the confined region used for mirror reflection and the
        evaluation grid.  If ``None`` they default to ``0`` and the box length
        along ``dim`` (taken from the final analysis frame's
        :attr:`~MDAnalysis.coordinates.timestep.Timestep.dimensions`).
    n_points : int, optional
        Number of evaluation grid points ``M``.  Default 400.
    bandwidth : {"auto", "silverman"} or float, optional
        Bandwidth selection method (see :func:`select_bandwidth`).
        Default ``"auto"``.
    grouping : {"atoms", "residues"}, optional
        ``"atoms"`` samples every atom's position each frame;
        ``"residues"`` samples the mass-weighted centre-of-mass of each
        residue in ``atomgroup`` each frame.  Default ``"atoms"``.
    chunk_size : int, optional
        Row-chunk size for the kernel matrix to bound memory.  Default
        50_000.

    Attributes
    ----------
    z_eval : numpy.ndarray
        ``(M,)`` evaluation grid (length).
    rho : numpy.ndarray
        ``(M,)`` normalised KDE density (1/length), integrating to ~1 over the
        unbounded support.  Multiply by the mean per-frame 2-D number density
        ``n_total / (n_frames_used * A)`` (where ``A`` is the cross-sectional
        area) to obtain a number density in particles/volume.
    n_eff : numpy.ndarray
        ``(M,)`` Kish effective sample size at each grid point.
    bandwidth : float
        Bandwidth used (length).
    n_total : int
        Total number of pooled samples.
    n_frames_used : int
        Number of analysis frames pooled.
    n_per_frame : float
        Mean number of samples per frame (``n_total / n_frames_used``).
    z_pooled : numpy.ndarray
        ``(n_total,)`` pooled sample positions (handy for diagnostics).

    Notes
    -----
    Confined-axis positions are wrapped into the primary box cell along
    ``dim`` before pooling, matching the behaviour of the original project
    script.  For ``grouping="residues"`` the centre-of-mass is computed with
    the atoms' topology masses; ensure masses are present in the topology for
    a true centre-of-mass (they default to 1.0, giving the geometric
    centroid).

    Examples
    --------
    ::

        import MDAnalysis as mda
        from qdiffusivity import TransverseDensityQKDE

        u = mda.Universe("topology.data", "trajectory.xtc")
        ag = u.select_atoms("type 1 2")
        kde = TransverseDensityQKDE(
            ag, dim=2, z_bot=10.0, z_top=90.0, grouping="residues",
        )
        kde.run()
        # number density (particles/Å^3): N_total / (n_frames * Lx * Ly) * rho
    """

    def __init__(
        self,
        atomgroup,
        *,
        dim: int = 2,
        z_bot=None,
        z_top=None,
        n_points: int = 400,
        bandwidth="auto",
        grouping: str = "atoms",
        chunk_size: int = 50_000,
    ):
        super().__init__(atomgroup.universe.trajectory)
        self._ag = atomgroup
        if dim not in (0, 1, 2):
            raise ValueError(f"dim must be 0, 1 or 2, got {dim}")
        self._dim = dim
        self._z_bot_user = z_bot
        self._z_top_user = z_top
        self._n_points = int(n_points)
        if grouping not in ("atoms", "residues"):
            raise ValueError(
                f"grouping must be 'atoms' or 'residues', got {grouping!r}"
            )
        self._grouping = grouping
        if isinstance(bandwidth, str) and bandwidth not in (
            "auto", "silverman"
        ):
            raise ValueError(
                f"bandwidth must be 'auto', 'silverman' or a float, "
                f"got {bandwidth!r}"
            )
        self._bandwidth = bandwidth
        self._chunk_size = int(chunk_size)

    def _prepare(self):
        if self._grouping == "residues":
            residx = self._ag.resindices
            self._res_unique, self._res_inv = np.unique(
                residx, return_inverse=True
            )
            self._n_res = self._res_unique.size
            self._masses = self._ag.masses.astype(np.float64)
            self._res_mass = np.zeros(self._n_res, dtype=np.float64)
            np.add.at(self._res_mass, self._res_inv, self._masses)
        self._z_frames = []

    def _single_frame(self):
        L = float(self._ts.dimensions[self._dim])
        if self._grouping == "residues":
            z_atoms = self._ag.positions[:, self._dim]
            masses = self._masses
            mz = np.zeros(self._n_res, dtype=np.float64)
            np.add.at(mz, self._res_inv, masses * z_atoms)
            com_z = mz / self._res_mass
            if L > 0:
                com_z %= L
            self._z_frames.append(com_z)
        else:
            z = self._ag.positions[:, self._dim].astype(np.float64).copy()
            if L > 0:
                z %= L
            self._z_frames.append(z)

    def _conclude(self):
        z_pooled = (
            np.concatenate(self._z_frames)
            if self._z_frames
            else np.empty(0, dtype=np.float64)
        )
        self.z_pooled = z_pooled
        self.n_total = int(z_pooled.size)
        self.n_frames_used = len(self._z_frames)
        self.n_per_frame = (
            self.n_total / self.n_frames_used if self.n_frames_used > 0 else 0.0
        )

        L = float(self._ts.dimensions[self._dim])
        z_bot = 0.0 if self._z_bot_user is None else float(self._z_bot_user)
        z_top = L if self._z_top_user is None else float(self._z_top_user)
        if z_top <= z_bot:
            raise ValueError(
                f"z_top ({z_top}) must be greater than z_bot ({z_bot})"
            )

        M = self._n_points
        self.z_eval = z_bot + (np.arange(M) + 0.5) * (z_top - z_bot) / M

        if z_pooled.size == 0:
            self.rho = np.zeros(M, dtype=np.float64)
            self.n_eff = np.zeros(M, dtype=np.float64)
            self.bandwidth = float("nan")
            return

        self.bandwidth = select_bandwidth(
            z_pooled, z_bot, z_top, method=self._bandwidth
        )
        self.rho, self.n_eff = kde_1d(
            z_pooled,
            self.z_eval,
            self.bandwidth,
            z_bot,
            z_top,
            chunk_size=self._chunk_size,
        )
