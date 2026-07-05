r"""Kernel density estimator for position-dependent diffusivity.

This module provides a kernel-weighted local estimator for the
perpendicular (transverse) and parallel diffusivities of nanoconfined
molecular dynamics simulations, generalising the per-project script
``zn-el/analysis/diff_kde/kde_diffusivity.py`` into a reusable
:class:`~MDAnalysis.analysis.base.AnalysisBase` class.

Theory
------
The estimator works in *u-space*, the CDF-uniformised coordinate
:math:`u = F(z) \in [0, 1]` built from the pooled equilibrium positions.
In u-space the equilibrium measure is uniform, so a single global
bandwidth is appropriate across the whole confined region (including
near the walls, where a z-space bandwidth would over-smooth the sparse
adsorption peaks and under-resolve the dense bulk).

* **CDF map.**  Each starting position :math:`z` is mapped to
  :math:`u = F(z) \in [0, 1]` via the empirical CDF (piecewise-linear
  interpolation through the sorted positions, using the midrank
  :math:`p = (i + 0.5)/n` so the CDF never returns exactly 0 or 1).

* **Perpendicular (transverse) estimator.**  For each trajectory
  increment with starting position :math:`z_j` and displacement
  :math:`\Delta z_j`, the z-space local estimator is
  :math:`\hat d_j = (\Delta z_j)^2 / (2\Delta t)`.  The kernel
  localises in u-space (variance equalisation) but the estimator is the
  z-space MLE, so :math:`D_\perp` is obtained directly — no
  :math:`\rho^{-2}` conversion is applied:

  .. math::

     \hat{D}_\perp(u_0) =
       \frac{\sum_j K_h(u_0 - u_j)\,(\Delta z_j)^2/(2\Delta t)}
            {\sum_j K_h(u_0 - u_j)}.

* **Parallel estimator.**  Only the *starting position* is mapped to
  u-space; the parallel displacement
  :math:`(\Delta x_j, \Delta y_j)` is not rescaled (parallel motion is
  unbounded):

  .. math::

     \hat{D}_\parallel(u_0) =
       \frac{\sum_j K_h(u_0 - u_j)\,[(\Delta x_j)^2 + (\Delta y_j)^2]
             /(4\Delta t)}
            {\sum_j K_h(u_0 - u_j)},

  where :math:`\langle \Delta x^2 + \Delta y^2\rangle = 4 D_\parallel
  \Delta t` gives the :math:`4\Delta t` denominator.

* **Kernel.**  Either the Gaussian kernel
  :math:`K_h(x) = \exp(-x^2/(2h^2))/(h\sqrt{2\pi})` (infinite support,
  smooth derivatives — the default) or the Epanechnikov kernel
  :math:`K_h(x) = \frac{3}{4h}(1 - (x/h)^2)` for :math:`|x| < h` (compact
  support, no leakage beyond the boundaries).

* **Bandwidth.**  ``"auto"`` uses a Sheather-Jones plug-in (a
  kernel-appropriate pilot density is built on a fine grid, its second
  derivative estimated by central differences, and the oracle bandwidth
  :math:`h^* = (\|K\|^2 / (N\,\mu_2(K)^2\,\widehat{\int[\hat f'']^2}))^{1/5}`
  evaluated); the Silverman rule of thumb is the fallback.
  ``"silverman"`` uses the rule of thumb directly, and a float fixes the
  bandwidth.

* **Boundary handling.**  Mirror reflection at :math:`u = 0` and
  :math:`u = 1`: for each data point at :math:`u_j`, mirror copies at
  :math:`-u_j` and :math:`2 - u_j` are added (with the same
  :math:`\hat d_j`) before kernel evaluation, so kernel mass leaking
  outside :math:`[0, 1]` is reflected back.

* **Evaluation grid.**  ``n_points`` cell-centred points uniform in
  u-space, :math:`u_m = (m + 0.5)/M` for :math:`m = 0, \ldots, M-1`,
  mapped back to z via :math:`z_m = F^{-1}(u_m)`.

* **Error bars.**  The weighted variance of :math:`\hat d_j` within the
  kernel window, divided by the Kish effective sample size
  :math:`N_{\mathrm{eff}}(u_0) = (\sum_j K_h)^2 / \sum_j K_h^2`, gives the
  standard error of the mean at each grid point.
"""

from __future__ import annotations

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.transformations import NoJump

from .density import epanechnikov_kernel

# Gaussian-kernel constants for the Sheather-Jones plug-in.
#   ||K||_2^2 = 1 / (2*sqrt(pi)),  mu_2(K) = 1
_GAU_KERNEL_NORM_SQ = 1.0 / (2.0 * np.sqrt(np.pi))
_GAU_KERNEL_MU2_SQ = 1.0

# Epanechnikov-kernel constants (reused from density.py for the plug-in).
#   ||K||_2^2 = 3/5,  mu_2(K) = 1/5
_EPA_KERNEL_NORM_SQ = 3.0 / 5.0
_EPA_KERNEL_MU2_SQ = (1.0 / 5.0) ** 2


def gaussian_kernel(x, h):
    r"""Gaussian kernel :math:`K_h(x) = \exp(-x^2/(2h^2))/(h\sqrt{2\pi})`.

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
    return np.exp(-0.5 * s * s) / (h * np.sqrt(2.0 * np.pi))


def _kernel(name):
    """Return ``(kernel_fn, norm_sq, mu2_sq)`` for ``name`` in
    ``{"gaussian", "epanechnikov"}``."""
    if name == "gaussian":
        return (
            gaussian_kernel,
            _GAU_KERNEL_NORM_SQ,
            _GAU_KERNEL_MU2_SQ,
        )
    if name == "epanechnikov":
        return (
            epanechnikov_kernel,
            _EPA_KERNEL_NORM_SQ,
            _EPA_KERNEL_MU2_SQ,
        )
    raise ValueError(
        f"kernel must be 'gaussian' or 'epanechnikov', "
        f"got {name!r}"
    )


def silverman_bw(u_data):
    r"""Silverman's rule of thumb (Gaussian reference).

    Returns :math:`h = 1.06\,\hat\sigma\,N^{-1/5}` with the robust
    scaled-IQR scale
    :math:`\hat\sigma = \min(\mathrm{std}, \mathrm{IQR}/1.34)`.

    Parameters
    ----------
    u_data : array_like
        1-D sample (u-space positions).

    Returns
    -------
    float
        Bandwidth ``h``.
    """
    u = np.asarray(u_data, dtype=float).ravel()
    n = u.size
    if n < 2:
        return 0.05
    std = float(np.std(u, ddof=1))
    q75, q25 = np.percentile(u, [75, 25])
    iqr = q75 - q25
    sigma_hat = min(std, iqr / 1.34) if iqr > 0 else std
    if sigma_hat <= 0:
        sigma_hat = std if std > 0 else 0.1
    h = 1.06 * sigma_hat * n ** (-1.0 / 5.0)
    return float(h)


def sheather_jones_bw(u_data, kernel="gaussian"):
    r"""Simplified Sheather-Jones plug-in bandwidth.

    Stage 1 — a pilot density is built by histogramming the u-data with
    ~50 bins over ``[0, 1]`` and smoothing with a kernel of Silverman
    bandwidth on a fine grid.

    Stage 2 — :math:`\hat f''` is estimated by central finite
    differences and :math:`\widehat{\int[\hat f'']^2}` computed.

    Stage 3 — the oracle bandwidth
    :math:`h^* = (\|K\|^2 / (N\,\mu_2(K)^2\,\widehat{\int[\hat f'']^2}))^{1/5}`
    is evaluated with the kernel-appropriate constants.  If the result is
    NaN, non-positive, or larger than the Silverman fallback, the latter
    is returned.

    Parameters
    ----------
    u_data : array_like
        1-D sample (u-space positions).
    kernel : {"gaussian", "epanechnikov"}, optional
        Kernel whose constants are used in the oracle formula.

    Returns
    -------
    float
        Bandwidth ``h``.
    """
    u = np.asarray(u_data, dtype=float).ravel()
    n = u.size
    if n < 2:
        return 0.05
    h_silver = silverman_bw(u)
    kernel_fn, norm_sq, mu2_sq = _kernel(kernel)

    n_hist = 50
    u_min, u_max = 0.0, 1.0
    counts, edges = np.histogram(u, bins=n_hist, range=(u_min, u_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    f_pilot_raw = counts.astype(float) / (n * bin_width)

    n_fine = 400
    u_fine = np.linspace(u_min, u_max, n_fine)
    du = u_fine[1] - u_fine[0]
    h0 = max(h_silver, du)
    diff = u_fine[:, None] - centers[None, :]
    kernel_vals = kernel_fn(diff, h0)
    f_pilot = kernel_vals @ f_pilot_raw

    integral = np.trapezoid(f_pilot, u_fine)
    if integral > 0:
        f_pilot = f_pilot / integral

    f_pp = np.empty_like(f_pilot)
    f_pp[1:-1] = (f_pilot[2:] - 2.0 * f_pilot[1:-1] + f_pilot[:-2]) / (du**2)
    f_pp[0] = f_pp[1]
    f_pp[-1] = f_pp[-2]

    int_fpp_sq = np.trapezoid(f_pp**2, u_fine)

    if int_fpp_sq <= 0 or not np.isfinite(int_fpp_sq):
        return h_silver
    h_sj = (norm_sq / (n * mu2_sq * int_fpp_sq)) ** 0.2
    h_sj *= 1.05
    if not np.isfinite(h_sj) or h_sj <= 0 or h_sj > h_silver:
        return h_silver
    return float(h_sj)


def select_diff_bandwidth(u_data, method="auto", kernel="gaussian"):
    """Select the diffusivity KDE bandwidth.

    Parameters
    ----------
    u_data : array_like
        1-D sample (u-space positions).
    method : {"auto", "silverman"} or float
        ``"auto"`` uses :func:`sheather_jones_bw` with a
        :func:`silverman_bw` fallback; ``"silverman"`` uses the rule of
        thumb directly; a float fixes the bandwidth.
    kernel : {"gaussian", "epanechnikov"}, optional
        Kernel whose constants are used by the plug-in.

    Returns
    -------
    float
        Bandwidth ``h``.
    """
    if method is None or method == "auto":
        h = sheather_jones_bw(u_data, kernel=kernel)
        if not np.isfinite(h) or h <= 0:
            h = silverman_bw(u_data)
        return float(h)
    if method == "silverman":
        return silverman_bw(u_data)
    return float(method)


def build_cdf(z_pooled):
    r"""Build a smooth CDF from pooled positions.

    Returns the closures ``P(z)`` (maps physical z to ``[0, 1]``),
    ``P_inv(p)`` (maps ``[0, 1]`` back to z), ``rho(z)`` (the equilibrium
    density, estimated as a Gaussian-smoothed histogram derivative of the
    CDF and interpolated back to arbitrary z), and ``rho_prime(z)`` (the
    spatial derivative of ``rho``, used by the Itô correction).

    Uses the empirical sorted positions with midrank
    ``p = (i + 0.5)/n`` so the CDF never returns exactly 0 or 1 at the
    extremes.  The density ``rho`` is computed from a coarse histogram
    (200 bins) Gaussian-smoothed to suppress Poisson noise before
    interpolation — the perpendicular KDE would otherwise double any
    noise in ``rho`` via the :math:`\rho^{-2}` conversion.  The
    derivative ``rho_prime`` is obtained from the same smoothed histogram
    by central finite differences and interpolation, so it inherits the
    noise suppression.

    Parameters
    ----------
    z_pooled : array_like
        ``(N,)`` pooled positions (e.g. all wrapped-z across all frames).

    Returns
    -------
    P : callable
        CDF closure ``P(z) -> [0, 1]``.
    P_inv : callable
        Inverse CDF closure ``P_inv(p) -> z``.
    rho : callable
        Equilibrium density closure ``rho(z) -> float`` (integrates to 1).
    rho_prime : callable
        Spatial derivative closure ``rho_prime(z) -> float`` (same units
        as ``rho`` per length).
    z_sorted : numpy.ndarray
        Sorted unique positions used to build the CDF.
    p_vals : numpy.ndarray
        Midrank probabilities aligned with ``z_sorted``.
    """
    z_sorted = np.sort(np.asarray(z_pooled, dtype=float).ravel())
    n = z_sorted.size
    if n == 0:
        raise ValueError("z_pooled is empty; cannot build a CDF")
    z_min = z_sorted[0]
    z_max = z_sorted[-1]
    p_vals = (np.arange(1, n + 1) - 0.5) / n

    def P(z):
        z = np.asarray(z, dtype=float)
        return np.interp(z, z_sorted, p_vals, left=0.0, right=1.0)

    def P_inv(p):
        p = np.asarray(p, dtype=float)
        return np.interp(p, p_vals, z_sorted, left=z_min, right=z_max)

    # Density estimate: a coarse histogram Gaussian-smoothed to suppress
    # Poisson noise before interpolation.
    n_bins = 200
    counts, edges = np.histogram(z_pooled, bins=n_bins, range=(z_min, z_max))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    rho_raw = counts.astype(float) / (n * bin_width)

    sigma_smooth = 2.0 * bin_width
    n_pad = int(3 * sigma_smooth / bin_width) + 1
    rho_padded = np.pad(rho_raw, n_pad, mode="edge")
    kernel_sigma = sigma_smooth / bin_width
    kernel_x = np.arange(-n_pad, n_pad + 1)
    kernel = np.exp(-0.5 * (kernel_x / kernel_sigma) ** 2)
    kernel /= kernel.sum()
    rho_smooth = np.convolve(rho_padded, kernel, mode="same")[n_pad:-n_pad]

    integral = np.trapezoid(rho_smooth, centers)
    if integral > 0:
        rho_smooth = rho_smooth / integral

    # Spatial derivative of rho via central finite differences on the
    # smoothed histogram grid, then interpolated back to arbitrary z.
    # Same noise-suppression as rho itself (the Gaussian smoothing
    # precedes the differencing, so the derivative is not dominated by
    # Poisson noise from the histogram bins).
    rho_prime_smooth = np.empty_like(rho_smooth)
    rho_prime_smooth[1:-1] = (
        rho_smooth[2:] - rho_smooth[:-2]
    ) / (2.0 * bin_width)
    rho_prime_smooth[0] = rho_prime_smooth[1]
    rho_prime_smooth[-1] = rho_prime_smooth[-2]

    def rho(z):
        z = np.asarray(z, dtype=float)
        return np.interp(
            z, centers, rho_smooth, left=rho_smooth[0], right=rho_smooth[-1]
        )

    def rho_prime(z):
        z = np.asarray(z, dtype=float)
        return np.interp(
            z,
            centers,
            rho_prime_smooth,
            left=rho_prime_smooth[0],
            right=rho_prime_smooth[-1],
        )

    return P, P_inv, rho, rho_prime, z_sorted, p_vals


def kde_estimate(
    u_data, d_data, u_eval, h, kernel="gaussian",
    chunk_size=50_000,
):
    r"""Kernel-weighted local estimator in u-space with mirror reflection.

    For each evaluation point :math:`u_0` in ``u_eval``,

    .. math::

       \hat{D}(u_0) = \frac{\sum_j K_h(u_0 - u_j)\,d_j}
                            {\sum_j K_h(u_0 - u_j)},

    where ``d_data`` already carries the :math:`1/(2\Delta t)` (or
    :math:`1/(4\Delta t)`) prefactor — i.e. the caller passes the local
    estimator values :math:`(\Delta z_j)^2/(2\Delta t)` or
    :math:`[(\Delta x_j)^2 + (\Delta y_j)^2]/(4\Delta t)`.

    Boundary handling: mirror reflection at :math:`u = 0` and :math:`u =
    1`.  For each data point at :math:`u_j`, mirror copies at
    :math:`-u_j` and :math:`2 - u_j` are added (with the same
    :math:`d_j`) before kernel evaluation.

    The :math:`O(N \cdot M)` kernel evaluation is performed in chunks of
    ``chunk_size`` increments to keep the broadcasting array
    memory-bounded.

    Parameters
    ----------
    u_data : array_like
        ``(N,)`` u-space positions of the increments.
    d_data : array_like
        ``(N,)`` local estimator values (already carrying the
        :math:`1/(2\Delta t)` or :math:`1/(4\Delta t)` prefactor).
    u_eval : array_like
        ``(M,)`` evaluation grid in u-space.
    h : float
        Bandwidth (must be positive).
    kernel : {"gaussian", "epanechnikov"}, optional
        Kernel function.  Default ``"gaussian"``.
    chunk_size : int, optional
        Row-chunk size for the kernel matrix to bound memory.  Default
        50_000.

    Returns
    -------
    D : numpy.ndarray
        ``(M,)`` kernel-weighted mean of ``d_data`` at each evaluation
        point.
    D_std : numpy.ndarray
        ``(M,)`` standard error of the mean (weighted standard deviation
        divided by :math:`\sqrt{N_{\mathrm{eff}}}`).
    n_eff : numpy.ndarray
        ``(M,)`` Kish effective sample size
        :math:`(\sum_j K_h)^2 / \sum_j K_h^2`.
    """
    u = np.asarray(u_data, dtype=float).ravel()
    d = np.asarray(d_data, dtype=float).ravel()
    u_eval = np.asarray(u_eval, dtype=float).ravel()
    n = u.size
    M = u_eval.size

    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {h}")
    kernel_fn, _, _ = _kernel(kernel)

    if n == 0 or M == 0:
        zeros = np.zeros(M, dtype=np.float64)
        return zeros, zeros, zeros

    u_aug = np.concatenate([u, -u, 2.0 - u])
    d_aug = np.concatenate([d, d, d])

    w_sum = np.zeros(M, dtype=np.float64)
    wd_sum = np.zeros(M, dtype=np.float64)
    wd2_sum = np.zeros(M, dtype=np.float64)
    w2_sum = np.zeros(M, dtype=np.float64)

    n_aug = u_aug.size
    for i0 in range(0, n_aug, chunk_size):
        i1 = min(i0 + chunk_size, n_aug)
        u_chunk = u_aug[i0:i1][:, None]
        d_chunk = d_aug[i0:i1][:, None]
        diff = u_eval[None, :] - u_chunk
        w = kernel_fn(diff, h)
        w_sum += w.sum(axis=0)
        wd = w * d_chunk
        wd_sum += wd.sum(axis=0)
        wd2_sum += (wd * d_chunk).sum(axis=0)
        w2_sum += (w * w).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(w_sum > 0, wd_sum / w_sum, np.nan)
        mean_d2 = np.where(w_sum > 0, wd2_sum / w_sum, np.nan)
        var_d = np.where(w_sum > 0, mean_d2 - D**2, 0.0)
        var_d = np.maximum(var_d, 0.0)
        n_eff = np.where(w2_sum > 0, w_sum**2 / w2_sum, 0.0)
        D_std = np.where(
            n_eff > 1, np.sqrt(var_d / np.maximum(n_eff, 1.0)), 0.0
        )
    return D, D_std, n_eff


class LocalDiffusivityKDE(AnalysisBase):
    r"""Kernel-weighted local estimator for transverse diffusivity.

    Pools per-frame positions of ``atomgroup``, builds a CDF-uniformised
    u-space from the equilibrium positions, and evaluates a
    kernel-weighted local estimator for the perpendicular
    (:math:`D_\perp`) and parallel (:math:`D_\parallel`) diffusivities on
    a uniform u-space grid mapped back to z.  Kernel mass leaking beyond
    :math:`u \in [0, 1]` is folded back by mirror reflection.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        Atoms whose positions are sampled.
    dim : int, optional
        Confined-axis index (0=x, 1=y, 2=z).  Default 2 (z).  The
        perpendicular displacement is taken along ``dim``; the parallel
        displacement is taken along the other two axes.
    n_points : int, optional
        Number of evaluation grid points ``M`` (uniform in u-space).
        Default 200.
    bandwidth : {"auto", "silverman"} or float, optional
        Bandwidth selection method (see
        :func:`select_diff_bandwidth`).  Default ``"auto"``.
    kernel : {"gaussian", "epanechnikov"}, optional
        Kernel function.  Default ``"gaussian"``.
    dt : float or None, optional
        Time step between consecutive frames, in the units the caller
        wishes the diffusivity to be expressed in (e.g. ps).  If ``None``
        (default) the trajectory's ``dt`` (:attr:`ts.dt`) is used.  The
        diffusivity is returned in (length² / time) with length in Å and
        time in whatever unit ``dt`` carries.
    ito_correction : bool, optional
        If ``True``, subtract the :math:`O(\Delta t)` Itô bias

        .. math::

           \text{bias}_\perp(z) = \frac{\Delta t}{2}\,\Phi(z)^2
           = \frac{\Delta t}{2}\left[D(z)\,\frac{\rho'(z)}{\rho(z)}\right]^2

        from the perpendicular estimator, where :math:`\Phi = -\beta D V'
        = D\,\rho'/\rho` in the isothermal (Hänggi–Klimontovich)
        convention.  The bias is computed from the uncorrected
        :math:`D_\perp` and the equilibrium density :math:`\rho` (and its
        derivative) produced by :func:`build_cdf`.  The parallel
        estimator has **zero** Itô bias (no parallel drift) and is left
        unchanged.  Default ``False`` (the bias is self-suppressing in
        the electrode geometry — see the project notes — and within
        the statistical error bars at typical frame spacings).

    Attributes
    ----------
    z_eval : numpy.ndarray
        ``(M,)`` evaluation grid in z-space.
    u_eval : numpy.ndarray
        ``(M,)`` evaluation grid in u-space (uniform, cell-centred).
    D_perp : numpy.ndarray
        ``(M,)`` perpendicular diffusivity (length²/time), clipped to
        be non-negative.  When ``ito_correction=True`` the Itô bias
        (see :attr:`ito_bias`) is subtracted *before* clipping.
    D_perp_std : numpy.ndarray
        ``(M,)`` standard error of :math:`D_\perp`.
    n_eff_perp : numpy.ndarray
        ``(M,)`` Kish effective sample size for :math:`D_\perp`.
    ito_bias : numpy.ndarray or None
        ``(M,)`` Itô bias :math:`\frac{\Delta t}{2}\Phi^2` subtracted
        from :math:`D_\perp` when ``ito_correction=True``; ``None``
        otherwise.
    D_para : numpy.ndarray
        ``(M,)`` parallel diffusivity (length²/time).
    D_para_std : numpy.ndarray
        ``(M,)`` standard error of :math:`D_\parallel`.
    n_eff_para : numpy.ndarray
        ``(M,)`` Kish effective sample size for :math:`D_\parallel`.
    bandwidth : float
        Bandwidth used (in u-space units).
    n_increments : int
        Total number of trajectory increments pooled
        (``n_frames_used - 1`` per atom, summed over atoms).
    n_frames_used : int
        Number of frames pooled.
    dt : float
        Time step used in the estimator (length²/time units).
    P, P_inv, rho : callables
        CDF, inverse-CDF and equilibrium density closures (see
        :func:`build_cdf`).
    rho_prime : callable
        Spatial derivative of the equilibrium density (see
        :func:`build_cdf`).

    Notes
    -----
    Parallel (x/y) positions are unwrapped with MDAnalysis's ``NoJump``
    transformation so that multi-frame displacements across periodic
    boundaries are captured.  The transformation is attached in place
    (guarded so it is only attached once per Universe) and the full
    trajectory is iterated from frame 0 (``NoJump`` is stateful).

    Examples
    --------
    ::

        import MDAnalysis as mda
        from qdiffusivity import LocalDiffusivityKDE

        u = mda.Universe("topology.data", "trajectory.xtc")
        ag = u.select_atoms("type 1 2")
        kde = LocalDiffusivityKDE(
            ag, dim=2, n_points=200, bandwidth="auto", kernel="gaussian",
        )
        kde.run()
        # D_perp, D_para are in Å²/ps if the trajectory dt is in ps.
    """

    def __init__(
        self,
        atomgroup,
        *,
        dim: int = 2,
        n_points: int = 200,
        bandwidth="auto",
        kernel: str = "gaussian",
        dt=None,
        ito_correction: bool = False,
        chunk_size: int = 50_000,
    ):
        super().__init__(atomgroup.universe.trajectory)
        self._ag = atomgroup
        if dim not in (0, 1, 2):
            raise ValueError(f"dim must be 0, 1 or 2, got {dim}")
        self._dim = dim
        self._para_dims = tuple(i for i in (0, 1, 2) if i != dim)
        self._n_points = int(n_points)
        if isinstance(bandwidth, str) and bandwidth not in (
            "auto", "silverman"
        ):
            raise ValueError(
                f"bandwidth must be 'auto', 'silverman' or a float, "
                f"got {bandwidth!r}"
            )
        self._bandwidth = bandwidth
        if kernel not in ("gaussian", "epanechnikov"):
            raise ValueError(
                f"kernel must be 'gaussian' or 'epanechnikov', got {kernel!r}"
            )
        self._kernel = kernel
        self._dt = dt
        self._ito_correction = bool(ito_correction)
        self._chunk_size = int(chunk_size)

    def _prepare(self):
        self._pos_frames = []

    def _single_frame(self):
        # Snapshot the positions so later NoJump unwrapping does not
        # mutate the stored array.  We store the wrapped positions here
        # and unwrap lazily in _conclude via NoJump.
        self._pos_frames.append(self._ag.positions.copy())

    def _conclude(self):
        pos = np.asarray(self._pos_frames, dtype=np.float64)
        # (n_frames, n_atoms, 3) — these are wrapped; we need unwrapped
        # parallel coords and wrapped confined coord.  Re-iterate with
        # NoJump to unwrap x/y, keeping z wrapped.
        n_frames = pos.shape[0]
        self.n_frames_used = n_frames
        if n_frames < 2:
            raise ValueError(
                f"need at least 2 frames to form a displacement; got {n_frames}"
            )

        # Determine dt.
        dt = self._dt if self._dt is not None else float(self._ts.dt)
        self.dt = dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Unwrap the parallel coordinates via NoJump.  We attach the
        # transformation to the universe's trajectory (guarded so it is
        # only attached once) and re-iterate from frame 0.
        u = self._ag.universe
        if not getattr(u, "_qdiff_nojump_applied", False):
            u.trajectory[0]
            u.trajectory.add_transformations(NoJump(self._ag))
            u._qdiff_nojump_applied = True

        pos_unwrapped = np.empty_like(pos)
        Lz = None
        k = 0
        for idx in range(0, u.trajectory.n_frames):
            u.trajectory[idx]
            if k >= n_frames:
                break
            pos_unwrapped[k] = self._ag.positions
            Lz = u.dimensions[self._dim]
            k += 1
        if Lz is not None and Lz > 0:
            pos_unwrapped[:, :, self._dim] %= Lz

        # Pass 1: build CDF from pooled wrapped-z.
        z_pooled = pos_unwrapped[:, :, self._dim].ravel()
        self.P, self.P_inv, self.rho, self.rho_prime, _, _ = build_cdf(z_pooled)

        # Evaluation grid: uniform in u-space, mapped back to z.
        M = self._n_points
        self.u_eval = (np.arange(M) + 0.5) / M
        self.z_eval = self.P_inv(self.u_eval)

        # u-positions of all increments (starting frame of each pair).
        z_start = pos_unwrapped[:-1, :, self._dim]  # (n_frames-1, n_atoms)
        u_start = self.P(z_start.ravel())

        # Bandwidth from the u-positions of all increments.
        self.bandwidth = select_diff_bandwidth(
            u_start, method=self._bandwidth, kernel=self._kernel
        )

        # Perpendicular: z-space local estimator (Δz)²/(2Δt), kernel
        # weighting in u-space.  No ρ⁻² conversion.
        dz = np.diff(pos_unwrapped[:, :, self._dim], axis=0)
        d_perp_local = (dz**2) / (2.0 * dt)
        self.n_increments = int(d_perp_local.size)

        self.D_perp, self.D_perp_std, self.n_eff_perp = kde_estimate(
            u_start.ravel(),
            d_perp_local.ravel(),
            self.u_eval,
            self.bandwidth,
            kernel=self._kernel,
            chunk_size=self._chunk_size,
        )

        # Itô bias of the perpendicular estimator:
        #   bias(z) = (Δt/2) Φ(z)^2,  Φ = -β D V' = D ρ'/ρ
        # in the isothermal convention (β·k_BT = 1).  Computed from the
        # uncorrected D_perp and the equilibrium density.  The parallel
        # estimator has zero Itô bias (no parallel drift).
        if self._ito_correction:
            rho_eval = self.rho(self.z_eval)
            rho_prime_eval = self.rho_prime(self.z_eval)
            with np.errstate(divide="ignore", invalid="ignore"):
                phi = np.where(
                    rho_eval > 0,
                    self.D_perp * rho_prime_eval / rho_eval,
                    0.0,
                )
            self.ito_bias = 0.5 * dt * phi**2
            self.D_perp = self.D_perp - self.ito_bias
        else:
            self.ito_bias = None

        self.D_perp = np.clip(self.D_perp, 0, None)

        # Parallel: Δx² + Δy², only z_start mapped to u.  Parallel
        # displacement is not rescaled (parallel motion unbounded).
        d_para_acc = np.zeros_like(dz)
        for d in self._para_dims:
            diff_d = np.diff(pos_unwrapped[:, :, d], axis=0)
            d_para_acc += diff_d**2
        d_para_local = d_para_acc / (4.0 * dt)

        self.D_para, self.D_para_std, self.n_eff_para = kde_estimate(
            u_start.ravel(),
            d_para_local.ravel(),
            self.u_eval,
            self.bandwidth,
            kernel=self._kernel,
            chunk_size=self._chunk_size,
        )
