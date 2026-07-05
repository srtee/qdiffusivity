r"""CDF-binned transverse density and diffusivity profiles.

This module provides binned (histogram-style) counterparts to the KDE
classes in :mod:`qdiffusivity.density` and
:mod:`qdiffusivity.diffusivity`.  Binning is always in *u-space* — the
CDF-uniformised coordinate :math:`u = F(z) \in [0, 1]` built from the
pooled equilibrium positions — so that bins are naturally finer where
the particle density is high (adsorption peaks) and coarser where it is
low, and every bin receives a comparable number of samples.  This is
the same equal-population strategy used by the project's quantile
scripts.

Two assignment schemes are supported:

* **Cloud-in-cell (CIC)** — used for integer ``bins`` (N uniform
  u-space bins).  Each sample is linearly split between the two nearest
  bin centres, avoiding the discontinuities that hard binning would
  introduce at the (population-balanced) bin edges.

* **Hard assignment** — used when explicit bin edges (an array in
  ``[0, 1]``) are supplied.  Each sample falls in exactly one bin
  (``np.digitize``), matching the standard histogram behaviour.

The perpendicular diffusivity estimator supports the same
:math:`O(\Delta t)` Itô bias correction as the KDE diffusivity class,
gated on ``ito_correction=False`` by default.
"""

from __future__ import annotations

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.transformations import NoJump

from .diffusivity import build_cdf


def resolve_bins(bins, n_default=30):
    """Resolve a ``bins`` spec into ``(n_bins, edges, use_cic)``.

    Parameters
    ----------
    bins : int, array_like, or "quantile"
        ``int`` — N uniform u-space bins in [0, 1] (CIC assignment).
        ``"quantile"`` — shortcut for the default ``n_default`` uniform
        u-space bins (CIC).
        ``array_like`` — explicit u-space bin edges in ``[0, 1]`` (hard
        assignment via ``np.digitize``).  Must be sorted and span a
        sub-interval of ``[0, 1]``.
    n_default : int, optional
        Number of bins when ``bins == "quantile"``.  Default 30.

    Returns
    -------
    n_bins : int
        Number of bins.
    edges : numpy.ndarray
        ``(n_bins + 1,)`` u-space bin edges in ``[0, 1]`` (uniform for
        int / ``"quantile"``; as supplied for an array, padded with
        0 and 1 if needed).
    use_cic : bool
        ``True`` for CIC assignment (int / ``"quantile"``);
        ``False`` for hard assignment (explicit edges).
    """
    if isinstance(bins, str):
        if bins != "quantile":
            raise ValueError(f"bins string must be 'quantile', got {bins!r}")
        n = int(n_default)
        edges = np.linspace(0.0, 1.0, n + 1)
        return n, edges, True

    if isinstance(bins, (int, np.integer)):
        n = int(bins)
        if n < 1:
            raise ValueError(f"bins must be >= 1, got {n}")
        edges = np.linspace(0.0, 1.0, n + 1)
        return n, edges, True

    arr = np.asarray(bins, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError("explicit bin edges must have >= 2 values")
    if np.any(np.diff(arr) <= 0):
        raise ValueError("explicit bin edges must be strictly increasing")
    if arr[0] < 0.0 or arr[-1] > 1.0:
        raise ValueError("explicit bin edges must lie in [0, 1]")
    # Pad with 0 / 1 if the user supplied interior edges only.
    if arr[0] > 0.0:
        arr = np.concatenate([[0.0], arr])
    if arr[-1] < 1.0:
        arr = np.concatenate([arr, [1.0]])
    n = arr.size - 1
    return n, arr, False


def cic_assign(u_data, n_bins):
    """Cloud-in-cell assignment of u-space samples to uniform bins.

    Each sample at fractional bin index ``q = u * n_bins - 0.5`` is split
    between the two nearest bin centres: weight ``1 - frac`` to bin
    ``k0 = floor(q)`` and weight ``frac`` to bin ``k0 + 1``.  Samples
    outside ``[0, n_bins - 1]`` (in the outer half of the first or last
    bin) are mirror-reflected back into the interior, so each boundary
    bin receives its correct share rather than a half- or double-weight
    artefact.

    Parameters
    ----------
    u_data : array_like
        ``(N,)`` u-space positions in ``[0, 1]``.
    n_bins : int
        Number of uniform u-space bins.

    Returns
    -------
    k0, k1 : numpy.ndarray
        ``(N,)`` int arrays of the two bin indices each sample
        contributes to (clipped to ``[0, n_bins - 1]``).
    w0, w1 : numpy.ndarray
        ``(N,)`` float arrays of the CIC weights for ``k0`` and ``k1``
        (``w0 + w1 == 1``).
    """
    u = np.asarray(u_data, dtype=float).ravel()
    q = u * n_bins - 0.5
    # Mirror reflection at the outer boundaries.
    q = np.where(q < 0.0, -q, q)
    q = np.where(q > n_bins - 1.0, 2.0 * (n_bins - 1.0) - q, q)
    k0 = np.floor(q).astype(int)
    frac = q - k0
    k0 = np.clip(k0, 0, n_bins - 1)
    k1 = np.clip(k0 + 1, 0, n_bins - 1)
    w0 = 1.0 - frac
    w1 = frac
    return k0, k1, w0, w1


def _bin_centers_from_edges(edges):
    """Return the midpoints of ``edges``."""
    return 0.5 * (edges[:-1] + edges[1:])


class TransverseDensityBinned(AnalysisBase):
    r"""CDF-binned transverse number-density profile.

    Pools per-frame positions of ``atomgroup`` along ``dim`` (the
    confined axis), builds a CDF-uniformised u-space, and assigns each
    sample to u-space bins via cloud-in-cell (CIC) or hard assignment.
    The bin population is converted to a number density via the
    cross-sectional area and the number of analysis frames.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        Atoms whose positions are sampled (or, with
        ``grouping="residues"``, the atoms whose residue centre-of-mass
        positions are sampled).
    dim : int, optional
        Confined-axis index (0=x, 1=y, 2=z).  Default 2 (z).
    z_bot, z_top : float, optional
        Boundaries of the confined region.  If ``None`` they default to
        ``0`` and the box length along ``dim``.
    bins : int, "quantile", or array_like, optional
        ``int`` — N uniform u-space bins (CIC assignment).  Default 30.
        ``"quantile"`` — shortcut for 30 uniform u-space bins.
        ``array_like`` — explicit u-space edges in ``[0, 1]`` (hard
        assignment).
    grouping : {"atoms", "residues"}, optional
        ``"atoms"`` samples every atom's position each frame;
        ``"residues"`` samples the mass-weighted centre-of-mass of
        each residue each frame.  Default ``"atoms"``.

    Attributes
    ----------
    z_centers : numpy.ndarray
        ``(n_bins,)`` bin centres in z-space (via the inverse CDF).
    u_centers : numpy.ndarray
        ``(n_bins,)`` bin centres in u-space.
    density : numpy.ndarray
        ``(n_bins,)`` number density (particles per volume), computed
        as ``(N_total / (n_frames_used * A)) * (bin_population /
        bin_width_in_u) * rho(z_center)`` — i.e. the CDF bin population
        rescaled by the local Jacobian :math:`\rho = du/dz`.
    n_per_bin : numpy.ndarray
        ``(n_bins,)`` raw (weighted) population per bin.
    n_eff : numpy.ndarray
        ``(n_bins,)`` effective sample size (sum of CIC weights, or
        integer count for hard assignment).
    bin_edges_u : numpy.ndarray
        ``(n_bins + 1,)`` u-space bin edges.
    n_total : int
        Total number of pooled samples.
    n_frames_used : int
        Number of analysis frames pooled.
    n_per_frame : float
        Mean number of samples per frame.
    P, P_inv, rho : callables
        CDF, inverse-CDF and equilibrium density closures.

    Notes
    -----
    Confined-axis positions are wrapped into the primary box cell along
    ``dim`` before pooling.  For ``grouping="residues"`` the
    centre-of-mass is computed with the atoms' topology masses.

    Examples
    --------
    ::

        import MDAnalysis as mda
        from qdiffusivity import TransverseDensityBinned

        u = mda.Universe("topology.data", "trajectory.xtc")
        ag = u.select_atoms("type 1 2")
        binned = TransverseDensityBinned(
            ag, dim=2, z_bot=10.0, z_top=90.0, bins=30,
        )
        binned.run()
        # binned.density is in particles/Å^3.
    """

    def __init__(
        self,
        atomgroup,
        *,
        dim: int = 2,
        z_bot=None,
        z_top=None,
        bins=30,
        grouping: str = "atoms",
    ):
        super().__init__(atomgroup.universe.trajectory)
        self._ag = atomgroup
        if dim not in (0, 1, 2):
            raise ValueError(f"dim must be 0, 1 or 2, got {dim}")
        self._dim = dim
        self._z_bot_user = z_bot
        self._z_top_user = z_top
        self._n_bins, self._edges_u, self._use_cic = resolve_bins(bins)
        if grouping not in ("atoms", "residues"):
            raise ValueError(
                f"grouping must be 'atoms' or 'residues', got {grouping!r}"
            )
        self._grouping = grouping

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
        self._u_frames = []

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
            self._u_frames.append(com_z)
        else:
            z = self._ag.positions[:, self._dim].astype(np.float64).copy()
            if L > 0:
                z %= L
            self._u_frames.append(z)

    def _conclude(self):
        z_frames = (
            np.concatenate(self._u_frames)
            if self._u_frames
            else np.empty(0, dtype=np.float64)
        )
        self.n_total = int(z_frames.size)
        self.n_frames_used = len(self._u_frames)
        self.n_per_frame = (
            self.n_total / self.n_frames_used if self.n_frames_used > 0 else 0.0
        )

        self.P, self.P_inv, self.rho, _, _, _ = build_cdf(z_frames)
        self.u_centers = _bin_centers_from_edges(self._edges_u)
        self.z_centers = self.P_inv(self.u_centers)
        n_bins = self._n_bins

        if z_frames.size == 0:
            self.density = np.zeros(n_bins, dtype=np.float64)
            self.n_per_bin = np.zeros(n_bins, dtype=np.float64)
            self.n_eff = np.zeros(n_bins, dtype=np.float64)
            return

        u_samples = self.P(z_frames)

        if self._use_cic:
            k0, k1, w0, w1 = cic_assign(u_samples, n_bins)
            pop = np.zeros(n_bins, dtype=np.float64)
            np.add.at(pop, k0, w0)
            np.add.at(pop, k1, w1)
            self.n_eff = pop.copy()
        else:
            # Hard assignment via digitize.  Samples exactly on the
            # upper edge (u == 1) are folded into the last bin.
            idx = np.clip(
                np.digitize(u_samples, self._edges_u) - 1,
                0,
                n_bins - 1,
            )
            pop = np.bincount(idx, minlength=n_bins).astype(np.float64)
            self.n_eff = pop.copy()

        self.n_per_bin = pop

        # Convert bin population to a number density.
        #   n(z) = (N_total / (n_frames * A)) * rho(z)
        # because in u-space the population per bin is ~N_total * du,
        # and du = rho(z) * dz, so the z-space number density is
        # (N_total / (n_frames * A)) * (pop / N_total) / dz
        # = pop / (n_frames * A * dz), with dz = du / rho(z).
        A = 1.0
        dims = self._ag.universe.dimensions
        if dims is not None:
            para = [i for i in (0, 1, 2) if i != self._dim]
            A = float(dims[para[0]] * dims[para[1]])
        rho_centers = self.rho(self.z_centers)
        du = np.diff(self._edges_u)
        with np.errstate(divide="ignore", invalid="ignore"):
            dz = np.where(rho_centers > 0, du / rho_centers, 0.0)
            self.density = np.where(
                dz > 0,
                pop / (self.n_frames_used * A * dz),
                0.0,
            )


class TransverseDiffusivityBinned(AnalysisBase):
    r"""CDF-binned transverse diffusivity estimator.

    Pools per-frame positions of ``atomgroup``, builds a CDF-uniformised
    u-space, and evaluates a binned local estimator for the perpendicular
    (:math:`D_\perp`) and parallel (:math:`D_\parallel`) diffusivities.
    Samples are assigned to u-space bins via cloud-in-cell (CIC) for
    integer ``bins`` or hard assignment for explicit edges.

    The perpendicular estimator uses the z-space local estimator
    :math:`(\Delta z)^2/(2\Delta t)`, binned in u-space.  The parallel
    estimator uses
    :math:`(\Delta x^2+\Delta y^2)/(4\Delta t)`, binned by the
    starting position in u-space.

    Parameters
    ----------
    atomgroup : MDAnalysis.core.groups.AtomGroup
        Atoms whose positions are sampled.
    dim : int, optional
        Confined-axis index (0=x, 1=y, 2=z).  Default 2 (z).
    n_points : int, optional
        Number of evaluation grid points ``M`` (uniform in u-space).
        Default 200.
    bins : int, "quantile", or array_like, optional
        ``int`` — N uniform u-space bins (CIC assignment).  Default 30.
        ``"quantile"`` — shortcut for 30 uniform u-space bins.
        ``array_like`` — explicit u-space edges in ``[0, 1]`` (hard
        assignment).
    dt : float or None, optional
        Time step between consecutive frames.  If ``None`` (default) the
        trajectory's ``dt`` (:attr:`ts.dt`) is used.
    ito_correction : bool, optional
        If ``True``, subtract the :math:`O(\Delta t)` Itô bias
        :math:`\frac{\Delta t}{2}\Phi(z)^2` (with
        :math:`\Phi = D\,\rho'/\rho` in the isothermal convention) from
        the perpendicular estimator.  Default ``False``.

    Attributes
    ----------
    z_centers : numpy.ndarray
        ``(n_bins,)`` bin centres in z-space (via the inverse CDF).
    u_centers : numpy.ndarray
        ``(n_bins,)`` bin centres in u-space.
    D_perp : numpy.ndarray
        ``(n_bins,)`` perpendicular diffusivity (length²/time), clipped
        to be non-negative.  When ``ito_correction=True`` the Itô bias
        is subtracted before clipping.
    D_perp_std : numpy.ndarray
        ``(n_bins,)`` standard error of :math:`D_\perp`.
    n_eff_perp : numpy.ndarray
        ``(n_bins,)`` effective sample size for :math:`D_\perp`.
    D_para : numpy.ndarray
        ``(n_bins,)`` parallel diffusivity (length²/time).
    D_para_std : numpy.ndarray
        ``(n_bins,)`` standard error of :math:`D_\parallel`.
    n_eff_para : numpy.ndarray
        ``(n_bins,)`` effective sample size for :math:`D_\parallel`.
    ito_bias : numpy.ndarray or None
        ``(n_bins,)`` Itô bias subtracted from :math:`D_\perp` when
        ``ito_correction=True``; ``None`` otherwise.
    bin_edges_u : numpy.ndarray
        ``(n_bins + 1,)`` u-space bin edges.
    n_increments : int
        Total number of trajectory increments pooled.
    n_frames_used : int
        Number of frames pooled.
    dt : float
        Time step used in the estimator.
    P, P_inv, rho : callables
        CDF, inverse-CDF and equilibrium density closures.

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
        from qdiffusivity import TransverseDiffusivityBinned

        u = mda.Universe("topology.data", "trajectory.xtc")
        ag = u.select_atoms("type 1 2")
        binned = TransverseDiffusivityBinned(
            ag, dim=2, bins=30, ito_correction=True,
        )
        binned.run()
    """

    def __init__(
        self,
        atomgroup,
        *,
        dim: int = 2,
        bins=30,
        dt=None,
        ito_correction: bool = False,
    ):
        super().__init__(atomgroup.universe.trajectory)
        self._ag = atomgroup
        if dim not in (0, 1, 2):
            raise ValueError(f"dim must be 0, 1 or 2, got {dim}")
        self._dim = dim
        self._para_dims = tuple(i for i in (0, 1, 2) if i != dim)
        self._n_bins, self._edges_u, self._use_cic = resolve_bins(bins)
        self._dt = dt
        self._ito_correction = bool(ito_correction)

    def _prepare(self):
        self._pos_frames = []

    def _single_frame(self):
        self._pos_frames.append(self._ag.positions.copy())

    def _conclude(self):
        pos = np.asarray(self._pos_frames, dtype=np.float64)
        n_frames = pos.shape[0]
        self.n_frames_used = n_frames
        if n_frames < 2:
            raise ValueError(
                f"need at least 2 frames to form a displacement; got {n_frames}"
            )

        dt = self._dt if self._dt is not None else float(self._ts.dt)
        self.dt = dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Unwrap parallel coordinates via NoJump.
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

        # Build CDF from pooled wrapped-z.
        z_pooled = pos_unwrapped[:, :, self._dim].ravel()
        self.P, self.P_inv, self.rho, self.rho_prime, _, _ = build_cdf(z_pooled)

        self.u_centers = _bin_centers_from_edges(self._edges_u)
        self.z_centers = self.P_inv(self.u_centers)
        n_bins = self._n_bins

        # u-positions of all increments (starting frame of each pair).
        z_start = pos_unwrapped[:-1, :, self._dim]
        u_start = self.P(z_start.ravel())

        # Perpendicular: z-space local estimator (Δz)²/(2Δt).
        dz = np.diff(pos_unwrapped[:, :, self._dim], axis=0)
        d_perp_local = (dz**2) / (2.0 * dt)
        self.n_increments = int(d_perp_local.size)

        self.D_perp, self.D_perp_std, self.n_eff_perp = _bin_local_estimator(
            u_start,
            d_perp_local.ravel(),
            self._edges_u,
            n_bins,
            self._use_cic,
        )

        # Itô bias correction (perpendicular only; parallel has none).
        if self._ito_correction:
            rho_eval = self.rho(self.z_centers)
            rho_prime_eval = self.rho_prime(self.z_centers)
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

        # Parallel: Δx² + Δy², only z_start mapped to u.
        d_para_acc = np.zeros_like(dz)
        for d in self._para_dims:
            diff_d = np.diff(pos_unwrapped[:, :, d], axis=0)
            d_para_acc += diff_d**2
        d_para_local = d_para_acc / (4.0 * dt)

        self.D_para, self.D_para_std, self.n_eff_para = _bin_local_estimator(
            u_start,
            d_para_local.ravel(),
            self._edges_u,
            n_bins,
            self._use_cic,
        )


def _bin_local_estimator(u_data, d_data, edges_u, n_bins, use_cic):
    """Bin a local estimator in u-space and return (D, D_std, n_eff).

    Parameters
    ----------
    u_data : array_like
        ``(N,)`` u-space positions of the increments.
    d_data : array_like
        ``(N,)`` local estimator values (already carrying the
        ``1/(2Δt)`` or ``1/(4Δt)`` prefactor).
    edges_u : array_like
        ``(n_bins + 1,)`` u-space bin edges in ``[0, 1]``.
    n_bins : int
        Number of bins.
    use_cic : bool
        ``True`` for CIC assignment, ``False`` for hard assignment.

    Returns
    -------
    D : numpy.ndarray
        ``(n_bins,)`` weighted mean of ``d_data`` per bin.
    D_std : numpy.ndarray
        ``(n_bins,)`` standard error of the mean per bin.
    n_eff : numpy.ndarray
        ``(n_bins,)`` effective sample size per bin.
    """
    u = np.asarray(u_data, dtype=float).ravel()
    d = np.asarray(d_data, dtype=float).ravel()
    n_bins = int(n_bins)

    if u.size == 0:
        zeros = np.zeros(n_bins, dtype=np.float64)
        return zeros, zeros, zeros

    D_sum = np.zeros(n_bins, dtype=np.float64)
    D2_sum = np.zeros(n_bins, dtype=np.float64)
    w_sum = np.zeros(n_bins, dtype=np.float64)

    if use_cic:
        k0, k1, w0, w1 = cic_assign(u, n_bins)
        np.add.at(D_sum, k0, d * w0)
        np.add.at(D_sum, k1, d * w1)
        np.add.at(D2_sum, k0, d * d * w0)
        np.add.at(D2_sum, k1, d * d * w1)
        np.add.at(w_sum, k0, w0)
        np.add.at(w_sum, k1, w1)
    else:
        idx = np.clip(np.digitize(u, edges_u) - 1, 0, n_bins - 1)
        np.add.at(D_sum, idx, d)
        np.add.at(D2_sum, idx, d * d)
        np.add.at(w_sum, idx, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(w_sum > 0, D_sum / w_sum, np.nan)
        mean_d2 = np.where(w_sum > 0, D2_sum / w_sum, np.nan)
        var_d = np.where(w_sum > 0, mean_d2 - D**2, 0.0)
        var_d = np.maximum(var_d, 0.0)
        n_eff = w_sum
        D_std = np.where(
            n_eff > 1,
            np.sqrt(var_d / np.maximum(n_eff, 1.0)),
            0.0,
        )
    return D, D_std, n_eff
