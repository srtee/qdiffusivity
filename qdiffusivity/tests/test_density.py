"""Unit and regression tests for the density KDE module."""

import numpy as np
import pytest

import qdiffusivity
from qdiffusivity.density import (
    TransverseDensityQKDE,
    epanechnikov_kernel,
    kde_1d,
    select_bandwidth,
    sheather_jones_bw,
    silverman_bw,
)

# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


def test_epanechnikov_kernel_normalised():
    x = np.linspace(-2.0, 2.0, 200_001)
    h = 1.0
    integral = np.trapezoid(epanechnikov_kernel(x, h), x)
    assert integral == pytest.approx(1.0, abs=1e-4)


def test_epanechnikov_kernel_compact_support():
    h = 1.5
    x = np.array([-1.6, -1.5, -0.7, 0.0, 0.7, 1.5, 1.6])
    k = epanechnikov_kernel(x, h)
    assert k[0] == 0.0
    assert k[-1] == 0.0
    assert k[1] == pytest.approx(0.0, abs=1e-12)
    assert k[2] > 0.0
    assert k[4] > 0.0
    # Peak at zero
    assert k[3] == pytest.approx(0.75 / h)


def test_epanechnikov_kernel_nonneg():
    x = np.linspace(-3.0, 3.0, 1001)
    assert np.all(epanechnikov_kernel(x, 2.0) >= 0.0)


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------


def test_silverman_bw_basic():
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 1.0, size=10_000)
    h = silverman_bw(z)
    assert h > 0
    assert h == pytest.approx(1.06 * 1.0 * 10_000 ** (-0.2), rel=0.3)


def test_silverman_bw_small_input():
    assert silverman_bw(np.array([1.0])) == 0.1
    assert silverman_bw(np.array([])) == 0.1


def test_sheather_jones_bw_positive():
    rng = np.random.default_rng(1)
    z = rng.normal(5.0, 2.0, size=5_000)
    h = sheather_jones_bw(z, -10.0, 20.0)
    assert np.isfinite(h)
    assert h > 0


def test_sheather_jones_bw_smaller_than_silverman_for_bimodal():
    rng = np.random.default_rng(2)
    z = np.concatenate(
        [rng.normal(-3.0, 0.5, size=2_000), rng.normal(3.0, 0.5, size=2_000)]
    )
    h_sj = sheather_jones_bw(z, -10.0, 10.0)
    h_silver = silverman_bw(z)
    assert h_sj > 0
    assert np.isfinite(h_sj)
    assert h_sj <= h_silver


def test_select_bandwidth_methods():
    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 1.0, size=2_000)
    h_auto = select_bandwidth(z, -5.0, 5.0, method="auto")
    h_silver = select_bandwidth(z, -5.0, 5.0, method="silverman")
    h_fixed = select_bandwidth(z, -5.0, 5.0, method=0.42)
    assert h_auto > 0
    assert h_silver > 0
    assert h_fixed == pytest.approx(0.42)


def test_select_bandwidth_invalid_method():
    with pytest.raises(ValueError):
        select_bandwidth(np.array([1.0, 2.0]), 0.0, 5.0, method="notamethod")


# ---------------------------------------------------------------------------
# kde_1d
# ---------------------------------------------------------------------------


def test_kde_1d_integrates_to_one():
    rng = np.random.default_rng(4)
    z = rng.normal(50.0, 5.0, size=20_000)
    z_eval = np.linspace(20.0, 80.0, 2001)
    h = 0.5
    rho, n_eff = kde_1d(z, z_eval, h, 0.0, 100.0)
    assert rho.shape == z_eval.shape
    assert n_eff.shape == z_eval.shape
    integral = np.trapezoid(rho, z_eval)
    # Mirror reflection preserves the integral on the bounded grid.
    assert integral == pytest.approx(1.0, abs=0.02)


def test_kde_1d_rejects_nonpositive_bandwidth():
    with pytest.raises(ValueError):
        kde_1d(np.array([1.0, 2.0]), np.array([1.5]), 0.0, 0.0, 5.0)
    with pytest.raises(ValueError):
        kde_1d(np.array([1.0, 2.0]), np.array([1.5]), -1.0, 0.0, 5.0)


def test_kde_1d_n_eff_finite_and_positive_in_well_sampled_region():
    rng = np.random.default_rng(5)
    z = rng.normal(50.0, 2.0, size=10_000)
    z_eval = np.linspace(45.0, 55.0, 21)
    rho, n_eff = kde_1d(z, z_eval, 0.5, 0.0, 100.0)
    assert np.all(n_eff > 0)
    assert np.all(np.isfinite(n_eff))
    assert np.all(n_eff <= z.size)


def test_kde_1d_empty_data():
    rho, n_eff = kde_1d(
        np.array([]), np.linspace(0.0, 10.0, 11), 0.5, 0.0, 10.0
    )
    assert np.all(rho == 0.0)
    assert np.all(n_eff == 0.0)


# ---------------------------------------------------------------------------
# TransverseDensityQKDE AnalysisBase
# ---------------------------------------------------------------------------


def _make_confined_universe(
    n_atoms, n_res, n_frames, Lx=20.0, Ly=20.0, Lz=100.0, seed=0
):
    """Build a small Universe with masses and a multi-frame trajectory.

    Atoms are assigned to residues round-robin; all atoms share a residue
    mass so the residue COM is well-defined.
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    u = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_res,
        atom_resindex=np.arange(n_atoms) % n_res,
        trajectory=True,
    )
    u.add_TopologyAttr("masses", values=np.ones(n_atoms))
    u.add_TopologyAttr("types", values=["X"] * n_atoms)
    u.add_TopologyAttr("resids", values=np.arange(1, n_res + 1))
    u.add_TopologyAttr("resnames", values=["RES"] * n_res)

    rng = np.random.default_rng(seed)
    pos = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
    base = rng.uniform(0.0, Lz, size=n_atoms)
    for f in range(n_frames):
        pos[f, :, 0] = rng.uniform(0.0, Lx, size=n_atoms)
        pos[f, :, 1] = rng.uniform(0.0, Ly, size=n_atoms)
        # Diffuse a bit in z but keep inside the box.
        base = (base + rng.normal(0.0, 1.0, size=n_atoms)) % Lz
        pos[f, :, 2] = base

    reader = MemoryReader(
        pos, dimensions=np.tile([Lx, Ly, Lz, 90.0, 90.0, 90.0], (n_frames, 1))
    )
    u.trajectory = reader
    return u


def test_density_kde_runs_and_attrs():
    u = _make_confined_universe(n_atoms=200, n_res=10, n_frames=5, Lz=100.0)
    ag = u.select_atoms("all")
    kde = TransverseDensityQKDE(
        ag, dim=2, z_bot=0.0, z_top=100.0, n_points=80, bandwidth=0.5
    )
    kde.run()
    assert kde.rho.shape == (80,)
    assert kde.n_eff.shape == (80,)
    assert kde.z_eval.shape == (80,)
    assert kde.bandwidth == pytest.approx(0.5)
    assert kde.n_total == 200 * 5
    assert kde.n_frames_used == 5
    assert kde.n_per_frame == pytest.approx(200.0)
    assert kde.z_pooled.size == 200 * 5


def test_density_kde_rho_positive_and_normalised():
    u = _make_confined_universe(
        n_atoms=300, n_res=15, n_frames=4, Lz=50.0, seed=1
    )
    ag = u.select_atoms("all")
    kde = TransverseDensityQKDE(
        ag, dim=2, z_bot=0.0, z_top=50.0, n_points=200, bandwidth=0.4
    )
    kde.run()
    assert np.all(kde.rho >= 0.0)
    # The KDE integrates to ~1 over the evaluation grid.
    integral = np.trapezoid(kde.rho, kde.z_eval)
    assert integral == pytest.approx(1.0, abs=0.05)


def test_density_kde_auto_bandwidth():
    u = _make_confined_universe(
        n_atoms=200, n_res=10, n_frames=3, Lz=80.0, seed=2
    )
    ag = u.select_atoms("all")
    kde = TransverseDensityQKDE(
        ag, dim=2, z_bot=0.0, z_top=80.0, n_points=100, bandwidth="auto"
    )
    kde.run()
    assert np.isfinite(kde.bandwidth)
    assert kde.bandwidth > 0


def test_density_kde_grouping_residues():
    u = _make_confined_universe(
        n_atoms=300, n_res=30, n_frames=3, Lz=60.0, seed=3
    )
    ag = u.select_atoms("all")
    kde = TransverseDensityQKDE(
        ag,
        dim=2,
        z_bot=0.0,
        z_top=60.0,
        n_points=50,
        grouping="residues",
        bandwidth=0.5,
    )
    kde.run()
    # Residue COM pooling yields one sample per residue per frame.
    assert kde.n_total == 30 * 3
    assert kde.n_per_frame == pytest.approx(30.0)


def test_density_kde_dim_validation():
    u = _make_confined_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    with pytest.raises(ValueError):
        TransverseDensityQKDE(u.select_atoms("all"), dim=5)


def test_density_kde_grouping_validation():
    u = _make_confined_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    with pytest.raises(ValueError):
        TransverseDensityQKDE(u.select_atoms("all"), grouping="molecules")


def test_density_kde_bandwidth_validation():
    u = _make_confined_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    with pytest.raises(ValueError):
        TransverseDensityQKDE(u.select_atoms("all"), bandwidth="notamethod")


def test_density_kde_auto_z_boundaries():
    u = _make_confined_universe(n_atoms=100, n_res=10, n_frames=2, Lz=40.0)
    ag = u.select_atoms("all")
    kde = TransverseDensityQKDE(ag, dim=2, n_points=40, bandwidth=0.3)
    kde.run()
    assert kde.z_eval[0] > 0.0
    assert kde.z_eval[-1] < 40.0
    assert kde.z_eval[-1] - kde.z_eval[0] < 40.0


def test_density_kde_inverted_boundaries_raises():
    u = _make_confined_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    ag = u.select_atoms("all")
    kde = TransverseDensityQKDE(
        ag, dim=2, z_bot=10.0, z_top=0.0, n_points=20, bandwidth=0.5
    )
    with pytest.raises(ValueError):
        kde.run()


def test_density_kde_exposed_from_package():
    assert hasattr(qdiffusivity, "TransverseDensityQKDE")
    assert hasattr(qdiffusivity, "epanechnikov_kernel")
    assert hasattr(qdiffusivity, "kde_1d")
    assert hasattr(qdiffusivity, "select_bandwidth")
