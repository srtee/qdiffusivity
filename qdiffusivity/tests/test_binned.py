"""Unit and regression tests for the binned density/diffusivity module."""

import numpy as np
import pytest

import qdiffusivity
from qdiffusivity.binned import (
    TransverseDensityBinned,
    TransverseDiffusivityBinned,
    cic_assign,
    resolve_bins,
)

# ---------------------------------------------------------------------------
# resolve_bins
# ---------------------------------------------------------------------------


def test_resolve_bins_int():
    n, edges, cic = resolve_bins(20)
    assert n == 20
    assert edges.shape == (21,)
    assert edges[0] == 0.0
    assert edges[-1] == 1.0
    assert np.allclose(np.diff(edges), 1.0 / 20)
    assert cic is True


def test_resolve_bins_quantile_string():
    n, edges, cic = resolve_bins("quantile")
    assert n == 30
    assert edges.shape == (31,)
    assert cic is True


def test_resolve_bins_quantile_string_custom_default():
    n, _, _ = resolve_bins("quantile", n_default=15)
    assert n == 15


def test_resolve_bins_explicit_edges():
    edges_in = np.array([0.1, 0.3, 0.7, 0.9])
    n, edges, cic = resolve_bins(edges_in)
    assert n == 5  # padded with 0 and 1
    assert edges[0] == 0.0
    assert edges[-1] == 1.0
    assert cic is False


def test_resolve_bins_explicit_edges_full():
    edges_in = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    n, edges, cic = resolve_bins(edges_in)
    assert n == 4
    assert np.allclose(edges, edges_in)
    assert cic is False


def test_resolve_bins_invalid_string():
    with pytest.raises(ValueError):
        resolve_bins("bogus")


def test_resolve_bins_invalid_int():
    with pytest.raises(ValueError):
        resolve_bins(0)


def test_resolve_bins_edges_not_increasing():
    with pytest.raises(ValueError):
        resolve_bins(np.array([0.0, 0.5, 0.3, 1.0]))


def test_resolve_bins_edges_out_of_range():
    with pytest.raises(ValueError):
        resolve_bins(np.array([-0.1, 0.5, 1.0]))
    with pytest.raises(ValueError):
        resolve_bins(np.array([0.0, 0.5, 1.1]))


# ---------------------------------------------------------------------------
# cic_assign
# ---------------------------------------------------------------------------


def test_cic_assign_weights_sum_to_one():
    rng = np.random.default_rng(40)
    u = rng.uniform(0.0, 1.0, size=5_000)
    k0, k1, w0, w1 = cic_assign(u, 20)
    assert np.allclose(w0 + w1, 1.0)
    assert np.all(k0 >= 0) and np.all(k0 < 20)
    assert np.all(k1 >= 0) and np.all(k1 < 20)


def test_cic_assign_center_goes_to_single_bin():
    # u exactly at a bin center -> all weight to that bin.
    u = np.array([0.525])  # center of bin 10 (for n=20: 10.5/20=0.525)
    k0, k1, w0, w1 = cic_assign(u, 20)
    assert k0[0] == 10
    assert w0[0] == pytest.approx(1.0)
    assert w1[0] == pytest.approx(0.0)


def test_cic_assign_edge_splits_evenly():
    # u exactly at a bin edge (midway between two centers) -> 50/50.
    u = np.array([0.5])  # edge between bin 9 and 10 (for n=20)
    k0, k1, w0, w1 = cic_assign(u, 20)
    assert w0[0] == pytest.approx(0.5)
    assert w1[0] == pytest.approx(0.5)


def test_cic_assign_mirror_boundary():
    # u near 0 should be reflected, not clamped.
    u = np.array([0.01])
    k0, k1, w0, w1 = cic_assign(u, 20)
    # q = 0.01*20 - 0.5 = -0.3 -> reflected to 0.3
    # k0=0, frac=0.3 -> w0=0.7, w1=0.3
    assert k0[0] == 0
    assert w0[0] == pytest.approx(0.7)


def test_cic_assign_total_population():
    # Total CIC weight should equal the number of samples.
    rng = np.random.default_rng(41)
    u = rng.uniform(0.0, 1.0, size=10_000)
    k0, k1, w0, w1 = cic_assign(u, 20)
    pop = np.zeros(20)
    np.add.at(pop, k0, w0)
    np.add.at(pop, k1, w1)
    assert pop.sum() == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# Shared universe builders
# ---------------------------------------------------------------------------


def _make_density_universe(
    n_atoms, n_res, n_frames, Lx=20.0, Ly=20.0, Lz=100.0, seed=0
):
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
        base = (base + rng.normal(0.0, 1.0, size=n_atoms)) % Lz
        pos[f, :, 2] = base

    reader = MemoryReader(
        pos,
        dimensions=np.tile([Lx, Ly, Lz, 90.0, 90.0, 90.0], (n_frames, 1)),
    )
    u.trajectory = reader
    return u


def _make_diffusion_universe(
    n_atoms,
    n_frames,
    Lx=30.0,
    Ly=30.0,
    Lz=100.0,
    D_perp=0.05,
    D_para=0.1,
    seed=0,
):
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    u = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_atoms,
        atom_resindex=np.arange(n_atoms),
        trajectory=True,
    )
    u.add_TopologyAttr("masses", values=np.ones(n_atoms))
    u.add_TopologyAttr("types", values=["X"] * n_atoms)
    u.add_TopologyAttr("resids", values=np.arange(1, n_atoms + 1))
    u.add_TopologyAttr("resnames", values=["RES"] * n_atoms)

    rng = np.random.default_rng(seed)
    pos = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
    z = rng.uniform(0.0, Lz, size=n_atoms)
    x = rng.uniform(0.0, Lx, size=n_atoms)
    y = rng.uniform(0.0, Ly, size=n_atoms)
    dt = 1.0
    sigma_perp = np.sqrt(2.0 * D_perp * dt)
    sigma_para = np.sqrt(2.0 * D_para * dt)
    for f in range(n_frames):
        x = (x + rng.normal(0.0, sigma_para, size=n_atoms)) % Lx
        y = (y + rng.normal(0.0, sigma_para, size=n_atoms)) % Ly
        z = (z + rng.normal(0.0, sigma_perp, size=n_atoms)) % Lz
        pos[f, :, 0] = x
        pos[f, :, 1] = y
        pos[f, :, 2] = z

    dims = np.tile([Lx, Ly, Lz, 90.0, 90.0, 90.0], (n_frames, 1))
    u.trajectory = MemoryReader(pos, dimensions=dims)
    return u


# ---------------------------------------------------------------------------
# TransverseDensityBinned
# ---------------------------------------------------------------------------


def test_density_binned_runs_and_attrs():
    u = _make_density_universe(
        n_atoms=200, n_res=10, n_frames=5, Lz=100.0, seed=50
    )
    ag = u.select_atoms("all")
    binned = TransverseDensityBinned(ag, dim=2, z_bot=0.0, z_top=100.0, bins=20)
    binned.run()
    assert binned.density.shape == (20,)
    assert binned.u_centers.shape == (20,)
    assert binned.z_centers.shape == (20,)
    assert binned.n_total == 200 * 5
    assert binned.n_frames_used == 5
    assert binned.n_per_frame == pytest.approx(200.0)


def test_density_binned_density_positive():
    u = _make_density_universe(
        n_atoms=300, n_res=15, n_frames=4, Lz=50.0, seed=51
    )
    ag = u.select_atoms("all")
    binned = TransverseDensityBinned(ag, dim=2, z_bot=0.0, z_top=50.0, bins=15)
    binned.run()
    assert np.all(binned.density >= 0.0)


def test_density_binned_quantile_string():
    u = _make_density_universe(
        n_atoms=100, n_res=10, n_frames=3, Lz=60.0, seed=52
    )
    ag = u.select_atoms("all")
    binned = TransverseDensityBinned(ag, dim=2, bins="quantile")
    binned.run()
    assert binned.density.shape == (30,)


def test_density_binned_explicit_edges():
    u = _make_density_universe(
        n_atoms=100, n_res=10, n_frames=3, Lz=60.0, seed=53
    )
    ag = u.select_atoms("all")
    edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    binned = TransverseDensityBinned(ag, dim=2, bins=edges)
    binned.run()
    assert binned.density.shape == (4,)


def test_density_binned_grouping_residues():
    u = _make_density_universe(
        n_atoms=300, n_res=30, n_frames=3, Lz=60.0, seed=54
    )
    ag = u.select_atoms("all")
    binned = TransverseDensityBinned(ag, dim=2, bins=10, grouping="residues")
    binned.run()
    assert binned.n_total == 30 * 3


def test_density_binned_dim_validation():
    u = _make_density_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    with pytest.raises(ValueError):
        TransverseDensityBinned(u.select_atoms("all"), dim=5)


def test_density_binned_grouping_validation():
    u = _make_density_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    with pytest.raises(ValueError):
        TransverseDensityBinned(u.select_atoms("all"), grouping="molecules")


# ---------------------------------------------------------------------------
# TransverseDiffusivityBinned
# ---------------------------------------------------------------------------


def test_diffusion_binned_runs_and_attrs():
    u = _make_diffusion_universe(n_atoms=100, n_frames=20, Lz=80.0, seed=60)
    ag = u.select_atoms("all")
    binned = TransverseDiffusivityBinned(ag, dim=2, bins=20)
    binned.run()
    assert binned.D_perp.shape == (20,)
    assert binned.D_para.shape == (20,)
    assert binned.z_centers.shape == (20,)
    assert binned.n_increments == 100 * 19
    assert binned.n_frames_used == 20
    assert binned.dt == pytest.approx(1.0)


def test_diffusion_binned_recovers_known_diffusivity():
    D_perp_true = 0.05
    D_para_true = 0.1
    u = _make_diffusion_universe(
        n_atoms=400,
        n_frames=40,
        Lz=60.0,
        D_perp=D_perp_true,
        D_para=D_para_true,
        seed=61,
    )
    ag = u.select_atoms("all")
    binned = TransverseDiffusivityBinned(ag, dim=2, bins=20)
    binned.run()
    bulk = (binned.z_centers > 20.0) & (binned.z_centers < 40.0)
    valid = binned.n_eff_perp > 5
    bulk_perp = np.median(binned.D_perp[bulk & valid])
    bulk_para = np.median(binned.D_para[bulk & valid])
    assert bulk_perp == pytest.approx(D_perp_true, rel=0.3)
    assert bulk_para == pytest.approx(D_para_true, rel=0.3)


def test_diffusion_binned_quantile_string():
    u = _make_diffusion_universe(n_atoms=100, n_frames=10, Lz=50.0, seed=62)
    ag = u.select_atoms("all")
    binned = TransverseDiffusivityBinned(ag, dim=2, bins="quantile")
    binned.run()
    assert binned.D_perp.shape == (30,)


def test_diffusion_binned_explicit_edges():
    u = _make_diffusion_universe(n_atoms=100, n_frames=10, Lz=50.0, seed=63)
    ag = u.select_atoms("all")
    edges = np.array([0.1, 0.3, 0.7, 0.9])
    binned = TransverseDiffusivityBinned(ag, dim=2, bins=edges)
    binned.run()
    assert binned.D_perp.shape == (5,)  # padded with 0 and 1


def test_diffusion_binned_dim_validation():
    u = _make_diffusion_universe(n_atoms=10, n_frames=3, Lz=10.0)
    with pytest.raises(ValueError):
        TransverseDiffusivityBinned(u.select_atoms("all"), dim=5)


def test_diffusion_binned_single_frame_raises():
    u = _make_diffusion_universe(n_atoms=10, n_frames=1, Lz=10.0)
    binned = TransverseDiffusivityBinned(u.select_atoms("all"), bins=10)
    with pytest.raises(ValueError):
        binned.run()


def test_diffusion_binned_ito_correction_default_off():
    u = _make_diffusion_universe(n_atoms=50, n_frames=5, Lz=40.0, seed=64)
    ag = u.select_atoms("all")
    binned = TransverseDiffusivityBinned(ag, dim=2, bins=10)
    binned.run()
    assert binned.ito_bias is None


def test_diffusion_binned_ito_correction_on():
    u = _make_diffusion_universe(n_atoms=100, n_frames=10, Lz=50.0, seed=65)
    ag = u.select_atoms("all")
    binned = TransverseDiffusivityBinned(
        ag, dim=2, bins=10, ito_correction=True
    )
    binned.run()
    assert binned.ito_bias is not None
    assert binned.ito_bias.shape == (10,)
    assert np.all(binned.ito_bias >= 0.0)


def test_diffusion_binned_ito_reduces_D_perp():
    u = _make_diffusion_universe(n_atoms=200, n_frames=20, Lz=60.0, seed=66)
    ag = u.select_atoms("all")
    b_unc = TransverseDiffusivityBinned(ag, dim=2, bins=15)
    b_unc.run()
    b_cor = TransverseDiffusivityBinned(ag, dim=2, bins=15, ito_correction=True)
    b_cor.run()
    assert np.all(b_cor.D_perp <= b_unc.D_perp + 1e-12)


def test_diffusion_binned_ito_no_effect_on_D_para():
    u = _make_diffusion_universe(n_atoms=100, n_frames=10, Lz=50.0, seed=67)
    ag = u.select_atoms("all")
    b_unc = TransverseDiffusivityBinned(ag, dim=2, bins=10)
    b_unc.run()
    b_cor = TransverseDiffusivityBinned(ag, dim=2, bins=10, ito_correction=True)
    b_cor.run()
    assert np.allclose(b_cor.D_para, b_unc.D_para)


def test_diffusion_binned_explicit_dt():
    u = _make_diffusion_universe(n_atoms=50, n_frames=5, Lz=40.0, seed=68)
    ag = u.select_atoms("all")
    binned = TransverseDiffusivityBinned(ag, dim=2, bins=10, dt=2.0)
    binned.run()
    assert binned.dt == pytest.approx(2.0)


def test_binned_exposed_from_package():
    assert hasattr(qdiffusivity, "TransverseDensityBinned")
    assert hasattr(qdiffusivity, "TransverseDiffusivityBinned")
    assert hasattr(qdiffusivity, "cic_assign")
    assert hasattr(qdiffusivity, "resolve_bins")
