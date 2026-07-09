"""Tests for the binned density/diffusivity module."""

import numpy as np
import pytest

from qdiffusivity.binned import (
    LocalDiffusivityQBinned,
    TransverseMassDensityQBinned,
    TransverseNumDensityQBinned,
    cic_assign,
    resolve_bins,
)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def test_resolve_bins():
    """int, 'quantile', explicit edges (padded/full), and invalid specs."""
    n, edges, cic = resolve_bins(20)
    assert n == 20 and edges.shape == (21,) and cic is True
    n, _, cic = resolve_bins("quantile")
    assert n == 30 and cic is True
    assert resolve_bins("quantile", n_default=15)[0] == 15
    n, edges, cic = resolve_bins(np.array([0.1, 0.3, 0.7, 0.9]))
    assert n == 5 and edges[0] == 0.0 and edges[-1] == 1.0 and cic is False
    n, edges, cic = resolve_bins(np.array([0, 0.25, 0.5, 0.75, 1.0]))
    assert n == 4 and cic is False
    for bad in [
        "bogus",
        0,
        np.array([0, 0.5, 0.3, 1.0]),
        np.array([-0.1, 0.5, 1.0]),
        np.array([0, 0.5, 1.1]),
    ]:
        with pytest.raises(ValueError):
            resolve_bins(bad)


def test_cic_assign():
    """Weights sum to 1, center -> single bin, edge -> 50/50, mirror,
    total population preserved."""
    rng = np.random.default_rng(40)
    u = rng.uniform(0.0, 1.0, size=5_000)
    k0, k1, w0, w1 = cic_assign(u, 20)
    assert np.allclose(w0 + w1, 1.0)
    assert np.all(k0 >= 0) and np.all(k0 < 20)
    _, _, w0c, w1c = cic_assign(np.array([0.525]), 20)
    assert w0c[0] == pytest.approx(1.0) and w1c[0] == pytest.approx(0.0)
    _, _, w0e, w1e = cic_assign(np.array([0.5]), 20)
    assert w0e[0] == pytest.approx(0.5) and w1e[0] == pytest.approx(0.5)
    k0m, _, w0m, _ = cic_assign(np.array([0.01]), 20)
    assert k0m[0] == 0 and w0m[0] == pytest.approx(0.7)
    pop = np.zeros(20)
    np.add.at(pop, k0, w0)
    np.add.at(pop, k1, w1)
    assert pop.sum() == pytest.approx(5_000.0)


@pytest.mark.parametrize(
    "cls",
    [TransverseNumDensityQBinned, TransverseMassDensityQBinned],
)
@pytest.mark.parametrize(
    "bins,expected_n",
    [(20, 20), ("quantile", 30), (np.array([0, 0.25, 0.5, 0.75, 1.0]), 4)],
)
@pytest.mark.parametrize("grouping", ["atoms", "residues"])
def test_density_binned(density_universe, cls, bins, expected_n, grouping):
    """Run, attributes, density positive, all bin specs, grouping."""
    u = density_universe(n_atoms=200, n_res=20, n_frames=5, Lz=100.0, seed=50)
    binned = cls(
        u.select_atoms("all"),
        dim=2,
        z_bot=0.0,
        z_top=100.0,
        bins=bins,
        grouping=grouping,
    )
    binned.run()
    assert binned.density.shape == (expected_n,)
    assert binned.z_centers.shape == (expected_n,)
    assert np.all(binned.density >= 0.0)
    assert binned.n_frames_used == 5


def test_num_vs_mass_density_binned(density_universe):
    """Mass density = num density * mean mass / N_A * 1e24."""
    u = density_universe(n_atoms=200, n_res=200, n_frames=5, Lz=100.0, seed=50)
    ag = u.select_atoms("all")
    nd = TransverseNumDensityQBinned(
        ag,
        dim=2,
        z_bot=0.0,
        z_top=100.0,
        bins=20,
    )
    nd.run()
    md = TransverseMassDensityQBinned(
        ag,
        dim=2,
        z_bot=0.0,
        z_top=100.0,
        bins=20,
    )
    md.run()
    mean_mass = float(np.mean(ag.masses))
    from MDAnalysis.units import constants

    conv = constants["N_Avogadro"] * 1e-24
    ratio = md.density / nd.density
    expected = mean_mass / conv
    assert np.allclose(ratio[nd.density > 0], expected, rtol=0.01)


@pytest.mark.parametrize(
    "bins,expected_n",
    [(20, 20), ("quantile", 30), (np.array([0.1, 0.3, 0.7, 0.9]), 5)],
)
def test_diffusivity_binned(diff_universe, bins, expected_n):
    """Run, attributes, known-D recovery, all bin specs."""
    D_perp_true, D_para_true = 0.05, 0.1
    u = diff_universe(
        n_atoms=400,
        n_frames=40,
        Lz=60.0,
        D_perp=D_perp_true,
        D_para=D_para_true,
        seed=61,
    )
    binned = LocalDiffusivityQBinned(u.select_atoms("all"), dim=2, bins=bins)
    binned.run()
    assert binned.D_perp.shape == (expected_n,)
    assert binned.n_increments == 400 * 39
    assert binned.dt == pytest.approx(1.0)
    bulk = (binned.z_centers > 20) & (binned.z_centers < 40)
    valid = binned.n_eff_perp > 5
    assert np.median(binned.D_perp[bulk & valid]) == pytest.approx(D_perp_true, rel=0.3)
    assert np.median(binned.D_para[bulk & valid]) == pytest.approx(D_para_true, rel=0.3)


@pytest.mark.parametrize(
    "cls,kwargs,use_diff",
    [
        (TransverseNumDensityQBinned, {"dim": 5}, False),
        (TransverseNumDensityQBinned, {"grouping": "molecules"}, False),
        (TransverseMassDensityQBinned, {"dim": 5}, False),
        (LocalDiffusivityQBinned, {"dim": 5}, True),
    ],
)
def test_binned_validation(density_universe, diff_universe, cls, kwargs, use_diff):
    if use_diff:
        u = diff_universe(n_atoms=10, n_frames=3, Lz=10.0)
    else:
        u = density_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    with pytest.raises(ValueError):
        cls(u.select_atoms("all"), **kwargs)


def test_diffusivity_binned_single_frame_raises(diff_universe):
    u = diff_universe(n_atoms=10, n_frames=1, Lz=10.0)
    binned = LocalDiffusivityQBinned(u.select_atoms("all"), bins=10)
    with pytest.raises(ValueError):
        binned.run()


def test_diffusivity_binned_explicit_dt(diff_universe):
    u = diff_universe(n_atoms=50, n_frames=5, Lz=40.0, seed=68)
    binned = LocalDiffusivityQBinned(
        u.select_atoms("all"),
        dim=2,
        bins=10,
        dt=2.0,
    )
    binned.run()
    assert binned.dt == pytest.approx(2.0)


@pytest.mark.parametrize("ito", [False, True])
def test_diffusivity_binned_ito(diff_universe, ito):
    """Default-off (None), on (finite non-neg), reduces D_perp, no effect
    on D_para."""
    u = diff_universe(n_atoms=200, n_frames=20, Lz=60.0, seed=66)
    ag = u.select_atoms("all")
    binned = LocalDiffusivityQBinned(ag, dim=2, bins=15, ito_correction=ito)
    binned.run()
    if not ito:
        assert binned.ito_bias is None
        return
    assert binned.ito_bias is not None
    assert np.all(binned.ito_bias >= 0)
    b_unc = LocalDiffusivityQBinned(ag, dim=2, bins=15)
    b_unc.run()
    assert np.all(binned.D_perp <= b_unc.D_perp + 1e-12)
    assert np.allclose(binned.D_para, b_unc.D_para)


def test_diffusivity_binned_density_result(diff_universe):
    """Passing a pre-computed density_result gives the same D as
    the auto-run path."""
    u = diff_universe(
        n_atoms=200,
        n_frames=20,
        Lz=60.0,
        D_perp=0.05,
        D_para=0.1,
        seed=72,
    )
    ag = u.select_atoms("all")

    dens = TransverseNumDensityQBinned(
        ag,
        dim=2,
        bins=20,
    )
    dens.run()
    assert hasattr(dens, "P") and callable(dens.P)
    assert hasattr(dens, "rho_prime") and callable(dens.rho_prime)

    b_pre = LocalDiffusivityQBinned(
        ag,
        dim=2,
        bins=20,
        density_result=dens,
    )
    b_pre.run()
    b_auto = LocalDiffusivityQBinned(
        ag,
        dim=2,
        bins=20,
    )
    b_auto.run()
    assert np.allclose(b_pre.D_perp, b_auto.D_perp, rtol=1e-10)
    assert np.allclose(b_pre.D_para, b_auto.D_para, rtol=1e-10)
